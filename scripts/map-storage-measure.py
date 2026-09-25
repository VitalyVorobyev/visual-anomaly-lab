#!/usr/bin/env -S uv run --project backend python
"""Measure how a run's anomaly maps could be stored, against how they are stored now.

Protocol and decision rule: `docs/measurements.md`, "Anomaly-map storage". One public VisA
class is imported with its official split, prepared under an `identity` region profile
(deciding) and a `foreground_threshold` profile (reported, so a real crop and its NaN border
are exercised), and scored by `pixel_reference` — numpy only, so the whole measurement is
torch-free. While the infer job runs, `InferContext.write_map` is wrapped to keep a copy of
every map in the prepared frame, before the pinned transform projects it.

Every candidate format is then written from those arrays and read back:

* `a_npy`      — the projected source-frame map, float32 `.npy` (what older runs hold);
* `b_npz32`    — the same array, `np.savez_compressed`;
* `b_npz16`    — the same array as float16, `np.savez_compressed` (answers the tolerance
  question; a narrowing cast is not exact);
* `c_npz`      — the prepared-frame map plus its pinned `SpatialTransform`, `np.savez`,
  projected on read;
* `c_npz_z`    — the same, `np.savez_compressed` (the format `InferContext.write_map` writes).

Measured per format: bytes on disk per map, write ms (encode and save; the projection that
every format pays at write time for the display range and the peak is reported once), read
ms (warm, decode to the source-frame float32 array), overlay ms (decode plus
`render_anomaly_map` at the source size, which is the heatmap route's server work), the
evaluator's wall time and traced peak memory over the whole run, whether every decoded map
is bit-identical to `a_npy`, and whether the evaluator's metrics are identical.

Every candidate is read by `anomaly_lab.map_files.read_map`, the one reader every map consumer
uses, which decodes all five; the evaluator runs on each format by repointing
`image_result.map_path` at that format's files.

    ./scripts/map-storage-measure.py --data-dir /tmp/map-storage

The directory must be absent or empty. Source images stay read-only under `--datasets`.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import platform
import statistics
import sys
import time
import tracemalloc
from collections.abc import Callable, Iterator
from importlib.metadata import version
from pathlib import Path
from typing import Any
from unittest import mock

import numpy as np

from anomaly_lab.config import Settings
from anomaly_lab.datasets.commit import commit_manifest
from anomaly_lab.datasets.reference_packs import pack_specs, scan_spec
from anomaly_lab.datasets.splitting import SplitParams, SplitStrategy, plan_imported_split
from anomaly_lab.db.connection import connection
from anomaly_lab.db.migrate import apply_migrations
from anomaly_lab.db.repositories import experiments as experiments_repo
from anomaly_lab.db.repositories import region_profiles as profiles_repo
from anomaly_lab.db.repositories import results as results_repo
from anomaly_lab.db.repositories import splits as splits_repo
from anomaly_lab.domain.entities import JobKind
from anomaly_lab.eval.evaluators import evaluator_for
from anomaly_lab.eval.runner import EvalConfig
from anomaly_lab.experiments.infer import run_infer_job
from anomaly_lab.experiments.train import run_train_job
from anomaly_lab.jobs.context import JobContext
from anomaly_lab.map_files import read_map
from anomaly_lab.media.overlay import read_display_range, render_anomaly_map
from anomaly_lab.models.base import InferContext, evenly_spaced
from anomaly_lab.models.preprocessing import PreprocessingConfig
from anomaly_lab.models.registry import get_model_class
from anomaly_lab.regions.preparation import (
    PreparedRegionBuild,
    load_prepared_build,
    read_build_summary,
    run_region_prepare_job,
)
from anomaly_lab.regions.transform import SpatialTransform

REPOSITORY = Path(__file__).resolve().parent.parent
METHOD = "pixel_reference"
SEED = 20260812
PREPARED_SIZE = 256
PADDING_FRACTION = 0.05
PROFILES: tuple[tuple[str, str, bool], ...] = (
    ("identity", "identity", True),
    ("threshold", "foreground_threshold", False),
)
OVERLAY_IMAGES = 20
READ_REPEATS = 3

_NP_SAVE = np.save

Encoder = Callable[[Path, np.ndarray, np.ndarray, SpatialTransform], None]


# -- the candidate formats --------------------------------------------------------------


def _save_npy(path: Path, source: np.ndarray, prepared: np.ndarray, t: SpatialTransform) -> None:
    _NP_SAVE(path, source)


def _save_npz32(path: Path, source: np.ndarray, prepared: np.ndarray, t: SpatialTransform) -> None:
    np.savez_compressed(path, map=source)


def _save_npz16(path: Path, source: np.ndarray, prepared: np.ndarray, t: SpatialTransform) -> None:
    np.savez_compressed(path, map=source.astype(np.float16))


def _transform_bytes(transform: SpatialTransform) -> np.ndarray:
    return np.frombuffer(transform.model_dump_json().encode("utf-8"), dtype=np.uint8)


def _save_prepared(
    path: Path, source: np.ndarray, prepared: np.ndarray, t: SpatialTransform
) -> None:
    np.savez(path, map=prepared, transform=_transform_bytes(t))


def _save_prepared_z(
    path: Path, source: np.ndarray, prepared: np.ndarray, t: SpatialTransform
) -> None:
    np.savez_compressed(path, map=prepared, transform=_transform_bytes(t))


FORMATS: dict[str, tuple[str, Encoder]] = {
    "a_npy": (".npy", _save_npy),
    "b_npz32": (".npz", _save_npz32),
    "b_npz16": (".npz", _save_npz16),
    "c_npz": (".npz", _save_prepared),
    "c_npz_z": (".npz", _save_prepared_z),
}


# -- catalogue setup ----------------------------------------------------------------------


def _settings(data_dir: Path, datasets: Path) -> Settings:
    return Settings(
        data_dir=data_dir.resolve(), reference_datasets_dir=datasets.resolve(), dev_cors=False
    )


def _import_class(settings: Settings, category: str) -> tuple[int, int]:
    visa = next(pack for pack in pack_specs(settings) if pack.key == "visa")
    spec = next((item for item in visa.datasets if item.key == f"visa:{category}"), None)
    if spec is None:
        raise ValueError(f"unknown VisA category {category!r}")
    manifest = scan_spec(spec, lambda fraction, message: None)
    with connection(settings.db_path) as conn:
        committed = commit_manifest(conn, settings, manifest)
        params = SplitParams(strategy=SplitStrategy.IMPORTED, manifest_id=committed.manifest_id)
        assignments = plan_imported_split(
            conn, committed.dataset_id, manifest, seed=SEED, holdout_from_train=0.0
        )
        split = splits_repo.create_split(
            conn,
            committed.dataset_id,
            name="official 1cls",
            strategy=SplitStrategy.IMPORTED.value,
            seed=SEED,
            params=params.model_dump(mode="json"),
            assignments=assignments,
        )
    return committed.dataset_id, split.id


def _build_profile(
    settings: Settings, dataset_id: int, label: str, extractor: str, job_id: int
) -> PreparedRegionBuild:
    with connection(settings.db_path) as conn:
        profile = profiles_repo.create_revision(
            conn,
            dataset_id=dataset_id,
            name=f"map storage {label}",
            extractor_type=extractor,
            extractor_config={},
            prepared_width=PREPARED_SIZE,
            prepared_height=PREPARED_SIZE,
            padding_fraction=PADDING_FRACTION,
            seed=SEED,
        )
    run_region_prepare_job(
        JobContext(
            job_id=job_id,
            kind=JobKind.REGION_PREPARE,
            params={"dataset_id": dataset_id, "profile_id": profile.id, "mode": "build"},
            settings=settings,
        )
    )
    summary = read_build_summary(settings, profile.id)
    if summary is None or summary.failed:
        raise RuntimeError(f"profile {label} did not build cleanly")
    return load_prepared_build(settings, profile, manifest_sha256=summary.manifest_sha256)


def _create_experiment(
    settings: Settings, dataset_id: int, split_id: int, label: str, build: PreparedRegionBuild
) -> int:
    config = get_model_class(METHOD).config_model().model_validate({}).model_dump(mode="json")
    preprocessing = PreprocessingConfig(width=PREPARED_SIZE, height=PREPARED_SIZE)
    with connection(settings.db_path) as conn:
        experiment = experiments_repo.create_experiment(
            conn,
            name=f"map storage · {label}",
            dataset_id=dataset_id,
            split_id=split_id,
            region_profile_id=build.profile.id,
            region_manifest_sha256=build.summary.manifest_sha256,
            model_type=METHOD,
            model_config=config,
            preprocessing_config=preprocessing.model_dump(mode="json"),
            eval_config=EvalConfig().model_dump(mode="json"),
            artifact_dir="",
            notes="Map storage measurement.",
        )
        artifact_dir = settings.experiment_dir(experiment.id)
        conn.execute(
            "UPDATE experiment SET artifact_dir = ? WHERE id = ?",
            (str(artifact_dir), experiment.id),
        )
    artifact_dir.mkdir(parents=True, exist_ok=True)
    return experiment.id


@contextlib.contextmanager
def _capturing_prepared(target: Path) -> Iterator[None]:
    """Keep every map as the plugin handed it over, before the pinned projection."""
    original = InferContext.write_map
    target.mkdir(parents=True, exist_ok=True)

    def write_map(self: InferContext, image_id: int, array: np.ndarray) -> Path:
        _NP_SAVE(target / f"{image_id}.npy", np.ascontiguousarray(np.squeeze(array), np.float32))
        return original(self, image_id, array)

    with mock.patch.object(InferContext, "write_map", write_map):
        yield


def _score(settings: Settings, experiment_id: int, capture: Path, log: Any) -> dict[str, Any]:
    with contextlib.redirect_stdout(log):
        run_train_job(
            JobContext(
                job_id=10_000 + experiment_id * 2,
                kind=JobKind.TRAIN,
                params={"experiment_id": experiment_id, "diagnostics": False},
                settings=settings,
            )
        )
        with _capturing_prepared(capture):
            started = time.perf_counter()
            infer = run_infer_job(
                JobContext(
                    job_id=10_001 + experiment_id * 2,
                    kind=JobKind.INFER,
                    params={
                        "experiment_id": experiment_id,
                        "subsets": ["test"],
                        "diagnostics": False,
                        "diagnostic_images": 0,
                    },
                    settings=settings,
                )
            )
    infer["wall_seconds"] = time.perf_counter() - started
    return infer


# -- measuring ----------------------------------------------------------------------------


def _ms(values: list[float]) -> dict[str, float]:
    return {
        "median": statistics.median(values) * 1000.0,
        "mean": statistics.fmean(values) * 1000.0,
        "max": max(values) * 1000.0,
    }


def _bit_identical(left: np.ndarray, right: np.ndarray) -> bool:
    return (
        left.dtype == right.dtype
        and left.shape == right.shape
        and left.tobytes() == right.tobytes()
    )


def _max_abs_difference(left: np.ndarray, right: np.ndarray) -> float:
    both = np.isfinite(left) & np.isfinite(right)
    if not both.any():
        return 0.0
    return float(np.max(np.abs(left[both].astype(np.float64) - right[both].astype(np.float64))))


def _evaluate(settings: Settings, experiment_id: int) -> tuple[dict[str, Any], float]:
    with connection(settings.db_path) as conn:
        experiment = experiments_repo.get_experiment(conn, experiment_id)
        assert experiment is not None
        started = time.perf_counter()
        metrics = evaluator_for(experiment.task).evaluate(conn, experiment)
        elapsed = time.perf_counter() - started
    return {subset.value: found for subset, found in metrics.items()}, elapsed


def _repoint(settings: Settings, experiment_id: int, paths: dict[int, str]) -> None:
    with connection(settings.db_path) as conn:
        conn.executemany(
            "UPDATE image_result SET map_path = ? WHERE experiment_id = ? AND image_id = ?",
            [(path, experiment_id, image_id) for image_id, path in paths.items()],
        )


def _measure_leg(
    settings: Settings,
    experiment_id: int,
    build: PreparedRegionBuild,
    capture: Path,
    root: Path,
) -> dict[str, Any]:
    with connection(settings.db_path) as conn:
        scored = [
            image
            for image in results_repo.list_scored_images(conn, experiment_id)
            if image.map_path
        ]
    stored = {image.image_id: str(image.map_path) for image in scored}
    ids = sorted(stored)
    sources = {image_id: read_map(stored[image_id]) for image_id in ids}
    prepared = {image_id: read_map(capture / f"{image_id}.npy") for image_id in ids}

    projection_seconds: list[float] = []
    for image_id in ids:
        started = time.perf_counter()
        projected = build.transform_for(image_id).project_map(prepared[image_id])
        projection_seconds.append(time.perf_counter() - started)
        if not _bit_identical(projected, sources[image_id]):
            raise RuntimeError(f"re-projecting image {image_id} does not reproduce its stored map")

    maps_dir = settings.experiment_dir(experiment_id) / "maps"
    value_range = read_display_range(maps_dir)
    overlay_ids = [ids[index] for index in evenly_spaced(len(ids), OVERLAY_IMAGES)]
    with connection(settings.db_path) as conn:
        sizes = {
            image.image_id: (image.width, image.height)
            for image in results_repo.list_scored_images(conn, experiment_id)
        }

    report: dict[str, Any] = {
        "images": len(ids),
        "source_shape": list(next(iter(sources.values())).shape),
        "prepared_shape": list(next(iter(prepared.values())).shape),
        "projection_ms": _ms(projection_seconds),
        "formats": {},
    }
    baseline_metrics: dict[str, Any] | None = None
    for name, (suffix, encode) in FORMATS.items():
        directory = root / name
        directory.mkdir(parents=True, exist_ok=True)
        paths = {image_id: directory / f"{image_id}{suffix}" for image_id in ids}

        write_seconds = []
        for image_id in ids:
            transform = build.transform_for(image_id)
            started = time.perf_counter()
            encode(paths[image_id], sources[image_id], prepared[image_id], transform)
            write_seconds.append(time.perf_counter() - started)
        sizes_on_disk = [paths[image_id].stat().st_size for image_id in ids]

        identical = 0
        worst = 0.0
        for image_id in ids:
            decoded = read_map(paths[image_id])
            if _bit_identical(decoded, sources[image_id]):
                identical += 1
            else:
                worst = max(worst, _max_abs_difference(decoded, sources[image_id]))

        read_seconds: list[float] = []
        for _ in range(READ_REPEATS):
            for image_id in ids:
                started = time.perf_counter()
                read_map(paths[image_id])
                read_seconds.append(time.perf_counter() - started)

        overlay_seconds: list[float] = []
        for _ in range(READ_REPEATS):
            for image_id in overlay_ids:
                started = time.perf_counter()
                render_anomaly_map(
                    read_map(paths[image_id]), value_range=value_range, size=sizes[image_id]
                )
                overlay_seconds.append(time.perf_counter() - started)

        _repoint(settings, experiment_id, {key: str(value) for key, value in paths.items()})
        try:
            metrics, evaluate_seconds = _evaluate(settings, experiment_id)
            tracemalloc.start()
            _evaluate(settings, experiment_id)
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
        finally:
            _repoint(settings, experiment_id, stored)
        if baseline_metrics is None:
            baseline_metrics = metrics
        same_metrics = json.dumps(metrics, sort_keys=True) == json.dumps(
            baseline_metrics, sort_keys=True
        )
        headline = metrics.get("test", {})
        report["formats"][name] = {
            "bytes_per_map": statistics.fmean(sizes_on_disk),
            "bytes_total": sum(sizes_on_disk),
            "write_ms": _ms(write_seconds),
            "read_ms": _ms(read_seconds),
            "overlay_ms": _ms(overlay_seconds),
            "evaluate_seconds": evaluate_seconds,
            "evaluate_peak_traced_bytes": peak,
            "maps_bit_identical": identical,
            "max_abs_difference": worst,
            "metrics_identical": same_metrics,
            "pixel_roc_auc": headline.get("pixel", {}).get("pixel_roc_auc"),
            "au_pro": headline.get("pixel", {}).get("au_pro"),
            "image_roc_auc": headline.get("image_roc_auc"),
        }
        print(f"  {name}: {json.dumps(report['formats'][name])}", file=sys.stderr)
    return report


# -- the rule -----------------------------------------------------------------------------


def _decision(leg: dict[str, Any], others: list[dict[str, Any]]) -> dict[str, Any]:
    formats = leg["formats"]
    base = formats["a_npy"]
    verdicts: dict[str, Any] = {}
    for name, found in formats.items():
        if name == "a_npy":
            continue
        exact = (
            found["maps_bit_identical"] == leg["images"]
            and found["metrics_identical"]
            and all(
                other["formats"][name]["maps_bit_identical"] == other["images"]
                and other["formats"][name]["metrics_identical"]
                for other in others
            )
        )
        overlay_ratio = found["overlay_ms"]["median"] / base["overlay_ms"]["median"]
        evaluate_ratio = found["evaluate_seconds"] / base["evaluate_seconds"]
        memory_ratio = found["evaluate_peak_traced_bytes"] / base["evaluate_peak_traced_bytes"]
        size_ratio = found["bytes_per_map"] / base["bytes_per_map"]
        checks = {
            "exact": exact,
            "overlay_within_1_5x": overlay_ratio <= 1.5,
            "evaluate_within_2x": evaluate_ratio <= 2.0,
            "memory_within_1_5x": memory_ratio <= 1.5,
            "at_most_half_the_bytes": size_ratio <= 0.5,
        }
        verdicts[name] = {
            "size_ratio": size_ratio,
            "overlay_ratio": overlay_ratio,
            "evaluate_ratio": evaluate_ratio,
            "memory_ratio": memory_ratio,
            "checks": checks,
            "eligible": all(checks.values()),
        }
    eligible = [name for name, verdict in verdicts.items() if verdict["eligible"]]
    adopted = min(eligible, key=lambda name: formats[name]["bytes_per_map"]) if eligible else None
    return {"formats": verdicts, "adopted": adopted}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--datasets", type=Path, default=REPOSITORY / "datasets")
    parser.add_argument("--category", default="candle")
    args = parser.parse_args(argv)

    data_dir: Path = args.data_dir
    if data_dir.exists() and any(data_dir.iterdir()):
        raise SystemExit(f"--data-dir must be absent or empty: {data_dir}")
    data_dir.mkdir(parents=True, exist_ok=True)
    settings = _settings(data_dir, args.datasets)
    settings.ensure_directories()
    apply_migrations(settings.db_path)

    report: dict[str, Any] = {
        "category": args.category,
        "method": METHOD,
        "prepared_size": PREPARED_SIZE,
        "seed": SEED,
        "host": {"platform": platform.platform(), "python": platform.python_version()},
        "packages": {name: version(name) for name in ("numpy", "pillow")},
        "legs": {},
    }
    print(f"Importing VisA {args.category}...", file=sys.stderr)
    dataset_id, split_id = _import_class(settings, args.category)
    with (data_dir / "measure.log").open("a", encoding="utf-8") as log:
        for job_id, (label, extractor, _deciding) in enumerate(PROFILES, start=1):
            print(f"Preparing and scoring under {label}...", file=sys.stderr)
            with contextlib.redirect_stdout(log):
                build = _build_profile(settings, dataset_id, label, extractor, job_id)
            experiment_id = _create_experiment(settings, dataset_id, split_id, label, build)
            capture = data_dir / "capture" / str(experiment_id)
            infer = _score(settings, experiment_id, capture, log)
            print(f"Measuring {label} (experiment {experiment_id})...", file=sys.stderr)
            leg = _measure_leg(
                settings, experiment_id, build, capture, data_dir / "formats" / str(experiment_id)
            )
            leg["infer_seconds"] = infer["wall_seconds"]
            report["legs"][label] = leg

    deciding = next(label for label, _, deciding in PROFILES if deciding)
    others = [leg for label, leg in report["legs"].items() if label != deciding]
    report["decision"] = _decision(report["legs"][deciding], others)
    output = data_dir / "result.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report["decision"], indent=2, sort_keys=True))
    print(f"Full evidence: {output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
