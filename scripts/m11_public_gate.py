#!/usr/bin/env -S uv run --project backend --extra dl python
"""Benchmark one M11 candidate against PatchCore on two public VisA classes.

This is M11 evidence, not an import API.  It creates an isolated app-data directory,
registers each class's official one-class split, builds one identity prepared-input
revision, then runs PatchCore and the selected candidate against exactly the same pixels.
Each experiment runs in a fresh process so peak RSS is comparable.

    ./scripts/dinomaly-public-gate.py --data-dir /tmp/dinomaly-public-gate
    ./scripts/glass-public-gate.py --data-dir /tmp/glass-public-gate

Source images stay read-only under ``/datasets``.  The output directory contains the
isolated database, prepared pixels, checkpoints, maps, job log and ``result.json``.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import platform
import resource
import subprocess
import sys
import time
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from statistics import mean
from typing import Any

from anomaly_lab.config import Settings
from anomaly_lab.datasets.commit import commit_manifest
from anomaly_lab.datasets.reference_packs import pack_specs, scan_spec
from anomaly_lab.datasets.splitting import (
    SplitParams,
    SplitStrategy,
    plan_imported_split,
)
from anomaly_lab.db.connection import connection
from anomaly_lab.db.migrate import apply_migrations
from anomaly_lab.db.repositories import experiments as experiments_repo
from anomaly_lab.db.repositories import region_profiles as profiles_repo
from anomaly_lab.db.repositories import splits as splits_repo
from anomaly_lab.domain.entities import JobKind
from anomaly_lab.eval.runner import EvalConfig
from anomaly_lab.experiments.infer import run_infer_job
from anomaly_lab.experiments.train import run_train_job
from anomaly_lab.jobs.context import JobContext
from anomaly_lab.models.preprocessing import PreprocessingConfig
from anomaly_lab.models.registry import get_model_class
from anomaly_lab.owned_storage import path_usage
from anomaly_lab.regions.preparation import (
    PreparedRegionBuild,
    load_prepared_build,
    read_build_summary,
    run_region_prepare_job,
)

REPOSITORY = Path(__file__).resolve().parent.parent
DEFAULT_CATEGORIES = ("candle", "pcb1")
SEED = 20260812
PATCHCORE_CONFIG: dict[str, Any] = {
    "backbone": "wide_resnet50_2",
    "layer_set": "layer2+layer3",
    "pretrained_backbone": True,
    "allow_downloads": True,
    "coreset_ratio": 0.1,
    "num_neighbors": 9,
    "max_bank_images": 256,
    "max_candidate_vectors": 50_000,
    "blur_sigma": 4,
    "feature_batch_size": 4,
    "seed": SEED,
}
QUALITY_FLOORS = {
    "mean_image_roc_auc": 0.80,
    "mean_pixel_roc_auc": 0.85,
    "mean_au_pro": 0.60,
}


@dataclass(frozen=True)
class CandidateSpec:
    """One reproducible candidate protocol; benchmark machinery stays method-agnostic."""

    key: str
    label: str
    family: str
    prepared_size: int
    config: dict[str, Any]
    # Which config field a `--steps` smoke override writes to, or None for a method with
    # no step budget.  Without this the override would be silently ignored: the config
    # models default to pydantic's extra="ignore", so an unknown "max_steps" key would
    # validate cleanly and the smoke run would quietly be a full run.
    step_field: str | None = "max_steps"


CANDIDATES = {
    "glass_anomalib": CandidateSpec(
        key="glass_anomalib",
        label="GLASS",
        family="learned-synthesis",
        prepared_size=288,
        config={
            "max_steps": 5_000,
            "center_images": 128,
            "center_refresh_steps": 392,
            "mining_steps": 20,
            "synthesis_anchor": "sample",
            "learning_rate": 1e-4,
            "allow_downloads": True,
            "seed": SEED,
        },
    ),
    # The wrapper's exact protocol row — 392 px, its pinned encoder, 5000 steps, the shared
    # seed — so the custom implementation's gate reads directly against the wrapper's
    # recorded numbers in measurements.md.  Parity retires the wrapper (ADR-0008/0029).
    "dinomaly_custom": CandidateSpec(
        key="dinomaly_custom",
        label="Dinomaly (ours)",
        family="transformer-reconstruction",
        prepared_size=392,
        config={
            "encoder": "dinov2_vit_s14_reg4",
            "max_steps": 5_000,
            "allow_downloads": True,
            "seed": SEED,
        },
    ),
    # The DINO patch-memory legs share one prepared size, 448: the only size divisible by
    # both patch sizes (14*32 = 16*28), so the DINOv2 and DINOv3 rows see identical pixels.
    # The recorded promotion verdict is the DINOv2 leg — its weights are ungated, so anyone
    # can reproduce it; the DINOv3 leg needs the Meta licence and an HF_TOKEN and is
    # recorded as a second row, not as the gate.  global_knn is the gate mode: VisA is not
    # a registered benchmark, so the per-position modes are measured as a separate ablation
    # (dino_memory_local, pcb1) rather than asked a question VisA does not pose.
    "dino_memory": CandidateSpec(
        key="dino_memory",
        label="DINO patch memory",
        family="frozen-backbone patch memory",
        prepared_size=448,
        config={
            "backbone": "dinov2_vit_s14_reg4",
            "layers": "last_two",
            "pretrained_backbone": True,
            "allow_downloads": True,
            "scoring": "global_knn",
            "num_neighbors": 1,
            "coreset_ratio": 0.1,
            "max_bank_images": 256,
            "max_candidate_vectors": 50_000,
            "channel_fusion": "per_image",
            "blur_sigma": 4.0,
            "score_percentile": 100.0,
            "feature_batch_size": 4,
            "seed": SEED,
        },
        step_field=None,
    ),
    "dino_memory_v3": CandidateSpec(
        key="dino_memory",
        label="DINO patch memory (DINOv3)",
        family="frozen-backbone patch memory",
        prepared_size=448,
        config={
            "backbone": "dinov3_vit_s16",
            "layers": "last_two",
            "pretrained_backbone": True,
            "allow_downloads": True,
            "scoring": "global_knn",
            "num_neighbors": 1,
            "coreset_ratio": 0.1,
            "max_bank_images": 256,
            "max_candidate_vectors": 50_000,
            "channel_fusion": "per_image",
            "blur_sigma": 4.0,
            "score_percentile": 100.0,
            "feature_batch_size": 4,
            "seed": SEED,
        },
        step_field=None,
    ),
    "dino_memory_local": CandidateSpec(
        key="dino_memory",
        label="DINO patch memory (per-position)",
        family="frozen-backbone patch memory",
        prepared_size=448,
        config={
            "backbone": "dinov2_vit_s14_reg4",
            "layers": "last_two",
            "pretrained_backbone": True,
            "allow_downloads": True,
            "scoring": "local_knn",
            "window_radius": 1,
            "num_neighbors": 1,
            "max_bank_images": 256,
            "per_position_images": 64,
            "channel_fusion": "per_image",
            "blur_sigma": 4.0,
            "score_percentile": 100.0,
            "feature_batch_size": 4,
            "seed": SEED,
        },
        step_field=None,
    ),
}


def _settings(data_dir: Path) -> Settings:
    return Settings(
        data_dir=data_dir.resolve(),
        reference_datasets_dir=(REPOSITORY / "datasets").resolve(),
        dev_cors=False,
    )


def _empty_destination(path: Path) -> None:
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise ValueError(f"--data-dir must be absent or empty: {path}")
    path.mkdir(parents=True, exist_ok=True)


def _packages() -> dict[str, str | None]:
    resolved: dict[str, str | None] = {}
    for package in ("anomalib", "torch", "torchvision", "timm", "numpy", "pillow"):
        try:
            resolved[package] = version(package)
        except PackageNotFoundError:
            resolved[package] = None
    return resolved


def _official_dataset(settings: Settings, category: str, log: Any) -> tuple[int, int]:
    visa = next(pack for pack in pack_specs(settings) if pack.key == "visa")
    spec = next((item for item in visa.datasets if item.key == f"visa:{category}"), None)
    if spec is None:
        raise ValueError(f"unknown VisA category {category!r}")
    missing = [path for path in visa.required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"VisA pack is incomplete; missing {missing[0]}")

    print(f"Scanning VisA {category}...", file=sys.stderr)
    started = time.perf_counter()
    manifest = scan_spec(spec, lambda _fraction, _message: None)
    with connection(settings.db_path) as conn:
        committed = commit_manifest(conn, settings, manifest)
        params = SplitParams(strategy=SplitStrategy.IMPORTED, manifest_id=committed.manifest_id)
        assignments = plan_imported_split(
            conn,
            committed.dataset_id,
            manifest,
            seed=SEED,
            holdout_from_train=0.0,
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
    log.write(
        json.dumps(
            {
                "event": "import",
                "category": category,
                "dataset_id": committed.dataset_id,
                "split_id": split.id,
                "samples": len(manifest.samples),
                "elapsed_seconds": time.perf_counter() - started,
            }
        )
        + "\n"
    )
    log.flush()
    return committed.dataset_id, split.id


def _identity_profile(
    settings: Settings,
    *,
    category: str,
    dataset_id: int,
    prepared_size: int,
    job_id: int,
    log: Any,
) -> PreparedRegionBuild:
    with connection(settings.db_path) as conn:
        profile = profiles_repo.create_revision(
            conn,
            dataset_id=dataset_id,
            name=f"M11 identity {prepared_size}",
            extractor_type="identity",
            extractor_config={},
            prepared_width=prepared_size,
            prepared_height=prepared_size,
            padding_fraction=0.0,
            seed=SEED,
        )
    print(f"Preparing {category} at {prepared_size}px...", file=sys.stderr)
    with contextlib.redirect_stdout(log):
        run_region_prepare_job(
            JobContext(
                job_id=job_id,
                kind=JobKind.REGION_PREPARE,
                params={
                    "dataset_id": dataset_id,
                    "profile_id": profile.id,
                    "mode": "build",
                },
                settings=settings,
            )
        )
    summary = read_build_summary(settings, profile.id)
    if summary is None or summary.failed:
        raise RuntimeError(
            f"identity build for {category} failed "
            f"({None if summary is None else summary.failed} images)"
        )
    return load_prepared_build(settings, profile, manifest_sha256=summary.manifest_sha256)


def _create_experiment(
    settings: Settings,
    *,
    category: str,
    dataset_id: int,
    split_id: int,
    method: str,
    method_config: dict[str, Any],
    build: PreparedRegionBuild,
    candidate: CandidateSpec,
) -> int:
    model_config = (
        get_model_class(method).config_model().model_validate(method_config).model_dump(mode="json")
    )
    preprocessing = PreprocessingConfig(
        width=build.profile.prepared_width,
        height=build.profile.prepared_height,
    ).model_dump(mode="json")
    with connection(settings.db_path) as conn:
        experiment = experiments_repo.create_experiment(
            conn,
            name=f"M11 gate · {category} · {method}",
            dataset_id=dataset_id,
            split_id=split_id,
            region_profile_id=build.profile.id,
            region_manifest_sha256=build.summary.manifest_sha256,
            model_type=method,
            model_config=model_config,
            preprocessing_config=preprocessing,
            eval_config=EvalConfig().model_dump(mode="json"),
            artifact_dir="",
            notes=f"Reproducible M11 {candidate.label} public quality gate.",
        )
        artifact_dir = settings.experiment_dir(experiment.id)
        conn.execute(
            "UPDATE experiment SET artifact_dir = ? WHERE id = ?",
            (str(artifact_dir), experiment.id),
        )
    artifact_dir.mkdir(parents=True, exist_ok=True)
    return experiment.id


def _peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if platform.system() == "Darwin" else value * 1024


def _child(data_dir: Path, experiment_id: int) -> int:
    settings = _settings(data_dir)
    apply_migrations(settings.db_path)
    log_path = data_dir / "gate.log"
    started = time.perf_counter()
    with log_path.open("a", encoding="utf-8") as log, contextlib.redirect_stdout(log):
        train_started = time.perf_counter()
        train = run_train_job(
            JobContext(
                job_id=20_000 + experiment_id * 2,
                kind=JobKind.TRAIN,
                params={"experiment_id": experiment_id, "diagnostics": False},
                settings=settings,
            )
        )
        train_seconds = time.perf_counter() - train_started
        infer_started = time.perf_counter()
        infer = run_infer_job(
            JobContext(
                job_id=20_001 + experiment_id * 2,
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
        infer_seconds = time.perf_counter() - infer_started
    usage = path_usage(settings.experiment_dir(experiment_id))
    print(
        json.dumps(
            {
                "experiment_id": experiment_id,
                "train": train,
                "infer": infer,
                "train_seconds": train_seconds,
                "infer_seconds": infer_seconds,
                "total_seconds": time.perf_counter() - started,
                "peak_rss_bytes": _peak_rss_bytes(),
                "artifact_files": usage.files,
                "artifact_bytes": usage.bytes,
            },
            sort_keys=True,
        )
    )
    return 0


def _execute(data_dir: Path, experiment_id: int, log: Any) -> dict[str, Any]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--data-dir",
        str(data_dir),
        "--_experiment-id",
        str(experiment_id),
    ]
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    log.write(completed.stderr)
    log.flush()
    if completed.returncode:
        raise RuntimeError(
            f"experiment {experiment_id} failed with exit {completed.returncode}; see gate.log"
        )
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if len(lines) != 1:
        raise RuntimeError(f"experiment {experiment_id} emitted an invalid child report")
    return dict(json.loads(lines[0]))


def _metric(run: dict[str, Any], name: str) -> float | None:
    metrics = run.get("infer", {}).get("metrics", {}).get("test", {})
    value = metrics.get(name)
    if value is None:
        value = metrics.get("pixel", {}).get(name)
    return None if value is None else float(value)


def _mean_metric(categories: dict[str, Any], method: str, name: str) -> float | None:
    values = [
        value for legs in categories.values() if (value := _metric(legs[method], name)) is not None
    ]
    return mean(values) if values else None


def _decision(categories: dict[str, Any], candidate_spec: CandidateSpec) -> dict[str, Any]:
    names = ("image_roc_auc", "pixel_roc_auc", "au_pro")
    candidate = {name: _mean_metric(categories, candidate_spec.key, name) for name in names}
    baseline = {name: _mean_metric(categories, "patchcore_anomalib", name) for name in names}
    clears = (
        candidate["image_roc_auc"] is not None
        and candidate["image_roc_auc"] >= QUALITY_FLOORS["mean_image_roc_auc"]
        and candidate["pixel_roc_auc"] is not None
        and candidate["pixel_roc_auc"] >= QUALITY_FLOORS["mean_pixel_roc_auc"]
        and candidate["au_pro"] is not None
        and candidate["au_pro"] >= QUALITY_FLOORS["mean_au_pro"]
    )
    return {
        "quality_floors": QUALITY_FLOORS,
        "candidate": candidate_spec.key,
        "candidate_mean": candidate,
        "patchcore_mean": baseline,
        "candidate_minus_patchcore": {
            name: (
                candidate[name] - baseline[name]
                if candidate[name] is not None and baseline[name] is not None
                else None
            )
            for name in names
        },
        "quality_floors_clear": clears,
        "recommendation": (
            f"promote {candidate_spec.key} as the {candidate_spec.family} reference"
            if clears
            else f"do not promote {candidate_spec.label}; investigate quality first"
        ),
    }


def _run(
    data_dir: Path,
    categories: tuple[str, ...],
    steps: int | None,
    candidate_spec: CandidateSpec,
) -> int:
    _empty_destination(data_dir)
    settings = _settings(data_dir)
    settings.ensure_directories()
    apply_migrations(settings.db_path)
    candidate_config = dict(candidate_spec.config)
    if steps is not None:
        if candidate_spec.step_field is None:
            msg = f"{candidate_spec.label} has no step budget; --steps does not apply"
            raise ValueError(msg)
        candidate_config[candidate_spec.step_field] = steps
    method_configs = {
        "patchcore_anomalib": dict(PATCHCORE_CONFIG),
        candidate_spec.key: candidate_config,
    }
    report: dict[str, Any] = {
        "schema_version": 2,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source": "VisA official 1cls split",
        "categories": {},
        "candidate": candidate_spec.key,
        "methods": method_configs,
        "prepared_size": {
            "width": candidate_spec.prepared_size,
            "height": candidate_spec.prepared_size,
        },
        "seed": SEED,
        "host": {"platform": platform.platform(), "python": platform.python_version()},
        "packages": _packages(),
    }
    log_path = data_dir / "gate.log"
    with log_path.open("a", encoding="utf-8") as log:
        for job_id, category in enumerate(categories, start=1):
            dataset_id, split_id = _official_dataset(settings, category, log)
            build = _identity_profile(
                settings,
                category=category,
                dataset_id=dataset_id,
                prepared_size=candidate_spec.prepared_size,
                job_id=job_id,
                log=log,
            )
            legs: dict[str, Any] = {}
            report["categories"][category] = legs
            for method, method_config in method_configs.items():
                experiment_id = _create_experiment(
                    settings,
                    category=category,
                    dataset_id=dataset_id,
                    split_id=split_id,
                    method=method,
                    method_config=method_config,
                    build=build,
                    candidate=candidate_spec,
                )
                print(
                    f"Running {category} / {method} as experiment {experiment_id}...",
                    file=sys.stderr,
                )
                legs[method] = _execute(data_dir, experiment_id, log)
                (data_dir / "result.partial.json").write_text(
                    json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
                )
    report["decision"] = _decision(report["categories"], candidate_spec)
    output = data_dir / "result.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report["decision"], indent=2, sort_keys=True))
    print(f"Full evidence: {output}", file=sys.stderr)
    return 0 if report["decision"]["quality_floors_clear"] else 1


def main(
    argv: list[str] | None = None,
    *,
    default_candidate: str = "dinomaly_custom",
) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument(
        "--candidate",
        choices=tuple(CANDIDATES),
        default=default_candidate,
        help=f"candidate method (default: {default_candidate})",
    )
    parser.add_argument(
        "--category",
        action="append",
        choices=tuple(
            spec.key.split(":", 1)[1]
            for pack in pack_specs(_settings(Path("/tmp/m11-gate-catalog")))
            if pack.key == "visa"
            for spec in pack.datasets
        ),
        dest="categories",
        help="VisA class to include; repeat for more than one (default: candle and pcb1).",
    )
    parser.add_argument(
        "--steps",
        type=int,
        help="Override 5000 only for a smoke run; recorded quality gates use 5000.",
    )
    parser.add_argument("--_experiment-id", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args._experiment_id is not None:
        return _child(args.data_dir, args._experiment_id)
    categories = tuple(args.categories or DEFAULT_CATEGORIES)
    candidate = CANDIDATES[args.candidate]
    if args.steps is not None and candidate.step_field is None:
        parser.error(f"--steps does not apply to {args.candidate}: it has no step budget")
    # A recorded gate needs two classes; a smoke run is marked by --steps.  A candidate
    # with no step budget has no smoke knob, so its single-class runs are ablation legs
    # (e.g. the per-position mode on pcb1) rather than an under-declared gate.
    if len(categories) < 2 and args.steps is None and candidate.step_field is not None:
        parser.error("the quality gate requires at least two VisA classes")
    return _run(args.data_dir, categories, args.steps, candidate)


if __name__ == "__main__":
    raise SystemExit(main())
