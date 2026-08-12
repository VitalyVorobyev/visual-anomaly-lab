#!/usr/bin/env -S uv run --project backend --extra dl python
"""Run the M10 identity-versus-localisation value gate on public VisA classes.

This is a controlled experiment, not a benchmark-import path. It creates an isolated app
data directory, adopts each class's official one-class split, builds full-frame and
localised region profiles, and runs the same PatchCore configuration and seed on both.
The localiser is either the classical foreground-threshold baseline or MobileSAM. Each
experiment runs in a fresh process so peak resident memory is comparable.

    ./scripts/region-value-gate.py --data-dir /tmp/anomaly-region-gate

The directory must be absent or empty. Source images remain read-only under ``/datasets``;
prepared pixels, models, maps, logs and ``result.json`` live under ``--data-dir``.
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
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from statistics import mean, median
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
from anomaly_lab.model_assets.catalog import get_spec
from anomaly_lab.model_assets.store import set_external_source
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
METHOD = "patchcore_anomalib"
SEED = 20260812
PREPARED_SIZE = 256
PADDING_FRACTION = 0.05
LOCALIZER_CONFIGS: dict[str, dict[str, Any]] = {
    "foreground_threshold": {},
    "mobile_sam": {
        "points_per_side": 4,
        "points_per_batch": 16,
        "predicted_iou_threshold": 0.72,
        "stability_threshold": 0.82,
        "min_area_fraction": 0.02,
        "max_area_fraction": 0.98,
    },
}
MODEL_CONFIG: dict[str, Any] = {
    "backbone": "wide_resnet50_2",
    "layer_set": "layer2+layer3",
    "pretrained_backbone": True,
    "allow_downloads": True,
    "coreset_ratio": 0.1,
    "num_neighbors": 9,
    "max_bank_images": 256,
    "max_candidate_vectors": 50_000,
    "blur_sigma": 4,
    "feature_batch_size": 8,
    "seed": SEED,
}
DEFAULT_CRITERION = {
    "minimum_mean_pixel_roc_auc_gain": 0.01,
    "minimum_mean_au_pro_gain": 0.01,
    "maximum_single_class_primary_loss": 0.02,
    "required_build_failure_rate": 0.0,
}


def _package_versions() -> dict[str, str | None]:
    packages = (
        "anomalib",
        "torch",
        "torchvision",
        "timm",
        "numpy",
        "pillow",
        "mobile-sam",
    )
    resolved: dict[str, str | None] = {}
    for package in packages:
        try:
            resolved[package] = version(package)
        except PackageNotFoundError:
            resolved[package] = None
    return resolved


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


def _official_dataset(
    settings: Settings, category: str, *, log: Any
) -> tuple[int, int]:
    visa = next(pack for pack in pack_specs(settings) if pack.key == "visa")
    spec = next(
        (item for item in visa.datasets if item.key == f"visa:{category}"), None
    )
    if spec is None:
        raise ValueError(f"unknown VisA category {category!r}")
    missing = [path for path in visa.required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"VisA pack is incomplete; missing {missing[0]}")

    print(f"Scanning VisA {category}...", file=sys.stderr)
    started = time.perf_counter()
    manifest = scan_spec(spec, lambda fraction, message: None)
    with connection(settings.db_path) as conn:
        committed = commit_manifest(conn, settings, manifest)
        params = SplitParams(
            strategy=SplitStrategy.IMPORTED,
            manifest_id=committed.manifest_id,
        )
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


def _build_profile(
    settings: Settings,
    *,
    category: str,
    dataset_id: int,
    label: str,
    extractor_type: str,
    extractor_config: dict[str, Any],
    job_id: int,
    log: Any,
) -> tuple[PreparedRegionBuild | None, dict[str, Any]]:
    with connection(settings.db_path) as conn:
        profile = profiles_repo.create_revision(
            conn,
            dataset_id=dataset_id,
            name=f"gate {label}",
            extractor_type=extractor_type,
            extractor_config=extractor_config,
            prepared_width=PREPARED_SIZE,
            prepared_height=PREPARED_SIZE,
            padding_fraction=PADDING_FRACTION,
            seed=SEED,
        )
    print(f"Preparing {category} / {label}...", file=sys.stderr)
    with contextlib.redirect_stdout(log):
        payload = run_region_prepare_job(
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
    if summary is None:
        raise RuntimeError(f"profile {profile.id} did not publish a build summary")
    report: dict[str, Any] = {
        "profile_id": profile.id,
        "extractor": extractor_type,
        "total": summary.total,
        "succeeded": summary.succeeded,
        "failed": summary.failed,
        "failure_rate": summary.failed / max(summary.total, 1),
        "elapsed_seconds": summary.elapsed_ms / 1000.0,
        "storage_files": summary.storage_files,
        "storage_bytes": summary.storage_bytes,
        "job_payload": payload,
    }
    if summary.failed:
        report["failure_examples"] = [
            {"image_id": item.image_id, "error": item.error}
            for item in summary.failure_examples
        ]
        return None, report
    build = load_prepared_build(
        settings, profile, manifest_sha256=summary.manifest_sha256
    )
    crop_fractions = [
        (entry.transform.crop_width * entry.transform.crop_height)
        / (entry.transform.source_width * entry.transform.source_height)
        for entry in build.entries.values()
        if entry.transform is not None
    ]
    extraction_fractions = [
        float(entry.extractor_metadata["coverage_fraction"])
        for entry in build.entries.values()
        if "coverage_fraction" in entry.extractor_metadata
    ]
    report["crop_fraction"] = _distribution(crop_fractions)
    report["extraction_fraction"] = _distribution(extraction_fractions)
    return build, report


def _distribution(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "median": None, "min": None, "max": None}
    return {
        "mean": mean(values),
        "median": median(values),
        "min": min(values),
        "max": max(values),
    }


def _create_experiment(
    settings: Settings,
    *,
    category: str,
    dataset_id: int,
    split_id: int,
    label: str,
    build: PreparedRegionBuild,
) -> int:
    model_config = (
        get_model_class(METHOD)
        .config_model()
        .model_validate(MODEL_CONFIG)
        .model_dump(mode="json")
    )
    preprocessing = PreprocessingConfig(
        width=build.profile.prepared_width,
        height=build.profile.prepared_height,
    ).model_dump(mode="json")
    evaluation = EvalConfig().model_dump(mode="json")
    with connection(settings.db_path) as conn:
        experiment = experiments_repo.create_experiment(
            conn,
            name=f"M10 gate · {category} · {label}",
            dataset_id=dataset_id,
            split_id=split_id,
            region_profile_id=build.profile.id,
            region_manifest_sha256=build.summary.manifest_sha256,
            model_type=METHOD,
            model_config=model_config,
            preprocessing_config=preprocessing,
            eval_config=evaluation,
            artifact_dir="",
            notes="Reproducible M10 region value gate.",
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


def _run_leg(data_dir: Path, experiment_id: int) -> int:
    settings = _settings(data_dir)
    apply_migrations(settings.db_path)
    log_path = data_dir / "gate.log"
    started = time.perf_counter()
    with log_path.open("a", encoding="utf-8") as log, contextlib.redirect_stdout(log):
        train_started = time.perf_counter()
        train = run_train_job(
            JobContext(
                job_id=10_000 + experiment_id * 2,
                kind=JobKind.TRAIN,
                params={"experiment_id": experiment_id, "diagnostics": False},
                settings=settings,
            )
        )
        train_seconds = time.perf_counter() - train_started
        infer_started = time.perf_counter()
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


def _prewarm_backbone(log: Any) -> float:
    """Resolve shared weights outside every timed leg."""
    print("Prewarming the shared PatchCore backbone...", file=sys.stderr)
    started = time.perf_counter()
    with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        import timm

        model = timm.create_model(MODEL_CONFIG["backbone"], pretrained=True)
        del model
    return time.perf_counter() - started


def _execute_leg(data_dir: Path, experiment_id: int, log: Any) -> dict[str, Any]:
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
    if completed.returncode != 0:
        raise RuntimeError(
            f"experiment {experiment_id} failed with exit {completed.returncode}; see gate.log"
        )
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if len(lines) != 1:
        raise RuntimeError(
            f"experiment {experiment_id} emitted an invalid child report"
        )
    return dict(json.loads(lines[0]))


def _metric(leg: dict[str, Any], name: str) -> float | None:
    metrics = leg.get("infer", {}).get("metrics", {}).get("test", {})
    value = metrics.get(name)
    if value is None:
        value = metrics.get("pixel", {}).get(name)
    return float(value) if value is not None else None


def _delta(
    localized: dict[str, Any], identity: dict[str, Any], name: str
) -> float | None:
    local_value = _metric(localized, name)
    identity_value = _metric(identity, name)
    if local_value is None or identity_value is None:
        return None
    return local_value - identity_value


def _decision(categories: dict[str, Any], localizer: str) -> dict[str, Any]:
    primary_names = ("pixel_roc_auc", "au_pro")
    deltas = {
        category: {
            name: _delta(legs["localized"]["run"], legs["identity"]["run"], name)
            for name in primary_names
        }
        for category, legs in categories.items()
        if "run" in legs.get("localized", {}) and "run" in legs.get("identity", {})
    }
    mean_deltas = {
        name: mean(value[name] for value in deltas.values() if value[name] is not None)
        if any(value[name] is not None for value in deltas.values())
        else None
        for name in primary_names
    }
    all_builds_complete = all(
        legs["localized"]["build"]["failure_rate"]
        == DEFAULT_CRITERION["required_build_failure_rate"]
        for legs in categories.values()
    )
    no_large_loss = all(
        value is None
        or value >= -DEFAULT_CRITERION["maximum_single_class_primary_loss"]
        for class_deltas in deltas.values()
        for value in class_deltas.values()
    )
    gains_clear = (
        mean_deltas["pixel_roc_auc"] is not None
        and mean_deltas["pixel_roc_auc"]
        >= DEFAULT_CRITERION["minimum_mean_pixel_roc_auc_gain"]
        and mean_deltas["au_pro"] is not None
        and mean_deltas["au_pro"] >= DEFAULT_CRITERION["minimum_mean_au_pro_gain"]
    )
    becomes_default = all_builds_complete and no_large_loss and gains_clear
    return {
        "criterion": DEFAULT_CRITERION,
        "per_class_primary_delta": deltas,
        "mean_primary_delta": mean_deltas,
        "all_builds_complete": all_builds_complete,
        "no_large_single_class_loss": no_large_loss,
        "required_mean_gains_clear": gains_clear,
        "localization_becomes_default": becomes_default,
        "recommendation": (
            f"{localizer} localization becomes the default spatial input"
            if becomes_default
            else f"identity remains the default; {localizer} localization stays opt-in"
        ),
    }


def _profile_specs(localizer: str) -> tuple[tuple[str, str, dict[str, Any]], ...]:
    return (
        ("identity", "identity", {}),
        ("localized", localizer, LOCALIZER_CONFIGS[localizer]),
    )


def _configure_mobile_sam_asset(
    settings: Settings, asset_path: Path | None
) -> dict[str, Any]:
    spec = get_spec("mobile-sam-vit-t")
    if spec is None:
        raise RuntimeError("the MobileSAM asset is absent from the fixed catalog")
    if asset_path is None:
        raise ValueError("--mobile-sam-asset is required when --localizer=mobile_sam")
    resolved = set_external_source(settings, spec, asset_path)
    return {
        "key": spec.key,
        "catalog_sha256": spec.sha256,
        "catalog_size": spec.expected_size,
        "source": resolved.source,
    }


def _run_gate(
    data_dir: Path,
    categories: tuple[str, ...],
    *,
    localizer: str,
    mobile_sam_asset: Path | None,
) -> int:
    _empty_destination(data_dir)
    settings = _settings(data_dir)
    settings.ensure_directories()
    asset_report = (
        _configure_mobile_sam_asset(settings, mobile_sam_asset)
        if localizer == "mobile_sam"
        else None
    )
    apply_migrations(settings.db_path)
    report: dict[str, Any] = {
        "schema_version": 1,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source": "VisA official 1cls split",
        "categories": {},
        "localizer": {"type": localizer, "config": LOCALIZER_CONFIGS[localizer]},
        "method": METHOD,
        "model_config": MODEL_CONFIG,
        "prepared_size": [PREPARED_SIZE, PREPARED_SIZE],
        "padding_fraction": PADDING_FRACTION,
        "seed": SEED,
        "host": {"platform": platform.platform(), "python": platform.python_version()},
        "packages": _package_versions(),
    }
    if asset_report is not None:
        report["localizer"]["asset"] = asset_report
    log_path = data_dir / "gate.log"
    with log_path.open("a", encoding="utf-8") as log:
        report["backbone_prewarm_seconds"] = _prewarm_backbone(log)
        next_job_id = 1
        for category in categories:
            dataset_id, split_id = _official_dataset(settings, category, log=log)
            category_report: dict[str, Any] = {}
            report["categories"][category] = category_report
            for label, extractor_type, extractor_config in _profile_specs(localizer):
                build, build_report = _build_profile(
                    settings,
                    category=category,
                    dataset_id=dataset_id,
                    label=label,
                    extractor_type=extractor_type,
                    extractor_config=extractor_config,
                    job_id=next_job_id,
                    log=log,
                )
                next_job_id += 1
                leg: dict[str, Any] = {"build": build_report}
                category_report[label] = leg
                if build is None:
                    continue
                experiment_id = _create_experiment(
                    settings,
                    category=category,
                    dataset_id=dataset_id,
                    split_id=split_id,
                    label=label,
                    build=build,
                )
                print(
                    f"Running {category} / {label} as experiment {experiment_id}...",
                    file=sys.stderr,
                )
                leg["run"] = _execute_leg(data_dir, experiment_id, log)
                (data_dir / "result.partial.json").write_text(
                    json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
                )
    report["decision"] = _decision(report["categories"], localizer)
    output = data_dir / "result.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report["decision"], indent=2, sort_keys=True))
    print(f"Full evidence: {output}", file=sys.stderr)
    return 0


def _summarize_existing(data_dir: Path) -> int:
    output = data_dir / "result.json"
    source = output if output.is_file() else data_dir / "result.partial.json"
    if not source.is_file():
        raise FileNotFoundError(f"no gate result under {data_dir}")
    report = json.loads(source.read_text(encoding="utf-8"))
    report["packages"] = _package_versions()
    localizer = str(report.get("localizer", {}).get("type", "foreground_threshold"))
    report["decision"] = _decision(report["categories"], localizer)
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report["decision"], indent=2, sort_keys=True))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument(
        "--category",
        action="append",
        choices=tuple(
            spec.key.split(":", 1)[1]
            for pack in pack_specs(_settings(Path("/tmp/anomaly-region-gate-catalog")))
            if pack.key == "visa"
            for spec in pack.datasets
        ),
        dest="categories",
        help="VisA class to include; repeat for more than one (default: candle and pcb1).",
    )
    parser.add_argument("--_experiment-id", type=int, help=argparse.SUPPRESS)
    parser.add_argument(
        "--localizer",
        choices=tuple(LOCALIZER_CONFIGS),
        default="foreground_threshold",
        help="Localisation profile paired against identity (default: foreground_threshold).",
    )
    parser.add_argument(
        "--mobile-sam-asset",
        type=Path,
        help="Verified MobileSAM checkpoint; required only for the mobile_sam localizer.",
    )
    parser.add_argument(
        "--summarize-existing",
        action="store_true",
        help="Recompute the decision from an existing result without rerunning experiments.",
    )
    args = parser.parse_args(argv)
    if args._experiment_id is not None:
        return _run_leg(args.data_dir, args._experiment_id)
    if args.summarize_existing:
        return _summarize_existing(args.data_dir.resolve())
    categories = tuple(args.categories or DEFAULT_CATEGORIES)
    if len(categories) < 2:
        parser.error("the value gate requires at least two VisA classes")
    return _run_gate(
        args.data_dir.resolve(),
        categories,
        localizer=args.localizer,
        mobile_sam_asset=args.mobile_sam_asset,
    )


if __name__ == "__main__":
    raise SystemExit(main())
