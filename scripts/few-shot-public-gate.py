#!/usr/bin/env -S uv run --project backend --extra dl python
"""Run the few-shot segmentation public gate (ADR-0040; protocol in docs/measurements.md).

VisA `candle` and `pcb1` at 448 x 448, target class `defect`. For every class, shot count and
seed, a `few_shot` split draws the references; each method then fits on them and segments
every other sample, in its own child process on the same immutable prepared pixels.

    ./scripts/few-shot-public-gate.py --data-dir /tmp/few-shot-gate
    ./scripts/few-shot-public-gate.py --leg calibration --data-dir /tmp/few-shot-calibration

`--leg methods` (the default) is the first gate: three methods at their shipped defaults over
k in {1, 2, 5, 10}. `--leg calibration` is its calibration leg: each method with `calibration`
at `none` and at `leave_one_out`, everything else at its default, over k in {5, 10} — one
reference cannot be left out, so at k = 1 the two are the same run.

The destination must be absent or empty; source images stay read-only under `--datasets-dir`.
It holds the isolated database, prepared pixels, `gate.log` and `result.json`, which is
rewritten after every run so an interrupted gate keeps what it measured. **Each run's maps
are deleted once its metrics are recorded**: at 448 px a run writes about 0.9 GB of them,
and 72 runs would not fit on a laptop. The metrics are what the gate decides on.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import m11_public_gate as harness
import numpy as np

from anomaly_lab.config import Settings
from anomaly_lab.datasets.splitting import SplitParams, SplitStrategy, plan_few_shot_split
from anomaly_lab.db.connection import connection
from anomaly_lab.db.migrate import apply_migrations
from anomaly_lab.db.repositories import experiments as experiments_repo
from anomaly_lab.db.repositories import splits as splits_repo
from anomaly_lab.domain.entities import Task
from anomaly_lab.eval.runner import EvalConfig
from anomaly_lab.models.preprocessing import PreprocessingConfig
from anomaly_lab.models.registry import get_model_class
from anomaly_lab.regions.preparation import PreparedRegionBuild

CATEGORIES = ("candle", "pcb1")
SEEDS = (0, 1, 2)
METHODS = ("color_prototype", "fss_dino", "proto_seg")
PREPARED_SIZE = 448
TARGET = "defect"
PRIMARY_SHOTS = 5
MARGIN = 0.02
LEG_SHOTS = {"methods": (1, 2, 5, 10), "calibration": (5, 10)}
LEG_VARIANTS: dict[str, tuple[str, ...]] = {
    "methods": ("default",),
    "calibration": ("none", "leave_one_out"),
}
# The calibration leg's rule, fixed before it ran (docs/measurements.md).
FPR_DROP = 0.20
IOU_TOLERANCE = 0.005
RECALL_KEPT = 0.5
STATE_FILES = {
    "color_prototype": "color_prototype.npz",
    "fss_dino": "fss_dino.npz",
    "proto_seg": "proto_seg.npz",
}
REPORTED = {
    "foreground_iou": "foreground_iou",
    "pixel_average_precision": "pixel_average_precision",
    "boundary_f1": "boundary_f1",
    "presence_roc_auc": "image_presence_roc_auc",
    "absent_false_positive_rate": "image_absent_false_positive_rate",
    "present_recall": "image_present_recall",
}


def _split(conn: Any, dataset_id: int, shots: int, seed: int) -> int:
    params = SplitParams(strategy=SplitStrategy.FEW_SHOT, label_key=TARGET, shots=shots)
    assignments = plan_few_shot_split(conn, dataset_id, seed=seed, label_key=TARGET, shots=shots)
    split = splits_repo.create_split(
        conn,
        dataset_id,
        name=f"{shots} shots · seed {seed}",
        strategy=SplitStrategy.FEW_SHOT.value,
        seed=seed,
        params=params.model_dump(mode="json"),
        assignments=assignments,
    )
    return split.id


def _experiment(
    settings: Any,
    *,
    dataset_id: int,
    split_id: int,
    method: str,
    variant: str,
    build: PreparedRegionBuild,
    name: str,
) -> int:
    overrides = {} if variant == "default" else {"calibration": variant}
    config = (
        get_model_class(method).config_model().model_validate(overrides).model_dump(mode="json")
    )
    preprocessing = PreprocessingConfig(width=PREPARED_SIZE, height=PREPARED_SIZE)
    with connection(settings.db_path) as conn:
        experiment = experiments_repo.create_experiment(
            conn,
            name=name,
            dataset_id=dataset_id,
            split_id=split_id,
            region_profile_id=build.profile.id,
            region_manifest_sha256=build.summary.manifest_sha256,
            model_type=method,
            task=Task.FEW_SHOT_SEGMENTATION.value,
            target_label=TARGET,
            model_config=config,
            preprocessing_config=preprocessing.model_dump(mode="json"),
            eval_config=EvalConfig().model_dump(mode="json"),
            artifact_dir="",
            notes="Few-shot segmentation public gate (docs/measurements.md).",
        )
        artifact_dir = settings.experiment_dir(experiment.id)
        conn.execute(
            "UPDATE experiment SET artifact_dir = ? WHERE id = ?",
            (str(artifact_dir), experiment.id),
        )
    artifact_dir.mkdir(parents=True, exist_ok=True)
    return experiment.id


def _row(report: dict[str, Any]) -> dict[str, Any]:
    metrics = report["infer"]["metrics"]["test"]
    row = {name: metrics.get(key) for name, key in REPORTED.items()}
    row["ms_per_image"] = metrics.get("timing", {}).get("mean_ms")
    row["train_seconds"] = report["train_seconds"]
    return row


def _scale(settings: Settings, experiment_id: int, method: str) -> list[float] | None:
    """The fitted Platt scale `[slope, bias]`, read back from the saved model."""
    found = sorted(settings.experiment_dir(experiment_id).rglob(STATE_FILES[method]))
    if not found:
        return None
    with np.load(found[0], allow_pickle=False) as stored:
        if "calibration" not in stored.files:
            return None
        return [float(value) for value in stored["calibration"]]


def _summary(runs: list[dict[str, Any]], leg: str) -> dict[str, Any]:
    """Per method, variant and shot count: mean over classes and seeds, and the seed spread."""
    table: dict[str, Any] = {}
    cells = [
        (method, variant, shots)
        for method in METHODS
        for variant in LEG_VARIANTS[leg]
        for shots in LEG_SHOTS[leg]
    ]
    for method, variant, shots in cells:
        legs = [
            run
            for run in runs
            if run["method"] == method and run["variant"] == variant and run["shots"] == shots
        ]
        if not legs:
            continue
        cell: dict[str, Any] = {"runs": len(legs)}
        for name in (*REPORTED, "ms_per_image"):
            values = [leg[name] for leg in legs if leg[name] is not None]
            cell[name] = mean(values) if values else None
        # Seed spread: per class, the deviation over seeds; then averaged over classes.
        spreads = []
        for category in CATEGORIES:
            ious = [
                leg["foreground_iou"]
                for leg in legs
                if leg["category"] == category and leg["foreground_iou"] is not None
            ]
            if len(ious) > 1:
                spreads.append(pstdev(ious))
        cell["iou_seed_spread"] = mean(spreads) if spreads else None
        key = method if leg == "methods" else f"{method}:{variant}"
        table.setdefault(key, {})[str(shots)] = cell
    return table


def _decision(table: dict[str, Any]) -> dict[str, Any]:
    def at(method: str, name: str) -> float | None:
        value = table.get(method, {}).get(str(PRIMARY_SHOTS), {}).get(name)
        return None if value is None else float(value)

    proto = at("proto_seg", "foreground_iou")
    fss = at("fss_dino", "foreground_iou")
    floor = at("color_prototype", "foreground_iou")
    proto_auc, fss_auc = at("proto_seg", "presence_roc_auc"), at("fss_dino", "presence_roc_auc")
    credible = fss is not None and floor is not None and fss > floor
    promoted = (
        proto is not None
        and fss is not None
        and proto - fss >= MARGIN
        and proto_auc is not None
        and fss_auc is not None
        and fss_auc - proto_auc <= MARGIN
    )
    return {
        "primary": f"foreground_iou at {PRIMARY_SHOTS} shots",
        "proto_seg": proto,
        "fss_dino": fss,
        "color_prototype": floor,
        "dino_credible": credible,
        "proto_seg_promoted": promoted,
        "default": "proto_seg" if promoted else "fss_dino",
    }


def _calibration_decision(table: dict[str, Any]) -> dict[str, Any]:
    """Per method at the primary shot count: does `leave_one_out` become its default?"""
    decisions: dict[str, Any] = {}
    for method in METHODS:

        def at(variant: str, name: str, method: str = method) -> float | None:
            cell = table.get(f"{method}:{variant}", {}).get(str(PRIMARY_SHOTS), {})
            value = cell.get(name)
            return None if value is None else float(value)

        fpr_none, fpr_loo = (
            at("none", "absent_false_positive_rate"),
            at("leave_one_out", "absent_false_positive_rate"),
        )
        iou_none, iou_loo = at("none", "foreground_iou"), at("leave_one_out", "foreground_iou")
        fpr_drop = None if fpr_none is None or fpr_loo is None else fpr_none - fpr_loo
        iou_change = None if iou_none is None or iou_loo is None else iou_loo - iou_none
        recall_none = at("none", "present_recall")
        recall_loo = at("leave_one_out", "present_recall")
        recall_kept = (
            recall_none is not None
            and recall_loo is not None
            and recall_loo >= RECALL_KEPT * recall_none
        )
        adopted = (
            fpr_drop is not None
            and iou_change is not None
            and fpr_drop >= FPR_DROP
            and iou_change >= -IOU_TOLERANCE
            and recall_kept
        )
        decisions[method] = {
            "absent_fpr_drop": fpr_drop,
            "foreground_iou_change": iou_change,
            "present_recall": {"none": recall_none, "leave_one_out": recall_loo},
            "presence_roc_auc_change": _change(at, "presence_roc_auc"),
            "pixel_average_precision_change": _change(at, "pixel_average_precision"),
            "leave_one_out_default": adopted,
        }
    return {
        "primary": f"at {PRIMARY_SHOTS} shots",
        "rule": (
            f"absent FPR falls by >= {FPR_DROP}, foreground IoU falls by <= {IOU_TOLERANCE}, "
            f"and present-image recall keeps >= {RECALL_KEPT} of its value"
        ),
        "methods": decisions,
    }


def _change(at: Any, name: str) -> float | None:
    before, after = at("none", name), at("leave_one_out", name)
    return None if before is None or after is None else after - before


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument(
        "--datasets-dir",
        type=Path,
        default=harness.REPOSITORY / "datasets",
        help="Where the public packs live (VisA_20220922 inside it); read only.",
    )
    parser.add_argument("--leg", choices=sorted(LEG_SHOTS), default="methods")
    parser.add_argument("--categories", nargs="+", default=list(CATEGORIES))
    parser.add_argument(
        "--shots", nargs="+", type=int, help="A smoke subset of the leg's shot counts."
    )
    parser.add_argument("--seeds", nargs="+", type=int, help="A smoke subset of the seeds.")
    args = parser.parse_args()
    leg: str = args.leg
    shot_counts: tuple[int, ...] = tuple(args.shots or LEG_SHOTS[leg])
    seeds: tuple[int, ...] = tuple(args.seeds or SEEDS)
    decide = _decision if leg == "methods" else _calibration_decision

    data_dir: Path = args.data_dir.resolve()
    harness._empty_destination(data_dir)
    # The children build their settings from the data directory alone; only the import
    # reads the packs, so only the parent needs to know where they are.
    settings = Settings(
        data_dir=data_dir, reference_datasets_dir=args.datasets_dir.resolve(), dev_cors=False
    )
    apply_migrations(settings.db_path)
    started = time.perf_counter()
    runs: list[dict[str, Any]] = []
    result_path = data_dir / "result.json"

    with (data_dir / "gate.log").open("a", encoding="utf-8") as log:
        for index, category in enumerate(args.categories):
            dataset_id, _official = harness._official_dataset(settings, category, log)
            build = harness._identity_profile(
                settings,
                category=category,
                dataset_id=dataset_id,
                prepared_size=PREPARED_SIZE,
                job_id=30_000 + index,
                log=log,
            )
            for shots in shot_counts:
                for seed in seeds:
                    with connection(settings.db_path) as conn:
                        split_id = _split(conn, dataset_id, shots, seed)
                    for method in METHODS:
                        for variant in LEG_VARIANTS[leg]:
                            label = method if variant == "default" else f"{method} ({variant})"
                            name = (
                                f"few-shot gate · {category} · {label} · {shots} shots · "
                                f"seed {seed}"
                            )
                            experiment_id = _experiment(
                                settings,
                                dataset_id=dataset_id,
                                split_id=split_id,
                                method=method,
                                variant=variant,
                                build=build,
                                name=name,
                            )
                            print(f"{name}...", file=sys.stderr)
                            report = harness._execute(data_dir, experiment_id, log)
                            shutil.rmtree(settings.experiment_dir(experiment_id) / "maps")
                            runs.append(
                                {
                                    "category": category,
                                    "method": method,
                                    "variant": variant,
                                    "shots": shots,
                                    "seed": seed,
                                    "experiment_id": experiment_id,
                                    "scale": _scale(settings, experiment_id, method),
                                    **_row(report),
                                }
                            )
                            table = _summary(runs, leg)
                            result_path.write_text(
                                json.dumps(
                                    {
                                        "protocol": {
                                            "leg": leg,
                                            "categories": args.categories,
                                            "shots": shot_counts,
                                            "seeds": seeds,
                                            "methods": METHODS,
                                            "variants": LEG_VARIANTS[leg],
                                            "prepared_size": PREPARED_SIZE,
                                            "target": TARGET,
                                        },
                                        "packages": harness._packages(),
                                        "elapsed_seconds": time.perf_counter() - started,
                                        "runs": runs,
                                        "summary": table,
                                        "decision": decide(table),
                                    },
                                    indent=2,
                                ),
                                encoding="utf-8",
                            )

    print(json.dumps(decide(_summary(runs, leg)), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
