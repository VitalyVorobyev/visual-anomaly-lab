#!/usr/bin/env -S uv run --project backend --extra dl python
"""Run the few-shot segmentation public gate (ADR-0040; protocol in docs/measurements.md).

VisA `candle` and `pcb1` at 448 x 448, target class `defect`. For every class, shot count and
seed, a `few_shot` split draws the references; each method then fits on them and segments
every other sample, in its own child process on the same immutable prepared pixels.

    ./scripts/few-shot-public-gate.py --data-dir /tmp/few-shot-gate

The destination must be absent or empty; source images stay read-only under `/datasets`.
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
SHOTS = (1, 2, 5, 10)
SEEDS = (0, 1, 2)
METHODS = ("color_prototype", "fss_dino", "proto_seg")
PREPARED_SIZE = 448
TARGET = "defect"
PRIMARY_SHOTS = 5
MARGIN = 0.02
REPORTED = {
    "foreground_iou": "foreground_iou",
    "pixel_average_precision": "pixel_average_precision",
    "boundary_f1": "boundary_f1",
    "presence_roc_auc": "image_presence_roc_auc",
    "absent_false_positive_rate": "image_absent_false_positive_rate",
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
    build: PreparedRegionBuild,
    name: str,
) -> int:
    config = get_model_class(method).config_model().model_validate({}).model_dump(mode="json")
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


def _summary(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Per method and shot count: mean over classes and seeds, and the spread across seeds."""
    table: dict[str, Any] = {}
    for method in METHODS:
        for shots in SHOTS:
            legs = [run for run in runs if run["method"] == method and run["shots"] == shots]
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
            table.setdefault(method, {})[str(shots)] = cell
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--categories", nargs="+", default=list(CATEGORIES))
    args = parser.parse_args()

    data_dir: Path = args.data_dir.resolve()
    harness._empty_destination(data_dir)
    settings = harness._settings(data_dir)
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
            for shots in SHOTS:
                for seed in SEEDS:
                    with connection(settings.db_path) as conn:
                        split_id = _split(conn, dataset_id, shots, seed)
                    for method in METHODS:
                        name = (
                            f"few-shot gate · {category} · {method} · {shots} shots · seed {seed}"
                        )
                        experiment_id = _experiment(
                            settings,
                            dataset_id=dataset_id,
                            split_id=split_id,
                            method=method,
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
                                "shots": shots,
                                "seed": seed,
                                "experiment_id": experiment_id,
                                **_row(report),
                            }
                        )
                        table = _summary(runs)
                        result_path.write_text(
                            json.dumps(
                                {
                                    "protocol": {
                                        "categories": args.categories,
                                        "shots": SHOTS,
                                        "seeds": SEEDS,
                                        "methods": METHODS,
                                        "prepared_size": PREPARED_SIZE,
                                        "target": TARGET,
                                    },
                                    "packages": harness._packages(),
                                    "elapsed_seconds": time.perf_counter() - started,
                                    "runs": runs,
                                    "summary": table,
                                    "decision": _decision(table),
                                },
                                indent=2,
                            ),
                            encoding="utf-8",
                        )

    print(json.dumps(_decision(_summary(runs)), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
