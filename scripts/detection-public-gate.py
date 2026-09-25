#!/usr/bin/env -S uv run --project backend --extra dl python
"""Run the public detection gate (ADR-0039; protocol in docs/measurements.md).

VisA `candle` and `pcb1` at 448 x 448, read as a detection benchmark of one class, `defect`:
each 8-connected component of a VisA mask is one truth box, exactly as the application
resolves an imported mask's boxes. For every class and seed, a `class_stratified` split draws
the samples into train and test; `color_detector` and `dino_linear_det` then fit on the train
subset at their shipped defaults and detect on the test subset, each in its own child process
on the same immutable prepared pixels.

    ./scripts/detection-public-gate.py --data-dir /tmp/detection-gate

The destination must be absent or empty; source images stay read-only under `--datasets-dir`
(the repository's `/datasets` by default). It holds the isolated database, prepared pixels,
`gate.log` and `result.json`, which is rewritten after every run so an interrupted gate keeps
what it measured. **Each run's maps and boxes are deleted once its metrics and per-sample
outcomes are recorded**; the metrics are what the gate decides on.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from collections import Counter
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import m11_public_gate as harness

from anomaly_lab.config import Settings
from anomaly_lab.datasets.splitting import (
    SplitParams,
    SplitStrategy,
    plan_class_stratified_split,
)
from anomaly_lab.db.connection import connection
from anomaly_lab.db.migrate import apply_migrations
from anomaly_lab.db.repositories import annotations as annotations_repo
from anomaly_lab.db.repositories import experiments as experiments_repo
from anomaly_lab.db.repositories import splits as splits_repo
from anomaly_lab.domain.entities import Subset, Task
from anomaly_lab.eval.detection import sample_outcomes
from anomaly_lab.eval.runner import EvalConfig
from anomaly_lab.experiments.service import pin_classes
from anomaly_lab.models.preprocessing import PreprocessingConfig
from anomaly_lab.models.registry import get_model_class
from anomaly_lab.regions.preparation import PreparedRegionBuild

CATEGORIES = ("candle", "pcb1")
SEEDS = (0, 1, 2)
FLOOR = "color_detector"
CANDIDATE = "dino_linear_det"
METHODS = (FLOOR, CANDIDATE)
PREPARED_SIZE = 448
MARGIN = 0.05
REPORTED = (
    "ap",
    "ap50",
    "ap75",
    "recall",
    "recall50",
    "f1_at_cut",
    "precision_at_cut",
    "recall_at_cut",
)


def _split(conn: Any, dataset_id: int, seed: int) -> int:
    classes = [label.key for label in annotations_repo.list_labels(conn, dataset_id)]
    params = SplitParams(strategy=SplitStrategy.CLASS_STRATIFIED, classes=classes)
    assignments = plan_class_stratified_split(
        conn,
        dataset_id,
        seed=seed,
        classes=classes,
        train_fraction=params.train_fraction,
        unlabeled_subset=params.unlabeled_subset,
    )
    split = splits_repo.create_split(
        conn,
        dataset_id,
        name=f"class stratified · seed {seed}",
        strategy=SplitStrategy.CLASS_STRATIFIED.value,
        seed=seed,
        params=params.model_dump(mode="json"),
        assignments=assignments,
    )
    return split.id


def _experiment(
    settings: Settings,
    *,
    dataset_id: int,
    split_id: int,
    method: str,
    seed: int,
    build: PreparedRegionBuild,
    name: str,
) -> int:
    config_model = get_model_class(method).config_model()
    # The method's seed follows the split's, where it has one; `color_detector` draws nothing.
    overrides: dict[str, Any] = {"seed": seed} if "seed" in config_model.model_fields else {}
    config = config_model.model_validate(overrides).model_dump(mode="json")
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
            task=Task.OBJECT_DETECTION.value,
            classes=pin_classes(conn, Task.OBJECT_DETECTION, dataset_id),
            model_config=config,
            preprocessing_config=preprocessing.model_dump(mode="json"),
            eval_config=EvalConfig().model_dump(mode="json"),
            artifact_dir="",
            notes="Public detection gate (docs/measurements.md).",
        )
        artifact_dir = settings.experiment_dir(experiment.id)
        conn.execute(
            "UPDATE experiment SET artifact_dir = ? WHERE id = ?",
            (str(artifact_dir), experiment.id),
        )
    artifact_dir.mkdir(parents=True, exist_ok=True)
    return experiment.id


def _outcomes(settings: Settings, experiment_id: int) -> dict[str, dict[str, int]]:
    """Per-sample outcomes of the test subset at its printed cut, by the sample's label."""
    with connection(settings.db_path) as conn:
        experiment = experiments_repo.get_experiment(conn, experiment_id)
        assert experiment is not None
        verdicts = sample_outcomes(conn, experiment, Subset.TEST).samples
    tally: dict[str, Counter[str]] = {}
    for verdict in verdicts:
        tally.setdefault(str(verdict.label), Counter())[verdict.outcome] += 1
    return {label: dict(sorted(counts.items())) for label, counts in sorted(tally.items())}


def _row(report: dict[str, Any]) -> dict[str, Any]:
    metrics = report["infer"]["metrics"]["test"]
    row: dict[str, Any] = {name: metrics.get(name) for name in REPORTED}
    for name in ("confidence_cut", "cut_rule", "truth_instances", "predicted_instances"):
        row[name] = metrics.get(name)
    row["images"] = metrics.get("images")
    row["ms_per_image"] = metrics.get("timing", {}).get("mean_ms")
    row["train_seconds"] = report["train_seconds"]
    row["infer_seconds"] = report["infer_seconds"]
    row["peak_rss_bytes"] = report["peak_rss_bytes"]
    return row


def _summary(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Per method and class: the mean over seeds and the spread across them."""
    table: dict[str, Any] = {}
    for method in METHODS:
        for category in dict.fromkeys(run["category"] for run in runs):
            legs = [run for run in runs if run["method"] == method and run["category"] == category]
            if not legs:
                continue
            cell: dict[str, Any] = {"runs": len(legs)}
            for name in (*REPORTED, "ms_per_image", "train_seconds", "peak_rss_bytes"):
                values = [leg[name] for leg in legs if leg[name] is not None]
                cell[name] = mean(values) if values else None
            aps = [leg["ap"] for leg in legs if leg["ap"] is not None]
            cell["ap_seed_spread"] = pstdev(aps) if len(aps) > 1 else None
            cell["truth_instances"] = sum(sum(leg["truth_instances"].values()) for leg in legs)
            cell["predicted_instances"] = sum(
                sum(leg["predicted_instances"].values()) for leg in legs
            )
            outcomes: dict[str, Counter[str]] = {}
            for leg in legs:
                for label, counts in leg["outcomes"].items():
                    outcomes.setdefault(label, Counter()).update(counts)
            cell["outcomes"] = {label: dict(sorted(c.items())) for label, c in outcomes.items()}
            table.setdefault(method, {})[category] = cell
    return table


def _checks(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Checks of construction: within a class and seed, both methods read the same truth."""
    checks: dict[str, Any] = {}
    for category in dict.fromkeys(run["category"] for run in runs):
        for seed in dict.fromkeys(run["seed"] for run in runs):
            legs = [run for run in runs if run["category"] == category and run["seed"] == seed]
            if len(legs) < len(METHODS):
                continue
            truth = {json.dumps(leg["truth_instances"], sort_keys=True) for leg in legs}
            images = {json.dumps(leg["images"], sort_keys=True) for leg in legs}
            checks[f"{category}:{seed}"] = {
                "truth_instances_identical": len(truth) == 1,
                "images_identical": len(images) == 1,
            }
    return checks


def _decision(table: dict[str, Any]) -> dict[str, Any]:
    def ap(method: str, category: str) -> float | None:
        value = table.get(method, {}).get(category, {}).get("ap")
        return None if value is None else float(value)

    leads: dict[str, float | None] = {}
    for category in CATEGORIES:
        candidate, floor = ap(CANDIDATE, category), ap(FLOOR, category)
        leads[category] = None if candidate is None or floor is None else candidate - floor
    promoted = all(lead is not None and lead >= MARGIN for lead in leads.values())
    return {
        "candidate": CANDIDATE,
        "primary": "test AP@[.5:.95] (COCO, class defect), mean over seeds",
        "margin": MARGIN,
        "lead_by_class": leads,
        "promoted": promoted,
        "maturity": "supported" if promoted else "experimental",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument(
        "--datasets-dir",
        type=Path,
        default=harness.REPOSITORY / "datasets",
        help="Where the public packs live (VisA_20220922 inside it); read only.",
    )
    parser.add_argument("--categories", nargs="+", default=list(CATEGORIES))
    parser.add_argument("--seeds", nargs="+", type=int, help="A smoke subset of the gate's seeds.")
    parser.add_argument(
        "--methods", nargs="+", choices=METHODS, help="A smoke subset of the gate's methods."
    )
    args = parser.parse_args()
    seeds: tuple[int, ...] = tuple(args.seeds or SEEDS)
    methods: tuple[str, ...] = tuple(args.methods or METHODS)

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
                job_id=50_000 + index,
                log=log,
            )
            for seed in seeds:
                with connection(settings.db_path) as conn:
                    split_id = _split(conn, dataset_id, seed)
                for method in methods:
                    name = f"detection gate · {category} · {method} · seed {seed}"
                    experiment_id = _experiment(
                        settings,
                        dataset_id=dataset_id,
                        split_id=split_id,
                        method=method,
                        seed=seed,
                        build=build,
                        name=name,
                    )
                    print(f"{name}...", file=sys.stderr)
                    report = harness._execute(data_dir, experiment_id, log)
                    outcomes = _outcomes(settings, experiment_id)
                    shutil.rmtree(settings.experiment_dir(experiment_id) / "maps")
                    runs.append(
                        {
                            "category": category,
                            "method": method,
                            "seed": seed,
                            "experiment_id": experiment_id,
                            **_row(report),
                            "outcomes": outcomes,
                        }
                    )
                    table = _summary(runs)
                    result_path.write_text(
                        json.dumps(
                            {
                                "protocol": {
                                    "categories": args.categories,
                                    "seeds": seeds,
                                    "methods": methods,
                                    "prepared_size": PREPARED_SIZE,
                                    "classes": ["defect"],
                                    "truth": "each 8-connected component of a VisA mask",
                                    "split": "class_stratified, shipped defaults",
                                },
                                "packages": harness._packages(),
                                "elapsed_seconds": time.perf_counter() - started,
                                "runs": runs,
                                "summary": table,
                                "checks": _checks(runs),
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
