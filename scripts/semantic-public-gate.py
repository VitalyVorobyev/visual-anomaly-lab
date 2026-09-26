#!/usr/bin/env -S uv run --project backend --extra dl python
"""Run the supervised segmentation public gate (ADR-0039; protocol in docs/measurements.md).

VisA `candle` and `pcb1` at 448 x 448, read as a semantic segmentation benchmark of one class,
`defect`, from VisA's pixel masks. For every class and seed, a `class_stratified` split draws
the annotated samples into train and test; each method then fits on the train subset and
segments the test subset, in its own child process on the same immutable prepared pixels.

    ./scripts/semantic-public-gate.py --data-dir /tmp/semantic-gate
    ./scripts/semantic-public-gate.py --gate bias --data-dir /tmp/semantic-bias-gate

`--gate sampling` (the default) is the first gate: both methods at their shipped defaults.
`--gate bias` is the logit-bias gate: `color_classifier` at its defaults and `dino_linear_seg`
under `per_class` sampling twice, with `logit_bias` `none` and `held_out_iou` — the same fitted
head, read with and without constants fitted for IoU on held-out folds of the training images.

The destination must be absent or empty; source images stay read-only under `--datasets-dir`
(the repository's `/datasets` by default). It holds the isolated database, prepared pixels,
`gate.log` and `result.json`, which is rewritten after every run so an interrupted gate keeps
what it measured. **Each run's maps are deleted once its metrics and per-sample outcomes are
recorded**; the metrics are what the gate decides on.
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
import numpy as np

from anomaly_lab.config import Settings
from anomaly_lab.datasets.splitting import (
    SplitParams,
    SplitStrategy,
    plan_class_stratified_split,
)
from anomaly_lab.db.connection import connection
from anomaly_lab.db.migrate import apply_schema
from anomaly_lab.db.repositories import annotations as annotations_repo
from anomaly_lab.db.repositories import experiments as experiments_repo
from anomaly_lab.db.repositories import splits as splits_repo
from anomaly_lab.domain.entities import Subset, Task
from anomaly_lab.eval.runner import EvalConfig
from anomaly_lab.eval.semantic import sample_outcomes
from anomaly_lab.experiments.service import pin_classes
from anomaly_lab.experiments.train import MODEL_SUBDIR
from anomaly_lab.models.preprocessing import PreprocessingConfig
from anomaly_lab.models.registry import get_model_class
from anomaly_lab.regions.preparation import PreparedRegionBuild

CATEGORIES = ("candle", "pcb1")
SEEDS = (0, 1, 2)
FLOOR = "color_classifier"
PER_CLASS: dict[str, str] = {"pixel_sampling": "per_class"}
# Each gate's runs: a label, the method, and the fields that differ from its shipped defaults.
GATES: dict[str, tuple[tuple[str, str, dict[str, str]], ...]] = {
    "sampling": ((FLOOR, FLOOR, {}), ("dino_linear_seg", "dino_linear_seg", {})),
    "bias": (
        (FLOOR, FLOOR, {}),
        ("dino_linear_seg:none", "dino_linear_seg", {**PER_CLASS, "logit_bias": "none"}),
        (
            "dino_linear_seg:held_out_iou",
            "dino_linear_seg",
            {**PER_CLASS, "logit_bias": "held_out_iou"},
        ),
    ),
}
CANDIDATES = {"sampling": "dino_linear_seg", "bias": "dino_linear_seg:held_out_iou"}
# The bias gate's default rule reads the first gate's shipped default (raster sampling, no
# bias) on the same splits: test mean IoU by class, docs/measurements.md, leg 1.
RASTER_DEFAULT_IOU = {"candle": 0.0829, "pcb1": 0.0388}
PREPARED_SIZE = 448
MARGIN = 0.05
REPORTED = (
    "mean_iou",
    "background_iou",
    "pixel_accuracy",
    "mean_class_accuracy",
    "frequency_weighted_iou",
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
    fields: dict[str, str],
    seed: int,
    build: PreparedRegionBuild,
    name: str,
) -> int:
    config_model = get_model_class(method).config_model()
    # The method's seed follows the split's, where it has one; `color_classifier` draws nothing.
    overrides: dict[str, Any] = {"seed": seed} if "seed" in config_model.model_fields else {}
    config = config_model.model_validate({**overrides, **fields}).model_dump(mode="json")
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
            task=Task.SEMANTIC_SEGMENTATION.value,
            classes=pin_classes(conn, Task.SEMANTIC_SEGMENTATION, dataset_id),
            model_config=config,
            preprocessing_config=preprocessing.model_dump(mode="json"),
            eval_config=EvalConfig().model_dump(mode="json"),
            artifact_dir="",
            notes="Supervised segmentation public gate (docs/measurements.md).",
        )
        artifact_dir = settings.experiment_dir(experiment.id)
        conn.execute(
            "UPDATE experiment SET artifact_dir = ? WHERE id = ?",
            (str(artifact_dir), experiment.id),
        )
    artifact_dir.mkdir(parents=True, exist_ok=True)
    return experiment.id


def _outcomes(settings: Settings, experiment_id: int) -> dict[str, dict[str, int]]:
    """Per-sample outcomes of the test subset, tallied by the sample's anomaly label."""
    with connection(settings.db_path) as conn:
        experiment = experiments_repo.get_experiment(conn, experiment_id)
        assert experiment is not None
        verdicts = sample_outcomes(conn, experiment, Subset.TEST).samples
    tally: dict[str, Counter[str]] = {}
    for verdict in verdicts:
        tally.setdefault(str(verdict.label), Counter())[verdict.outcome] += 1
    return {label: dict(sorted(counts.items())) for label, counts in sorted(tally.items())}


def _constants(artifact_dir: Path) -> dict[str, list[float]]:
    """The logit constants `dino_linear_seg` saved beside its head, background first."""
    state = artifact_dir / MODEL_SUBDIR / "dino_linear_seg.npz"
    if not state.exists():
        return {}
    with np.load(state, allow_pickle=False) as stored:
        return {
            key: [float(value) for value in stored[key]]
            for key in ("prior_shift", "held_out_bias")
            if key in stored.files
        }


def _row(report: dict[str, Any]) -> dict[str, Any]:
    metrics = report["infer"]["metrics"]["test"]
    row: dict[str, Any] = {name: metrics.get(name) for name in REPORTED}
    row["per_class_iou"] = metrics.get("per_class_iou")
    row["images"] = metrics.get("images")
    row["ms_per_image"] = metrics.get("timing", {}).get("mean_ms")
    row["train_seconds"] = report["train_seconds"]
    row["infer_seconds"] = report["infer_seconds"]
    row["peak_rss_bytes"] = report["peak_rss_bytes"]
    return row


def _summary(runs: list[dict[str, Any]], gate: str) -> dict[str, Any]:
    """Per method and class: the mean over seeds and the spread across them."""
    table: dict[str, Any] = {}
    for method, _, _ in GATES[gate]:
        for category in dict.fromkeys(run["category"] for run in runs):
            legs = [run for run in runs if run["method"] == method and run["category"] == category]
            if not legs:
                continue
            cell: dict[str, Any] = {"runs": len(legs)}
            for name in (*REPORTED, "ms_per_image", "train_seconds"):
                values = [leg[name] for leg in legs if leg[name] is not None]
                cell[name] = mean(values) if values else None
            ious = [leg["mean_iou"] for leg in legs if leg["mean_iou"] is not None]
            cell["mean_iou_seed_spread"] = pstdev(ious) if len(ious) > 1 else None
            outcomes: dict[str, Counter[str]] = {}
            for leg in legs:
                for label, counts in leg["outcomes"].items():
                    outcomes.setdefault(label, Counter()).update(counts)
            cell["outcomes"] = {label: dict(sorted(c.items())) for label, c in outcomes.items()}
            table.setdefault(method, {})[category] = cell
    return table


def _decision(table: dict[str, Any], gate: str) -> dict[str, Any]:
    def iou(method: str, category: str) -> float | None:
        value = table.get(method, {}).get(category, {}).get("mean_iou")
        return None if value is None else float(value)

    candidate_key = CANDIDATES[gate]
    leads: dict[str, float | None] = {}
    for category in CATEGORIES:
        candidate, floor = iou(candidate_key, category), iou(FLOOR, category)
        leads[category] = None if candidate is None or floor is None else candidate - floor
    promoted = all(lead is not None and lead >= MARGIN for lead in leads.values())
    decision: dict[str, Any] = {
        "gate": gate,
        "candidate": candidate_key,
        "primary": "test mean IoU (annotation classes; background excluded), mean over seeds",
        "margin": MARGIN,
        "lead_by_class": leads,
        "promoted": promoted,
        "maturity": "supported" if promoted else "experimental",
    }
    if gate == "bias":
        # The fitted variant becomes the default if it is promoted, or if it measures above
        # the raster default on the primary on both classes.
        above = {
            category: (value := iou(candidate_key, category)) is not None
            and value > RASTER_DEFAULT_IOU[category]
            for category in CATEGORIES
        }
        decision["above_raster_default_by_class"] = above
        decision["new_defaults"] = promoted or all(above.values())
    return decision


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument(
        "--datasets-dir",
        type=Path,
        default=harness.REPOSITORY / "datasets",
        help="Where the public packs live (VisA_20220922 inside it); read only.",
    )
    parser.add_argument("--gate", choices=sorted(GATES), default="sampling")
    parser.add_argument("--categories", nargs="+", default=list(CATEGORIES))
    parser.add_argument("--seeds", nargs="+", type=int, help="A smoke subset of the gate's seeds.")
    args = parser.parse_args()
    gate: str = args.gate
    seeds: tuple[int, ...] = tuple(args.seeds or SEEDS)

    data_dir: Path = args.data_dir.resolve()
    harness._empty_destination(data_dir)
    # The children build their settings from the data directory alone; only the import
    # reads the packs, so only the parent needs to know where they are.
    settings = Settings(
        data_dir=data_dir, reference_datasets_dir=args.datasets_dir.resolve(), dev_cors=False
    )
    apply_schema(settings.db_path)
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
                job_id=40_000 + index,
                log=log,
            )
            for seed in seeds:
                with connection(settings.db_path) as conn:
                    split_id = _split(conn, dataset_id, seed)
                for label, method, fields in GATES[gate]:
                    name = f"semantic gate · {gate} · {category} · {label} · seed {seed}"
                    experiment_id = _experiment(
                        settings,
                        dataset_id=dataset_id,
                        split_id=split_id,
                        method=method,
                        fields=fields,
                        seed=seed,
                        build=build,
                        name=name,
                    )
                    print(f"{name}...", file=sys.stderr)
                    report = harness._execute(data_dir, experiment_id, log)
                    outcomes = _outcomes(settings, experiment_id)
                    constants = _constants(settings.experiment_dir(experiment_id))
                    shutil.rmtree(settings.experiment_dir(experiment_id) / "maps")
                    runs.append(
                        {
                            "category": category,
                            "method": label,
                            "seed": seed,
                            "experiment_id": experiment_id,
                            **_row(report),
                            "outcomes": outcomes,
                            **constants,
                        }
                    )
                    table = _summary(runs, gate)
                    result_path.write_text(
                        json.dumps(
                            {
                                "protocol": {
                                    "gate": gate,
                                    "categories": args.categories,
                                    "seeds": seeds,
                                    "runs": [
                                        {"label": label, "method": method, "fields": fields}
                                        for label, method, fields in GATES[gate]
                                    ],
                                    "prepared_size": PREPARED_SIZE,
                                    "classes": ["defect"],
                                    "split": "class_stratified, shipped defaults",
                                },
                                "packages": harness._packages(),
                                "elapsed_seconds": time.perf_counter() - started,
                                "runs": runs,
                                "summary": table,
                                "decision": _decision(table, gate),
                            },
                            indent=2,
                        ),
                        encoding="utf-8",
                    )

    print(json.dumps(_decision(_summary(runs, gate), gate), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
