"""Which evaluator reads an experiment's results: one per task (ADR-0039).

An explicit table, for the reason `models/registry.py` and `jobs/handlers.py` are one:
reading this file tells you every task the workbench can evaluate. The `anomaly` evaluator
is `eval/runner.py`, unchanged — the registry is a seam in front of it, not a rewrite of
it, and its tests are the evidence that nothing moved.

A task with no entry here cannot be created (`experiments/service.create_experiment`
refuses it), so a run never reaches the end of inference with nobody to read its results.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from typing import Any, Protocol

from anomaly_lab.db.repositories import results as results_repo
from anomaly_lab.domain.entities import Experiment, Subset, Task
from anomaly_lab.eval import ground_truth, runner, segmentation


class Evaluator(Protocol):
    """Reads stored predictions and ground truth, and writes the metric sets."""

    @property
    def headline(self) -> str:
        """The metric the `infer` log names per subset — the task's one-line answer."""
        ...

    def evaluate_and_store(
        self, conn: sqlite3.Connection, experiment: Experiment
    ) -> dict[Subset, dict[str, Any]]:
        """Compute and persist every subset's metrics — the `infer` job's last act."""
        ...

    def evaluate(
        self, conn: sqlite3.Connection, experiment: Experiment
    ) -> dict[Subset, dict[str, Any]]:
        """Compute every subset's metrics without storing anything."""
        ...

    def current_digest(
        self, conn: sqlite3.Connection, experiment: Experiment, subset: Subset
    ) -> str:
        """The ground-truth digest a subset's metrics would carry now; a stored one that
        differs means the truth moved after they were computed."""
        ...


class AnomalyEvaluator:
    """Image- and pixel-level anomaly metrics over stored scores and maps (ADR-0011)."""

    headline = "sample_roc_auc"

    def evaluate_and_store(
        self, conn: sqlite3.Connection, experiment: Experiment
    ) -> dict[Subset, dict[str, Any]]:
        return runner.evaluate_and_store(conn, experiment)

    def evaluate(
        self, conn: sqlite3.Connection, experiment: Experiment
    ) -> dict[Subset, dict[str, Any]]:
        return runner.evaluate_experiment(conn, experiment)

    def current_digest(
        self, conn: sqlite3.Connection, experiment: Experiment, subset: Subset
    ) -> str:
        return ground_truth.current_digest(conn, experiment.id, subset)


class FewShotSegmentationEvaluator:
    """One class against background, per image, over stored maps or masks (ADR-0040)."""

    headline = "foreground_iou"

    def evaluate_and_store(
        self, conn: sqlite3.Connection, experiment: Experiment
    ) -> dict[Subset, dict[str, Any]]:
        # The sample rows are the ranked list and the gallery's order: presence scores,
        # aggregated as the anomaly runner aggregates any score.
        runner.rebuild_sample_results(conn, experiment)
        computed, digests = segmentation.evaluate(conn, experiment)
        results_repo.replace_metric_sets(
            conn, experiment.id, computed, ground_truth_digests=digests
        )
        return computed

    def evaluate(
        self, conn: sqlite3.Connection, experiment: Experiment
    ) -> dict[Subset, dict[str, Any]]:
        return segmentation.evaluate(conn, experiment)[0]

    def current_digest(
        self, conn: sqlite3.Connection, experiment: Experiment, subset: Subset
    ) -> str:
        return segmentation.current_digest(conn, experiment, subset)


EVALUATORS: dict[Task, Callable[[], Evaluator]] = {
    Task.ANOMALY: AnomalyEvaluator,
    Task.FEW_SHOT_SEGMENTATION: FewShotSegmentationEvaluator,
}


class UnsupportedTaskError(Exception):
    """No evaluator is registered for this task — a stored row from a newer build, say."""


def has_evaluator(task: Task) -> bool:
    return task in EVALUATORS


def evaluator_for(task: Task) -> Evaluator:
    factory = EVALUATORS.get(task)
    if factory is None:
        known = ", ".join(sorted(entry.value for entry in EVALUATORS))
        msg = f"no evaluator is registered for the task {task.value!r}; known tasks are {known}"
        raise UnsupportedTaskError(msg)
    return factory()
