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

from anomaly_lab.domain.entities import Experiment, Subset, Task
from anomaly_lab.eval import runner


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


EVALUATORS: dict[Task, Callable[[], Evaluator]] = {
    Task.ANOMALY: AnomalyEvaluator,
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
