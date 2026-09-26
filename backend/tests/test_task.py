"""The task seam (ADR-0039): stored on the experiment, declared by the method, read by the
evaluator registry — and invisible to every anomaly run that existed before it."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from anomaly_lab.domain.entities import Task
from anomaly_lab.eval import runner
from anomaly_lab.eval.evaluators import (
    EVALUATORS,
    AnomalyEvaluator,
    UnsupportedTaskError,
    evaluator_for,
    has_evaluator,
)
from anomaly_lab.models.base import Capabilities
from anomaly_lab.models.registry import describe_all

from .conftest import Fixture, create_experiment

FEW_SHOT_METHODS = {"color_prototype", "fss_dino", "proto_seg"}
SEGMENTATION_METHODS = {"color_classifier", "dino_linear_seg"}
DETECTION_METHODS = {"color_detector", "dino_linear_det"}


def test_every_method_written_before_tasks_is_an_anomaly_method() -> None:
    assert Capabilities().tasks == [Task.ANOMALY]
    for description in describe_all():
        if description.key in FEW_SHOT_METHODS:
            assert description.capabilities.tasks == [Task.FEW_SHOT_SEGMENTATION]
        elif description.key in SEGMENTATION_METHODS:
            assert description.capabilities.tasks == [Task.SEMANTIC_SEGMENTATION]
        elif description.key in DETECTION_METHODS:
            assert description.capabilities.tasks == [Task.OBJECT_DETECTION]
        else:
            assert Task.ANOMALY in description.capabilities.tasks, description.key


def test_every_declared_task_has_an_evaluator() -> None:
    for description in describe_all():
        for task in description.capabilities.tasks:
            assert has_evaluator(task), (description.key, task)


def test_the_anomaly_evaluator_is_the_runner_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def recorder(name: str) -> object:
        def record(conn: object, experiment: object) -> dict[str, object]:
            calls.append(name)
            return {}

        return record

    monkeypatch.setattr(runner, "evaluate_and_store", recorder("store"))
    monkeypatch.setattr(runner, "evaluate_experiment", recorder("read"))
    evaluator = evaluator_for(Task.ANOMALY)
    assert isinstance(evaluator, AnomalyEvaluator)
    evaluator.evaluate_and_store(None, None)  # type: ignore[arg-type]
    evaluator.evaluate(None, None)  # type: ignore[arg-type]
    assert calls == ["store", "read"]


def test_every_task_has_an_evaluator() -> None:
    assert set(EVALUATORS) == set(Task)
    assert evaluator_for(Task.OBJECT_DETECTION).headline == "ap"


def test_a_task_without_an_evaluator_is_named_not_guessed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delitem(EVALUATORS, Task.OBJECT_DETECTION)
    assert not has_evaluator(Task.OBJECT_DETECTION)
    with pytest.raises(UnsupportedTaskError, match="object_detection"):
        evaluator_for(Task.OBJECT_DETECTION)


def test_creation_refuses_a_task_without_an_evaluator(
    client: TestClient, seeded: Fixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    create_experiment(client, seeded)  # builds the region profile a run pins
    monkeypatch.delitem(EVALUATORS, Task.OBJECT_DETECTION)
    refused = client.post(
        "/api/experiments",
        json={
            "name": "boxes",
            "dataset_id": seeded.dataset_id,
            "split_id": seeded.split_id,
            "region_profile_id": seeded.region_profile_id,
            "model_type": "color_detector",
            "task": "object_detection",
            "config": {},
        },
    )
    assert refused.status_code == 422
    assert "no evaluator is registered for the task 'object_detection'" in refused.text
