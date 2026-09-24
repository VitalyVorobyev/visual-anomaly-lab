"""The task seam (ADR-0039): stored on the experiment, declared by the method, read by the
evaluator registry — and invisible to every anomaly run that existed before it."""

from __future__ import annotations

import sqlite3

import pytest

from anomaly_lab.config import Settings
from anomaly_lab.db.connection import connect
from anomaly_lab.db.migrate import apply_migrations_to, discover_migrations
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

FEW_SHOT_METHODS = {"color_prototype", "fss_dino", "proto_seg"}


def test_every_method_written_before_tasks_is_an_anomaly_method() -> None:
    assert Capabilities().tasks == [Task.ANOMALY]
    for description in describe_all():
        if description.key in FEW_SHOT_METHODS:
            assert description.capabilities.tasks == [Task.FEW_SHOT_SEGMENTATION]
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


def test_a_task_without_an_evaluator_is_named_not_guessed() -> None:
    assert set(EVALUATORS) == {Task.ANOMALY, Task.FEW_SHOT_SEGMENTATION}
    assert not has_evaluator(Task.OBJECT_DETECTION)
    with pytest.raises(UnsupportedTaskError, match="object_detection"):
        evaluator_for(Task.OBJECT_DETECTION)


def test_migration_021_makes_every_existing_run_an_anomaly_run(settings: Settings) -> None:
    with connect(settings.db_path) as conn:
        for migration in discover_migrations():
            if migration.number > 20:
                break
            conn.executescript(
                f"BEGIN;\n{migration.sql}\nPRAGMA user_version = {migration.number};\nCOMMIT;"
            )
        conn.execute("INSERT INTO dataset (name, root_path) VALUES ('d', '/d')")
        conn.execute(
            "INSERT INTO split (dataset_id, name, strategy, seed, params) "
            "VALUES (1, 's', 'imported', 0, '{}')"
        )
        conn.execute(
            "INSERT INTO region_profile_revision (dataset_id, name, revision_no, extractor_type, "
            "extractor_config, prepared_width, prepared_height, seed) "
            "VALUES (1, 'full frame', 1, 'identity', '{}', 8, 8, 17)"
        )
        conn.execute(
            "INSERT INTO experiment (name, dataset_id, split_id, region_profile_id, "
            "region_manifest_sha256, model_type, artifact_dir) "
            "VALUES ('e', 1, 1, 1, 'sha', 'pixel_reference', '/artifacts/1')"
        )

        assert apply_migrations_to(conn) >= 21
        assert conn.execute("SELECT task FROM experiment").fetchone()[0] == "anomaly"
        # Validated in Python, like `job.kind`: a new task is not a table rebuild.
        conn.execute("UPDATE experiment SET task = 'object_detection'")
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute("UPDATE experiment SET task = NULL")
