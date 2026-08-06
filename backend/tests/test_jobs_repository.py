"""Startup reconciliation of jobs left running by a crash (ADR-0009)."""

from __future__ import annotations

import sqlite3

from anomaly_lab.db.repositories.jobs import fail_stale_running_jobs
from anomaly_lab.domain.entities import JobKind, JobStatus


def _insert_job(conn: sqlite3.Connection, status: str, kind: str = "train") -> int:
    cursor = conn.execute(
        "INSERT INTO job (kind, status) VALUES (?, ?)",
        (kind, status),
    )
    return int(cursor.lastrowid or 0)


def test_running_jobs_become_failed(migrated_db: sqlite3.Connection) -> None:
    job_id = _insert_job(migrated_db, "running")

    reconciled = fail_stale_running_jobs(migrated_db)

    assert [job.id for job in reconciled] == [job_id]
    assert reconciled[0].status is JobStatus.FAILED
    assert reconciled[0].error
    assert reconciled[0].finished_at


def test_other_statuses_are_left_alone(migrated_db: sqlite3.Connection) -> None:
    for status in ("queued", "succeeded", "failed", "cancelled"):
        _insert_job(migrated_db, status)

    assert fail_stale_running_jobs(migrated_db) == []

    remaining = migrated_db.execute("SELECT status FROM job ORDER BY id").fetchall()
    assert [row["status"] for row in remaining] == ["queued", "succeeded", "failed", "cancelled"]


def test_is_a_no_op_on_an_empty_database(migrated_db: sqlite3.Connection) -> None:
    assert fail_stale_running_jobs(migrated_db) == []


def test_params_round_trip_as_json(migrated_db: sqlite3.Connection) -> None:
    """An import job records what it operates on; no experiment_id exists for it."""
    migrated_db.execute(
        "INSERT INTO job (kind, status, params) VALUES ('import', 'running', ?)",
        ('{"dataset_id": 3, "adapter": "channel_folders"}',),
    )

    job = fail_stale_running_jobs(migrated_db)[0]

    assert job.kind is JobKind.IMPORT
    assert job.experiment_id is None
    assert job.params == {"dataset_id": 3, "adapter": "channel_folders"}
