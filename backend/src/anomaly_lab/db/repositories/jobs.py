"""Job repository.

The `job` table is the queue's single source of truth (ADR-0009). The in-process runner
holds no pending list of its own — it asks this module for the next queued row — so the
"two sources of truth" the ADR accepts as a cost is reduced to just the currently running
child process.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Mapping
from typing import Any

from anomaly_lab.domain.entities import Job, JobKind, JobStatus

STALE_JOB_ERROR = "The backend restarted while this job was running; its worker process is gone."


def _to_job(row: sqlite3.Row) -> Job:
    return Job.model_validate(dict(row))


def _dump(payload: Mapping[str, Any] | None) -> str:
    return json.dumps(dict(payload or {}), sort_keys=True)


def create_job(
    conn: sqlite3.Connection,
    *,
    kind: JobKind,
    params: Mapping[str, Any] | None = None,
    experiment_id: int | None = None,
) -> Job:
    """Enqueue a job. It becomes visible — and cancellable — before it starts."""
    cursor = conn.execute(
        """
        INSERT INTO job (kind, experiment_id, status, params)
             VALUES (?, ?, 'queued', ?)
        """,
        (kind.value, experiment_id, _dump(params)),
    )
    created = get_job(conn, int(cursor.lastrowid or 0))
    if created is None:  # pragma: no cover - the insert above just succeeded
        msg = "the job row vanished immediately after insertion"
        raise RuntimeError(msg)
    return created


def get_job(conn: sqlite3.Connection, job_id: int) -> Job | None:
    row = conn.execute("SELECT * FROM job WHERE id = ?", (job_id,)).fetchone()
    return _to_job(row) if row is not None else None


def list_jobs(
    conn: sqlite3.Connection,
    *,
    kind: JobKind | None = None,
    status: JobStatus | None = None,
    limit: int = 50,
) -> list[Job]:
    clauses: list[str] = []
    params: list[object] = []
    if kind is not None:
        clauses.append("kind = ?")
        params.append(kind.value)
    if status is not None:
        clauses.append("status = ?")
        params.append(status.value)
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""

    rows = conn.execute(
        f"SELECT * FROM job {where} ORDER BY id DESC LIMIT ?",
        [*params, limit],
    ).fetchall()
    return [_to_job(row) for row in rows]


def list_jobs_for_experiment(
    conn: sqlite3.Connection,
    experiment_id: int,
    *,
    limit: int = 50,
) -> list[Job]:
    """An experiment's jobs, newest first — its history of training and scoring runs."""
    rows = conn.execute(
        "SELECT * FROM job WHERE experiment_id = ? ORDER BY id DESC LIMIT ?",
        (experiment_id, limit),
    ).fetchall()
    return [_to_job(row) for row in rows]


def next_queued_job(conn: sqlite3.Connection) -> Job | None:
    """The oldest queued job. FIFO, one at a time, no priority lane (ADR-0009)."""
    row = conn.execute("SELECT * FROM job WHERE status = 'queued' ORDER BY id LIMIT 1").fetchone()
    return _to_job(row) if row is not None else None


def mark_running(conn: sqlite3.Connection, job_id: int, *, log_path: str) -> None:
    conn.execute(
        """
        UPDATE job
           SET status = 'running',
               log_path = ?,
               started_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
         WHERE id = ?
        """,
        (log_path, job_id),
    )


def update_progress(
    conn: sqlite3.Connection,
    job_id: int,
    *,
    fraction: float,
    message: str | None,
) -> None:
    """Mirror a `progress` event onto the row so a REST poll is always accurate.

    Clamped rather than validated: a worker reporting 1.0000001 should not abort a job
    on a CHECK constraint.
    """
    conn.execute(
        """
        UPDATE job
           SET progress = ?,
               message = COALESCE(?, message)
         WHERE id = ?
        """,
        (min(max(fraction, 0.0), 1.0), message, job_id),
    )


def finish_job(
    conn: sqlite3.Connection,
    job_id: int,
    *,
    status: JobStatus,
    result: Mapping[str, Any] | None = None,
    error: str | None = None,
    message: str | None = None,
) -> None:
    """Record a terminal state. Succeeded jobs are pinned to full progress."""
    conn.execute(
        """
        UPDATE job
           SET status = ?,
               result = ?,
               error = ?,
               message = COALESCE(?, message),
               progress = CASE WHEN ? = 'succeeded' THEN 1.0 ELSE progress END,
               finished_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
         WHERE id = ?
        """,
        (status.value, _dump(result), error, message, status.value, job_id),
    )


def cancel_queued_job(conn: sqlite3.Connection, job_id: int) -> bool:
    """Cancel a job that has not started. Returns False if it was already running."""
    cursor = conn.execute(
        """
        UPDATE job
           SET status = 'cancelled',
               finished_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
         WHERE id = ? AND status = 'queued'
        """,
        (job_id,),
    )
    return cursor.rowcount > 0


def fail_stale_running_jobs(conn: sqlite3.Connection) -> list[Job]:
    """Reconcile jobs left `running` by a crash or a hard kill (ADR-0009).

    A worker is a child of the backend process, so any job still marked `running` at
    startup is one whose owner is provably gone. Transitioning it to `failed` here is
    what keeps the UI from showing a phantom running job forever.
    """
    ids = [
        int(row["id"])
        for row in conn.execute("SELECT id FROM job WHERE status = 'running' ORDER BY id")
    ]
    if not ids:
        return []

    conn.execute(
        """
        UPDATE job
           SET status = 'failed',
               error = ?,
               finished_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
         WHERE status = 'running'
        """,
        (STALE_JOB_ERROR,),
    )

    # Only placeholders are interpolated; every id is bound as a parameter.
    placeholders = ",".join("?" * len(ids))
    rows = conn.execute(
        f"SELECT * FROM job WHERE id IN ({placeholders}) ORDER BY id",
        ids,
    ).fetchall()
    return [_to_job(row) for row in rows]
