"""Job repository."""

from __future__ import annotations

import sqlite3

from anomaly_lab.domain.entities import Job

STALE_JOB_ERROR = "The backend restarted while this job was running; its worker process is gone."


def _to_job(row: sqlite3.Row) -> Job:
    return Job.model_validate(dict(row))


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
