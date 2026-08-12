"""Experiment repository.

An experiment freezes its configuration at creation. Nothing here offers a way to edit
`model_config`, `preprocessing_config` or `split_id` afterwards, because a stored metric
that was computed under different settings than the row claims is worse than no metric —
and the schema already refuses to let the dataset or split be deleted out from under one.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Mapping
from typing import Any, Literal

from anomaly_lab.domain.entities import Experiment, ExperimentStatus


def _to_experiment(row: sqlite3.Row) -> Experiment:
    return Experiment.model_validate(dict(row))


def get_experiment(conn: sqlite3.Connection, experiment_id: int) -> Experiment | None:
    row = conn.execute("SELECT * FROM experiment WHERE id = ?", (experiment_id,)).fetchone()
    return _to_experiment(row) if row is not None else None


def list_experiments(
    conn: sqlite3.Connection,
    *,
    dataset_id: int | None = None,
    model_type: str | None = None,
    status: ExperimentStatus | None = None,
    query: str | None = None,
    sort: Literal["newest", "oldest", "name"] = "newest",
    limit: int = 200,
) -> list[Experiment]:
    """Experiments matching the catalogue's composable filters."""
    clauses: list[str] = []
    params: list[object] = []
    if dataset_id is not None:
        clauses.append("dataset_id = ?")
        params.append(dataset_id)
    if model_type is not None:
        clauses.append("model_type = ?")
        params.append(model_type)
    if status is not None:
        clauses.append("status = ?")
        params.append(status.value)
    if query is not None and query.strip():
        # Search is intentionally limited to human-authored text. A method has its own
        # exact filter, and searching serialized configuration would make a typo look like
        # a supported query language.
        clauses.append("(name LIKE ? COLLATE NOCASE OR notes LIKE ? COLLATE NOCASE)")
        needle = f"%{query.strip()}%"
        params.extend([needle, needle])

    order_by = {
        "newest": "id DESC",
        "oldest": "id ASC",
        "name": "name COLLATE NOCASE ASC, id DESC",
    }[sort]
    where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
    params.append(limit)
    rows = conn.execute(
        f"SELECT * FROM experiment{where} ORDER BY {order_by} LIMIT ?",
        params,
    ).fetchall()
    return [_to_experiment(row) for row in rows]


def list_experiments_for_dataset(conn: sqlite3.Connection, dataset_id: int) -> list[Experiment]:
    """Every experiment owned by one dataset, without catalogue pagination."""
    rows = conn.execute(
        "SELECT * FROM experiment WHERE dataset_id = ? ORDER BY id", (dataset_id,)
    ).fetchall()
    return [_to_experiment(row) for row in rows]


def create_experiment(
    conn: sqlite3.Connection,
    *,
    name: str,
    dataset_id: int,
    split_id: int,
    model_type: str,
    model_config: Mapping[str, Any],
    preprocessing_config: Mapping[str, Any],
    eval_config: Mapping[str, Any],
    artifact_dir: str,
    notes: str | None = None,
) -> Experiment:
    cursor = conn.execute(
        """
        INSERT INTO experiment
               (name, dataset_id, split_id, model_type, model_config,
                preprocessing_config, eval_config, artifact_dir, notes)
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            name,
            dataset_id,
            split_id,
            model_type,
            json.dumps(dict(model_config), sort_keys=True),
            json.dumps(dict(preprocessing_config), sort_keys=True),
            json.dumps(dict(eval_config), sort_keys=True),
            artifact_dir,
            notes,
        ),
    )
    created = get_experiment(conn, int(cursor.lastrowid or 0))
    if created is None:  # pragma: no cover - the insert above just succeeded
        msg = "the experiment row vanished immediately after insertion"
        raise RuntimeError(msg)
    return created


def set_status(
    conn: sqlite3.Connection,
    experiment_id: int,
    status: ExperimentStatus,
) -> Experiment | None:
    conn.execute(
        "UPDATE experiment SET status = ? WHERE id = ?",
        (status.value, experiment_id),
    )
    return get_experiment(conn, experiment_id)


def fail_stale_training_experiments(conn: sqlite3.Connection) -> list[int]:
    """Reconcile experiments left mid-training by a process that is gone.

    The sibling of `jobs.fail_stale_running_jobs`, and needed for the same reason: only
    one job runs at a time, so at startup nothing is running, and an experiment still
    reading `training` is the record of a run that died. Without this it reads `training`
    for ever — which is not merely cosmetic, because `training` is indistinguishable on
    screen from a run that is genuinely in progress.

    Returns the ids it corrected, so startup can say what it found.
    """
    rows = conn.execute(
        "SELECT id FROM experiment WHERE status = ?",
        (ExperimentStatus.TRAINING.value,),
    ).fetchall()
    stale = [int(row["id"]) for row in rows]
    if stale:
        conn.execute(
            "UPDATE experiment SET status = ? WHERE status = ?",
            (ExperimentStatus.FAILED.value, ExperimentStatus.TRAINING.value),
        )
    return stale


def delete_experiment(conn: sqlite3.Connection, experiment_id: int) -> bool:
    """Delete an experiment and everything cascading from it.

    Artifacts on disk are *not* removed here: the repository owns rows, and a filesystem
    deletion inside a database transaction cannot be rolled back with it. The caller
    removes the directory after the transaction commits.
    """
    cursor = conn.execute("DELETE FROM experiment WHERE id = ?", (experiment_id,))
    return cursor.rowcount > 0
