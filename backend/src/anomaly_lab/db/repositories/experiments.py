"""Experiment repository.

An experiment freezes its configuration at creation. Nothing here offers a way to edit
`model_config`, `preprocessing_config` or `split_id` afterwards, because a stored metric
that was computed under different settings than the row claims is worse than no metric —
and the schema already refuses to let the dataset or split be deleted out from under one.
"""

from __future__ import annotations

import base64
import json
import sqlite3
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from anomaly_lab.domain.entities import Experiment, ExperimentStatus


def _to_experiment(row: sqlite3.Row) -> Experiment:
    return Experiment.model_validate(dict(row))


def get_experiment(conn: sqlite3.Connection, experiment_id: int) -> Experiment | None:
    row = conn.execute("SELECT * FROM experiment WHERE id = ?", (experiment_id,)).fetchone()
    return _to_experiment(row) if row is not None else None


# Each catalogue order is a column and a direction, with the id as the tie-break in the same
# direction, so a (value, id) pair is a total order a cursor can resume after.
_ORDERS: dict[str, tuple[str, str]] = {
    "newest": ("id", "DESC"),
    "oldest": ("id", "ASC"),
    "name": ("name COLLATE NOCASE", "ASC"),
    "method": ("model_type", "ASC"),
    "status": ("status", "ASC"),
}
ExperimentOrder = Literal["newest", "oldest", "name", "method", "status"]


@dataclass(frozen=True)
class ExperimentPage:
    items: list[Experiment]
    total: int
    """Every experiment the filters match, across all pages."""
    next_cursor: str | None
    """Where the next page starts, or `None` on the last one."""


def _encode_cursor(value: object, experiment_id: int) -> str:
    payload = json.dumps([value, experiment_id]).encode()
    return base64.urlsafe_b64encode(payload).decode().rstrip("=")


def _decode_cursor(cursor: str) -> tuple[object, int]:
    try:
        padded = cursor + "=" * (-len(cursor) % 4)
        value, experiment_id = json.loads(base64.urlsafe_b64decode(padded))
        return value, int(experiment_id)
    except (ValueError, TypeError) as exc:
        raise ValueError("the cursor is not one this catalogue issued") from exc


def list_experiments(
    conn: sqlite3.Connection,
    *,
    dataset_id: int | None = None,
    model_types: Sequence[str] = (),
    status: ExperimentStatus | None = None,
    query: str | None = None,
    created_from: str | None = None,
    created_to: str | None = None,
    sort: ExperimentOrder = "newest",
    limit: int = 200,
    cursor: str | None = None,
) -> ExperimentPage:
    """One page of the experiments matching the catalogue's composable filters.

    Keyset pagination rather than an offset: a run created while someone pages through the
    catalogue neither shifts every later page nor appears twice.
    """
    clauses: list[str] = []
    params: list[object] = []
    if dataset_id is not None:
        clauses.append("dataset_id = ?")
        params.append(dataset_id)
    if model_types:
        clauses.append(f"model_type IN ({','.join('?' * len(model_types))})")
        params.extend(model_types)
    if status is not None:
        clauses.append("status = ?")
        params.append(status.value)
    if query is not None and query.strip():
        # Search is intentionally limited to human-authored text — and the id, since a run is
        # named by its number everywhere else on screen. A method has its own exact filter, and
        # searching serialized configuration would make a typo look like a query language.
        needle = query.strip()
        text = "(name LIKE ? COLLATE NOCASE OR notes LIKE ? COLLATE NOCASE)"
        number = needle.removeprefix("#")
        if number.isdigit():
            clauses.append(f"({text} OR id = ?)")
            params.extend([f"%{needle}%", f"%{needle}%", int(number)])
        else:
            clauses.append(text)
            params.extend([f"%{needle}%", f"%{needle}%"])
    # Inclusive calendar days, compared against the stored UTC timestamp's date.
    if created_from is not None:
        clauses.append("substr(created_at, 1, 10) >= ?")
        params.append(created_from)
    if created_to is not None:
        clauses.append("substr(created_at, 1, 10) <= ?")
        params.append(created_to)

    where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
    total = int(conn.execute(f"SELECT COUNT(*) FROM experiment{where}", params).fetchone()[0])

    column, direction = _ORDERS[sort]
    keyset: list[str] = []
    keyset_params: list[object] = []
    if cursor is not None:
        value, last_id = _decode_cursor(cursor)
        beyond = ">" if direction == "ASC" else "<"
        if column == "id":
            keyset.append(f"id {beyond} ?")
            keyset_params.append(last_id)
        else:
            keyset.append(f"({column} {beyond} ? OR ({column} = ? AND id {beyond} ?))")
            keyset_params.extend([value, value, last_id])
    page_where = " WHERE " + " AND ".join(clauses + keyset) if clauses or keyset else ""
    rows = conn.execute(
        f"SELECT * FROM experiment{page_where} ORDER BY {column} {direction}, id {direction} "
        "LIMIT ?",
        [*params, *keyset_params, limit + 1],
    ).fetchall()

    items = [_to_experiment(row) for row in rows[:limit]]
    next_cursor = None
    if len(rows) > limit and items:
        last = rows[limit - 1]
        key = {"name COLLATE NOCASE": "name", "model_type": "model_type", "status": "status"}
        value = None if column == "id" else last[key[column]]
        next_cursor = _encode_cursor(value, int(last["id"]))
    return ExperimentPage(items=items, total=total, next_cursor=next_cursor)


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
    region_profile_id: int,
    region_manifest_sha256: str | None = None,
    model_type: str,
    model_config: Mapping[str, Any],
    task: str = "anomaly",
    target_label: str | None = None,
    classes: Sequence[str] = (),
    preprocessing_config: Mapping[str, Any],
    eval_config: Mapping[str, Any],
    artifact_dir: str,
    channels: Sequence[str] = (),
    notes: str | None = None,
) -> Experiment:
    cursor = conn.execute(
        """
        INSERT INTO experiment
               (name, dataset_id, split_id, region_profile_id,
                region_manifest_sha256, model_type, task, target_label, classes, model_config,
                preprocessing_config, eval_config, channels, artifact_dir, notes)
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            name,
            dataset_id,
            split_id,
            region_profile_id,
            region_manifest_sha256,
            model_type,
            task,
            target_label,
            json.dumps(list(classes)),
            json.dumps(dict(model_config), sort_keys=True),
            json.dumps(dict(preprocessing_config), sort_keys=True),
            json.dumps(dict(eval_config), sort_keys=True),
            json.dumps(list(channels)),
            artifact_dir,
            notes,
        ),
    )
    created = get_experiment(conn, int(cursor.lastrowid or 0))
    if created is None:  # pragma: no cover - the insert above just succeeded
        msg = "the experiment row vanished immediately after insertion"
        raise RuntimeError(msg)
    return created


class PinConflictError(ValueError):
    """An experiment already pins a different build; a pin is never moved."""


def pin_region_build(
    conn: sqlite3.Connection, experiment_id: int, manifest_sha256: str
) -> Experiment:
    """Pin the prepared-region build an experiment reads — once.

    Setting the pin a run already holds is a no-op; setting a different one is refused,
    because every stored score, map and checkpoint of the run was computed on the pinned
    pixels. The schema's trigger enforces the same rule beneath this function.
    """
    cursor = conn.execute(
        """
        UPDATE experiment SET region_manifest_sha256 = ?
         WHERE id = ? AND region_manifest_sha256 IS NULL
        """,
        (manifest_sha256, experiment_id),
    )
    current = get_experiment(conn, experiment_id)
    if current is None:
        raise ValueError(f"no experiment with id {experiment_id}")
    if cursor.rowcount == 0 and current.region_manifest_sha256 != manifest_sha256:
        raise PinConflictError(
            f"experiment {experiment_id} already pins region build "
            f"{current.region_manifest_sha256}; a pinned build never changes"
        )
    return current


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
