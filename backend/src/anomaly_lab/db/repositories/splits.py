"""Split repository.

Assignments are sample-level and nothing here offers an image-level equivalent — that
absence is the mechanism that makes cross-channel leakage impossible (ADR-0041).
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Mapping
from typing import Any

from anomaly_lab.db.connection import transaction
from anomaly_lab.domain.entities import Split, Subset


def _to_split(row: sqlite3.Row) -> Split:
    return Split.model_validate(dict(row))


def list_splits(conn: sqlite3.Connection, dataset_id: int) -> list[Split]:
    rows = conn.execute(
        "SELECT * FROM split WHERE dataset_id = ? ORDER BY id",
        (dataset_id,),
    ).fetchall()
    return [_to_split(row) for row in rows]


def get_split(conn: sqlite3.Connection, split_id: int) -> Split | None:
    row = conn.execute("SELECT * FROM split WHERE id = ?", (split_id,)).fetchone()
    return _to_split(row) if row is not None else None


def create_split(
    conn: sqlite3.Connection,
    dataset_id: int,
    *,
    name: str,
    strategy: str,
    seed: int,
    params: Mapping[str, Any],
    assignments: Mapping[int, Subset],
) -> Split:
    """Create a split and its assignments in one transaction.

    A split with only some of its assignments written would be silently wrong rather than
    visibly broken, so the two writes are atomic.
    """
    with transaction(conn):
        cursor = conn.execute(
            """
            INSERT INTO split (dataset_id, name, strategy, seed, params)
                 VALUES (?, ?, ?, ?, ?)
            """,
            (dataset_id, name, strategy, seed, json.dumps(dict(params), sort_keys=True)),
        )
        split_id = int(cursor.lastrowid or 0)
        conn.executemany(
            "INSERT INTO split_assignment (split_id, sample_id, subset) VALUES (?, ?, ?)",
            [(split_id, sample_id, subset.value) for sample_id, subset in assignments.items()],
        )

    created = get_split(conn, split_id)
    if created is None:  # pragma: no cover - the insert above just committed
        msg = "the split row vanished immediately after insertion"
        raise RuntimeError(msg)
    return created


def assignments(conn: sqlite3.Connection, split_id: int) -> dict[int, Subset]:
    """Every sample's subset in this split, keyed by sample id."""
    rows = conn.execute(
        "SELECT sample_id, subset FROM split_assignment WHERE split_id = ?",
        (split_id,),
    ).fetchall()
    return {int(row["sample_id"]): Subset(row["subset"]) for row in rows}


def experiments_using(conn: sqlite3.Connection, split_id: int) -> list[tuple[int, str]]:
    """The experiments that ran on this split, newest first.

    `experiment.split_id` is `ON DELETE RESTRICT`, so SQLite would refuse the delete on its
    own — but as an `IntegrityError` naming a constraint. Asking first turns it into a
    message naming the runs that hold the split.
    """
    rows = conn.execute(
        "SELECT id, name FROM experiment WHERE split_id = ? ORDER BY id DESC",
        (split_id,),
    ).fetchall()
    return [(int(row["id"]), str(row["name"])) for row in rows]


def delete_split(conn: sqlite3.Connection, split_id: int) -> bool:
    """Delete a split; its assignments go with it (`ON DELETE CASCADE`)."""
    cursor = conn.execute("DELETE FROM split WHERE id = ?", (split_id,))
    return cursor.rowcount > 0


def list_sample_ids(
    conn: sqlite3.Connection,
    split_id: int,
    subset: Subset | None = None,
) -> list[int]:
    if subset is None:
        rows = conn.execute(
            "SELECT sample_id FROM split_assignment WHERE split_id = ? ORDER BY sample_id",
            (split_id,),
        ).fetchall()
    else:
        rows = conn.execute(
            """
            SELECT sample_id
              FROM split_assignment
             WHERE split_id = ? AND subset = ?
             ORDER BY sample_id
            """,
            (split_id, subset.value),
        ).fetchall()
    return [int(row["sample_id"]) for row in rows]
