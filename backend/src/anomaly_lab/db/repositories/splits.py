"""Split repository.

Assignments are sample-level and nothing here offers an image-level equivalent — that
absence is the mechanism that makes cross-channel leakage impossible (ADR-0005).
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from anomaly_lab.domain.entities import Label, Split, Subset


def _to_split(row: sqlite3.Row) -> Split:
    return Split.model_validate(dict(row))


@dataclass(frozen=True)
class SubsetComposition:
    """What a subset actually contains, so a split can report itself honestly."""

    subset: Subset
    total: int
    normal: int
    defect: int
    unlabeled: int


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
    conn.execute("BEGIN")
    try:
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
    except Exception:
        conn.execute("ROLLBACK")
        raise
    conn.execute("COMMIT")

    created = get_split(conn, split_id)
    if created is None:  # pragma: no cover - the insert above just committed
        msg = "the split row vanished immediately after insertion"
        raise RuntimeError(msg)
    return created


def composition(conn: sqlite3.Connection, split_id: int) -> list[SubsetComposition]:
    """Per-subset counts by label, with every subset present even when empty."""
    counts: dict[Subset, dict[Label, int]] = {subset: dict.fromkeys(Label, 0) for subset in Subset}
    rows = conn.execute(
        """
        SELECT split_assignment.subset AS subset,
               sample.label            AS label,
               COUNT(*)                AS n
          FROM split_assignment
          JOIN sample ON sample.id = split_assignment.sample_id
         WHERE split_assignment.split_id = ?
         GROUP BY split_assignment.subset, sample.label
        """,
        (split_id,),
    ).fetchall()
    for row in rows:
        counts[Subset(row["subset"])][Label(row["label"])] = int(row["n"])

    return [
        SubsetComposition(
            subset=subset,
            total=sum(by_label.values()),
            normal=by_label[Label.NORMAL],
            defect=by_label[Label.DEFECT],
            unlabeled=by_label[Label.UNLABELED],
        )
        for subset, by_label in counts.items()
    ]


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
