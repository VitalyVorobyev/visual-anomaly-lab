"""Sample repository.

The sample is the unit of identity, labelling and splitting (ADR-0005), so every filter
this module offers is expressed over samples even when the user is thinking about images:
filtering by channel means "samples having an image in this channel", never "these
images".
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

from anomaly_lab.domain.entities import Label, LabelSource, Sample, Subset


def _to_sample(row: sqlite3.Row) -> Sample:
    return Sample.model_validate(dict(row))


@dataclass(frozen=True)
class SampleFilter:
    """Browser filters. Every field is optional and they compose with AND."""

    label: Label | None = None
    channel_id: int | None = None
    split_id: int | None = None
    subset: Subset | None = None


def _where(dataset_id: int, filters: SampleFilter) -> tuple[str, list[object]]:
    clauses = ["sample.dataset_id = ?"]
    params: list[object] = [dataset_id]

    if filters.label is not None:
        clauses.append("sample.label = ?")
        params.append(filters.label.value)

    if filters.channel_id is not None:
        clauses.append(
            "EXISTS (SELECT 1 FROM image"
            " WHERE image.sample_id = sample.id AND image.channel_id = ?)"
        )
        params.append(filters.channel_id)

    # A subset without a split is meaningless, so the split id carries the join and the
    # subset narrows it.
    if filters.split_id is not None:
        clauses.append(
            "EXISTS (SELECT 1 FROM split_assignment"
            " WHERE split_assignment.sample_id = sample.id"
            " AND split_assignment.split_id = ?"
            + (" AND split_assignment.subset = ?" if filters.subset is not None else "")
            + ")"
        )
        params.append(filters.split_id)
        if filters.subset is not None:
            params.append(filters.subset.value)

    return " AND ".join(clauses), params


def count_samples(
    conn: sqlite3.Connection,
    dataset_id: int,
    filters: SampleFilter | None = None,
) -> int:
    where, params = _where(dataset_id, filters or SampleFilter())
    row = conn.execute(f"SELECT COUNT(*) AS n FROM sample WHERE {where}", params).fetchone()
    return int(row["n"])


def list_samples(
    conn: sqlite3.Connection,
    dataset_id: int,
    filters: SampleFilter | None = None,
    *,
    limit: int = 200,
    offset: int = 0,
) -> list[Sample]:
    """A page of samples in stable order.

    Ordered by `(group_key, external_id)` rather than by id so that the grid's scroll
    position means the same thing after a re-import inserts new rows in the middle.
    """
    where, params = _where(dataset_id, filters or SampleFilter())
    rows = conn.execute(
        f"""
        SELECT *
          FROM sample
         WHERE {where}
         ORDER BY group_key, LENGTH(external_id), external_id
         LIMIT ? OFFSET ?
        """,
        [*params, limit, offset],
    ).fetchall()
    return [_to_sample(row) for row in rows]


def get_sample(conn: sqlite3.Connection, sample_id: int) -> Sample | None:
    row = conn.execute("SELECT * FROM sample WHERE id = ?", (sample_id,)).fetchone()
    return _to_sample(row) if row is not None else None


def find_sample(
    conn: sqlite3.Connection,
    dataset_id: int,
    *,
    group_key: str,
    external_id: str,
) -> Sample | None:
    row = conn.execute(
        """
        SELECT * FROM sample
         WHERE dataset_id = ? AND group_key = ? AND external_id = ?
        """,
        (dataset_id, group_key, external_id),
    ).fetchone()
    return _to_sample(row) if row is not None else None


def upsert_sample(
    conn: sqlite3.Connection,
    dataset_id: int,
    *,
    group_key: str,
    external_id: str,
    label: Label,
) -> tuple[Sample, bool]:
    """Insert or update a sample by its natural key. Returns `(sample, created)`.

    A label the operator set by hand is never overwritten by an imported guess: if
    `label_source` is already `manual`, the incoming label is ignored. This is what makes
    re-importing a corrected dataset safe (ADR-0013).
    """
    existing = find_sample(conn, dataset_id, group_key=group_key, external_id=external_id)
    if existing is None:
        cursor = conn.execute(
            """
            INSERT INTO sample (dataset_id, group_key, external_id, label, label_source)
                 VALUES (?, ?, ?, ?, ?)
            """,
            (dataset_id, group_key, external_id, label.value, LabelSource.IMPORT.value),
        )
        created = get_sample(conn, int(cursor.lastrowid or 0))
        if created is None:  # pragma: no cover - the insert above just succeeded
            msg = "the sample row vanished immediately after insertion"
            raise RuntimeError(msg)
        return created, True

    if existing.label_source is not LabelSource.MANUAL and existing.label is not label:
        conn.execute(
            "UPDATE sample SET label = ? WHERE id = ?",
            (label.value, existing.id),
        )
        refreshed = get_sample(conn, existing.id)
        if refreshed is not None:
            return refreshed, False

    return existing, False


def set_label(conn: sqlite3.Connection, sample_id: int, label: Label) -> Sample | None:
    """Record an operator's label, marking it as manual so re-import cannot undo it."""
    conn.execute(
        "UPDATE sample SET label = ?, label_source = ? WHERE id = ?",
        (label.value, LabelSource.MANUAL.value, sample_id),
    )
    return get_sample(conn, sample_id)


def list_group_keys(conn: sqlite3.Connection, dataset_id: int) -> list[str]:
    rows = conn.execute(
        "SELECT DISTINCT group_key FROM sample WHERE dataset_id = ? ORDER BY group_key",
        (dataset_id,),
    ).fetchall()
    return [str(row["group_key"]) for row in rows]
