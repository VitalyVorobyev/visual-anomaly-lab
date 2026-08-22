"""Sample repository.

The sample is the unit of identity, labelling and splitting (ADR-0005), so every filter
this module offers is expressed over samples even when the user is thinking about images:
filtering by channel means "samples having an image in this channel", never "these
images".
"""

from __future__ import annotations

import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass

from anomaly_lab.db.repositories import annotations as annotations_repo
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
    annotated: bool | None = None


_SAMPLE_MISSING_TRUTH = (
    "EXISTS (SELECT 1 FROM image"
    f"       WHERE image.sample_id = sample.id AND NOT {annotations_repo.IMAGE_HAS_TRUTH})"
)
_SAMPLE_HAS_IMAGES = "EXISTS (SELECT 1 FROM image WHERE image.sample_id = sample.id)"


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

    if filters.annotated is not None:
        # A sample counts as annotated only when *every* one of its images has truth: a
        # part whose dark-field view is still blank is unfinished work, and a queue that
        # called it done would quietly drop it.
        clauses.append(
            _SAMPLE_MISSING_TRUTH
            if not filters.annotated
            else f"{_SAMPLE_HAS_IMAGES} AND NOT {_SAMPLE_MISSING_TRUTH}"
        )

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
    notes: str | None = None,
) -> tuple[Sample, bool]:
    """Insert or update a sample by its natural key. Returns `(sample, created)`.

    A label the operator set by hand is never overwritten by an imported guess: if
    `label_source` is already `manual`, the incoming label is ignored. This is what makes
    re-importing a corrected dataset safe (handbook import.md).

    `notes` follows the same instinct one step further: an import writes them when it has
    something to say and **never clears them**, because `None` from an adapter means "this
    scan read no defect type", not "there is no defect type". Deleting on absence would
    make an import with the wrong options destructive.
    """
    existing = find_sample(conn, dataset_id, group_key=group_key, external_id=external_id)
    if existing is None:
        cursor = conn.execute(
            """
            INSERT INTO sample (dataset_id, group_key, external_id, label, label_source, notes)
                 VALUES (?, ?, ?, ?, ?, ?)
            """,
            (dataset_id, group_key, external_id, label.value, LabelSource.IMPORT.value, notes),
        )
        created = get_sample(conn, int(cursor.lastrowid or 0))
        if created is None:  # pragma: no cover - the insert above just succeeded
            msg = "the sample row vanished immediately after insertion"
            raise RuntimeError(msg)
        return created, True

    changed = False
    if existing.label_source is not LabelSource.MANUAL and existing.label is not label:
        conn.execute("UPDATE sample SET label = ? WHERE id = ?", (label.value, existing.id))
        changed = True
    if notes is not None and notes != existing.notes:
        conn.execute("UPDATE sample SET notes = ? WHERE id = ?", (notes, existing.id))
        changed = True

    if changed:
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


# SQLite caps host parameters per statement — 999 on builds predating 3.32 — and a
# selection can be larger than one page of the grid. Chunking keeps that limit from
# becoming the caller's problem.
_MAX_IDS_PER_STATEMENT = 500


def set_labels(
    conn: sqlite3.Connection,
    dataset_id: int,
    sample_ids: Sequence[int],
    label: Label,
) -> int:
    """Label an explicit selection, returning how many samples now carry the label.

    Scoped by `dataset_id` as well as by id, so a request cannot reach across datasets by
    naming ids belonging to another one. The chunks run inside one transaction: a bulk
    edit that half-applied would be worse than one that failed.
    """
    if not sample_ids:
        return 0

    updated = 0
    conn.execute("BEGIN")
    try:
        for start in range(0, len(sample_ids), _MAX_IDS_PER_STATEMENT):
            chunk = sample_ids[start : start + _MAX_IDS_PER_STATEMENT]
            placeholders = ", ".join("?" * len(chunk))
            cursor = conn.execute(
                f"""
                UPDATE sample
                   SET label = ?, label_source = ?
                 WHERE dataset_id = ? AND id IN ({placeholders})
                """,
                [label.value, LabelSource.MANUAL.value, dataset_id, *chunk],
            )
            updated += cursor.rowcount
    except Exception:
        conn.execute("ROLLBACK")
        raise
    conn.execute("COMMIT")
    return updated


def set_labels_matching(
    conn: sqlite3.Connection,
    dataset_id: int,
    filters: SampleFilter,
    label: Label,
) -> int:
    """Label every sample matching a browser filter, returning how many.

    Built from the same `_where` clause the grid pages with, so "label everything I am
    looking at" and "what I am looking at" cannot drift apart — which is the whole reason
    this is a filter rather than a list of ids the client assembled for itself.
    """
    where, params = _where(dataset_id, filters)
    cursor = conn.execute(
        f"UPDATE sample SET label = ?, label_source = ? WHERE {where}",
        [label.value, LabelSource.MANUAL.value, *params],
    )
    return cursor.rowcount


def list_group_keys(conn: sqlite3.Connection, dataset_id: int) -> list[str]:
    rows = conn.execute(
        "SELECT DISTINCT group_key FROM sample WHERE dataset_id = ? ORDER BY group_key",
        (dataset_id,),
    ).fetchall()
    return [str(row["group_key"]) for row in rows]
