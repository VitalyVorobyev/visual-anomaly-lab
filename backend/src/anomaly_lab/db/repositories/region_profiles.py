"""Immutable dataset-owned region profile revisions (ADR-0033)."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Mapping
from typing import Any

from anomaly_lab.domain.entities import RegionProfileRevision, SpatialResample


def _to_profile(row: sqlite3.Row) -> RegionProfileRevision:
    return RegionProfileRevision.model_validate(dict(row))


def get_profile(conn: sqlite3.Connection, profile_id: int) -> RegionProfileRevision | None:
    row = conn.execute(
        "SELECT * FROM region_profile_revision WHERE id = ?", (profile_id,)
    ).fetchone()
    return _to_profile(row) if row is not None else None


def list_profiles(conn: sqlite3.Connection, dataset_id: int) -> list[RegionProfileRevision]:
    rows = conn.execute(
        """
        SELECT * FROM region_profile_revision
         WHERE dataset_id = ?
         ORDER BY name COLLATE NOCASE, revision_no DESC, id DESC
        """,
        (dataset_id,),
    ).fetchall()
    return [_to_profile(row) for row in rows]


def experiments_pinning(conn: sqlite3.Connection, profile_id: int) -> list[tuple[int, str]]:
    """The experiments that reference this revision, newest first.

    `experiment.region_profile_id` is `ON DELETE RESTRICT`, so SQLite would refuse the
    delete on its own — but as an `IntegrityError` naming a constraint, which is not an
    answer anyone can act on. Asking first turns it into a message naming the runs that
    hold the profile.
    """
    rows = conn.execute(
        """
        SELECT id, name FROM experiment
         WHERE region_profile_id = ?
         ORDER BY id DESC
        """,
        (profile_id,),
    ).fetchall()
    return [(int(row["id"]), str(row["name"])) for row in rows]


def delete_revision(conn: sqlite3.Connection, profile_id: int) -> bool:
    """Delete one revision's row, and the `region_prepare` jobs that named it.

    Rows only. Prepared pixels live on disk and a filesystem deletion cannot be rolled
    back with this transaction, so the caller removes the directory after the commit.

    The job rows are swept here because nothing else references them: a `region_prepare`
    job carries its profile in `params` JSON, not in a foreign key, so no cascade reaches
    it and it would otherwise outlive the profile it describes.
    """
    conn.execute(
        """
        DELETE FROM job
         WHERE kind = 'region_prepare'
           AND CAST(json_extract(params, '$.profile_id') AS INTEGER) = ?
        """,
        (profile_id,),
    )
    cursor = conn.execute("DELETE FROM region_profile_revision WHERE id = ?", (profile_id,))
    return cursor.rowcount > 0


def create_revision(
    conn: sqlite3.Connection,
    *,
    dataset_id: int,
    name: str,
    extractor_type: str,
    extractor_config: Mapping[str, Any],
    prepared_width: int,
    prepared_height: int,
    padding_fraction: float,
    resample: SpatialResample = SpatialResample.BILINEAR,
    seed: int,
) -> RegionProfileRevision:
    row = conn.execute(
        """
        SELECT COALESCE(MAX(revision_no), 0) + 1 AS next_revision
          FROM region_profile_revision
         WHERE dataset_id = ? AND name = ? COLLATE NOCASE
        """,
        (dataset_id, name),
    ).fetchone()
    revision_no = int(row["next_revision"])
    cursor = conn.execute(
        """
        INSERT INTO region_profile_revision (
            dataset_id, name, revision_no, extractor_type, extractor_config,
            prepared_width, prepared_height, padding_fraction, failure_policy, seed, resample
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'fail', ?, ?)
        """,
        (
            dataset_id,
            name,
            revision_no,
            extractor_type,
            json.dumps(dict(extractor_config), sort_keys=True),
            prepared_width,
            prepared_height,
            padding_fraction,
            seed,
            resample.value,
        ),
    )
    created = get_profile(conn, int(cursor.lastrowid or 0))
    if created is None:  # pragma: no cover
        raise RuntimeError("region profile row vanished immediately after insertion")
    return created
