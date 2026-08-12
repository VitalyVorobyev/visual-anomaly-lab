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
