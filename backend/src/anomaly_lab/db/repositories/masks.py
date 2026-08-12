"""Mask repository.

Masks are pixel-level ground truth, one row per annotated image. The table has existed
since schema v1 and went unpopulated for as long as the only dataset on hand had no
annotations; a dataset that ships masks is what finally fills it, with no migration.

One thing differs from the image repository, forced by the original schema (ADR-0004):

  * There is no `UNIQUE (image_id, kind)` constraint, so the upsert reads before it
    writes rather than leaning on `ON CONFLICT`.
Migration 005 added a nullable digest without pretending old catalog rows had already
been verified. Annotation draft creation fills it when that source mask becomes an
explicit base.
"""

from __future__ import annotations

import sqlite3

from anomaly_lab.domain.entities import Mask, MaskKind

# Same reasoning as the image repository: one page of samples asks for its masks at once.
_MAX_BATCH = 900


def _to_mask(row: sqlite3.Row) -> Mask:
    return Mask.model_validate(dict(row))


def list_masks_for_images(conn: sqlite3.Connection, image_ids: list[int]) -> dict[int, list[Mask]]:
    """Masks for a set of images, grouped by image id. Images with none map to `[]`."""
    grouped: dict[int, list[Mask]] = {image_id: [] for image_id in image_ids}
    for start in range(0, len(image_ids), _MAX_BATCH):
        chunk = image_ids[start : start + _MAX_BATCH]
        if not chunk:
            continue
        placeholders = ",".join("?" * len(chunk))
        rows = conn.execute(
            f"SELECT * FROM mask WHERE image_id IN ({placeholders}) ORDER BY image_id, id",
            chunk,
        ).fetchall()
        for row in rows:
            grouped[int(row["image_id"])].append(_to_mask(row))
    return grouped


def list_masks_for_dataset(conn: sqlite3.Connection, dataset_id: int) -> list[Mask]:
    """Every mask in a dataset — what `verify` walks alongside the images."""
    rows = conn.execute(
        """
        SELECT mask.*
          FROM mask
          JOIN image  ON image.id = mask.image_id
          JOIN sample ON sample.id = image.sample_id
         WHERE sample.dataset_id = ?
         ORDER BY mask.id
        """,
        (dataset_id,),
    ).fetchall()
    return [_to_mask(row) for row in rows]


def count_masks_for_dataset(conn: sqlite3.Connection, dataset_id: int) -> int:
    """How many images in this dataset carry ground truth. Drives the pixel-metric path."""
    row = conn.execute(
        """
        SELECT COUNT(*) AS n
          FROM mask
          JOIN image  ON image.id = mask.image_id
          JOIN sample ON sample.id = image.sample_id
         WHERE sample.dataset_id = ?
        """,
        (dataset_id,),
    ).fetchone()
    return int(row["n"]) if row is not None else 0


def upsert_mask(
    conn: sqlite3.Connection,
    image_id: int,
    *,
    path: str,
    kind: MaskKind = MaskKind.GROUND_TRUTH,
    sha256: str | None = None,
) -> tuple[Mask, bool]:
    """Insert or repoint one image's mask of a given kind. Returns `(mask, created)`.

    Identity is `(image_id, kind)` rather than `(image_id, path)`: re-importing a dataset
    whose annotations moved should repoint the row it already has, not accumulate a second
    mask claiming the same thing about the same pixels.
    """
    existing = conn.execute(
        "SELECT * FROM mask WHERE image_id = ? AND kind = ? ORDER BY id LIMIT 1",
        (image_id, kind.value),
    ).fetchone()

    if existing is None:
        cursor = conn.execute(
            "INSERT INTO mask (image_id, path, kind, sha256) VALUES (?, ?, ?, ?)",
            (image_id, path, kind.value, sha256),
        )
        return (
            Mask(
                id=int(cursor.lastrowid or 0),
                image_id=image_id,
                path=path,
                kind=kind,
                sha256=sha256,
            ),
            True,
        )

    if existing["path"] != path or (sha256 is not None and existing["sha256"] != sha256):
        conn.execute(
            "UPDATE mask SET path = ?, sha256 = ? WHERE id = ?",
            (path, sha256, existing["id"]),
        )
        return (
            Mask(
                id=int(existing["id"]),
                image_id=image_id,
                path=path,
                kind=kind,
                sha256=sha256,
            ),
            False,
        )

    return _to_mask(existing), False


def get_mask_for_image(conn: sqlite3.Connection, image_id: int) -> Mask | None:
    row = conn.execute(
        "SELECT * FROM mask WHERE image_id = ? AND kind = ? ORDER BY id LIMIT 1",
        (image_id, MaskKind.GROUND_TRUTH.value),
    ).fetchone()
    return _to_mask(row) if row is not None else None


def record_sha256(conn: sqlite3.Connection, mask_id: int, sha256: str) -> None:
    conn.execute("UPDATE mask SET sha256 = ? WHERE id = ?", (sha256, mask_id))
