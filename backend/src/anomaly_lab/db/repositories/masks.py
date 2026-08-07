"""Mask repository.

Masks are pixel-level ground truth, one row per annotated image. The table has existed
since schema v1 and went unpopulated for as long as the only dataset on hand had no
annotations; a dataset that ships masks is what finally fills it, with no migration.

Two things differ from the image repository, both forced by the frozen schema (ADR-0004):

  * There is no `UNIQUE (image_id, kind)` constraint, so the upsert reads before it
    writes rather than leaning on `ON CONFLICT`.
  * There is no `sha256` column, so a mask can be checked for existence and not for
    drift. `verify` says exactly that rather than implying a check it did not make.
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
            "INSERT INTO mask (image_id, path, kind) VALUES (?, ?, ?)",
            (image_id, path, kind.value),
        )
        return Mask(id=int(cursor.lastrowid or 0), image_id=image_id, path=path, kind=kind), True

    if existing["path"] != path:
        conn.execute("UPDATE mask SET path = ? WHERE id = ?", (path, existing["id"]))
        return Mask(id=int(existing["id"]), image_id=image_id, path=path, kind=kind), False

    return _to_mask(existing), False
