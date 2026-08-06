"""Image repository."""

from __future__ import annotations

import sqlite3

from anomaly_lab.domain.entities import Image

# The grid asks for one page of samples and then all of their images at once. Batching
# keeps that a single query instead of one per sample, and SQLite's default parameter
# limit is comfortably above any page size we serve.
_MAX_BATCH = 900


def _to_image(row: sqlite3.Row) -> Image:
    return Image.model_validate(dict(row))


def get_image(conn: sqlite3.Connection, image_id: int) -> Image | None:
    row = conn.execute("SELECT * FROM image WHERE id = ?", (image_id,)).fetchone()
    return _to_image(row) if row is not None else None


def list_images_for_sample(conn: sqlite3.Connection, sample_id: int) -> list[Image]:
    rows = conn.execute(
        """
        SELECT image.*
          FROM image
          LEFT JOIN channel ON channel.id = image.channel_id
         WHERE image.sample_id = ?
         ORDER BY channel.position, channel.name, image.id
        """,
        (sample_id,),
    ).fetchall()
    return [_to_image(row) for row in rows]


def list_images_for_samples(
    conn: sqlite3.Connection, sample_ids: list[int]
) -> dict[int, list[Image]]:
    """Images for a page of samples, grouped by sample id, in channel display order."""
    grouped: dict[int, list[Image]] = {sample_id: [] for sample_id in sample_ids}
    for start in range(0, len(sample_ids), _MAX_BATCH):
        chunk = sample_ids[start : start + _MAX_BATCH]
        if not chunk:
            continue
        # Only placeholders are interpolated; every id is bound as a parameter.
        placeholders = ",".join("?" * len(chunk))
        rows = conn.execute(
            f"""
            SELECT image.*
              FROM image
              LEFT JOIN channel ON channel.id = image.channel_id
             WHERE image.sample_id IN ({placeholders})
             ORDER BY image.sample_id, channel.position, channel.name, image.id
            """,
            chunk,
        ).fetchall()
        for row in rows:
            grouped[int(row["sample_id"])].append(_to_image(row))
    return grouped


def list_images_for_dataset(conn: sqlite3.Connection, dataset_id: int) -> list[Image]:
    """Every image in a dataset — the input to the prewarm and verify jobs."""
    rows = conn.execute(
        """
        SELECT image.*
          FROM image
          JOIN sample ON sample.id = image.sample_id
         WHERE sample.dataset_id = ?
         ORDER BY image.id
        """,
        (dataset_id,),
    ).fetchall()
    return [_to_image(row) for row in rows]


def upsert_image(
    conn: sqlite3.Connection,
    sample_id: int,
    *,
    channel_id: int | None,
    path: str,
    width: int,
    height: int,
    bit_depth: int,
    file_size: int,
    sha256: str,
) -> tuple[Image, bool]:
    """Insert or update an image by `(sample_id, path)`. Returns `(image, created)`.

    `imported_at` is deliberately left alone on update: it records when the file first
    entered the catalog, not when it was last re-scanned.
    """
    row = conn.execute(
        "SELECT * FROM image WHERE sample_id = ? AND path = ?",
        (sample_id, path),
    ).fetchone()

    if row is None:
        cursor = conn.execute(
            """
            INSERT INTO image
                   (sample_id, channel_id, path, width, height, bit_depth, file_size, sha256)
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (sample_id, channel_id, path, width, height, bit_depth, file_size, sha256),
        )
        created = get_image(conn, int(cursor.lastrowid or 0))
        if created is None:  # pragma: no cover - the insert above just succeeded
            msg = "the image row vanished immediately after insertion"
            raise RuntimeError(msg)
        return created, True

    conn.execute(
        """
        UPDATE image
           SET channel_id = ?,
               width = ?,
               height = ?,
               bit_depth = ?,
               file_size = ?,
               sha256 = ?
         WHERE id = ?
        """,
        (channel_id, width, height, bit_depth, file_size, sha256, int(row["id"])),
    )
    updated = get_image(conn, int(row["id"]))
    if updated is None:  # pragma: no cover - the row was read a moment ago
        msg = "the image row vanished during update"
        raise RuntimeError(msg)
    return updated, False


def find_duplicate_hashes(conn: sqlite3.Connection, dataset_id: int) -> dict[str, list[str]]:
    """Paths sharing a `sha256` within one dataset — a warning, never an error."""
    rows = conn.execute(
        """
        SELECT image.sha256 AS sha256, image.path AS path
          FROM image
          JOIN sample ON sample.id = image.sample_id
         WHERE sample.dataset_id = ?
           AND image.sha256 IN (
                 SELECT i2.sha256
                   FROM image AS i2
                   JOIN sample AS s2 ON s2.id = i2.sample_id
                  WHERE s2.dataset_id = ?
                  GROUP BY i2.sha256
                 HAVING COUNT(*) > 1
               )
         ORDER BY image.sha256, image.path
        """,
        (dataset_id, dataset_id),
    ).fetchall()

    duplicates: dict[str, list[str]] = {}
    for row in rows:
        duplicates.setdefault(str(row["sha256"]), []).append(str(row["path"]))
    return duplicates
