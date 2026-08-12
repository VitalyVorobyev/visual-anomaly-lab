"""Dataset and channel repository."""

from __future__ import annotations

import sqlite3

from anomaly_lab.domain.entities import Channel, Dataset, Label


def _to_dataset(row: sqlite3.Row) -> Dataset:
    return Dataset.model_validate(dict(row))


def _to_channel(row: sqlite3.Row) -> Channel:
    return Channel.model_validate(dict(row))


def list_datasets(conn: sqlite3.Connection) -> list[Dataset]:
    rows = conn.execute("SELECT * FROM dataset ORDER BY id").fetchall()
    return [_to_dataset(row) for row in rows]


def get_dataset(conn: sqlite3.Connection, dataset_id: int) -> Dataset | None:
    row = conn.execute("SELECT * FROM dataset WHERE id = ?", (dataset_id,)).fetchone()
    return _to_dataset(row) if row is not None else None


def find_dataset_by_root(conn: sqlite3.Connection, root_path: str) -> Dataset | None:
    """Resolve the dataset a re-import should update (ADR-0013)."""
    row = conn.execute("SELECT * FROM dataset WHERE root_path = ?", (root_path,)).fetchone()
    return _to_dataset(row) if row is not None else None


def create_dataset(
    conn: sqlite3.Connection,
    *,
    name: str,
    root_path: str,
    adapter: str | None = None,
    manifest_path: str | None = None,
    notes: str | None = None,
) -> Dataset:
    cursor = conn.execute(
        """
        INSERT INTO dataset (name, root_path, adapter, manifest_path, notes)
             VALUES (?, ?, ?, ?, ?)
        """,
        (name, root_path, adapter, manifest_path, notes),
    )
    created = get_dataset(conn, int(cursor.lastrowid or 0))
    if created is None:  # pragma: no cover - the insert above just succeeded
        msg = "the dataset row vanished immediately after insertion"
        raise RuntimeError(msg)
    return created


def record_manifest(
    conn: sqlite3.Connection,
    dataset_id: int,
    *,
    adapter: str,
    manifest_path: str,
) -> None:
    """Point the dataset at the manifest that was last committed into it (ADR-0006)."""
    conn.execute(
        """
        UPDATE dataset
           SET adapter = ?,
               manifest_path = ?
         WHERE id = ?
        """,
        (adapter, manifest_path, dataset_id),
    )


def delete_dataset(conn: sqlite3.Connection, dataset_id: int) -> bool:
    """Delete a dataset and everything under it, children first.

    Deliberately explicit rather than leaning on `ON DELETE CASCADE`: `image.channel_id`
    is `RESTRICT`, and SQLite does not define the order in which cascades run, so the
    channel cascade can fire while images still reference those channels. Deleting in
    dependency order inside one transaction sidesteps the ordering question entirely.

    `experiment` is deliberately *not* deleted here — it references the dataset with
    `RESTRICT`, so a dataset with experiments attached refuses to delete (ADR-0005).
    """
    if get_dataset(conn, dataset_id) is None:
        return False

    conn.execute("BEGIN")
    try:
        delete_dataset_rows(conn, dataset_id)
    except Exception:
        conn.execute("ROLLBACK")
        raise
    conn.execute("COMMIT")
    return True


def delete_dataset_rows(
    conn: sqlite3.Connection, dataset_id: int, *, include_experiments: bool = False
) -> bool:
    """Delete one dataset's rows inside a transaction owned by the caller.

    The ordinary repository operation preserves experiments and therefore refuses on
    the schema's ``RESTRICT`` boundary.  The explicit destructive API opts into the
    larger cascade after it has previewed the consequences and excluded live work.
    """
    if get_dataset(conn, dataset_id) is None:
        return False
    if include_experiments:
        conn.execute(
            """
            DELETE FROM job
             WHERE experiment_id IS NULL
               AND CAST(json_extract(params, '$.dataset_id') AS INTEGER) = ?
            """,
            (dataset_id,),
        )
        conn.execute("DELETE FROM experiment WHERE dataset_id = ?", (dataset_id,))
    conn.execute(
        """
        DELETE FROM image
         WHERE sample_id IN (SELECT id FROM sample WHERE dataset_id = ?)
        """,
        (dataset_id,),
    )
    conn.execute("DELETE FROM sample WHERE dataset_id = ?", (dataset_id,))
    conn.execute("DELETE FROM channel WHERE dataset_id = ?", (dataset_id,))
    conn.execute("DELETE FROM split WHERE dataset_id = ?", (dataset_id,))
    conn.execute("DELETE FROM dataset WHERE id = ?", (dataset_id,))
    return True


def list_channels(conn: sqlite3.Connection, dataset_id: int) -> list[Channel]:
    """The dataset's channel dictionary, in display order.

    The length of this list is data. No caller may assume it, and none may cache a count.
    """
    rows = conn.execute(
        "SELECT * FROM channel WHERE dataset_id = ? ORDER BY position, name",
        (dataset_id,),
    ).fetchall()
    return [_to_channel(row) for row in rows]


def upsert_channel(
    conn: sqlite3.Connection,
    dataset_id: int,
    *,
    name: str,
    position: int,
) -> Channel:
    row = conn.execute(
        "SELECT * FROM channel WHERE dataset_id = ? AND name = ?",
        (dataset_id, name),
    ).fetchone()
    if row is not None:
        conn.execute(
            "UPDATE channel SET position = ? WHERE id = ?",
            (position, int(row["id"])),
        )
        return _to_channel(
            conn.execute("SELECT * FROM channel WHERE id = ?", (int(row["id"]),)).fetchone()
        )

    cursor = conn.execute(
        "INSERT INTO channel (dataset_id, name, position) VALUES (?, ?, ?)",
        (dataset_id, name, position),
    )
    return _to_channel(
        conn.execute("SELECT * FROM channel WHERE id = ?", (int(cursor.lastrowid or 0),)).fetchone()
    )


def label_counts(conn: sqlite3.Connection, dataset_id: int) -> dict[Label, int]:
    """Sample counts per label, with every label present even at zero."""
    counts = dict.fromkeys(Label, 0)
    rows = conn.execute(
        "SELECT label, COUNT(*) AS n FROM sample WHERE dataset_id = ? GROUP BY label",
        (dataset_id,),
    ).fetchall()
    for row in rows:
        counts[Label(row["label"])] = int(row["n"])
    return counts


def count_images(conn: sqlite3.Connection, dataset_id: int) -> int:
    row = conn.execute(
        """
        SELECT COUNT(*) AS n
          FROM image
          JOIN sample ON sample.id = image.sample_id
         WHERE sample.dataset_id = ?
        """,
        (dataset_id,),
    ).fetchone()
    return int(row["n"])
