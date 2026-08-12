"""Versioned annotation repository."""

from __future__ import annotations

import sqlite3

from anomaly_lab.domain.annotations import (
    AnnotationDocument,
    AnnotationDraft,
    AnnotationLabel,
    AnnotationRevision,
)


def _document(value: str) -> AnnotationDocument:
    return AnnotationDocument.model_validate_json(value)


def _draft(row: sqlite3.Row) -> AnnotationDraft:
    values = dict(row)
    values["document"] = _document(str(values["document"]))
    return AnnotationDraft.model_validate(values)


def _revision(row: sqlite3.Row) -> AnnotationRevision:
    values = dict(row)
    values["document"] = _document(str(values["document"]))
    return AnnotationRevision.model_validate(values)


def ensure_default_label(conn: sqlite3.Connection, dataset_id: int) -> None:
    conn.execute(
        """
        INSERT INTO annotation_label (dataset_id, key, name, color, position)
             VALUES (?, 'defect', 'Defect', '#ef4444', 0)
        ON CONFLICT (dataset_id, key) DO NOTHING
        """,
        (dataset_id,),
    )


def list_labels(conn: sqlite3.Connection, dataset_id: int) -> list[AnnotationLabel]:
    rows = conn.execute(
        "SELECT * FROM annotation_label WHERE dataset_id = ? ORDER BY position, id",
        (dataset_id,),
    ).fetchall()
    return [AnnotationLabel.model_validate(dict(row)) for row in rows]


def create_label(
    conn: sqlite3.Connection,
    dataset_id: int,
    *,
    key: str,
    name: str,
    color: str,
    position: int,
) -> AnnotationLabel:
    cursor = conn.execute(
        """
        INSERT INTO annotation_label (dataset_id, key, name, color, position)
             VALUES (?, ?, ?, ?, ?)
        """,
        (dataset_id, key, name, color.lower(), position),
    )
    row = conn.execute(
        "SELECT * FROM annotation_label WHERE id = ?", (int(cursor.lastrowid or 0),)
    ).fetchone()
    if row is None:  # pragma: no cover
        raise RuntimeError("annotation label vanished after insertion")
    return AnnotationLabel.model_validate(dict(row))


def update_label(
    conn: sqlite3.Connection,
    dataset_id: int,
    key: str,
    *,
    name: str,
    color: str,
    position: int,
) -> AnnotationLabel | None:
    cursor = conn.execute(
        """
        UPDATE annotation_label
           SET name = ?, color = ?, position = ?
         WHERE dataset_id = ? AND key = ?
        """,
        (name, color.lower(), position, dataset_id, key),
    )
    if cursor.rowcount != 1:
        return None
    row = conn.execute(
        "SELECT * FROM annotation_label WHERE dataset_id = ? AND key = ?",
        (dataset_id, key),
    ).fetchone()
    return AnnotationLabel.model_validate(dict(row)) if row is not None else None


def get_draft(conn: sqlite3.Connection, image_id: int) -> AnnotationDraft | None:
    row = conn.execute("SELECT * FROM annotation_draft WHERE image_id = ?", (image_id,)).fetchone()
    return _draft(row) if row is not None else None


def latest_revision(conn: sqlite3.Connection, image_id: int) -> AnnotationRevision | None:
    row = conn.execute(
        "SELECT * FROM annotation_revision WHERE image_id = ? ORDER BY revision_no DESC LIMIT 1",
        (image_id,),
    ).fetchone()
    return _revision(row) if row is not None else None


def create_draft(
    conn: sqlite3.Connection,
    image_id: int,
    document: AnnotationDocument,
    *,
    base_revision_id: int | None,
    source_mask_id: int | None,
    source_mask_path: str | None,
    source_mask_sha256: str | None,
) -> AnnotationDraft:
    conn.execute(
        """
        INSERT INTO annotation_draft
               (image_id, base_revision_id, document, source_mask_id,
                source_mask_path, source_mask_sha256)
             VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            image_id,
            base_revision_id,
            document.canonical_json(),
            source_mask_id,
            source_mask_path,
            source_mask_sha256,
        ),
    )
    created = get_draft(conn, image_id)
    if created is None:  # pragma: no cover
        raise RuntimeError("annotation draft vanished after insertion")
    return created


def update_draft(
    conn: sqlite3.Connection,
    image_id: int,
    expected_version: int,
    document: AnnotationDocument,
) -> AnnotationDraft | None:
    cursor = conn.execute(
        """
        UPDATE annotation_draft
           SET document = ?, version = version + 1,
               updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
         WHERE image_id = ? AND version = ?
        """,
        (document.canonical_json(), image_id, expected_version),
    )
    if cursor.rowcount != 1:
        return None
    return get_draft(conn, image_id)


def insert_revision(
    conn: sqlite3.Connection,
    draft: AnnotationDraft,
    *,
    document_sha256: str,
    mask_path: str,
    mask_sha256: str,
) -> AnnotationRevision:
    next_no = int(
        conn.execute(
            "SELECT COALESCE(MAX(revision_no), 0) + 1 FROM annotation_revision WHERE image_id = ?",
            (draft.image_id,),
        ).fetchone()[0]
    )
    cursor = conn.execute(
        """
        INSERT INTO annotation_revision
               (image_id, revision_no, document, document_sha256, mask_path, mask_sha256,
                source_mask_id, source_mask_path, source_mask_sha256)
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            draft.image_id,
            next_no,
            draft.document.canonical_json(),
            document_sha256,
            mask_path,
            mask_sha256,
            draft.source_mask_id,
            draft.source_mask_path,
            draft.source_mask_sha256,
        ),
    )
    conn.execute(
        "DELETE FROM annotation_draft WHERE image_id = ? AND version = ?",
        (draft.image_id, draft.version),
    )
    row = conn.execute(
        "SELECT * FROM annotation_revision WHERE id = ?", (int(cursor.lastrowid or 0),)
    ).fetchone()
    if row is None:  # pragma: no cover
        raise RuntimeError("annotation revision vanished after insertion")
    return _revision(row)


def list_revisions(conn: sqlite3.Connection, image_id: int) -> list[AnnotationRevision]:
    rows = conn.execute(
        "SELECT * FROM annotation_revision WHERE image_id = ? ORDER BY revision_no DESC",
        (image_id,),
    ).fetchall()
    return [_revision(row) for row in rows]


def get_revision(conn: sqlite3.Connection, revision_id: int) -> AnnotationRevision | None:
    row = conn.execute("SELECT * FROM annotation_revision WHERE id = ?", (revision_id,)).fetchone()
    return _revision(row) if row is not None else None


def delete_for_dataset(conn: sqlite3.Connection, dataset_id: int) -> None:
    image_select = (
        "SELECT image.id FROM image JOIN sample ON sample.id = image.sample_id "
        "WHERE sample.dataset_id = ?"
    )
    conn.execute(f"DELETE FROM annotation_draft WHERE image_id IN ({image_select})", (dataset_id,))
    conn.execute(
        f"DELETE FROM annotation_revision WHERE image_id IN ({image_select})", (dataset_id,)
    )
    conn.execute("DELETE FROM annotation_label WHERE dataset_id = ?", (dataset_id,))
