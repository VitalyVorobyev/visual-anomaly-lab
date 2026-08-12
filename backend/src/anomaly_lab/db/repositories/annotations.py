"""Versioned annotation repository."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from anomaly_lab.domain.annotations import (
    AnnotationDocument,
    AnnotationDraft,
    AnnotationLabel,
    AnnotationRevision,
)
from anomaly_lab.media.decode import sha256_of


class GroundTruthDriftError(RuntimeError):
    """Pinned truth bytes no longer match their recorded identity."""


@dataclass(frozen=True)
class GroundTruthMask:
    image_id: int
    record_id: int
    path: str
    sha256: str | None
    kind: Literal["revision", "source"]

    @property
    def identity(self) -> str:
        digest = self.sha256 or "unhashed"
        return f"{self.kind}:{self.record_id}:{digest}"


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


def resolve_ground_truth_masks(
    conn: sqlite3.Connection,
    image_ids: list[int],
    *,
    verify_bytes: bool = False,
) -> dict[int, GroundTruthMask]:
    """Resolve newest completed revision, then imported source mask, per image.

    `verify_bytes` is for consumers that will use the pixels. Metadata-only reads use
    recorded digests so experiment detail stays O(rows), not O(source bytes).
    """
    found: dict[int, GroundTruthMask] = {}
    for start in range(0, len(image_ids), 900):
        chunk = image_ids[start : start + 900]
        if not chunk:
            continue
        placeholders = ",".join("?" * len(chunk))
        revisions = conn.execute(
            f"""
            SELECT id, image_id, mask_path, mask_sha256
              FROM annotation_revision
             WHERE image_id IN ({placeholders})
             ORDER BY image_id, revision_no DESC
            """,
            chunk,
        ).fetchall()
        for row in revisions:
            image_id = int(row["image_id"])
            found.setdefault(
                image_id,
                GroundTruthMask(
                    image_id=image_id,
                    record_id=int(row["id"]),
                    path=str(row["mask_path"]),
                    sha256=str(row["mask_sha256"]),
                    kind="revision",
                ),
            )

        unresolved = [image_id for image_id in chunk if image_id not in found]
        if unresolved:
            source_placeholders = ",".join("?" * len(unresolved))
            sources = conn.execute(
                f"""
                SELECT id, image_id, path, sha256
                  FROM mask
                 WHERE kind = 'ground_truth' AND image_id IN ({source_placeholders})
                 ORDER BY image_id, id
                """,
                unresolved,
            ).fetchall()
            for row in sources:
                image_id = int(row["image_id"])
                found.setdefault(
                    image_id,
                    GroundTruthMask(
                        image_id=image_id,
                        record_id=int(row["id"]),
                        path=str(row["path"]),
                        sha256=str(row["sha256"]) if row["sha256"] is not None else None,
                        kind="source",
                    ),
                )

    if not verify_bytes:
        return found

    verified: dict[int, GroundTruthMask] = {}
    for image_id, truth in found.items():
        path = Path(truth.path)
        if not path.is_file():
            if truth.kind == "revision":
                raise GroundTruthDriftError(
                    f"completed annotation ground truth for image {image_id} is unavailable"
                )
            verified[image_id] = truth
            continue
        try:
            actual = sha256_of(path)
        except OSError as exc:
            if truth.kind == "revision":
                raise GroundTruthDriftError(
                    f"completed annotation ground truth for image {image_id} cannot be read"
                ) from exc
            verified[image_id] = truth
            continue
        if truth.sha256 is not None and actual != truth.sha256:
            raise GroundTruthDriftError(
                f"{truth.kind} ground truth for image {image_id} changed after it was pinned"
            )
        if truth.kind == "source" and truth.sha256 is None:
            conn.execute("UPDATE mask SET sha256 = ? WHERE id = ?", (actual, truth.record_id))
        verified[image_id] = GroundTruthMask(
            image_id=truth.image_id,
            record_id=truth.record_id,
            path=truth.path,
            sha256=actual,
            kind=truth.kind,
        )
    return verified


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
