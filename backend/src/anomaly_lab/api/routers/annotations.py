"""Editable annotation drafts and immutable completed revisions."""

from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Header, HTTPException, Request, Response
from fastapi.responses import FileResponse

from anomaly_lab.annotation_render import AnnotationRenderError, render_binary_mask
from anomaly_lab.config import Settings
from anomaly_lab.db.connection import connection
from anomaly_lab.db.repositories import annotations as annotations_repo
from anomaly_lab.db.repositories import datasets as datasets_repo
from anomaly_lab.db.repositories import images as images_repo
from anomaly_lab.db.repositories import masks as masks_repo
from anomaly_lab.domain.annotations import (
    AnnotationDocument,
    AnnotationDraft,
    AnnotationLabel,
    AnnotationLabelCreate,
    AnnotationLabelUpdate,
    AnnotationRevision,
)
from anomaly_lab.media.decode import sha256_of

router = APIRouter(tags=["annotations"])


def _etag(draft: AnnotationDraft) -> str:
    return f'"annotation-draft-{draft.image_id}-v{draft.version}"'


def _set_draft_etag(response: Response, draft: AnnotationDraft) -> None:
    response.headers["ETag"] = _etag(draft)
    response.headers["Cache-Control"] = "no-store"


def _require_if_match(value: str | None) -> str:
    if value is None:
        raise HTTPException(status_code=428, detail="If-Match is required for annotation writes")
    return value


def _image_dataset_id(conn: sqlite3.Connection, image_id: int) -> int:
    # Kept as a tiny SQL join here because an image's owning dataset is context for
    # taxonomy validation, not a new domain read model.
    row = conn.execute(
        """
        SELECT sample.dataset_id
          FROM image JOIN sample ON sample.id = image.sample_id
         WHERE image.id = ?
        """,
        (image_id,),
    ).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail=f"no image with id {image_id}")
    return int(row["dataset_id"])


def _validate_document(
    conn: sqlite3.Connection,
    image_id: int,
    dataset_id: int,
    document: AnnotationDocument,
    *,
    expected_base: str,
) -> None:
    image = images_repo.get_image(conn, image_id)
    if image is None:  # pragma: no cover - ownership join already found it
        raise HTTPException(status_code=404, detail=f"no image with id {image_id}")
    if (document.image_width, document.image_height) != (image.width, image.height):
        raise HTTPException(
            status_code=422,
            detail="annotation dimensions must match the source image exactly",
        )
    if document.base != expected_base:
        raise HTTPException(status_code=422, detail="a draft's base layer cannot be changed")
    known = {label.key for label in annotations_repo.list_labels(conn, dataset_id)}
    unknown = sorted({shape.label_key for shape in document.shapes} - known)
    if unknown:
        raise HTTPException(
            status_code=422, detail=f"unknown annotation labels: {', '.join(unknown)}"
        )


@router.get(
    "/api/datasets/{dataset_id}/annotation-labels",
    summary="The defect-label taxonomy for one dataset",
)
def list_annotation_labels(request: Request, dataset_id: int) -> list[AnnotationLabel]:
    settings: Settings = request.app.state.settings
    with connection(settings.db_path) as conn:
        if datasets_repo.get_dataset(conn, dataset_id) is None:
            raise HTTPException(status_code=404, detail=f"no dataset with id {dataset_id}")
        return annotations_repo.list_labels(conn, dataset_id)


@router.post(
    "/api/datasets/{dataset_id}/annotation-labels",
    summary="Add one stable class key to a dataset's defect taxonomy",
)
def create_annotation_label(
    request: Request, dataset_id: int, body: AnnotationLabelCreate
) -> AnnotationLabel:
    settings: Settings = request.app.state.settings
    with connection(settings.db_path) as conn:
        if datasets_repo.get_dataset(conn, dataset_id) is None:
            raise HTTPException(status_code=404, detail=f"no dataset with id {dataset_id}")
        annotations_repo.ensure_default_label(conn, dataset_id)
        try:
            return annotations_repo.create_label(conn, dataset_id, **body.model_dump(mode="python"))
        except sqlite3.IntegrityError as exc:
            raise HTTPException(
                status_code=409, detail=f"annotation label {body.key!r} exists"
            ) from exc


@router.put(
    "/api/datasets/{dataset_id}/annotation-labels/{key}",
    summary="Rename or recolour a class without changing its stable key",
)
def update_annotation_label(
    request: Request, dataset_id: int, key: str, body: AnnotationLabelUpdate
) -> AnnotationLabel:
    settings: Settings = request.app.state.settings
    with connection(settings.db_path) as conn:
        if datasets_repo.get_dataset(conn, dataset_id) is None:
            raise HTTPException(status_code=404, detail=f"no dataset with id {dataset_id}")
        updated = annotations_repo.update_label(
            conn, dataset_id, key, **body.model_dump(mode="python")
        )
    if updated is None:
        raise HTTPException(status_code=404, detail=f"no annotation label {key!r}")
    return updated


@router.post(
    "/api/images/{image_id}/annotations/draft",
    summary="Open an existing draft or start one from the latest truth",
)
def open_annotation_draft(request: Request, response: Response, image_id: int) -> AnnotationDraft:
    settings: Settings = request.app.state.settings
    with connection(settings.db_path) as conn:
        conn.execute("BEGIN IMMEDIATE")
        try:
            dataset_id = _image_dataset_id(conn, image_id)
            annotations_repo.ensure_default_label(conn, dataset_id)
            existing = annotations_repo.get_draft(conn, image_id)
            if existing is not None:
                conn.execute("COMMIT")
                _set_draft_etag(response, existing)
                return existing

            image = images_repo.get_image(conn, image_id)
            if image is None:  # pragma: no cover
                raise HTTPException(status_code=404, detail=f"no image with id {image_id}")
            latest = annotations_repo.latest_revision(conn, image_id)
            if latest is not None:
                document = latest.document
                base_revision_id = latest.id
                source_mask_id = latest.source_mask_id
                source_mask_path = latest.source_mask_path
                source_mask_sha256 = latest.source_mask_sha256
            else:
                source = masks_repo.get_mask_for_image(conn, image_id)
                source_mask_id = source.id if source else None
                source_mask_path = source.path if source else None
                source_mask_sha256 = None
                if source is not None:
                    path = Path(source.path)
                    if not path.is_file():
                        raise HTTPException(
                            status_code=409, detail="the source mask is unavailable"
                        )
                    actual = sha256_of(path)
                    if source.sha256 is not None and source.sha256 != actual:
                        raise HTTPException(
                            status_code=409,
                            detail="the imported source mask changed after it entered the catalog",
                        )
                    if source.sha256 is None:
                        masks_repo.record_sha256(conn, source.id, actual)
                    source_mask_sha256 = actual
                document = AnnotationDocument(
                    image_width=image.width,
                    image_height=image.height,
                    base="source_mask" if source is not None else "empty",
                )
                base_revision_id = None

            created = annotations_repo.create_draft(
                conn,
                image_id,
                document,
                base_revision_id=base_revision_id,
                source_mask_id=source_mask_id,
                source_mask_path=source_mask_path,
                source_mask_sha256=source_mask_sha256,
            )
            conn.execute("COMMIT")
        except BaseException:
            if conn.in_transaction:
                conn.execute("ROLLBACK")
            raise
    _set_draft_etag(response, created)
    return created


@router.get(
    "/api/images/{image_id}/annotations/draft",
    summary="Read the current editable annotation draft",
)
def get_annotation_draft(request: Request, response: Response, image_id: int) -> AnnotationDraft:
    settings: Settings = request.app.state.settings
    with connection(settings.db_path) as conn:
        draft = annotations_repo.get_draft(conn, image_id)
    if draft is None:
        raise HTTPException(status_code=404, detail=f"image {image_id} has no annotation draft")
    _set_draft_etag(response, draft)
    return draft


@router.put(
    "/api/images/{image_id}/annotations/draft",
    summary="Save a draft if the caller still owns the version it read",
)
def save_annotation_draft(
    request: Request,
    response: Response,
    image_id: int,
    document: AnnotationDocument,
    if_match: Annotated[str | None, Header(alias="If-Match")] = None,
) -> AnnotationDraft:
    expected = _require_if_match(if_match)
    settings: Settings = request.app.state.settings
    with connection(settings.db_path) as conn:
        conn.execute("BEGIN IMMEDIATE")
        try:
            dataset_id = _image_dataset_id(conn, image_id)
            current = annotations_repo.get_draft(conn, image_id)
            if current is None:
                raise HTTPException(
                    status_code=404, detail=f"image {image_id} has no annotation draft"
                )
            if expected != _etag(current):
                raise HTTPException(
                    status_code=412, detail="the annotation draft changed elsewhere"
                )
            _validate_document(
                conn, image_id, dataset_id, document, expected_base=current.document.base
            )
            saved = annotations_repo.update_draft(conn, image_id, current.version, document)
            if saved is None:  # pragma: no cover - BEGIN IMMEDIATE serialises writers
                raise HTTPException(
                    status_code=412, detail="the annotation draft changed elsewhere"
                )
            conn.execute("COMMIT")
        except BaseException:
            if conn.in_transaction:
                conn.execute("ROLLBACK")
            raise
    _set_draft_etag(response, saved)
    return saved


@router.post(
    "/api/images/{image_id}/annotations/complete",
    summary="Materialise and freeze the current draft as a revision",
)
def complete_annotation_draft(
    request: Request,
    image_id: int,
    if_match: Annotated[str | None, Header(alias="If-Match")] = None,
) -> AnnotationRevision:
    expected = _require_if_match(if_match)
    settings: Settings = request.app.state.settings
    destination: Path | None = None
    with connection(settings.db_path) as conn:
        conn.execute("BEGIN IMMEDIATE")
        try:
            dataset_id = _image_dataset_id(conn, image_id)
            draft = annotations_repo.get_draft(conn, image_id)
            if draft is None:
                raise HTTPException(
                    status_code=404, detail=f"image {image_id} has no annotation draft"
                )
            if expected != _etag(draft):
                raise HTTPException(
                    status_code=412, detail="the annotation draft changed elsewhere"
                )
            _validate_document(
                conn, image_id, dataset_id, draft.document, expected_base=draft.document.base
            )
            latest = annotations_repo.latest_revision(conn, image_id)
            next_no = (latest.revision_no if latest else 0) + 1
            destination = settings.annotation_image_dir(image_id) / f"revision-{next_no}.png"
            if settings.annotations_dir.is_symlink() or destination.parent.is_symlink():
                raise HTTPException(
                    status_code=409, detail="the app-owned annotation directory is unsafe"
                )
            try:
                mask_sha256 = render_binary_mask(
                    draft.document,
                    destination,
                    source_mask_path=Path(draft.source_mask_path)
                    if draft.source_mask_path
                    else None,
                    source_mask_sha256=draft.source_mask_sha256,
                )
            except AnnotationRenderError as exc:
                raise HTTPException(status_code=409, detail=str(exc)) from exc
            document_sha256 = hashlib.sha256(draft.document.canonical_json().encode()).hexdigest()
            revision = annotations_repo.insert_revision(
                conn,
                draft,
                document_sha256=document_sha256,
                mask_path=str(destination),
                mask_sha256=mask_sha256,
            )
            conn.execute("COMMIT")
        except BaseException:
            if conn.in_transaction:
                conn.execute("ROLLBACK")
            if destination is not None:
                destination.unlink(missing_ok=True)
            raise
    return revision


@router.get(
    "/api/images/{image_id}/annotations/revisions",
    summary="Completed annotation history, newest first",
)
def list_annotation_revisions(request: Request, image_id: int) -> list[AnnotationRevision]:
    settings: Settings = request.app.state.settings
    with connection(settings.db_path) as conn:
        _image_dataset_id(conn, image_id)
        return annotations_repo.list_revisions(conn, image_id)


@router.get(
    "/api/images/{image_id}/annotations/revisions/{revision_id}/mask",
    summary="The immutable binary PNG materialised for a completed revision",
)
def read_annotation_revision_mask(
    request: Request, image_id: int, revision_id: int
) -> FileResponse:
    settings: Settings = request.app.state.settings
    with connection(settings.db_path) as conn:
        revision = annotations_repo.get_revision(conn, revision_id)
    if revision is None or revision.image_id != image_id:
        raise HTTPException(status_code=404, detail="no such annotation revision")
    path = Path(revision.mask_path)
    expected = settings.annotation_image_dir(image_id)
    if (
        settings.annotations_dir.is_symlink()
        or expected.is_symlink()
        or path.parent != expected
        or path.name != f"revision-{revision.revision_no}.png"
    ):
        raise HTTPException(status_code=409, detail="the stored annotation path is unsafe")
    if not path.is_file() or sha256_of(path) != revision.mask_sha256:
        raise HTTPException(
            status_code=409, detail="the materialised annotation mask is unavailable"
        )
    return FileResponse(
        path,
        media_type="image/png",
        headers={
            "ETag": f'"annotation-mask-{revision.mask_sha256}"',
            "Cache-Control": "public, max-age=31536000, immutable",
        },
    )
