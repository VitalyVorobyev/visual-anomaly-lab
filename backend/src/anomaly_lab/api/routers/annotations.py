"""Editable annotation drafts and immutable completed revisions."""

from __future__ import annotations

import hashlib
import shutil
import sqlite3
from collections.abc import Callable
from pathlib import Path
from typing import Annotated

import numpy as np
from fastapi import APIRouter, Body, Header, HTTPException, Request, Response
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from anomaly_lab.annotation_bitmap import AnnotationBitmapError, decode_shape, encode_png
from anomaly_lab.annotation_interchange import (
    AnnotationInterchangeError,
    CocoDocument,
    LabelMeDocument,
    coco_from_mask,
    imported_document,
    labelme_from_mask,
    shapes_from_coco,
    shapes_from_labelme,
    shapes_from_png,
)
from anomaly_lab.annotation_render import AnnotationRenderError, render_binary_mask
from anomaly_lab.config import Settings
from anomaly_lab.db.connection import connection
from anomaly_lab.db.repositories import annotations as annotations_repo
from anomaly_lab.db.repositories import datasets as datasets_repo
from anomaly_lab.db.repositories import images as images_repo
from anomaly_lab.db.repositories import masks as masks_repo
from anomaly_lab.db.repositories import samples as samples_repo
from anomaly_lab.domain.annotations import (
    AnnotationDocument,
    AnnotationDraft,
    AnnotationLabel,
    AnnotationLabelCreate,
    AnnotationLabelUpdate,
    AnnotationRevision,
    AnnotationSampleDraft,
    AnnotationShape,
    BitmapShape,
)
from anomaly_lab.domain.entities import AnnotationScope, Image
from anomaly_lab.media.decode import UnreadableImageError, sha256_of
from anomaly_lab.models.preprocessing import load_mask
from anomaly_lab.schemas import API_MODEL_CONFIG

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


def _sample_etag(draft: AnnotationSampleDraft) -> str:
    # A namespace of its own, not `annotation-draft-…`: an image id and a sample id are
    # both small integers and would collide into a token that validates against the wrong
    # document.
    return f'"annotation-sample-draft-{draft.sample_id}-v{draft.version}"'


def _set_sample_draft_etag(response: Response, draft: AnnotationSampleDraft) -> None:
    response.headers["ETag"] = _sample_etag(draft)
    response.headers["Cache-Control"] = "no-store"


def _annotation_scope(conn: sqlite3.Connection, dataset_id: int) -> AnnotationScope:
    dataset = datasets_repo.get_dataset(conn, dataset_id)
    if dataset is None:  # pragma: no cover - callers resolved the id from a row
        raise HTTPException(status_code=404, detail=f"no dataset with id {dataset_id}")
    return dataset.annotation_scope


def _require_image_scope(conn: sqlite3.Connection, dataset_id: int) -> None:
    """Refuse a per-image write on a dataset whose truth is edited per sample.

    A 409 with a pointer rather than a quiet redirect: two writers editing the same part
    through two different scopes would each hold a valid ETag for a different document and
    neither would detect the other.
    """
    if _annotation_scope(conn, dataset_id) is AnnotationScope.SAMPLE:
        raise HTTPException(
            status_code=409,
            detail=(
                "this dataset annotates whole samples; use /api/samples/{sample_id}/annotations/…"
            ),
        )


def _require_sample_scope(conn: sqlite3.Connection, dataset_id: int) -> None:
    if _annotation_scope(conn, dataset_id) is AnnotationScope.IMAGE:
        raise HTTPException(
            status_code=409,
            detail=(
                "this dataset annotates individual images; use /api/images/{image_id}/annotations/…"
            ),
        )


def _sample_dataset_id(conn: sqlite3.Connection, sample_id: int) -> int:
    sample = samples_repo.get_sample(conn, sample_id)
    if sample is None:
        raise HTTPException(status_code=404, detail=f"no sample with id {sample_id}")
    return sample.dataset_id


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


def _validate_taxonomy(
    conn: sqlite3.Connection, dataset_id: int, document: AnnotationDocument
) -> None:
    """The checks that are about the document alone, not about which frame it belongs to.

    Split out because a sample-scoped document is validated against a *shared* frame that
    no single image owns, while these two checks are identical in both scopes.
    """
    known = {label.key for label in annotations_repo.list_labels(conn, dataset_id)}
    unknown = sorted({shape.label_key for shape in document.shapes} - known)
    if unknown:
        raise HTTPException(
            status_code=422, detail=f"unknown annotation labels: {', '.join(unknown)}"
        )
    for shape in document.shapes:
        if isinstance(shape, BitmapShape):
            try:
                decode_shape(shape)
            except AnnotationBitmapError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc


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
    _validate_taxonomy(conn, dataset_id, document)


def _save_import(
    request: Request,
    response: Response,
    image_id: int,
    expected: str,
    build_shapes: Callable[
        [tuple[int, int], dict[str, str]],
        list[AnnotationShape],
    ],
) -> AnnotationDraft:
    settings: Settings = request.app.state.settings
    with connection(settings.db_path) as conn:
        conn.execute("BEGIN IMMEDIATE")
        try:
            dataset_id = _image_dataset_id(conn, image_id)
            _require_image_scope(conn, dataset_id)
            current = annotations_repo.get_draft(conn, image_id)
            if current is None:
                raise HTTPException(
                    status_code=404, detail=f"image {image_id} has no annotation draft"
                )
            if expected != _etag(current):
                raise HTTPException(
                    status_code=412, detail="the annotation draft changed elsewhere"
                )
            labels = annotations_repo.list_labels(conn, dataset_id)
            known = {label.key: label.name for label in labels}
            default_key = labels[0].key if labels else "defect"
            try:
                shapes = build_shapes(
                    (current.document.image_width, current.document.image_height), known
                )
                document = imported_document(
                    current.document,
                    shapes,
                    clear_label_key=default_key,
                )
            except AnnotationInterchangeError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
            _validate_document(
                conn,
                image_id,
                dataset_id,
                document,
                expected_base=current.document.base,
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


def _resolved_mask(
    conn: sqlite3.Connection,
    image_id: int,
    *,
    size: tuple[int, int],
) -> np.ndarray:
    try:
        truth = annotations_repo.resolve_ground_truth_masks(
            conn, [image_id], verify_bytes=True
        ).get(image_id)
    except annotations_repo.GroundTruthDriftError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if truth is None:
        raise HTTPException(status_code=404, detail=f"image {image_id} has no completed annotation")
    try:
        mask = load_mask(Path(truth.path))
    except UnreadableImageError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if mask.shape != (size[1], size[0]):
        raise HTTPException(
            status_code=409,
            detail="the resolved annotation mask does not match the source image dimensions",
        )
    return mask


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
            _require_image_scope(conn, dataset_id)
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
            _require_image_scope(conn, dataset_id)
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
            _require_image_scope(conn, dataset_id)
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


@router.put(
    "/api/images/{image_id}/annotations/draft/import/png",
    summary="Replace a draft's editable layers from a binary PNG mask",
)
def import_annotation_png(
    request: Request,
    response: Response,
    image_id: int,
    payload: Annotated[bytes, Body(media_type="image/png")],
    label_key: str = "defect",
    if_match: Annotated[str | None, Header(alias="If-Match")] = None,
) -> AnnotationDraft:
    expected = _require_if_match(if_match)
    return _save_import(
        request,
        response,
        image_id,
        expected,
        lambda size, known: shapes_from_png(
            payload,
            size=size,
            label_key=(label_key if label_key in known else (_raise_unknown_label(label_key))),
        ),
    )


def _raise_unknown_label(label_key: str) -> str:
    raise AnnotationInterchangeError(
        f"annotation label {label_key!r} is not in this dataset's taxonomy"
    )


@router.put(
    "/api/images/{image_id}/annotations/draft/import/labelme",
    summary="Replace a draft's editable layers from LabelMe polygon or mask shapes",
)
def import_annotation_labelme(
    request: Request,
    response: Response,
    image_id: int,
    payload: LabelMeDocument,
    if_match: Annotated[str | None, Header(alias="If-Match")] = None,
) -> AnnotationDraft:
    return _save_import(
        request,
        response,
        image_id,
        _require_if_match(if_match),
        lambda size, known: shapes_from_labelme(
            payload,
            size=size,
            known_labels=known,
        ),
    )


@router.put(
    "/api/images/{image_id}/annotations/draft/import/coco",
    summary="Replace a draft's editable layers from one-image COCO polygons or RLE",
)
def import_annotation_coco(
    request: Request,
    response: Response,
    image_id: int,
    payload: CocoDocument,
    if_match: Annotated[str | None, Header(alias="If-Match")] = None,
) -> AnnotationDraft:
    return _save_import(
        request,
        response,
        image_id,
        _require_if_match(if_match),
        lambda size, known: shapes_from_coco(
            payload,
            size=size,
            known_labels=known,
        ),
    )


def _export_context(
    request: Request,
    image_id: int,
) -> tuple[np.ndarray, str, str]:
    settings: Settings = request.app.state.settings
    with connection(settings.db_path) as conn:
        dataset_id = _image_dataset_id(conn, image_id)
        image = images_repo.get_image(conn, image_id)
        if image is None:  # pragma: no cover - ownership join already found it
            raise HTTPException(status_code=404, detail=f"no image with id {image_id}")
        labels = annotations_repo.list_labels(conn, dataset_id)
        label = next(
            (item for item in labels if item.key == "defect"), labels[0] if labels else None
        )
        mask = _resolved_mask(conn, image_id, size=(image.width, image.height))
    return mask, Path(image.path).name, label.key if label is not None else "defect"


@router.get(
    "/api/images/{image_id}/annotations/export/png",
    summary="Export the current completed truth as a binary PNG mask",
    response_class=Response,
    responses={200: {"content": {"image/png": {}}}},
)
def export_annotation_png(request: Request, image_id: int) -> Response:
    mask, _, _ = _export_context(request, image_id)
    return Response(
        content=encode_png(mask),
        media_type="image/png",
        headers={"Content-Disposition": f'attachment; filename="annotation-{image_id}.png"'},
    )


@router.get(
    "/api/images/{image_id}/annotations/export/labelme",
    summary="Export the current completed truth as a LabelMe mask annotation",
)
def export_annotation_labelme(request: Request, image_id: int) -> LabelMeDocument:
    mask, image_path, label = _export_context(request, image_id)
    return labelme_from_mask(mask, image_path=image_path, label=label)


@router.get(
    "/api/images/{image_id}/annotations/export/coco",
    summary="Export the current completed truth as one-image COCO RLE",
)
def export_annotation_coco(request: Request, image_id: int) -> CocoDocument:
    mask, image_path, label = _export_context(request, image_id)
    return coco_from_mask(mask, image_path=image_path, label=label)


# --- Sample-scoped editing (ADR-0036) --------------------------------------------------
#
# One document, edited once, materialised onto every image of the sample. Truth below this
# boundary stays image-keyed: `resolve_ground_truth_masks`, pixel metrics, `has_mask`, the
# `MetricSet` digest and all three interchange formats never learn that scope exists.


class AnnotationScopeState(BaseModel):
    """A dataset's annotation scope, and everything that would stop it changing."""

    model_config = API_MODEL_CONFIG

    dataset_id: int
    scope: AnnotationScope
    samples: int
    multi_image_samples: int
    open_drafts: int = Field(
        description="Drafts open in the current scope. Any scope change requires zero."
    )
    can_use_sample_scope: bool
    blockers: list[str] = Field(
        default_factory=list,
        description="Every reason sample scope is unavailable, reported together.",
    )


class AnnotationScopeUpdate(BaseModel):
    model_config = API_MODEL_CONFIG

    scope: AnnotationScope


def _scope_state(conn: sqlite3.Connection, dataset_id: int) -> AnnotationScopeState:
    """Answer the whole question at once rather than failing on the first obstacle.

    An operator who fixes one blocker and is then told about a second has learned the
    system does not know its own mind. Every reason is collected before any is reported.
    """
    scope = _annotation_scope(conn, dataset_id)
    blockers: list[str] = []

    masks = masks_repo.count_masks_for_dataset(conn, dataset_id)
    if masks:
        blockers.append(
            f"{masks} imported source masks are pinned to individual images; a document "
            "shared across channels cannot carry one image's provenance"
        )

    mixed = annotations_repo.samples_with_mixed_dimensions(conn, dataset_id)
    if mixed:
        shown = ", ".join(mixed[:5])
        rest = f" and {len(mixed) - 5} more" if len(mixed) > 5 else ""
        blockers.append(
            f"{len(mixed)} samples mix image dimensions ({shown}{rest}); a shared document "
            "pins one source frame"
        )

    open_drafts = (
        annotations_repo.count_open_sample_drafts(conn, dataset_id)
        if scope is AnnotationScope.SAMPLE
        else annotations_repo.count_open_image_drafts(conn, dataset_id)
    )
    if open_drafts and scope is AnnotationScope.IMAGE:
        blockers.append(
            f"{open_drafts} image drafts are open; complete or discard them first, because "
            "sample scope would leave them unreachable"
        )

    return AnnotationScopeState(
        dataset_id=dataset_id,
        scope=scope,
        samples=samples_repo.count_samples(conn, dataset_id, samples_repo.SampleFilter()),
        multi_image_samples=annotations_repo.count_multi_image_samples(conn, dataset_id),
        open_drafts=open_drafts,
        can_use_sample_scope=not blockers,
        blockers=blockers,
    )


@router.get(
    "/api/datasets/{dataset_id}/annotation-scope",
    summary="Whether this dataset annotates images or whole samples",
)
def read_annotation_scope(request: Request, dataset_id: int) -> AnnotationScopeState:
    settings: Settings = request.app.state.settings
    with connection(settings.db_path) as conn:
        if datasets_repo.get_dataset(conn, dataset_id) is None:
            raise HTTPException(status_code=404, detail=f"no dataset with id {dataset_id}")
        return _scope_state(conn, dataset_id)


@router.put(
    "/api/datasets/{dataset_id}/annotation-scope",
    summary="Move a dataset between per-image and per-sample annotation editing",
)
def set_annotation_scope(
    request: Request, dataset_id: int, body: AnnotationScopeUpdate
) -> AnnotationScopeState:
    """Completed revisions are untouched in either direction.

    They are immutable and image-keyed, so they stay valid truth whichever scope produced
    them. Only the editing surface moves.
    """
    settings: Settings = request.app.state.settings
    with connection(settings.db_path) as conn:
        conn.execute("BEGIN IMMEDIATE")
        try:
            if datasets_repo.get_dataset(conn, dataset_id) is None:
                raise HTTPException(status_code=404, detail=f"no dataset with id {dataset_id}")
            state = _scope_state(conn, dataset_id)
            if body.scope is not state.scope:
                if state.open_drafts:
                    raise HTTPException(
                        status_code=409,
                        detail=(
                            f"{state.open_drafts} annotation drafts are open; complete or "
                            "discard them before changing scope"
                        ),
                    )
                if body.scope is AnnotationScope.SAMPLE and state.blockers:
                    raise HTTPException(status_code=409, detail="; ".join(state.blockers))
                datasets_repo.set_annotation_scope(conn, dataset_id, body.scope)
            updated = _scope_state(conn, dataset_id)
            conn.execute("COMMIT")
        except BaseException:
            if conn.in_transaction:
                conn.execute("ROLLBACK")
            raise
    return updated


def _shared_frame(images: list[Image]) -> tuple[int, int]:
    if not images:
        raise HTTPException(status_code=404, detail="this sample has no images to annotate")
    frames = {(image.width, image.height) for image in images}
    if len(frames) != 1:
        raise HTTPException(
            status_code=409,
            detail="this sample's images do not share one source frame",
        )
    return frames.pop()


@router.post(
    "/api/samples/{sample_id}/annotations/draft",
    summary="Open the sample's shared draft, or start one from its latest truth",
)
def open_sample_annotation_draft(
    request: Request, response: Response, sample_id: int
) -> AnnotationSampleDraft:
    settings: Settings = request.app.state.settings
    with connection(settings.db_path) as conn:
        conn.execute("BEGIN IMMEDIATE")
        try:
            dataset_id = _sample_dataset_id(conn, sample_id)
            _require_sample_scope(conn, dataset_id)
            annotations_repo.ensure_default_label(conn, dataset_id)
            existing = annotations_repo.get_sample_draft(conn, sample_id)
            if existing is not None:
                conn.execute("COMMIT")
                _set_sample_draft_etag(response, existing)
                return existing

            width, height = _shared_frame(images_repo.list_images_for_sample(conn, sample_id))
            latest = annotations_repo.latest_revision_for_sample(conn, sample_id)
            document = AnnotationDocument(image_width=width, image_height=height)
            if latest is not None:
                if latest.document.base != "empty":
                    raise HTTPException(
                        status_code=409,
                        detail=(
                            "this sample's newest revision was opened on an imported source "
                            "mask, which belongs to one image and cannot be shared"
                        ),
                    )
                if (latest.document.image_width, latest.document.image_height) != (width, height):
                    raise HTTPException(
                        status_code=409,
                        detail="this sample's newest revision was drawn in a different frame",
                    )
                document = latest.document

            created = annotations_repo.create_sample_draft(conn, sample_id, document)
            conn.execute("COMMIT")
        except BaseException:
            if conn.in_transaction:
                conn.execute("ROLLBACK")
            raise
    _set_sample_draft_etag(response, created)
    return created


@router.get(
    "/api/samples/{sample_id}/annotations/draft",
    summary="Read the sample's current shared draft",
)
def get_sample_annotation_draft(
    request: Request, response: Response, sample_id: int
) -> AnnotationSampleDraft:
    settings: Settings = request.app.state.settings
    with connection(settings.db_path) as conn:
        draft = annotations_repo.get_sample_draft(conn, sample_id)
    if draft is None:
        raise HTTPException(status_code=404, detail=f"sample {sample_id} has no annotation draft")
    _set_sample_draft_etag(response, draft)
    return draft


@router.put(
    "/api/samples/{sample_id}/annotations/draft",
    summary="Save the shared draft if the caller still owns the version it read",
)
def save_sample_annotation_draft(
    request: Request,
    response: Response,
    sample_id: int,
    document: AnnotationDocument,
    if_match: Annotated[str | None, Header(alias="If-Match")] = None,
) -> AnnotationSampleDraft:
    expected = _require_if_match(if_match)
    settings: Settings = request.app.state.settings
    with connection(settings.db_path) as conn:
        conn.execute("BEGIN IMMEDIATE")
        try:
            dataset_id = _sample_dataset_id(conn, sample_id)
            _require_sample_scope(conn, dataset_id)
            current = annotations_repo.get_sample_draft(conn, sample_id)
            if current is None:
                raise HTTPException(
                    status_code=404, detail=f"sample {sample_id} has no annotation draft"
                )
            if expected != _sample_etag(current):
                raise HTTPException(
                    status_code=412, detail="the annotation draft changed elsewhere"
                )
            _validate_sample_document(conn, sample_id, dataset_id, document)
            saved = annotations_repo.update_sample_draft(conn, sample_id, current.version, document)
            if saved is None:  # pragma: no cover - BEGIN IMMEDIATE serialises writers
                raise HTTPException(
                    status_code=412, detail="the annotation draft changed elsewhere"
                )
            conn.execute("COMMIT")
        except BaseException:
            if conn.in_transaction:
                conn.execute("ROLLBACK")
            raise
    _set_sample_draft_etag(response, saved)
    return saved


def _validate_sample_document(
    conn: sqlite3.Connection,
    sample_id: int,
    dataset_id: int,
    document: AnnotationDocument,
) -> None:
    width, height = _shared_frame(images_repo.list_images_for_sample(conn, sample_id))
    if (document.image_width, document.image_height) != (width, height):
        raise HTTPException(
            status_code=422,
            detail="annotation dimensions must match the sample's shared frame exactly",
        )
    if document.base != "empty":
        raise HTTPException(
            status_code=422,
            detail="a sample-scoped document has no base layer to inherit",
        )
    _validate_taxonomy(conn, dataset_id, document)


@router.post(
    "/api/samples/{sample_id}/annotations/complete",
    summary="Freeze the shared draft as one revision per image of the sample",
)
def complete_sample_annotation_draft(
    request: Request,
    sample_id: int,
    if_match: Annotated[str | None, Header(alias="If-Match")] = None,
) -> list[AnnotationRevision]:
    """Render once, then write the same bytes to every image's own revision path.

    The digest is therefore identical across the fan-out, which is what makes "these
    channels share one truth" checkable after the fact rather than merely intended. Each
    image keeps its own `revision_no`, so one that carries earlier image-scoped history
    continues counting from where it stopped.
    """
    expected = _require_if_match(if_match)
    settings: Settings = request.app.state.settings
    written: list[Path] = []
    with connection(settings.db_path) as conn:
        conn.execute("BEGIN IMMEDIATE")
        try:
            dataset_id = _sample_dataset_id(conn, sample_id)
            _require_sample_scope(conn, dataset_id)
            draft = annotations_repo.get_sample_draft(conn, sample_id)
            if draft is None:
                raise HTTPException(
                    status_code=404, detail=f"sample {sample_id} has no annotation draft"
                )
            if expected != _sample_etag(draft):
                raise HTTPException(
                    status_code=412, detail="the annotation draft changed elsewhere"
                )
            _validate_sample_document(conn, sample_id, dataset_id, draft.document)
            if settings.annotations_dir.is_symlink():
                raise HTTPException(
                    status_code=409, detail="the app-owned annotation directory is unsafe"
                )

            images = images_repo.list_images_for_sample(conn, sample_id)
            document_sha256 = hashlib.sha256(draft.document.canonical_json().encode()).hexdigest()
            rendered: Path | None = None
            mask_sha256 = ""
            revisions: list[AnnotationRevision] = []
            for image in images:
                next_no = annotations_repo.next_revision_no(conn, image.id)
                destination = settings.annotation_image_dir(image.id) / f"revision-{next_no}.png"
                if destination.parent.is_symlink():
                    raise HTTPException(
                        status_code=409, detail="the app-owned annotation directory is unsafe"
                    )
                if rendered is None:
                    try:
                        mask_sha256 = render_binary_mask(
                            draft.document,
                            destination,
                            source_mask_path=None,
                            source_mask_sha256=None,
                        )
                    except AnnotationRenderError as exc:
                        raise HTTPException(status_code=409, detail=str(exc)) from exc
                    rendered = destination
                else:
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(rendered, destination)
                written.append(destination)
                revisions.append(
                    annotations_repo.insert_shared_revision(
                        conn,
                        image.id,
                        draft.document,
                        document_sha256=document_sha256,
                        mask_path=str(destination),
                        mask_sha256=mask_sha256,
                    )
                )
            annotations_repo.delete_sample_draft(conn, sample_id)
            conn.execute("COMMIT")
        except BaseException:
            if conn.in_transaction:
                conn.execute("ROLLBACK")
            for path in written:
                path.unlink(missing_ok=True)
            raise
    return revisions
