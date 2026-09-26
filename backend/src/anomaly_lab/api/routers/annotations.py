"""Editable annotation drafts and immutable completed revisions.

The HTTP edge only: headers in (`If-Match`, `If-None-Match`), headers out (`ETag`,
`Cache-Control`), bytes shaped into responses. Every check, transaction and file write is in
`anomaly_lab.annotations.service`, and its refusals become status codes in `api/errors.py`.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Body, Header, HTTPException, Request, Response
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from anomaly_lab.annotation_bitmap import encode_png
from anomaly_lab.annotation_interchange import (
    CocoDocument,
    LabelMeDocument,
    coco_from_mask,
    labelme_from_mask,
)
from anomaly_lab.annotations import service
from anomaly_lab.annotations.service import (
    IMAGE_DRAFTS,
    SAMPLE_DRAFTS,
    AnnotationDraftState,
    AnnotationSampleDraftState,
    AnnotationScopeState,
    ClassCoverage,
    CopyRegionsResult,
)
from anomaly_lab.config import Settings
from anomaly_lab.domain.annotations import (
    AnnotationDocument,
    AnnotationDraft,
    AnnotationLabel,
    AnnotationLabelCreate,
    AnnotationLabelUpdate,
    AnnotationRevision,
    AnnotationSampleDraft,
)
from anomaly_lab.domain.entities import AnnotationScope
from anomaly_lab.schemas import API_MODEL_CONFIG

router = APIRouter(tags=["annotations"])


def _settings(request: Request) -> Settings:
    settings: Settings = request.app.state.settings
    return settings


def _set_etag(response: Response, etag: str) -> None:
    response.headers["ETag"] = etag
    response.headers["Cache-Control"] = "no-store"


def _require_if_match(value: str | None) -> str:
    if value is None:
        raise HTTPException(status_code=428, detail="If-Match is required for annotation writes")
    return value


def _require_if_none_match(value: str | None) -> None:
    """`If-None-Match: *` is what makes draft creation create-only.

    Without it the route would have to be an upsert, and an upsert is how the previous design
    lost work: a second window holding a stale seed would POST, be handed the *first* window's
    saved draft together with a currently-valid ETag for a document it had never read, and then
    overwrite it with a PUT that no precondition could refuse.
    """
    if value is None:
        raise HTTPException(
            status_code=428,
            detail="If-None-Match: * is required when creating an annotation draft",
        )
    if value.strip() != "*":
        raise HTTPException(
            status_code=422, detail="If-None-Match must be * when creating an annotation draft"
        )


def _discard_precondition(value: str | None) -> str | None:
    """The version a discard must match, or `None` for `If-Match: *`.

    `*` matches any current representation (RFC 9110), which is the deliberate force-discard:
    the caller has been shown that the draft moved and chose to drop it anyway.
    """
    expected = _require_if_match(value)
    return None if expected.strip() == "*" else expected


@router.get(
    "/api/datasets/{dataset_id}/annotation-labels",
    summary="The defect-label taxonomy for one dataset",
)
def list_annotation_labels(request: Request, dataset_id: int) -> list[AnnotationLabel]:
    return service.list_labels(_settings(request), dataset_id)


@router.post(
    "/api/datasets/{dataset_id}/annotation-labels",
    summary="Add one stable class key to a dataset's defect taxonomy",
)
def create_annotation_label(
    request: Request, dataset_id: int, body: AnnotationLabelCreate
) -> AnnotationLabel:
    return service.create_label(_settings(request), dataset_id, body)


@router.get(
    "/api/datasets/{dataset_id}/annotation-labels/coverage",
    summary="How many samples show, lack, or have no answer for each class",
)
def get_class_coverage(request: Request, dataset_id: int) -> list[ClassCoverage]:
    """Per class, in samples: what a few-shot run can take references from and test on."""
    return service.class_coverage(_settings(request), dataset_id)


@router.put(
    "/api/datasets/{dataset_id}/annotation-labels/{key}",
    summary="Rename or recolour a class without changing its stable key",
)
def update_annotation_label(
    request: Request, dataset_id: int, key: str, body: AnnotationLabelUpdate
) -> AnnotationLabel:
    return service.update_label(_settings(request), dataset_id, key, body)


@router.get(
    "/api/images/{image_id}/annotations/draft",
    summary="The draft in progress, or the document a new one would start from",
)
def get_annotation_draft(
    request: Request, response: Response, image_id: int
) -> AnnotationDraftState:
    """Read-or-seed, and never a write.

    A read that created the draft would persist a row every time an image was opened, and
    every such row would count as unsaved work forever. The first save creates it.
    """
    state, etag = IMAGE_DRAFTS.read(_settings(request), image_id)
    if etag is not None:
        _set_etag(response, etag)
    return state


@router.post(
    "/api/images/{image_id}/annotations/draft",
    summary="Create a draft from the first save, refusing if one already exists",
    status_code=201,
)
def create_annotation_draft(
    request: Request,
    response: Response,
    image_id: int,
    document: AnnotationDocument,
    if_none_match: Annotated[str | None, Header(alias="If-None-Match")] = None,
) -> AnnotationDraft:
    _require_if_none_match(if_none_match)
    created = IMAGE_DRAFTS.create(_settings(request), image_id, document)
    _set_etag(response, IMAGE_DRAFTS.etag(created))
    return created


@router.delete(
    "/api/images/{image_id}/annotations/draft",
    summary="Throw away a draft without completing it",
    status_code=204,
)
def discard_annotation_draft(
    request: Request,
    image_id: int,
    if_match: Annotated[str | None, Header(alias="If-Match")] = None,
) -> Response:
    IMAGE_DRAFTS.discard(_settings(request), image_id, _discard_precondition(if_match))
    return Response(status_code=204)


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
    saved = IMAGE_DRAFTS.save(_settings(request), image_id, document, expected)
    _set_etag(response, IMAGE_DRAFTS.etag(saved))
    return saved


class CopyRegionsRequest(BaseModel):
    """Which sibling channels receive a copy of this image's regions."""

    model_config = API_MODEL_CONFIG

    target_image_ids: list[int] = Field(min_length=1)


@router.post(
    "/api/images/{image_id}/annotations/copy-regions",
    summary="Append this image's regions to other channels of the same sample",
)
def copy_annotation_regions(
    request: Request,
    image_id: int,
    body: CopyRegionsRequest,
    if_match: Annotated[str | None, Header(alias="If-Match")] = None,
) -> CopyRegionsResult:
    """Copy, never move and never replace.

    Appending is what lets the targets go unguarded: an operation that only adds cannot lose
    what is already there, so a target draft saved in another window survives intact. The
    *source* still carries a precondition, because copying a stale document into three channels
    is exactly the mistake an editor that has fallen behind would make.

    Equal dimensions are a refusal rather than a rescale. An annotation is in source-image
    pixels and never leaves that frame (ADR-0032); scaling one into a differently sized channel
    would silently invent geometry nobody drew.
    """
    expected = _require_if_match(if_match)
    return IMAGE_DRAFTS.copy_regions(_settings(request), image_id, body.target_image_ids, expected)


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
    return IMAGE_DRAFTS.complete(_settings(request), image_id, expected)


@router.get(
    "/api/images/{image_id}/annotations/revisions",
    summary="Completed annotation history, newest first",
)
def list_annotation_revisions(request: Request, image_id: int) -> list[AnnotationRevision]:
    return service.list_revisions(_settings(request), image_id)


@router.get(
    "/api/images/{image_id}/annotations/revisions/{revision_id}/mask",
    summary="The immutable binary PNG materialised for a completed revision",
)
def read_annotation_revision_mask(
    request: Request, image_id: int, revision_id: int
) -> FileResponse:
    revision = service.revision_mask(_settings(request), image_id, revision_id)
    return FileResponse(
        revision.mask_path,
        media_type="image/png",
        headers={
            "ETag": f'"annotation-mask-{revision.mask_sha256}"',
            "Cache-Control": "public, max-age=31536000, immutable",
        },
    )


@router.get(
    "/api/images/{image_id}/annotations/source-mask",
    summary="The imported mask a `source_mask` document is based on, as a binary PNG",
    response_class=Response,
    responses={200: {"content": {"image/png": {}}}},
)
def read_annotation_source_mask(request: Request, image_id: int) -> Response:
    """The base layer of a `base="source_mask"` document, and nothing else.

    The editor used to draw this base from `GET /api/images/{id}/mask`, which answers a
    different question — the *current* truth, newest completed revision first. Once an
    image had a revision the "base" on screen was that revision's outline, already
    containing the regions being edited above it, and after an import that subtracts the
    whole base it showed truth the document no longer had. A document's base is the
    imported mask it pins, so that is what this serves: source-sized 0/255, verified
    against the digest the catalog recorded, and never a revision.
    """
    content, sha256 = service.source_mask_png(_settings(request), image_id)
    return Response(
        content=content,
        media_type="image/png",
        headers={
            "ETag": f'"annotation-source-mask-{sha256}"',
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
    return _import(request, response, image_id, expected, service.from_png(payload, label_key))


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
    expected = _require_if_match(if_match)
    return _import(request, response, image_id, expected, service.from_labelme(payload))


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
    expected = _require_if_match(if_match)
    return _import(request, response, image_id, expected, service.from_coco(payload))


def _import(
    request: Request,
    response: Response,
    image_id: int,
    expected: str,
    build_shapes: service.ShapeBuilder,
) -> AnnotationDraft:
    saved = IMAGE_DRAFTS.import_file(_settings(request), image_id, expected, build_shapes)
    _set_etag(response, IMAGE_DRAFTS.etag(saved))
    return saved


@router.get(
    "/api/images/{image_id}/annotations/export/png",
    summary="Export the current completed truth as a binary PNG mask",
    response_class=Response,
    responses={200: {"content": {"image/png": {}}}},
)
def export_annotation_png(request: Request, image_id: int) -> Response:
    mask, _, _ = service.export_context(_settings(request), image_id)
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
    mask, image_path, label = service.export_context(_settings(request), image_id)
    return labelme_from_mask(mask, image_path=image_path, label=label)


@router.get(
    "/api/images/{image_id}/annotations/export/coco",
    summary="Export the current completed truth as one-image COCO RLE",
)
def export_annotation_coco(request: Request, image_id: int) -> CocoDocument:
    mask, image_path, label = service.export_context(_settings(request), image_id)
    return coco_from_mask(mask, image_path=image_path, label=label)


# --- Sample-scoped editing (ADR-0036) --------------------------------------------------
#
# The same lifecycle as above behind `SAMPLE_DRAFTS`; see the service for what differs.


class AnnotationScopeUpdate(BaseModel):
    model_config = API_MODEL_CONFIG

    scope: AnnotationScope


@router.get(
    "/api/datasets/{dataset_id}/annotation-scope",
    summary="Whether this dataset annotates images or whole samples",
)
def read_annotation_scope(request: Request, dataset_id: int) -> AnnotationScopeState:
    return service.read_scope(_settings(request), dataset_id)


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
    return service.set_scope(_settings(request), dataset_id, body.scope)


@router.get(
    "/api/samples/{sample_id}/annotations/draft",
    summary="The shared draft in progress, or the document a new one would start from",
)
def get_sample_annotation_draft(
    request: Request, response: Response, sample_id: int
) -> AnnotationSampleDraftState:
    state, etag = SAMPLE_DRAFTS.read(_settings(request), sample_id)
    if etag is not None:
        _set_etag(response, etag)
    return state


@router.post(
    "/api/samples/{sample_id}/annotations/draft",
    summary="Create the shared draft from the first save, refusing if one already exists",
    status_code=201,
)
def create_sample_annotation_draft(
    request: Request,
    response: Response,
    sample_id: int,
    document: AnnotationDocument,
    if_none_match: Annotated[str | None, Header(alias="If-None-Match")] = None,
) -> AnnotationSampleDraft:
    _require_if_none_match(if_none_match)
    created = SAMPLE_DRAFTS.create(_settings(request), sample_id, document)
    _set_etag(response, SAMPLE_DRAFTS.etag(created))
    return created


@router.delete(
    "/api/samples/{sample_id}/annotations/draft",
    summary="Throw away the shared draft without completing it",
    status_code=204,
)
def discard_sample_annotation_draft(
    request: Request,
    sample_id: int,
    if_match: Annotated[str | None, Header(alias="If-Match")] = None,
) -> Response:
    SAMPLE_DRAFTS.discard(_settings(request), sample_id, _discard_precondition(if_match))
    return Response(status_code=204)


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
    saved = SAMPLE_DRAFTS.save(_settings(request), sample_id, document, expected)
    _set_etag(response, SAMPLE_DRAFTS.etag(saved))
    return saved


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
    return SAMPLE_DRAFTS.complete(_settings(request), sample_id, expected)
