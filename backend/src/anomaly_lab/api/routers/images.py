"""Image delivery.

**Files are served only by `image_id`, resolved through the database — never by a path
the client supplied** (§11). That is the whole of the path-traversal story here: there is
no request that can name a file, so no request can name the wrong one.

Every response is content-addressed. The `ETag` is derived from the image's `sha256` and
the tier, and imported files are immutable, so `Cache-Control: immutable` is a statement
of fact rather than a hope: a client that has the bytes never needs to ask for them again.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path

import numpy as np
from fastapi import APIRouter, HTTPException, Query, Request, Response
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from anomaly_lab.api.routers.jobs import JobSummary, summary_of
from anomaly_lab.config import Settings
from anomaly_lab.db.connection import connection
from anomaly_lab.db.repositories import annotations as annotations_repo
from anomaly_lab.db.repositories import datasets as datasets_repo
from anomaly_lab.db.repositories import images as images_repo
from anomaly_lab.db.repositories import results as results_repo
from anomaly_lab.domain.entities import Image, JobKind
from anomaly_lab.jobs.queue import JobQueue
from anomaly_lab.media.cache import TIERS, ImageTier, ensure_cached, etag_for, render
from anomaly_lab.media.decode import UnreadableImageError
from anomaly_lab.media.overlay import (
    read_display_range,
    render_anomaly_map,
    render_mask_contour,
    render_prediction_region,
)
from anomaly_lab.media.prewarm import PrewarmParams
from anomaly_lab.media.values import encode_plane
from anomaly_lab.models.preprocessing import load_mask
from anomaly_lab.schemas import API_MODEL_CONFIG

router = APIRouter(prefix="/api/images", tags=["images"])

# A year, which is as close to "forever" as the header allows. Safe because the cache key
# is the content hash: different bytes are a different URL response, never a stale one.
IMMUTABLE_CACHE_CONTROL = "public, max-age=31536000, immutable"


class MapRender(StrEnum):
    """How an anomaly map should be drawn.

    A heatmap answers "how anomalous, everywhere"; a segmentation answers "where, exactly".
    Only the second has an edge, and only something with an edge can be laid against a
    ground-truth outline and read as agreement or disagreement.
    """

    HEATMAP = "heatmap"
    REGION = "region"
    CONTOUR = "contour"


class PrewarmRequest(BaseModel):
    model_config = API_MODEL_CONFIG

    dataset_id: int
    tiers: list[ImageTier] | None = Field(
        default=None,
        description="Which tiers to render. Defaults to every cacheable tier.",
    )


def _load_image(request: Request, image_id: int) -> tuple[Image, Settings]:
    settings: Settings = request.app.state.settings
    with connection(settings.db_path) as conn:
        image = images_repo.get_image(conn, image_id)
    if image is None:
        raise HTTPException(status_code=404, detail=f"no image with id {image_id}")
    return image, settings


@router.post("/prewarm", summary="Render a dataset's cached tiers up front")
def start_prewarm(request: Request, body: PrewarmRequest) -> JobSummary:
    """Start a pre-warm job so the first browse of a dataset is not the slowest one."""
    settings: Settings = request.app.state.settings
    with connection(settings.db_path) as conn:
        if datasets_repo.get_dataset(conn, body.dataset_id) is None:
            raise HTTPException(status_code=404, detail=f"no dataset with id {body.dataset_id}")

    params = PrewarmParams(
        dataset_id=body.dataset_id,
        **({"tiers": body.tiers} if body.tiers is not None else {}),
    )
    queue: JobQueue = request.app.state.job_queue
    return summary_of(queue.enqueue(kind=JobKind.PREWARM, params=params.model_dump(mode="json")))


@router.get(
    "/{image_id}/anomaly-map",
    summary="One experiment's anomaly map for an image, colormapped",
    response_class=Response,
    responses={200: {"content": {"image/png": {}}}},
)
def read_anomaly_map(
    request: Request,
    image_id: int,
    experiment_id: int = Query(description="Which experiment's map to render."),
    native: bool = Query(
        default=True,
        description="Resample the map to the source image's pixel grid so it overlays exactly.",
    ),
    alpha: bool = Query(
        default=True,
        description=(
            "Scale opacity with the score, for laying over the source image. Set false "
            "for a standalone panel, where a low-scoring map would otherwise be invisible."
        ),
    ),
    render: MapRender = Query(
        default=MapRender.HEATMAP,
        description=(
            "`heatmap` colormaps the whole map. `region` and `contour` draw only where it "
            "crosses `threshold`, which is the model's own segmentation."
        ),
    ),
    threshold: float | None = Query(
        default=None,
        description=(
            "Cut in **map units**, required by `region` and `contour`. A display decision "
            "only: no metric is computed from it."
        ),
    ),
) -> Response:
    """Render a stored float32 map as a PNG (ADR-0007).

    Every map of one experiment is stretched over the **same** range, read from the
    range file the inference job wrote. Normalizing each map to its own extremes would
    make a clean part look as alarming as a defective one, which is precisely the
    comparison the overlay exists to support.

    `alpha` is the *overlay* decision, and it is wrong outside an overlay. Laid over a
    photograph, score-driven alpha is what keeps the quiet regions from being tinted; in
    a diagnostics panel beside an opaque per-branch map, it makes a clean image look
    blank and puts the two panes on visibly different scales — which defeats the
    comparison the panel exists for.

    `render` chooses between a heatmap and a **segmentation**. The heatmap shows how much,
    everywhere; the segmentation shows where, with an edge — which is the only form that
    can be laid against a ground-truth outline and read as agreement or disagreement.

    The numbers behind the picture — this map's extremes and the run's range — are served
    as JSON on the experiment's per-sample image route, not as headers here. An `<img>` tag
    cannot read a response header, and this endpoint exists to be an `img src`.
    """
    if render is not MapRender.HEATMAP and threshold is None:
        raise HTTPException(
            status_code=422,
            detail=f"render={render.value} needs a threshold, in the map's own units",
        )

    image, settings = _load_image(request, image_id)
    with connection(settings.db_path) as conn:
        stored = results_repo.get_image_result(conn, experiment_id, image_id)
    if stored is None or not stored.map_path:
        raise HTTPException(
            status_code=404,
            detail=f"experiment {experiment_id} has no anomaly map for image {image_id}",
        )

    try:
        array = np.load(stored.map_path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        # The artifact directory is deletable by design, so a missing map is an expected
        # state rather than corruption — 410, the same answer a missing source file gets.
        raise HTTPException(
            status_code=410,
            detail=f"the anomaly map file for image {image_id} is no longer readable",
        ) from exc

    maps_dir = settings.experiment_dir(experiment_id) / "maps"
    value_range = read_display_range(maps_dir)
    size = (image.width, image.height) if native else None

    if render is MapRender.HEATMAP:
        payload = render_anomaly_map(
            array, value_range=value_range, size=size, alpha_follows_score=alpha
        )
    else:
        payload = render_prediction_region(
            array,
            float(threshold or 0.0),
            size=size,
            outline=render is MapRender.CONTOUR,
        )

    tag = f"{render.value}-{threshold}" if render is not MapRender.HEATMAP else f"h{int(alpha)}"
    return Response(
        content=payload,
        media_type="image/png",
        headers=_headers(f'W/"map-{experiment_id}-{image.sha256[:16]}-{int(native)}-{tag}"'),
    )


@router.get(
    "/{image_id}/anomaly-map/values",
    summary="The anomaly map's own numbers, for a readout under the cursor",
    response_class=Response,
    responses={200: {"content": {"application/octet-stream": {}}}},
)
def read_anomaly_map_values(request: Request, image_id: int, experiment_id: int) -> Response:
    """The stored map as a float32 plane (ADR-0023).

    Resolved through `image_result` exactly as the PNG above is, so this inherits the same
    property: no request can name a file. It exists to be **read**, never drawn — the
    colormap and the run-wide display range stay on this side, in one language.
    """
    image, settings = _load_image(request, image_id)
    with connection(settings.db_path) as conn:
        stored = results_repo.get_image_result(conn, experiment_id, image_id)
    if stored is None or not stored.map_path:
        raise HTTPException(
            status_code=404,
            detail=f"experiment {experiment_id} has no anomaly map for image {image_id}",
        )

    try:
        array = np.load(stored.map_path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise HTTPException(
            status_code=410,
            detail=f"the anomaly map file for image {image_id} is no longer readable",
        ) from exc

    return Response(
        content=encode_plane(array),
        media_type="application/octet-stream",
        # Weak, and revalidated: re-running inference overwrites this file in place, so the
        # same reasoning as the PNG's ETag applies (ADR-0019).
        headers={
            "ETag": f'W/"values-{experiment_id}-{image.sha256[:16]}"',
            "Cache-Control": "no-cache",
        },
    )


@router.get(
    "/{image_id}/mask",
    summary="Ground-truth mask outline for an image",
    response_class=Response,
    responses={200: {"content": {"image/png": {}}}},
)
def read_mask(request: Request, image_id: int) -> Response:
    """The ground-truth outline as a transparent PNG, ready to lay over the source.

    An outline rather than a filled region: filling the mask hides the pixels the reader
    is trying to compare the model's map against.
    """
    image, settings = _load_image(request, image_id)
    with connection(settings.db_path) as conn:
        try:
            truth = annotations_repo.resolve_ground_truth_masks(
                conn, [image_id], verify_bytes=True
            ).get(image_id)
        except annotations_repo.GroundTruthDriftError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
    if truth is None:
        raise HTTPException(status_code=404, detail=f"image {image_id} has no ground-truth mask")

    try:
        mask = load_mask(Path(truth.path))
    except UnreadableImageError as exc:
        raise HTTPException(status_code=410, detail=str(exc)) from exc

    payload = render_mask_contour(mask, size=(image.width, image.height))
    return Response(
        content=payload,
        media_type="image/png",
        headers=_headers(f'"ground-truth-{truth.sha256}"'),
    )


@router.get(
    "/{image_id}/{tier}",
    summary="One rendered tier of an image",
    response_class=Response,
    responses={
        200: {"content": {"image/webp": {}, "image/png": {}}},
        304: {"description": "The client's copy is current."},
    },
)
def read_tier(request: Request, image_id: int, tier: ImageTier) -> Response:
    """Serve `thumb`, `preview` or `full`.

    `thumb` and `preview` are rendered once and cached; `full` is rendered per request,
    because caching a lossless copy of every image would cost most of a gigabyte per
    dataset to avoid re-rendering something that is looked at once.
    """
    image, settings = _load_image(request, image_id)
    etag = etag_for(image, tier)

    # Answered before any decoding, which is what makes a scroll back through the grid
    # free rather than merely fast.
    if request.headers.get("if-none-match") == etag:
        return Response(status_code=304, headers=_headers(etag))

    try:
        if TIERS[tier].cached:
            path = ensure_cached(settings, image, tier)
            return FileResponse(
                path,
                media_type=TIERS[tier].media_type,
                headers=_headers(etag),
            )
        return Response(
            content=render(image, tier),
            media_type=TIERS[tier].media_type,
            headers=_headers(etag),
        )
    except UnreadableImageError as exc:
        # The catalog references files in place, so a source file can disappear or be
        # replaced between import and now. `verify` is how that is found deliberately;
        # this is how it surfaces when someone simply opens the image.
        raise HTTPException(status_code=410, detail=str(exc)) from exc


def _headers(etag: str) -> dict[str, str]:
    return {"ETag": etag, "Cache-Control": IMMUTABLE_CACHE_CONTROL}
