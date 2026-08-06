"""Image delivery.

**Files are served only by `image_id`, resolved through the database — never by a path
the client supplied** (§11). That is the whole of the path-traversal story here: there is
no request that can name a file, so no request can name the wrong one.

Every response is content-addressed. The `ETag` is derived from the image's `sha256` and
the tier, and imported files are immutable, so `Cache-Control: immutable` is a statement
of fact rather than a hope: a client that has the bytes never needs to ask for them again.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from anomaly_lab.api.routers.jobs import JobSummary, summary_of
from anomaly_lab.config import Settings
from anomaly_lab.db.connection import connection
from anomaly_lab.db.repositories import datasets as datasets_repo
from anomaly_lab.db.repositories import images as images_repo
from anomaly_lab.domain.entities import Image, JobKind
from anomaly_lab.jobs.queue import JobQueue
from anomaly_lab.media.cache import TIERS, ImageTier, ensure_cached, etag_for, render
from anomaly_lab.media.decode import UnreadableImageError
from anomaly_lab.media.prewarm import PrewarmParams
from anomaly_lab.schemas import API_MODEL_CONFIG

router = APIRouter(prefix="/api/images", tags=["images"])

# A year, which is as close to "forever" as the header allows. Safe because the cache key
# is the content hash: different bytes are a different URL response, never a stale one.
IMMUTABLE_CACHE_CONTROL = "public, max-age=31536000, immutable"


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
