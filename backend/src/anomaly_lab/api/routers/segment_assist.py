"""Prompt-guided, non-destructive contour suggestions for the annotation editor."""

from __future__ import annotations

import asyncio
import importlib.util
import time
from typing import Literal

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field, model_validator

from anomaly_lab.config import Settings
from anomaly_lab.db.connection import connection
from anomaly_lab.db.repositories import annotations as annotations_repo
from anomaly_lab.db.repositories import images as images_repo
from anomaly_lab.db.repositories import jobs as jobs_repo
from anomaly_lab.domain.annotations import BitmapShape
from anomaly_lab.domain.entities import Image
from anomaly_lab.jobs.queue import JobQueue
from anomaly_lab.jobs.resident import ResidentError, ResidentWorker
from anomaly_lab.model_assets.catalog import get_spec
from anomaly_lab.model_assets.store import resolve_asset
from anomaly_lab.schemas import API_MODEL_CONFIG

router = APIRouter(tags=["segment-assist"])
ASSET_KEY = "mobile-sam-vit-t"


class AssistPoint(BaseModel):
    model_config = API_MODEL_CONFIG

    x: float = Field(ge=0)
    y: float = Field(ge=0)
    kind: Literal["positive", "negative"] = "positive"


class AssistBox(BaseModel):
    model_config = API_MODEL_CONFIG

    x0: float = Field(ge=0)
    y0: float = Field(ge=0)
    x1: float = Field(ge=0)
    y1: float = Field(ge=0)

    @model_validator(mode="after")
    def ordered(self) -> AssistBox:
        if self.x1 <= self.x0 or self.y1 <= self.y0:
            raise ValueError("box end must be below and to the right of its start")
        return self


class SegmentAssistRequest(BaseModel):
    model_config = API_MODEL_CONFIG

    points: list[AssistPoint] = Field(default_factory=list, max_length=32)
    box: AssistBox | None = None
    label_key: str = Field(min_length=1, max_length=64, pattern=r"^[a-z][a-z0-9_-]*$")
    operation: Literal["add", "subtract"] = "add"

    @model_validator(mode="after")
    def has_prompt(self) -> SegmentAssistRequest:
        if not self.points and self.box is None:
            raise ValueError("at least one point or a box is required")
        return self


class SegmentCandidate(BaseModel):
    model_config = API_MODEL_CONFIG

    shape: BitmapShape
    score: float = Field(description="MobileSAM's predicted mask quality; not a probability.")
    area: int = Field(ge=1, description="Candidate area in source-image pixels.")


class SegmentAssistResponse(BaseModel):
    model_config = API_MODEL_CONFIG

    image_id: int
    device: Literal["mps", "cpu"]
    warm: bool
    elapsed_ms: float
    candidates: list[SegmentCandidate] = Field(max_length=3)


class SegmentAssistCapability(BaseModel):
    model_config = API_MODEL_CONFIG

    provider: Literal["mobile_sam"] = "mobile_sam"
    title: str = "MobileSAM · TinyViT"
    asset_key: str = ASSET_KEY
    asset_status: Literal["missing", "invalid", "ready"]
    runtime_available: bool
    available: bool
    reason: str | None = None


@router.get("/api/segment-assist", summary="Whether prompt-guided contour assistance is ready")
def segment_assist_capability(request: Request) -> SegmentAssistCapability:
    settings: Settings = request.app.state.settings
    spec = get_spec(ASSET_KEY)
    if spec is None:  # pragma: no cover - fixed catalog and fixed consumer
        raise HTTPException(status_code=500, detail="MobileSAM asset is not catalogued")
    resolved = resolve_asset(settings, spec)
    runtime = (
        importlib.util.find_spec("torch") is not None
        and importlib.util.find_spec("mobile_sam") is not None
    )
    status: Literal["missing", "invalid", "ready"] = (
        "ready" if resolved.ready else "missing" if resolved.reason == "missing" else "invalid"
    )
    reason = None
    if not runtime:
        reason = "Install the backend's 'dl' extra to enable MobileSAM."
    elif not resolved.ready:
        reason = f"The MobileSAM asset is {resolved.reason}."
    return SegmentAssistCapability(
        asset_status=status,
        runtime_available=runtime,
        available=runtime and resolved.ready,
        reason=reason,
    )


@router.post(
    "/api/images/{image_id}/segment-assist",
    summary="Suggest source-coordinate masks from point and box prompts",
)
async def segment_assist(
    request: Request, image_id: int, body: SegmentAssistRequest
) -> SegmentAssistResponse:
    settings: Settings = request.app.state.settings
    image = await asyncio.to_thread(_validate_request, request, settings, image_id, body)
    spec = get_spec(ASSET_KEY)
    if spec is None:  # pragma: no cover
        raise HTTPException(status_code=500, detail="MobileSAM asset is not catalogued")
    resolved = await asyncio.to_thread(resolve_asset, settings, spec)
    if not resolved.ready:
        raise HTTPException(
            status_code=409,
            detail={
                "code": "model_asset_not_ready",
                "asset_key": ASSET_KEY,
                "reason": resolved.reason,
            },
        )

    resident: ResidentWorker = request.app.state.resident
    started = time.perf_counter()
    try:
        result, warm = await resident.segment(
            asset_key=ASSET_KEY,
            asset_path=resolved.path,
            image_id=image.id,
            points=[point.model_dump(mode="json") for point in body.points],
            box=body.box.model_dump(mode="json") if body.box else None,
            label_key=body.label_key,
            operation=body.operation,
        )
    except ResidentError as exc:
        detail = str(exc)
        tail = resident.stderr_tail()
        raise HTTPException(
            status_code=503, detail=f"{detail}\n{tail}" if tail else detail
        ) from exc

    try:
        return SegmentAssistResponse.model_validate(
            {
                **result,
                "warm": warm,
                "elapsed_ms": (time.perf_counter() - started) * 1000.0,
            }
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=503, detail="MobileSAM returned an invalid response"
        ) from exc


def _validate_request(
    request: Request,
    settings: Settings,
    image_id: int,
    body: SegmentAssistRequest,
) -> Image:
    queue: JobQueue = request.app.state.job_queue
    with connection(settings.db_path) as conn:
        job = jobs_repo.running_job(conn)
        if job is None and queue.current_job_id is not None:
            job = jobs_repo.get_job(conn, queue.current_job_id)
        if job is not None:
            raise HTTPException(
                status_code=409,
                detail=f"a {job.kind.value} job (id {job.id}) is running; try again when it ends",
            )
        image = images_repo.get_image(conn, image_id)
        if image is None:
            raise HTTPException(status_code=404, detail=f"no image with id {image_id}")
        row = conn.execute(
            "SELECT dataset_id FROM sample WHERE id = ?", (image.sample_id,)
        ).fetchone()
        dataset_id = int(row["dataset_id"])
        labels = {label.key for label in annotations_repo.list_labels(conn, dataset_id)}
    if body.label_key not in labels:
        raise HTTPException(status_code=422, detail=f"unknown annotation label {body.label_key!r}")
    for point in body.points:
        if point.x > image.width or point.y > image.height:
            raise HTTPException(status_code=422, detail="point lies outside the source image")
    if body.box and (body.box.x1 > image.width or body.box.y1 > image.height):
        raise HTTPException(status_code=422, detail="box lies outside the source image")
    return image
