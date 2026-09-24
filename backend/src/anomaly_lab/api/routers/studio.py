"""The reference studio's live preview (ADR-0040): one image segmented by the current references.

The HTTP edge only. What the preview is fitted on and why a request is refused is
`experiments/preview.py`; who holds the fitted method is the resident worker (ADR-0026),
under the same lock as every other resident request, so a preview can never share the
accelerator with a running job.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

import numpy as np
from fastapi import APIRouter, HTTPException, Request, Response
from pydantic import BaseModel, Field

from anomaly_lab.config import Settings
from anomaly_lab.experiments.preview import (
    MAX_REFERENCES,
    PreviewError,
    PreviewSpec,
    preview_dir,
    resolve,
)
from anomaly_lab.experiments.service import refuse_while_a_job_runs
from anomaly_lab.jobs.queue import JobQueue
from anomaly_lab.jobs.resident import ResidentError, ResidentWorker
from anomaly_lab.media.overlay import render_anomaly_map
from anomaly_lab.schemas import API_MODEL_CONFIG

router = APIRouter(tags=["studio"])


class PreviewRequest(BaseModel):
    model_config = API_MODEL_CONFIG

    class_key: str
    method: str = Field(description="A method that declares `few_shot_segmentation`.")
    profile_id: int = Field(description="The built region profile the method reads.")
    references: list[int] = Field(
        min_length=1, max_length=MAX_REFERENCES, description="Reference sample ids."
    )
    image_id: int = Field(description="The image to segment.")


class PreviewResult(BaseModel):
    model_config = API_MODEL_CONFIG

    image_id: int
    score: float = Field(description="Presence confidence, in this preview's own units.")
    foreground_share: float = Field(
        description="Share of the image at or above 0.5 foreground probability."
    )
    generation: str = Field(description="Identifies the fitted preview; part of the map's URL.")
    map_url: str
    warm: bool
    elapsed_ms: float


def _settings(request: Request) -> Settings:
    settings: Settings = request.app.state.settings
    return settings


@router.post(
    "/api/datasets/{dataset_id}/studio/preview",
    summary="Segment one image with a few-shot method fitted on the studio's references",
)
async def preview(request: Request, dataset_id: int, body: PreviewRequest) -> PreviewResult:
    settings = _settings(request)
    queue: JobQueue = request.app.state.job_queue
    resident: ResidentWorker = request.app.state.resident
    spec = PreviewSpec(
        dataset_id=dataset_id,
        class_key=body.class_key,
        method=body.method,
        profile_id=body.profile_id,
        references=body.references,
    )
    try:
        resolved = await asyncio.to_thread(resolve, settings, spec)
    except PreviewError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    await asyncio.to_thread(refuse_while_a_job_runs, settings, queue)

    started = time.perf_counter()
    try:
        result, warm = await resident.preview(
            spec_json=spec.canonical(), generation=resolved.generation, image_id=body.image_id
        )
    except ResidentError as exc:
        tail = resident.stderr_tail()
        raise HTTPException(status_code=503, detail=f"{exc}\n{tail}" if tail else str(exc)) from exc
    return PreviewResult(
        image_id=body.image_id,
        score=float(result.get("score", 0.0)),  # type: ignore[arg-type]
        foreground_share=float(result.get("foreground_share", 0.0)),  # type: ignore[arg-type]
        generation=resolved.generation,
        map_url=f"/api/studio/previews/{resolved.generation}/{body.image_id}.png",
        warm=warm,
        elapsed_ms=(time.perf_counter() - started) * 1000.0,
    )


@router.get(
    "/api/studio/previews/{generation}/{image_id}.png",
    summary="A preview's foreground probability as an overlay",
    response_class=Response,
    responses={200: {"content": {"image/png": {}}}},
)
def preview_map(request: Request, generation: str, image_id: int) -> Response:
    """The map one preview wrote, coloured on the fixed probability range [0, 1]."""
    if not generation.isalnum():
        raise HTTPException(status_code=404, detail="no such preview")
    path: Path = preview_dir(_settings(request), generation) / "maps" / f"{image_id}.npy"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="no such preview")
    values = np.load(path, allow_pickle=False)
    payload = render_anomaly_map(values, value_range=(0.0, 1.0))
    return Response(
        content=payload,
        media_type="image/png",
        headers={"Cache-Control": "no-store"},
    )
