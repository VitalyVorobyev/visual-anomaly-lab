"""The Prepare screen's live stage: an unsaved recipe on one image, and its sampled check.

Tuning a profile is a loop of small changes, so neither of these writes a revision. The
live preview answers on the request path — in a thread for the classical extractors,
through the resident MobileSAM for the learned one (ADR-0026) — and the sampled check is
the same `region_prepare` preview job a saved revision gets, carrying the recipe instead
of a profile id.
"""

from __future__ import annotations

import asyncio
import sqlite3
import time
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, Field

from anomaly_lab.api.routers.jobs import JobSummary, summary_of
from anomaly_lab.api.routers.region_profiles import PreparationSize, validated_recipe
from anomaly_lab.api.routers.segment_assist import refuse_while_a_job_runs
from anomaly_lab.config import Settings
from anomaly_lab.db.connection import connection
from anomaly_lab.db.repositories import datasets as datasets_repo
from anomaly_lab.db.repositories import images as images_repo
from anomaly_lab.domain.entities import Image as ImageEntity
from anomaly_lab.domain.entities import JobKind, SampleAlignment
from anomaly_lab.jobs.queue import JobQueue
from anomaly_lab.jobs.resident import ResidentError, ResidentWorker
from anomaly_lab.model_assets.catalog import get_spec
from anomaly_lab.model_assets.store import resolve_asset
from anomaly_lab.regions.base import RegionExtraction
from anomaly_lab.regions.live import (
    LiveSourceTooLargeError,
    RegionLivePreview,
    compose,
    locate_members,
    located_from,
    members_for,
    require_live_size,
)
from anomaly_lab.regions.mobile_sam import ASSET_KEY as MOBILE_SAM_ASSET
from anomaly_lab.regions.preparation import (
    PREVIEW_LIMIT,
    RegionRecipe,
    Size,
    preview_selection,
    resolve_assets,
)
from anomaly_lab.regions.registry import build as build_extractor
from anomaly_lab.regions.registry import get_extractor_class
from anomaly_lab.schemas import API_MODEL_CONFIG

router = APIRouter(tags=["region-profiles"])

# Extractors answered by the resident worker rather than in the API process.
RESIDENT_EXTRACTORS = {"mobile_sam": MOBILE_SAM_ASSET}


class RegionPreviewRequest(RegionRecipe):
    """An unsaved profile, one image of the dataset, and the size to prepare it at."""

    image_id: int
    width: int = Field(ge=8, le=2048, description="Prepared frame width in pixels.")
    height: int = Field(ge=8, le=2048, description="Prepared frame height in pixels.")


class RegionCheckRequest(RegionRecipe):
    """An unsaved profile to run over the sampled images, at one size."""

    width: int = Field(ge=8, le=2048, description="Prepared frame width in pixels.")
    height: int = Field(ge=8, le=2048, description="Prepared frame height in pixels.")


class RegionPreviewImage(BaseModel):
    """One image the live stage can step to, named the way the browser names it."""

    model_config = API_MODEL_CONFIG

    image_id: int
    sample_id: int
    group_key: str
    external_id: str
    channel: str | None
    width: int
    height: int


class RegionPreviewImages(BaseModel):
    model_config = API_MODEL_CONFIG

    total: int = Field(description="Images in the dataset; the list is spread evenly across them.")
    images: list[RegionPreviewImage]


AlignmentQuery = Annotated[
    SampleAlignment,
    Query(description="Spread over whole samples (union) or over images and channels."),
]
LimitQuery = Annotated[int, Query(ge=1, le=96, description="How many images to return.")]


@router.get(
    "/api/datasets/{dataset_id}/region-preview/images",
    summary="Images to step through on the live Prepare stage, evenly spaced over the dataset",
)
def region_preview_images(
    request: Request,
    dataset_id: int,
    alignment: AlignmentQuery = SampleAlignment.PER_IMAGE,
    limit: LimitQuery = PREVIEW_LIMIT,
) -> RegionPreviewImages:
    settings: Settings = request.app.state.settings
    with connection(settings.db_path) as conn:
        _require_dataset(conn, dataset_id)
        images = images_repo.list_images_for_dataset(conn, dataset_id)
        chosen = [images[index].id for index in preview_selection(images, alignment, limit)]
        # A union selection keeps whole samples; the stage steps through one image of each.
        if alignment is SampleAlignment.UNION:
            by_id = {image.id: image for image in images}
            seen: set[int] = set()
            unique: list[int] = []
            for image_id in chosen:
                sample_id = by_id[image_id].sample_id
                if sample_id not in seen:
                    seen.add(sample_id)
                    unique.append(image_id)
            chosen = unique
        return RegionPreviewImages(total=len(images), images=_describe(conn, chosen))


@router.get(
    "/api/datasets/{dataset_id}/region-preview/random",
    summary="One image of the dataset, chosen at random, for the live Prepare stage",
)
def region_preview_random(request: Request, dataset_id: int) -> RegionPreviewImage:
    settings: Settings = request.app.state.settings
    with connection(settings.db_path) as conn:
        _require_dataset(conn, dataset_id)
        row = conn.execute(
            """
            SELECT image.id FROM image JOIN sample ON sample.id = image.sample_id
             WHERE sample.dataset_id = ? ORDER BY random() LIMIT 1
            """,
            (dataset_id,),
        ).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="this dataset has no images")
        return _describe(conn, [int(row["id"])])[0]


@router.post(
    "/api/datasets/{dataset_id}/region-preview",
    summary="Prepare one image under an unsaved region profile, synchronously",
)
async def region_preview(
    request: Request, dataset_id: int, body: RegionPreviewRequest
) -> RegionLivePreview:
    started = time.perf_counter()
    recipe = _recipe_of(body)
    availability = get_extractor_class(recipe.extractor_type).availability()
    if not availability.available:
        raise HTTPException(
            status_code=422,
            detail=availability.reason or f"{recipe.extractor_type} is not available",
        )
    settings: Settings = request.app.state.settings
    size: Size = (body.width, body.height)
    target, members = await asyncio.to_thread(
        _target_and_members, request, settings, dataset_id, body.image_id, recipe
    )
    try:
        require_live_size(members)
    except LiveSourceTooLargeError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    asset_key = RESIDENT_EXTRACTORS.get(recipe.extractor_type)
    if asset_key is None:
        return await asyncio.to_thread(
            _local_preview, settings, recipe, size, target, members, started
        )

    resident: ResidentWorker = request.app.state.resident
    asset_path = await asyncio.to_thread(_ready_asset, settings, asset_key)
    located = []
    for member in members:
        try:
            answer, _warm = await resident.region(
                asset_key=asset_key,
                asset_path=asset_path,
                image_id=member.id,
                config=recipe.extractor_config,
            )
        except ResidentError as exc:
            tail = resident.stderr_tail()
            detail = f"{exc}\n{tail}" if tail else str(exc)
            raise HTTPException(status_code=503, detail=detail) from exc
        extraction, error = _resident_extraction(answer)
        located.append(located_from(recipe, size, member, extraction, error))
    return await asyncio.to_thread(compose, recipe, size, target, located, started=started)


@router.post(
    "/api/datasets/{dataset_id}/region-check",
    summary="Queue the sampled preview of an unsaved region profile",
)
def region_check(request: Request, dataset_id: int, body: RegionCheckRequest) -> JobSummary:
    recipe = _recipe_of(body)
    settings: Settings = request.app.state.settings
    with connection(settings.db_path) as conn:
        _require_dataset(conn, dataset_id)
    size = PreparationSize(width=body.width, height=body.height)
    queue: JobQueue = request.app.state.job_queue
    return summary_of(
        queue.enqueue(
            kind=JobKind.REGION_PREPARE,
            params={
                "dataset_id": dataset_id,
                "mode": "preview",
                "recipe": recipe.model_dump(mode="json"),
                "width": size.width,
                "height": size.height,
            },
        )
    )


def _recipe_of(body: RegionRecipe) -> RegionRecipe:
    """Just the recipe of a request, its extractor config validated with defaults filled in."""
    recipe = RegionRecipe.model_validate(body.model_dump(include=set(RegionRecipe.model_fields)))
    return recipe.model_copy(update={"extractor_config": validated_recipe(recipe)})


def _require_dataset(conn: sqlite3.Connection, dataset_id: int) -> None:
    if datasets_repo.get_dataset(conn, dataset_id) is None:
        raise HTTPException(status_code=404, detail=f"no dataset with id {dataset_id}")


def _describe(conn: sqlite3.Connection, image_ids: list[int]) -> list[RegionPreviewImage]:
    if not image_ids:
        return []
    marks = ",".join("?" for _ in image_ids)
    rows = conn.execute(
        f"""
        SELECT image.id, image.sample_id, image.width, image.height,
               sample.group_key, sample.external_id, channel.name AS channel
          FROM image
          JOIN sample ON sample.id = image.sample_id
          LEFT JOIN channel ON channel.id = image.channel_id
         WHERE image.id IN ({marks})
        """,
        image_ids,
    ).fetchall()
    by_id = {
        int(row["id"]): RegionPreviewImage(
            image_id=int(row["id"]),
            sample_id=int(row["sample_id"]),
            group_key=str(row["group_key"]),
            external_id=str(row["external_id"]),
            channel=row["channel"],
            width=int(row["width"]),
            height=int(row["height"]),
        )
        for row in rows
    }
    return [by_id[image_id] for image_id in image_ids if image_id in by_id]


def _target_and_members(
    request: Request,
    settings: Settings,
    dataset_id: int,
    image_id: int,
    recipe: RegionRecipe,
) -> tuple[ImageEntity, list[ImageEntity]]:
    with connection(settings.db_path) as conn:
        if recipe.extractor_type in RESIDENT_EXTRACTORS:
            refuse_while_a_job_runs(request, conn)
        target = images_repo.get_image(conn, image_id)
        if target is None or images_repo.dataset_id_of(conn, image_id) != dataset_id:
            raise HTTPException(
                status_code=404, detail=f"no image {image_id} in dataset {dataset_id}"
            )
        sample_images = (
            images_repo.list_images_for_sample(conn, target.sample_id)
            if recipe.sample_alignment is SampleAlignment.UNION
            else [target]
        )
    return target, members_for(recipe, target, sample_images)


def _local_preview(
    settings: Settings,
    recipe: RegionRecipe,
    size: Size,
    target: ImageEntity,
    members: list[ImageEntity],
    started: float,
) -> RegionLivePreview:
    extractor = build_extractor(
        recipe.extractor_type,
        recipe.extractor_config,
        assets=resolve_assets(settings, recipe.extractor_type),
    )
    located = locate_members(recipe, size, members, extractor)
    return compose(recipe, size, target, located, started=started)


def _ready_asset(settings: Settings, asset_key: str) -> Path:
    spec = get_spec(asset_key)
    if spec is None:  # pragma: no cover - fixed catalog and fixed consumer
        raise HTTPException(status_code=500, detail=f"{asset_key} is not catalogued")
    resolved = resolve_asset(settings, spec)
    if not resolved.ready:
        raise HTTPException(
            status_code=409,
            detail=f"The {spec.title} model asset is {resolved.reason}; install it to preview.",
        )
    return resolved.path


def _resident_extraction(
    answer: dict[str, object],
) -> tuple[RegionExtraction | None, str | None]:
    error = answer.get("error")
    if isinstance(error, str):
        return None, error
    try:
        return RegionExtraction.model_validate(answer.get("extraction")), None
    except ValueError as exc:
        raise HTTPException(status_code=503, detail="MobileSAM returned an invalid region") from exc
