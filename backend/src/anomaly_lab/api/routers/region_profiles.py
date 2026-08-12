"""Region extractor catalogue and immutable dataset profile revisions."""

from __future__ import annotations

import sqlite3
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, ValidationError, field_validator

from anomaly_lab.api.routers.jobs import JobSummary, summary_of
from anomaly_lab.config import Settings
from anomaly_lab.db.connection import connection
from anomaly_lab.db.repositories import datasets as datasets_repo
from anomaly_lab.db.repositories import region_profiles as profiles_repo
from anomaly_lab.domain.entities import JobKind, RegionProfileRevision, SpatialResample
from anomaly_lab.jobs.queue import JobQueue
from anomaly_lab.regions.base import RegionExtractorDescription
from anomaly_lab.regions.preparation import RegionBuildSummary, read_build_summary
from anomaly_lab.regions.registry import (
    UnknownRegionExtractorError,
    describe_all,
    validate_config,
)
from anomaly_lab.schemas import API_MODEL_CONFIG

router = APIRouter(tags=["region-profiles"])


class RegionProfileCreate(BaseModel):
    model_config = API_MODEL_CONFIG

    name: str = Field(min_length=1, max_length=120)
    extractor_type: str = Field(min_length=1, max_length=80)
    extractor_config: dict[str, Any] = Field(default_factory=dict)
    prepared_width: int = Field(default=256, ge=8, le=2048)
    prepared_height: int = Field(default=256, ge=8, le=2048)
    padding_fraction: float = Field(default=0.05, ge=0.0, le=1.0)
    resample: SpatialResample = SpatialResample.BILINEAR
    seed: int = 17

    @field_validator("name")
    @classmethod
    def _meaningful_name(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("profile name cannot be blank")
        return stripped


@router.get("/api/region-extractors", summary="Every registered region extractor and its schema")
def list_region_extractors() -> list[RegionExtractorDescription]:
    return describe_all()


@router.get(
    "/api/datasets/{dataset_id}/region-profiles",
    summary="Immutable region profile revisions owned by one dataset",
)
def list_region_profiles(request: Request, dataset_id: int) -> list[RegionProfileRevision]:
    settings: Settings = request.app.state.settings
    with connection(settings.db_path) as conn:
        if datasets_repo.get_dataset(conn, dataset_id) is None:
            raise HTTPException(status_code=404, detail=f"no dataset with id {dataset_id}")
        return profiles_repo.list_profiles(conn, dataset_id)


@router.post(
    "/api/datasets/{dataset_id}/region-profiles",
    summary="Append an immutable region profile revision",
)
def create_region_profile(
    request: Request, dataset_id: int, body: RegionProfileCreate
) -> RegionProfileRevision:
    try:
        validated = validate_config(body.extractor_type, body.extractor_config)
    except UnknownRegionExtractorError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.errors(include_url=False)) from exc

    settings: Settings = request.app.state.settings
    with connection(settings.db_path) as conn:
        conn.execute("BEGIN IMMEDIATE")
        try:
            if datasets_repo.get_dataset(conn, dataset_id) is None:
                raise HTTPException(status_code=404, detail=f"no dataset with id {dataset_id}")
            created = profiles_repo.create_revision(
                conn,
                dataset_id=dataset_id,
                name=body.name,
                extractor_type=body.extractor_type,
                extractor_config=validated.model_dump(mode="json"),
                prepared_width=body.prepared_width,
                prepared_height=body.prepared_height,
                padding_fraction=body.padding_fraction,
                resample=body.resample,
                seed=body.seed,
            )
            conn.execute("COMMIT")
        except sqlite3.IntegrityError as exc:
            conn.execute("ROLLBACK")
            raise HTTPException(
                status_code=409, detail="region profile revision conflicts"
            ) from exc
        except BaseException:
            if conn.in_transaction:
                conn.execute("ROLLBACK")
            raise
    return created


@router.get("/api/region-profiles/{profile_id}", summary="One immutable region profile revision")
def get_region_profile(request: Request, profile_id: int) -> RegionProfileRevision:
    settings: Settings = request.app.state.settings
    with connection(settings.db_path) as conn:
        profile = profiles_repo.get_profile(conn, profile_id)
    if profile is None:
        raise HTTPException(status_code=404, detail=f"no region profile with id {profile_id}")
    return profile


def _require_profile(request: Request, profile_id: int) -> RegionProfileRevision:
    settings: Settings = request.app.state.settings
    with connection(settings.db_path) as conn:
        profile = profiles_repo.get_profile(conn, profile_id)
    if profile is None:
        raise HTTPException(status_code=404, detail=f"no region profile with id {profile_id}")
    return profile


def _enqueue_preparation(request: Request, profile_id: int, mode: str) -> JobSummary:
    profile = _require_profile(request, profile_id)
    settings: Settings = request.app.state.settings
    with connection(settings.db_path) as conn:
        active = conn.execute(
            """
            SELECT id FROM job
             WHERE kind = 'region_prepare'
               AND status IN ('queued', 'running')
               AND CAST(json_extract(params, '$.profile_id') AS INTEGER) = ?
             LIMIT 1
            """,
            (profile_id,),
        ).fetchone()
    if active is not None:
        raise HTTPException(
            status_code=409,
            detail=f"region profile {profile_id} already has active job {int(active['id'])}",
        )
    queue: JobQueue = request.app.state.job_queue
    return summary_of(
        queue.enqueue(
            kind=JobKind.REGION_PREPARE,
            params={"dataset_id": profile.dataset_id, "profile_id": profile.id, "mode": mode},
        )
    )


@router.post("/api/region-profiles/{profile_id}/preview", summary="Preview a profile on 24 images")
def preview_region_profile(request: Request, profile_id: int) -> JobSummary:
    return _enqueue_preparation(request, profile_id, "preview")


@router.post("/api/region-profiles/{profile_id}/build", summary="Prepare a profile for every image")
def build_region_profile(request: Request, profile_id: int) -> JobSummary:
    return _enqueue_preparation(request, profile_id, "build")


@router.get(
    "/api/region-profiles/{profile_id}/build",
    summary="The latest complete profile build report",
)
def get_region_profile_build(request: Request, profile_id: int) -> RegionBuildSummary:
    _require_profile(request, profile_id)
    settings: Settings = request.app.state.settings
    summary = read_build_summary(settings, profile_id)
    if summary is None:
        raise HTTPException(status_code=404, detail=f"region profile {profile_id} is not built")
    return summary


@router.get(
    "/api/region-profiles/{profile_id}/prepared/{image_id}",
    summary="One lossless prepared image from a completed profile build",
    response_class=FileResponse,
)
def get_prepared_image(request: Request, profile_id: int, image_id: int) -> FileResponse:
    profile = _require_profile(request, profile_id)
    settings: Settings = request.app.state.settings
    with connection(settings.db_path) as conn:
        row = conn.execute(
            """
            SELECT 1 FROM image JOIN sample ON sample.id = image.sample_id
             WHERE image.id = ? AND sample.dataset_id = ?
            """,
            (image_id, profile.dataset_id),
        ).fetchone()
    if row is None:
        raise HTTPException(
            status_code=404, detail="image does not belong to this profile's dataset"
        )
    path = settings.region_profile_dir(profile_id) / "images" / f"{image_id}.png"
    if settings.region_profiles_dir.is_symlink() or path.is_symlink() or not path.is_file():
        raise HTTPException(
            status_code=404, detail=f"image {image_id} was not prepared successfully"
        )
    return FileResponse(path, media_type="image/png")
