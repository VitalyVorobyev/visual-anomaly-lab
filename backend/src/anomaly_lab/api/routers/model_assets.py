"""Catalog, acquisition and local-source overrides for executable model assets."""

from __future__ import annotations

import asyncio
from enum import StrEnum
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from anomaly_lab.api.routers.jobs import JobSummary, summary_of
from anomaly_lab.config import Settings
from anomaly_lab.db.connection import connection
from anomaly_lab.db.repositories import jobs as jobs_repo
from anomaly_lab.domain.entities import JobKind, JobStatus
from anomaly_lab.jobs.queue import JobQueue
from anomaly_lab.jobs.resident import ResidentWorker
from anomaly_lab.model_assets.catalog import SPECS, ModelAssetSpec, get_spec
from anomaly_lab.model_assets.store import (
    clear_external_source,
    managed_path,
    resolve_asset,
    set_external_source,
)
from anomaly_lab.schemas import API_MODEL_CONFIG

router = APIRouter(prefix="/api/model-assets", tags=["model-assets"])


class ModelAssetStatus(StrEnum):
    MISSING = "missing"
    READY = "ready"
    INVALID = "invalid"


class ModelAssetSource(StrEnum):
    MANAGED = "managed"
    EXTERNAL = "external"


class ModelAssetInfo(BaseModel):
    model_config = API_MODEL_CONFIG

    key: str
    title: str
    purpose: str
    status: ModelAssetStatus
    source: ModelAssetSource
    path: str
    size: int | None
    expected_size: int
    sha256: str
    reason: str | None = None
    license_name: str
    license_url: str
    project_url: str
    active_job: JobSummary | None = None


class ModelAssetCatalog(BaseModel):
    model_config = API_MODEL_CONFIG

    assets: list[ModelAssetInfo] = Field(default_factory=list)


class InstallModelAssetRequest(BaseModel):
    model_config = API_MODEL_CONFIG

    license_accepted: bool = False


class ExternalModelAssetSource(BaseModel):
    model_config = API_MODEL_CONFIG

    path: str = Field(min_length=1)


def _require_spec(key: str) -> ModelAssetSpec:
    spec = get_spec(key)
    if spec is None:
        raise HTTPException(status_code=404, detail="model asset not found")
    return spec


def _active_downloads(settings: Settings) -> dict[str, JobSummary]:
    with connection(settings.db_path) as conn:
        jobs = [
            *jobs_repo.list_jobs(
                conn, kind=JobKind.MODEL_ASSET_DOWNLOAD, status=JobStatus.QUEUED
            ),
            *jobs_repo.list_jobs(
                conn, kind=JobKind.MODEL_ASSET_DOWNLOAD, status=JobStatus.RUNNING
            ),
        ]
    return {
        str(job.params.get("asset_key")): summary_of(job)
        for job in jobs
        if job.params.get("asset_key")
    }


def _info(settings: Settings, spec: ModelAssetSpec, active: JobSummary | None) -> ModelAssetInfo:
    resolved = resolve_asset(settings, spec)
    status = (
        ModelAssetStatus.READY
        if resolved.ready
        else ModelAssetStatus.MISSING
        if resolved.reason == "missing"
        else ModelAssetStatus.INVALID
    )
    return ModelAssetInfo(
        key=spec.key,
        title=spec.title,
        purpose=spec.purpose,
        status=status,
        source=ModelAssetSource(resolved.source),
        path=str(resolved.path),
        size=resolved.size,
        expected_size=spec.expected_size,
        sha256=spec.sha256,
        reason=resolved.reason,
        license_name=spec.license_name,
        license_url=spec.license_url,
        project_url=spec.project_url,
        active_job=active,
    )


@router.get("", summary="List licensed model assets and their verified local state")
def list_model_assets(request: Request) -> ModelAssetCatalog:
    settings: Settings = request.app.state.settings
    active = _active_downloads(settings)
    return ModelAssetCatalog(
        assets=[_info(settings, spec, active.get(spec.key)) for spec in SPECS]
    )


@router.post("/{asset_key}/install", summary="Download and verify a catalogued model asset")
def install_model_asset(
    request: Request, asset_key: str, body: InstallModelAssetRequest
) -> JobSummary:
    spec = _require_spec(asset_key)
    settings: Settings = request.app.state.settings
    if not body.license_accepted:
        raise HTTPException(status_code=422, detail="the asset licence must be accepted")
    resolved = resolve_asset(settings, spec)
    if resolved.source == "external":
        raise HTTPException(status_code=409, detail="clear the external source before installing")
    if resolved.ready:
        raise HTTPException(status_code=409, detail="model asset is already installed")
    if spec.key in _active_downloads(settings):
        raise HTTPException(status_code=409, detail="model asset download is already running")
    queue: JobQueue = request.app.state.job_queue
    return summary_of(
        queue.enqueue(kind=JobKind.MODEL_ASSET_DOWNLOAD, params={"asset_key": spec.key})
    )


@router.put("/{asset_key}/source", summary="Use a verified model asset from an external path")
def set_model_asset_source(
    request: Request, asset_key: str, body: ExternalModelAssetSource
) -> ModelAssetInfo:
    spec = _require_spec(asset_key)
    settings: Settings = request.app.state.settings
    if spec.key in _active_downloads(settings):
        raise HTTPException(status_code=409, detail="model asset download is running")
    try:
        set_external_source(settings, spec, Path(body.path))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return _info(settings, spec, None)


@router.delete("/{asset_key}/source", summary="Stop using an external model asset path")
def clear_model_asset_source(request: Request, asset_key: str) -> ModelAssetInfo:
    spec = _require_spec(asset_key)
    settings: Settings = request.app.state.settings
    clear_external_source(settings, spec)
    return _info(settings, spec, _active_downloads(settings).get(spec.key))


@router.delete("/{asset_key}", summary="Remove an app-managed model asset")
async def remove_model_asset(request: Request, asset_key: str) -> ModelAssetInfo:
    spec = _require_spec(asset_key)
    settings: Settings = request.app.state.settings
    if spec.key in _active_downloads(settings):
        raise HTTPException(status_code=409, detail="cancel the model asset download first")
    resolved = resolve_asset(settings, spec)
    if resolved.source == "external":
        raise HTTPException(
            status_code=409, detail="clear the external source; external files are never deleted"
        )
    resident: ResidentWorker = request.app.state.resident
    async with resident.eviction_guard():
        await asyncio.to_thread(managed_path(settings, spec).unlink, missing_ok=True)
    return _info(settings, spec, None)
