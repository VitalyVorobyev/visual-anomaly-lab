"""Local public benchmark discovery and one-action registration."""

from __future__ import annotations

from enum import StrEnum

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from anomaly_lab.api.routers.jobs import JobSummary, summary_of
from anomaly_lab.config import Settings
from anomaly_lab.datasets.reference_packs import (
    RegisterReferencePacksParams,
    is_present,
    pack_specs,
    registered_dataset_id,
)
from anomaly_lab.db.connection import connection
from anomaly_lab.db.repositories import datasets as datasets_repo
from anomaly_lab.db.repositories import jobs as jobs_repo
from anomaly_lab.domain.entities import JobKind, JobStatus
from anomaly_lab.jobs.queue import JobQueue
from anomaly_lab.schemas import API_MODEL_CONFIG

router = APIRouter(prefix="/api/reference-packs", tags=["reference-packs"])


class ReferencePackStatus(StrEnum):
    ABSENT = "absent"
    INCOMPLETE = "incomplete"
    AVAILABLE = "available"
    REGISTERED = "registered"


class ReferenceDatasetInfo(BaseModel):
    model_config = API_MODEL_CONFIG

    key: str
    name: str
    registered_dataset_id: int | None = None


class ReferencePackInfo(BaseModel):
    model_config = API_MODEL_CONFIG

    key: str
    title: str
    root_path: str
    status: ReferencePackStatus
    datasets: list[ReferenceDatasetInfo] = Field(default_factory=list)
    missing: list[str] = Field(default_factory=list)
    install_url: str


class ReferencePackCatalog(BaseModel):
    model_config = API_MODEL_CONFIG

    packs: list[ReferencePackInfo] = Field(default_factory=list)
    available_datasets: int
    pending_datasets: int


def _catalog(settings: Settings) -> ReferencePackCatalog:
    with connection(settings.db_path) as conn:
        registered = datasets_repo.list_datasets(conn)

    packs: list[ReferencePackInfo] = []
    available = pending = 0
    for pack in pack_specs(settings):
        missing = [str(path) for path in pack.required if not is_present(path)]
        datasets = [
            ReferenceDatasetInfo(
                key=spec.key,
                name=spec.name,
                registered_dataset_id=registered_dataset_id(spec, registered),
            )
            for spec in pack.datasets
        ]
        if not pack.root.is_dir():
            status = ReferencePackStatus.ABSENT
        elif missing:
            status = ReferencePackStatus.INCOMPLETE
        elif all(item.registered_dataset_id is not None for item in datasets):
            status = ReferencePackStatus.REGISTERED
        else:
            status = ReferencePackStatus.AVAILABLE
            available += len(datasets)
            pending += sum(item.registered_dataset_id is None for item in datasets)
        packs.append(
            ReferencePackInfo(
                key=pack.key,
                title=pack.title,
                root_path=str(pack.root),
                status=status,
                datasets=datasets,
                missing=missing,
                install_url=pack.install_url,
            )
        )
    return ReferencePackCatalog(packs=packs, available_datasets=available, pending_datasets=pending)


@router.get("", summary="Discover local public reference dataset packs")
def list_reference_packs(request: Request) -> ReferencePackCatalog:
    return _catalog(request.app.state.settings)


@router.post("/register", summary="Register every missing dataset from selected local packs")
def register_reference_packs(request: Request, body: RegisterReferencePacksParams) -> JobSummary:
    settings: Settings = request.app.state.settings
    catalog = _catalog(settings)
    selected = [pack for pack in catalog.packs if pack.key in set(body.pack_keys)]
    unknown = sorted(set(body.pack_keys) - {pack.key for pack in catalog.packs})
    if unknown:
        raise HTTPException(status_code=422, detail=f"unknown packs: {', '.join(unknown)}")
    unavailable = [pack.title for pack in selected if pack.status in {"absent", "incomplete"}]
    if unavailable:
        raise HTTPException(
            status_code=409,
            detail=f"reference packs are not ready: {', '.join(unavailable)}",
        )
    if not any(
        dataset.registered_dataset_id is None for pack in selected for dataset in pack.datasets
    ):
        raise HTTPException(
            status_code=409, detail="all selected reference datasets are registered"
        )

    with connection(settings.db_path) as conn:
        active = [
            *jobs_repo.list_jobs(conn, kind=JobKind.REFERENCE_IMPORT, status=JobStatus.QUEUED),
            *jobs_repo.list_jobs(conn, kind=JobKind.REFERENCE_IMPORT, status=JobStatus.RUNNING),
        ]
    if active:
        raise HTTPException(status_code=409, detail="reference registration is already running")

    queue: JobQueue = request.app.state.job_queue
    return summary_of(
        queue.enqueue(
            kind=JobKind.REFERENCE_IMPORT,
            params=body.model_dump(),
        )
    )
