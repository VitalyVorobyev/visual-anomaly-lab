"""Import: scan, review, commit, verify (ADR-0006).

Two stages, deliberately not one. A scan proposes and writes nothing; a human reads the
proposal; a commit writes rows. A one-shot importer that guessed silently would produce a
corrupt catalog discovered only much later, when several experiments had already been run
against it.

The module is named `import_` because `import` is a keyword.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from anomaly_lab.api.routers.jobs import JobSummary, summary_of
from anomaly_lab.config import Settings
from anomaly_lab.datasets.adapters.base import (
    UnknownAdapterError,
    get_adapter,
    parse_options,
    registered_adapters,
)
from anomaly_lab.datasets.commit import CommitError, commit_manifest
from anomaly_lab.datasets.manifest import Manifest
from anomaly_lab.datasets.scan import DEFAULT_ADAPTER, ScanParams
from anomaly_lab.datasets.storage import (
    ManifestNotFoundError,
    UnsupportedManifestVersionError,
    load_manifest,
)
from anomaly_lab.db.connection import connection
from anomaly_lab.db.repositories import datasets as datasets_repo
from anomaly_lab.domain.entities import JobKind
from anomaly_lab.jobs.queue import JobQueue

router = APIRouter(prefix="/api/import", tags=["import"])


class AdapterInfo(BaseModel):
    """One registered adapter, with the schema its options form is generated from."""

    model_config = ConfigDict(frozen=True)

    name: str
    summary: str
    options_schema: dict[str, Any]


class ScanRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    root_path: str
    dataset_name: str
    adapter: str = DEFAULT_ADAPTER
    options: dict[str, Any] = Field(default_factory=dict)


class CommitRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    manifest: Manifest
    dataset_id: int | None = Field(
        default=None,
        description=(
            "Commit into this dataset explicitly. Omit to resolve by the manifest's root "
            "path, which is what makes re-importing a directory idempotent."
        ),
    )


class CommitResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    dataset_id: int
    dataset_created: bool
    manifest_id: str
    samples_created: int
    samples_updated: int
    images_created: int
    images_updated: int
    channels: int
    missing_paths: list[str] = Field(
        default_factory=list,
        description="Recorded images this manifest no longer mentions. Reported, not deleted.",
    )


class VerifyRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    dataset_id: int


@router.get("/adapters", summary="Import adapters and their option schemas")
def list_adapters() -> list[AdapterInfo]:
    """List the registered adapters. Their JSON Schemas drive the import form."""
    return [
        AdapterInfo(
            name=adapter.name(),
            summary=adapter.summary(),
            options_schema=adapter.options_model().model_json_schema(),
        )
        for adapter in registered_adapters()
    ]


@router.post("/scan", summary="Scan a directory into a reviewable manifest")
def start_scan(request: Request, body: ScanRequest) -> JobSummary:
    """Start a scan job. Nothing is written to the catalog by this or by the job.

    The directory and the adapter options are validated here so that an obvious mistake
    is a 400 with a message, rather than a job that starts and immediately fails.
    """
    root = Path(body.root_path).expanduser()
    if not root.is_dir():
        raise HTTPException(status_code=400, detail=f"{body.root_path} is not a directory")

    try:
        adapter = get_adapter(body.adapter)
        parse_options(adapter, body.options)
    except UnknownAdapterError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=f"invalid adapter options: {exc}") from exc

    queue: JobQueue = request.app.state.job_queue
    params = ScanParams(
        root_path=str(root),
        dataset_name=body.dataset_name,
        adapter=body.adapter,
        options=body.options,
    )
    return summary_of(queue.enqueue(kind=JobKind.IMPORT, params=params.model_dump()))


@router.get("/manifests/{manifest_id}", summary="Read a manifest back for review")
def read_manifest(request: Request, manifest_id: str) -> Manifest:
    """Fetch a scan's proposal. The scan job's result carries the id."""
    settings: Settings = request.app.state.settings
    try:
        return load_manifest(settings, manifest_id)
    except ManifestNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except UnsupportedManifestVersionError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/commit", summary="Commit an accepted manifest into the catalog")
def commit(request: Request, body: CommitRequest) -> CommitResponse:
    """Create or update the dataset this manifest describes.

    Idempotent: committing the same manifest twice leaves the same rows. The manifest is
    stored verbatim, so what was accepted stays answerable later.
    """
    settings: Settings = request.app.state.settings
    with connection(settings.db_path) as conn:
        try:
            result = commit_manifest(conn, settings, body.manifest, dataset_id=body.dataset_id)
        except CommitError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    return CommitResponse(
        dataset_id=result.dataset_id,
        dataset_created=result.dataset_created,
        manifest_id=result.manifest_id,
        samples_created=result.samples_created,
        samples_updated=result.samples_updated,
        images_created=result.images_created,
        images_updated=result.images_updated,
        channels=result.channels,
        missing_paths=result.missing_paths,
    )


@router.post("/verify", summary="Re-check a dataset's files against their hashes")
def start_verify(request: Request, body: VerifyRequest) -> JobSummary:
    """Start a verify job. It reports drift and never repairs it."""
    settings: Settings = request.app.state.settings
    with connection(settings.db_path) as conn:
        if datasets_repo.get_dataset(conn, body.dataset_id) is None:
            raise HTTPException(status_code=404, detail=f"no dataset with id {body.dataset_id}")

    queue: JobQueue = request.app.state.job_queue
    return summary_of(queue.enqueue(kind=JobKind.VERIFY, params={"dataset_id": body.dataset_id}))
