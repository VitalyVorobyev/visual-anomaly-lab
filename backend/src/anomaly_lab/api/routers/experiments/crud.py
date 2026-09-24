"""The experiment catalogue: the method picker, create, search, read and delete."""

from __future__ import annotations

from fastapi import APIRouter, Query, Request

from anomaly_lab.api.routers.experiments.views import (
    CreateExperimentRequest,
    ExperimentDeletionPreview,
    ExperimentDeletionResult,
    ExperimentDetail,
    ExperimentSort,
    ExperimentSummary,
    MethodCatalog,
    detail,
    load,
    summary,
)
from anomaly_lab.api.routers.jobs import summary_of
from anomaly_lab.config import Settings
from anomaly_lab.db.connection import connection
from anomaly_lab.db.repositories import experiments as experiments_repo
from anomaly_lab.domain.entities import ExperimentStatus
from anomaly_lab.eval.runner import EvalConfig
from anomaly_lab.experiments import service
from anomaly_lab.jobs.queue import JobQueue
from anomaly_lab.jobs.resident import ResidentWorker
from anomaly_lab.models.preprocessing import PreprocessingOptions
from anomaly_lab.models.registry import describe_all

router = APIRouter(prefix="/api/experiments", tags=["experiments"])


@router.get("/model-types", summary="Every registered method, with its configuration schema")
def list_model_types() -> MethodCatalog:
    """The method picker's whole data source.

    Each entry carries the JSON Schema of the method's own config model, so the form is
    generated rather than written. A method whose optional dependencies are missing is
    listed with `availability.available = false` and the reason, rather than hidden —
    "why can't I pick EfficientAD" should be answerable from the screen.
    """
    return MethodCatalog(
        methods=describe_all(),
        preprocessing_schema=PreprocessingOptions.model_json_schema(),
        evaluation_schema=EvalConfig.model_json_schema(),
    )


@router.post("", summary="Create an experiment with its configuration frozen")
def create_experiment(request: Request, body: CreateExperimentRequest) -> ExperimentDetail:
    """Validate a configuration against its method's schema and record it.

    Validation happens here rather than at job time so a typo is a 422 on the create
    screen instead of a failed job discovered ten minutes later.
    """
    settings: Settings = request.app.state.settings
    experiment = service.create_experiment(
        settings,
        name=body.name,
        dataset_id=body.dataset_id,
        split_id=body.split_id,
        region_profile_id=body.region_profile_id,
        model_type=body.model_type,
        config=body.config,
        preprocessing=body.preprocessing,
        evaluation=body.evaluation,
        channels=body.channels,
        notes=body.notes,
    )
    with connection(settings.db_path) as conn:
        return detail(conn, experiment)


@router.get("", summary="Search and filter experiments")
def list_experiments(
    request: Request,
    dataset_id: int | None = Query(default=None),
    model_type: str | None = Query(default=None),
    status: ExperimentStatus | None = Query(default=None),
    q: str | None = Query(default=None, max_length=200),
    sort: ExperimentSort = Query(default=ExperimentSort.NEWEST),
    limit: int = Query(default=100, ge=1, le=500),
) -> list[ExperimentSummary]:
    settings: Settings = request.app.state.settings
    with connection(settings.db_path) as conn:
        found = experiments_repo.list_experiments(
            conn,
            dataset_id=dataset_id,
            model_type=model_type,
            status=status,
            query=q,
            sort=sort.value,
            limit=limit,
        )
        return [summary(conn, experiment) for experiment in found]


@router.get("/{experiment_id}", summary="One experiment, with its metrics and job history")
def get_experiment(request: Request, experiment_id: int) -> ExperimentDetail:
    experiment, settings = load(request, experiment_id)
    with connection(settings.db_path) as conn:
        return detail(conn, experiment)


@router.get(
    "/{experiment_id}/deletion-preview",
    summary="Preview the app-owned records and artifacts an experiment deletion removes",
)
def preview_experiment_deletion(request: Request, experiment_id: int) -> ExperimentDeletionPreview:
    experiment, settings = load(request, experiment_id)
    resident: ResidentWorker = request.app.state.resident
    preview = service.preview_deletion(settings, resident, experiment)
    return ExperimentDeletionPreview(
        experiment_id=experiment.id,
        name=experiment.name,
        generated_files=preview.usage.files,
        generated_bytes=preview.usage.bytes,
        active_jobs=[summary_of(job) for job in preview.active_jobs],
        resident_loaded=preview.resident_loaded,
        artifact_location_safe=preview.artifact_location_safe,
        can_delete=preview.blocker is None,
        blocker=preview.blocker,
    )


@router.delete("/{experiment_id}", summary="Delete an experiment and its artifacts")
async def delete_experiment(request: Request, experiment_id: int) -> ExperimentDeletionResult:
    """Remove the rows, then the directory — in that order, and never the other way.

    The filesystem cannot join a database transaction, so the deletion that can be rolled
    back goes first. A leftover directory is inert; a row pointing at deleted artifacts
    is a broken screen.
    """
    experiment, settings = load(request, experiment_id)
    queue: JobQueue = request.app.state.job_queue
    resident: ResidentWorker = request.app.state.resident
    outcome = await service.delete_experiment(settings, queue, resident, experiment)
    return ExperimentDeletionResult(
        deleted=outcome.deleted,
        artifacts_removed=outcome.artifacts_removed,
        freed_files=outcome.freed.files,
        freed_bytes=outcome.freed.bytes,
        artifact_error=outcome.artifact_error,
    )
