"""Work on an experiment: queue training, scoring and export, or re-read stored scores."""

from __future__ import annotations

from fastapi import APIRouter, Request

from anomaly_lab.api.routers.experiments.views import MetricSummary, load, metric_summaries
from anomaly_lab.api.routers.jobs import JobSummary, summary_of
from anomaly_lab.db.connection import connection
from anomaly_lab.deployment.export import ExportParams
from anomaly_lab.domain.entities import JobKind
from anomaly_lab.experiments import service
from anomaly_lab.experiments.infer import InferParams
from anomaly_lab.experiments.train import TrainParams
from anomaly_lab.jobs.queue import JobQueue

router = APIRouter(prefix="/api/experiments", tags=["experiments"])


@router.post("/{experiment_id}/train", summary="Queue a training job")
def start_train(
    request: Request, experiment_id: int, body: TrainParams | None = None
) -> JobSummary:
    experiment, _ = load(request, experiment_id)
    params = body or TrainParams(experiment_id=experiment.id)
    params = params.model_copy(update={"experiment_id": experiment.id})

    # Refused here rather than ten minutes later inside a worker. A form that cannot
    # succeed should fail as a form, which is the same reasoning `create_experiment`
    # applies to a config that does not validate against its method's schema.
    if params.additional_steps is not None:
        service.refuse_impossible_resume(experiment)

    queue: JobQueue = request.app.state.job_queue
    return summary_of(
        queue.enqueue(
            kind=JobKind.TRAIN,
            params=params.model_dump(mode="json"),
            experiment_id=experiment.id,
        )
    )


@router.post("/{experiment_id}/infer", summary="Queue an inference and evaluation job")
def start_infer(
    request: Request, experiment_id: int, body: InferParams | None = None
) -> JobSummary:
    experiment, _ = load(request, experiment_id)
    params = body or InferParams(experiment_id=experiment.id)
    params = params.model_copy(update={"experiment_id": experiment.id})
    queue: JobQueue = request.app.state.job_queue
    return summary_of(
        queue.enqueue(
            kind=JobKind.INFER,
            params=params.model_dump(mode="json"),
            experiment_id=experiment.id,
        )
    )


@router.post("/{experiment_id}/export", summary="Queue a verified portable-model export")
def start_export(
    request: Request,
    experiment_id: int,
    body: ExportParams | None = None,
) -> JobSummary:
    experiment, _ = load(request, experiment_id)
    params = body or ExportParams(experiment_id=experiment.id)
    params = params.model_copy(update={"experiment_id": experiment.id})
    service.refuse_impossible_export(experiment, params)

    queue: JobQueue = request.app.state.job_queue
    return summary_of(
        queue.enqueue(
            kind=JobKind.EXPORT,
            params=params.model_dump(mode="json"),
            experiment_id=experiment.id,
        )
    )


@router.post("/{experiment_id}/reevaluate", summary="Recompute metrics from stored scores")
def reevaluate(request: Request, experiment_id: int) -> list[MetricSummary]:
    """Re-read the results without re-running inference.

    Cheap because nothing about evaluation depends on a model (ADR-0011), and useful
    because it is how a changed aggregation mode is applied to a finished experiment.
    """
    experiment, settings = load(request, experiment_id)
    service.reevaluate(settings, experiment)
    with connection(settings.db_path) as conn:
        return metric_summaries(conn, experiment)
