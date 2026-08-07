"""Experiments: create, train, score, and read the results.

The route surface deliberately mentions no method by name. It serves the registry's
descriptions — including each method's JSON Schema — and the frontend builds its form
from them, which is what makes "add a method without touching the rest of the app" true
in practice rather than aspirational (ADR-0007).

Threshold-dependent numbers are computed per request rather than stored, so the slider is
a filter over a few hundred floats and never a database write (ADR-0011).
"""

from __future__ import annotations

import sqlite3
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, Field

from anomaly_lab.api.routers.jobs import JobSummary, summary_of
from anomaly_lab.config import Settings
from anomaly_lab.db.connection import connection
from anomaly_lab.db.repositories import datasets as datasets_repo
from anomaly_lab.db.repositories import experiments as experiments_repo
from anomaly_lab.db.repositories import jobs as jobs_repo
from anomaly_lab.db.repositories import results as results_repo
from anomaly_lab.db.repositories import splits as splits_repo
from anomaly_lab.domain.entities import (
    Experiment,
    ExperimentStatus,
    JobKind,
    Subset,
)
from anomaly_lab.eval.runner import EvalConfig, evaluate_and_store
from anomaly_lab.eval.threshold import (
    SampleVerdict,
    ThresholdReport,
    classify,
    report,
    suggest_threshold,
)
from anomaly_lab.experiments.infer import InferParams
from anomaly_lab.experiments.train import TrainParams
from anomaly_lab.jobs.queue import JobQueue
from anomaly_lab.models.base import ModelDescription
from anomaly_lab.models.diagnostics import DiagnosticIndex, load_index
from anomaly_lab.models.preprocessing import PreprocessingConfig
from anomaly_lab.models.registry import UnknownModelError, describe_all, get_model_class
from anomaly_lab.schemas import API_MODEL_CONFIG

router = APIRouter(prefix="/api/experiments", tags=["experiments"])


class MethodCatalog(BaseModel):
    """Everything the create screen needs, in one round trip."""

    model_config = API_MODEL_CONFIG

    methods: list[ModelDescription]
    preprocessing_schema: dict[str, Any] = Field(
        description="JSON Schema for the preprocessing every method is made to share."
    )
    evaluation_schema: dict[str, Any] = Field(
        description="JSON Schema for how the stored scores are read back."
    )


class CreateExperimentRequest(BaseModel):
    model_config = API_MODEL_CONFIG

    name: str
    dataset_id: int
    split_id: int
    model_type: str
    config: dict[str, Any] = Field(default_factory=dict)
    preprocessing: dict[str, Any] = Field(default_factory=dict)
    evaluation: dict[str, Any] = Field(default_factory=dict)
    notes: str | None = None


class MetricSummary(BaseModel):
    model_config = API_MODEL_CONFIG

    subset: Subset
    metrics: dict[str, Any] = Field(default_factory=dict)
    computed_at: str


class ExperimentSummary(BaseModel):
    model_config = API_MODEL_CONFIG

    id: int
    name: str
    dataset_id: int
    split_id: int
    model_type: str
    status: ExperimentStatus
    created_at: str
    notes: str | None = None
    headline_roc_auc: float | None = Field(
        default=None,
        description="Sample-level ROC-AUC on test, or on the best subset scored so far.",
    )


class ExperimentDetail(ExperimentSummary):
    config: dict[str, Any] = Field(default_factory=dict)
    preprocessing: dict[str, Any] = Field(default_factory=dict)
    evaluation: dict[str, Any] = Field(default_factory=dict)
    artifact_dir: str
    dataset_name: str | None = None
    split_name: str | None = None
    metrics: list[MetricSummary] = Field(default_factory=list)
    scored_subsets: list[Subset] = Field(default_factory=list)
    jobs: list[JobSummary] = Field(default_factory=list)
    produces_anomaly_map: bool = True
    produces_diagnostics: bool = False


class ResultsPage(BaseModel):
    """Ranked samples for one subset, with a starting threshold and its rationale."""

    model_config = API_MODEL_CONFIG

    experiment_id: int
    subset: Subset | None = None
    suggested_threshold: float
    threshold_rationale: str
    score_min: float = 0.0
    score_max: float = 0.0
    samples: list[SampleVerdict] = Field(default_factory=list)


class ImageScore(BaseModel):
    """One image of one sample, as the result viewer draws it."""

    model_config = API_MODEL_CONFIG

    image_id: int
    channel: str | None = None
    score: float
    inference_ms: float
    has_map: bool = False
    has_mask: bool = False


def _headline(metric_sets: list[MetricSummary]) -> float | None:
    """Test if there is one, otherwise whatever was scored — never a blend of subsets."""
    by_subset = {found.subset: found.metrics for found in metric_sets}
    for subset in (Subset.TEST, Subset.VAL, Subset.TRAIN):
        metrics = by_subset.get(subset)
        if metrics and metrics.get("sample_roc_auc") is not None:
            return float(metrics["sample_roc_auc"])
    return None


def _metric_summaries(conn: sqlite3.Connection, experiment_id: int) -> list[MetricSummary]:
    return [
        MetricSummary(subset=found.subset, metrics=found.metrics, computed_at=found.computed_at)
        for found in results_repo.list_metric_sets(conn, experiment_id)
    ]


def _summary(conn: sqlite3.Connection, experiment: Experiment) -> ExperimentSummary:
    return ExperimentSummary(
        id=experiment.id,
        name=experiment.name,
        dataset_id=experiment.dataset_id,
        split_id=experiment.split_id,
        model_type=experiment.model_type,
        status=experiment.status,
        created_at=experiment.created_at,
        notes=experiment.notes,
        headline_roc_auc=_headline(_metric_summaries(conn, experiment.id)),
    )


def _detail(conn: sqlite3.Connection, experiment: Experiment) -> ExperimentDetail:
    dataset = datasets_repo.get_dataset(conn, experiment.dataset_id)
    split = splits_repo.get_split(conn, experiment.split_id)
    metrics = _metric_summaries(conn, experiment.id)

    try:
        capabilities = get_model_class(experiment.model_type).capabilities()
        produces_map = capabilities.produces_anomaly_map
        produces_diagnostics = capabilities.produces_diagnostics
    except UnknownModelError:
        # A method can be removed from the registry while its experiments remain. The
        # record stays readable; only the capability-driven affordances go away.
        produces_map = False
        produces_diagnostics = False

    return ExperimentDetail(
        **_summary(conn, experiment).model_dump(),
        config=experiment.model_config_,
        preprocessing=experiment.preprocessing_config,
        evaluation=experiment.eval_config,
        artifact_dir=experiment.artifact_dir,
        dataset_name=dataset.name if dataset else None,
        split_name=split.name if split else None,
        metrics=metrics,
        scored_subsets=results_repo.scored_subsets(conn, experiment.id),
        jobs=[summary_of(job) for job in jobs_repo.list_jobs_for_experiment(conn, experiment.id)],
        produces_anomaly_map=produces_map,
        produces_diagnostics=produces_diagnostics,
    )


def _load(request: Request, experiment_id: int) -> tuple[Experiment, Settings]:
    settings: Settings = request.app.state.settings
    with connection(settings.db_path) as conn:
        experiment = experiments_repo.get_experiment(conn, experiment_id)
    if experiment is None:
        raise HTTPException(status_code=404, detail=f"no experiment with id {experiment_id}")
    return experiment, settings


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
        preprocessing_schema=PreprocessingConfig.model_json_schema(),
        evaluation_schema=EvalConfig.model_json_schema(),
    )


@router.post("", summary="Create an experiment with its configuration frozen")
def create_experiment(request: Request, body: CreateExperimentRequest) -> ExperimentDetail:
    """Validate a configuration against its method's schema and record it.

    Validation happens here rather than at job time so a typo is a 422 on the create
    screen instead of a failed job discovered ten minutes later.
    """
    settings: Settings = request.app.state.settings

    try:
        model_class = get_model_class(body.model_type)
    except UnknownModelError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    try:
        config = model_class.config_model().model_validate(body.config).model_dump(mode="json")
        preprocessing = PreprocessingConfig.model_validate(body.preprocessing).model_dump(
            mode="json"
        )
        evaluation = EvalConfig.model_validate(body.evaluation).model_dump(mode="json")
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    with connection(settings.db_path) as conn:
        if datasets_repo.get_dataset(conn, body.dataset_id) is None:
            raise HTTPException(status_code=404, detail=f"no dataset with id {body.dataset_id}")
        split = splits_repo.get_split(conn, body.split_id)
        if split is None:
            raise HTTPException(status_code=404, detail=f"no split with id {body.split_id}")
        if split.dataset_id != body.dataset_id:
            raise HTTPException(
                status_code=422,
                detail=f"split {body.split_id} belongs to dataset {split.dataset_id}",
            )

        # The directory is named after the row, so it cannot be built until the row
        # exists; created first, then recorded, then made.
        experiment = experiments_repo.create_experiment(
            conn,
            name=body.name,
            dataset_id=body.dataset_id,
            split_id=body.split_id,
            model_type=body.model_type,
            model_config=config,
            preprocessing_config=preprocessing,
            eval_config=evaluation,
            artifact_dir="",
            notes=body.notes,
        )
        artifact_dir = settings.experiment_dir(experiment.id)
        conn.execute(
            "UPDATE experiment SET artifact_dir = ? WHERE id = ?",
            (str(artifact_dir), experiment.id),
        )
        artifact_dir.mkdir(parents=True, exist_ok=True)

        stored = experiments_repo.get_experiment(conn, experiment.id)
        if stored is None:  # pragma: no cover - inserted a moment ago
            raise HTTPException(status_code=500, detail="the experiment vanished after creation")
        return _detail(conn, stored)


@router.get("", summary="Experiments, newest first")
def list_experiments(
    request: Request,
    dataset_id: int | None = Query(default=None),
    limit: int = Query(default=100, ge=1, le=500),
) -> list[ExperimentSummary]:
    settings: Settings = request.app.state.settings
    with connection(settings.db_path) as conn:
        found = experiments_repo.list_experiments(conn, dataset_id=dataset_id, limit=limit)
        return [_summary(conn, experiment) for experiment in found]


@router.get("/{experiment_id}", summary="One experiment, with its metrics and job history")
def get_experiment(request: Request, experiment_id: int) -> ExperimentDetail:
    experiment, settings = _load(request, experiment_id)
    with connection(settings.db_path) as conn:
        return _detail(conn, experiment)


@router.delete("/{experiment_id}", summary="Delete an experiment and its artifacts")
def delete_experiment(request: Request, experiment_id: int) -> dict[str, bool]:
    """Remove the rows, then the directory — in that order, and never the other way.

    The filesystem cannot join a database transaction, so the deletion that can be rolled
    back goes first. A leftover directory is inert; a row pointing at deleted artifacts
    is a broken screen.
    """
    import shutil

    experiment, settings = _load(request, experiment_id)
    with connection(settings.db_path) as conn:
        deleted = experiments_repo.delete_experiment(conn, experiment_id)
    if deleted and experiment.artifact_dir:
        shutil.rmtree(experiment.artifact_dir, ignore_errors=True)
    return {"deleted": deleted}


@router.post("/{experiment_id}/train", summary="Queue a training job")
def start_train(
    request: Request, experiment_id: int, body: TrainParams | None = None
) -> JobSummary:
    experiment, _ = _load(request, experiment_id)
    params = body or TrainParams(experiment_id=experiment.id)
    params = params.model_copy(update={"experiment_id": experiment.id})
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
    experiment, _ = _load(request, experiment_id)
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


@router.post("/{experiment_id}/reevaluate", summary="Recompute metrics from stored scores")
def reevaluate(request: Request, experiment_id: int) -> list[MetricSummary]:
    """Re-read the results without re-running inference.

    Cheap because nothing about evaluation depends on a model (ADR-0011), and useful
    because it is how a changed aggregation mode is applied to a finished experiment.
    """
    experiment, settings = _load(request, experiment_id)
    with connection(settings.db_path) as conn:
        evaluate_and_store(conn, experiment)
        return _metric_summaries(conn, experiment.id)


@router.get("/{experiment_id}/results", summary="Ranked samples for one subset")
def get_results(
    request: Request,
    experiment_id: int,
    subset: Subset | None = Query(default=None),
) -> ResultsPage:
    experiment, settings = _load(request, experiment_id)
    with connection(settings.db_path) as conn:
        samples = results_repo.list_scored_samples(conn, experiment.id, subset=subset)

    threshold, rationale = suggest_threshold(samples)
    scores = [sample.agg_score for sample in samples]
    return ResultsPage(
        experiment_id=experiment.id,
        subset=subset,
        suggested_threshold=threshold,
        threshold_rationale=rationale,
        score_min=min(scores) if scores else 0.0,
        score_max=max(scores) if scores else 0.0,
        samples=classify(samples, threshold),
    )


@router.get("/{experiment_id}/threshold", summary="Confusion matrix at one threshold")
def get_threshold(
    request: Request,
    experiment_id: int,
    value: float = Query(description="Scores at or above this are predicted defective."),
    subset: Subset | None = Query(default=None),
) -> ThresholdReport:
    """Recomputed on every slider move, from persisted scores. Nothing is written.

    Returns the classified rows alongside the counts so the client never has to apply the
    threshold rule itself — see `ThresholdReport` for why that matters.
    """
    experiment, settings = _load(request, experiment_id)
    with connection(settings.db_path) as conn:
        samples = results_repo.list_scored_samples(conn, experiment.id, subset=subset)
    return report(samples, value, include_samples=True)


@router.get("/{experiment_id}/samples/{sample_id}/images", summary="Per-image scores of a sample")
def get_sample_images(request: Request, experiment_id: int, sample_id: int) -> list[ImageScore]:
    """What the result viewer needs to draw one part across its channels."""
    experiment, settings = _load(request, experiment_id)
    with connection(settings.db_path) as conn:
        scored = [
            image
            for image in results_repo.list_scored_images(conn, experiment.id)
            if image.sample_id == sample_id
        ]
        masks = results_repo.masks_for_images(conn, [image.image_id for image in scored])

    return [
        ImageScore(
            image_id=image.image_id,
            channel=image.channel,
            score=image.score,
            inference_ms=image.inference_ms,
            has_map=image.map_path is not None,
            has_mask=image.image_id in masks,
        )
        for image in scored
    ]


@router.get("/{experiment_id}/diagnostics", summary="What this run recorded about itself")
def get_diagnostics(request: Request, experiment_id: int) -> DiagnosticIndex:
    """The self-describing index a model wrote (ADR-0018).

    Returned verbatim. The UI renders by `kind` and never by method name, which is what
    makes a future method's diagnostics work here with no change.
    """
    experiment, settings = _load(request, experiment_id)
    return load_index(settings.experiment_dir(experiment.id) / "diagnostics")
