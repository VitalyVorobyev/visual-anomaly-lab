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
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import APIRouter, HTTPException, Query, Request, Response
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
    Label,
    Subset,
)
from anomaly_lab.eval.metrics import pr_curve, roc_curve
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
from anomaly_lab.media.overlay import read_display_range, render_anomaly_map, render_rgb_image
from anomaly_lab.models.base import ModelDescription, evenly_spaced
from anomaly_lab.models.diagnostics import (
    DiagnosticEntry,
    DiagnosticIndex,
    DiagnosticKind,
    load_index,
)
from anomaly_lab.models.preprocessing import PreprocessingConfig
from anomaly_lab.models.registry import UnknownModelError, describe_all, get_model_class
from anomaly_lab.schemas import API_MODEL_CONFIG

router = APIRouter(prefix="/api/experiments", tags=["experiments"])

# A ROC curve has one point per distinct score, so a large test set produces more points
# than a chart has pixels. Capped, and the cap is reported rather than applied silently.
CURVE_POINT_LIMIT = 2000


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


class MapScale(BaseModel):
    """The numbers a rendered map is drawn against.

    Served as JSON because an `<img>` tag cannot read a response header, and the map
    endpoint exists to be an `img src`. Without these on screen, a map that is genuinely
    cold looks exactly like one that failed to render — which is what score-driven alpha
    does to every low-scoring image (ADR-0019).
    """

    model_config = API_MODEL_CONFIG

    low: float
    high: float


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
    map_range: MapScale | None = None
    """
    The run-wide display range every one of this run's maps is drawn against (ADR-0019).

    A segmentation threshold has to come from *this*, not from the image on screen: a cut
    derived per image is a different cut on every image, so two samples' predicted regions
    would not be comparable — the same mistake the run-wide range exists to prevent for the
    heatmap.
    """


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


class Curve(BaseModel):
    """One plotted curve, downsampled to a drawable number of points."""

    model_config = API_MODEL_CONFIG

    x: list[float] = Field(description="False-positive rate for ROC; recall for PR.")
    y: list[float] = Field(description="True-positive rate for ROC; precision for PR.")
    t: list[float] = Field(
        default_factory=list,
        description=(
            "The score at each point, so precision and recall can be drawn against the "
            "threshold rather than only against each other. Empty for ROC: that curve "
            "carries two synthetic endpoints whose thresholds are infinite, and JSON has "
            "no way to say so."
        ),
    )
    total: int = Field(description="Points before downsampling.")
    dropped: int = Field(default=0, description="Points not returned, so a cap is visible.")


class CurveSet(BaseModel):
    """The curves behind one subset's headline numbers.

    Every field is `None` when the subset cannot support that curve — one class present,
    nothing scored. A fabricated chance diagonal would be a picture of a claim nobody
    made (§8).
    """

    model_config = API_MODEL_CONFIG

    experiment_id: int
    subset: Subset | None = None
    sample_roc: Curve | None = None
    sample_pr: Curve | None = None
    image_roc: Curve | None = None
    image_pr: Curve | None = None


class ArtifactFile(BaseModel):
    model_config = API_MODEL_CONFIG

    name: str
    bytes: int


class ArtifactGroup(BaseModel):
    """One subdirectory of a run's output."""

    model_config = API_MODEL_CONFIG

    name: str
    title: str
    path: str
    file_count: int
    total_bytes: int
    files: list[ArtifactFile] = Field(
        default_factory=list,
        description="Empty when the group is large enough that only its count is useful.",
    )


class ArtifactListing(BaseModel):
    """Where a run's output is, and what it weighs."""

    model_config = API_MODEL_CONFIG

    root: str
    exists: bool
    total_bytes: int
    groups: list[ArtifactGroup] = Field(default_factory=list)


class SamplePreview(BaseModel):
    """One image standing for one sample, so a gallery tile needs no request of its own."""

    model_config = API_MODEL_CONFIG

    sample_id: int
    image_id: int
    has_map: bool = False
    has_mask: bool = False
    width: int = 0
    height: int = 0


class ImageScore(BaseModel):
    """One image of one sample, as the result viewer draws it."""

    model_config = API_MODEL_CONFIG

    image_id: int
    channel: str | None = None
    score: float
    inference_ms: float
    has_map: bool = False
    has_mask: bool = False
    width: int = 0
    height: int = 0
    """
    The source image's pixel dimensions, so a viewer can shape its frame before the
    picture arrives. Without them the canvas is laid out square and reflows on load, or —
    worse — is drawn full-width with the image letterboxed inside it, which spends the
    window on black bars on exactly the screen that exists to show a photograph.
    """
    map_scale: MapScale | None = None
    """This image's own extremes. `None` when the map file could not be read."""


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
        map_range=_run_map_range(experiment.artifact_dir),
    )


def _run_map_range(artifact_dir: str) -> MapScale | None:
    """The run-wide range from `maps/range.json`, or `None` before anything is scored."""
    found = read_display_range(Path(artifact_dir) / "maps")
    return None if found is None else MapScale(low=found[0], high=found[1])


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


@router.get("/{experiment_id}/artifacts", summary="What this run left on disk")
def get_artifacts(request: Request, experiment_id: int) -> ArtifactListing:
    """Everything under the experiment's directory, grouped and sized.

    A *listing*, not a download and not a mount. ADR-0019 ruled out serving the artifact
    directory statically, and one of its stated reasons was that doing so exposes the
    checkpoints; nothing here changes that. What it fixes is the other half of the
    problem, which is that a run could spend eleven minutes producing a 31 MB checkpoint
    and then not say where it was — the path was in `ExperimentDetail` all along and no
    screen showed it.

    Opening the directory is the desktop shell's job (ADR-0014), and a browser gets the
    path as text, which is a different affordance rather than a broken one.
    """
    experiment, _ = _load(request, experiment_id)
    root = Path(experiment.artifact_dir)

    groups = [
        _artifact_group(root, "model", "Trained weights"),
        _artifact_group(root, "maps", "Anomaly maps", summarize_from=32),
        _artifact_group(root, "diagnostics", "Diagnostics", summarize_from=32),
        _artifact_group(root, "logs", "Job logs"),
    ]
    return ArtifactListing(
        root=str(root),
        exists=root.is_dir(),
        total_bytes=sum(group.total_bytes for group in groups),
        groups=[group for group in groups if group.file_count > 0],
    )


def _artifact_group(root: Path, name: str, title: str, *, summarize_from: int = 0) -> ArtifactGroup:
    """One subdirectory, with its files listed or merely counted.

    A run writes one file per scored image into `maps/`, so listing every one of them
    would be five hundred rows answering a question nobody asked. Past `summarize_from`
    the group reports its count and total size and stops naming names.
    """
    directory = root / name
    files = sorted(path for path in directory.rglob("*") if path.is_file())
    sizes = [path.stat().st_size for path in files]
    listed: list[ArtifactFile] = []
    if summarize_from == 0 or len(files) <= summarize_from:
        listed = [
            ArtifactFile(name=str(path.relative_to(directory)), bytes=size)
            for path, size in zip(files, sizes, strict=True)
        ]
    return ArtifactGroup(
        name=name,
        title=title,
        path=str(directory),
        file_count=len(files),
        total_bytes=sum(sizes),
        files=listed,
    )


@router.get("/{experiment_id}/previews", summary="One representative image per scored sample")
def get_sample_previews(
    request: Request,
    experiment_id: int,
    subset: Subset | None = Query(default=None),
) -> list[SamplePreview]:
    """What a gallery needs to draw a tile per sample, without one request per tile.

    Deliberately not part of the threshold report. That response is recomputed on every
    slider tick, and none of this changes when the threshold moves — folding it in would
    resend a few hundred unchanging rows per tick. Here it is one request per subset,
    cached by the client for as long as the run's results stand.

    One image per sample, the first by channel order. A grouped sample is several
    photographs of one part and a tile is one thumbnail; which channel it shows is a
    presentation choice, and the sample page is where all of them are.
    """
    experiment, settings = _load(request, experiment_id)
    with connection(settings.db_path) as conn:
        scored = results_repo.list_scored_images(conn, experiment.id, subset=subset)
        masks = results_repo.masks_for_images(conn, [image.image_id for image in scored])

    seen: dict[int, SamplePreview] = {}
    for image in scored:
        if image.sample_id in seen:
            continue
        seen[image.sample_id] = SamplePreview(
            sample_id=image.sample_id,
            image_id=image.image_id,
            has_map=image.map_path is not None,
            has_mask=image.image_id in masks,
            width=image.width,
            height=image.height,
        )
    return list(seen.values())


@router.get("/{experiment_id}/curves", summary="ROC and PR curves for one subset")
def get_curves(
    request: Request,
    experiment_id: int,
    subset: Subset | None = Query(default=None),
) -> CurveSet:
    """The arrays behind the headline numbers, for the benchmark charts.

    Recomputed from the stored scores on every request — the same read the threshold
    endpoint does, over a few hundred floats — rather than persisted. Nothing here is
    threshold-dependent and nothing is written (ADR-0011).

    Pixel-level curves are deliberately absent. The pixel accumulator streams its
    histograms and discards them by design (ADR-0017), so drawing that curve would mean
    re-reading every anomaly map — the expensive pass this layer exists to avoid.
    """
    experiment, settings = _load(request, experiment_id)
    with connection(settings.db_path) as conn:
        samples = results_repo.list_scored_samples(conn, experiment.id, subset=subset)
        images = results_repo.list_scored_images(conn, experiment.id, subset=subset)

    sample_labels, sample_scores = _labelled(
        [(row.label, row.agg_score) for row in samples],
    )
    image_labels, image_scores = _labelled([(row.label, row.score) for row in images])

    return CurveSet(
        experiment_id=experiment.id,
        subset=subset,
        sample_roc=_curve(roc_curve(sample_labels, sample_scores)),
        sample_pr=_curve(pr_curve(sample_labels, sample_scores)),
        image_roc=_curve(roc_curve(image_labels, image_scores)),
        image_pr=_curve(pr_curve(image_labels, image_scores)),
    )


def _labelled(rows: list[tuple[Label, float]]) -> tuple[np.ndarray, np.ndarray]:
    """Labels and scores as arrays, with unlabeled rows left out.

    An unlabeled sample has no ground truth, so it cannot be a point on a ROC curve. It
    is dropped here rather than counted as normal, which is what the evaluation layer
    does with the same rows.
    """
    labelled = [(label, score) for label, score in rows if label is not Label.UNLABELED]
    return (
        np.array([label is Label.DEFECT for label, _ in labelled], dtype=bool),
        np.array([score for _, score in labelled], dtype=np.float64),
    )


def _curve(
    arrays: tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray] | None,
) -> Curve | None:
    """Downsample one curve to a drawable number of points, saying what was dropped.

    A third array, where the curve has one, is the score at each point and rides the
    **same** `kept` indices — a `t` sampled independently would label the wrong points.
    """
    if arrays is None:
        return None
    x, y = arrays[0], arrays[1]
    threshold = arrays[2] if len(arrays) == 3 else None
    kept = evenly_spaced(x.size, CURVE_POINT_LIMIT)
    return Curve(
        x=[float(x[index]) for index in kept],
        y=[float(y[index]) for index in kept],
        t=[] if threshold is None else [float(threshold[index]) for index in kept],
        total=int(x.size),
        dropped=int(x.size) - len(kept),
    )


def _map_scale(map_path: str | None) -> MapScale | None:
    """This map's own extremes, or `None` if it cannot be read.

    One `.npy` read per image of the sample being viewed — a few hundred kilobytes for the
    one part on screen, not a scan of the run.
    """
    if not map_path:
        return None
    try:
        array = np.load(map_path, allow_pickle=False)
    except (OSError, ValueError):
        # Deletable by design; the caller renders the absence rather than failing.
        return None
    return MapScale(low=float(np.min(array)), high=float(np.max(array)))


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
            width=image.width,
            height=image.height,
            map_scale=_map_scale(image.map_path),
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


@router.get(
    "/{experiment_id}/diagnostics/payload",
    summary="One diagnostic array, rendered",
    response_class=Response,
    responses={
        200: {"content": {"image/png": {}}},
        304: {"description": "The client's copy is current."},
    },
)
def read_diagnostic_payload(
    request: Request,
    experiment_id: int,
    key: str = Query(description="The diagnostic's key, as the index reports it."),
    image_id: int | None = Query(
        default=None,
        description="For a per-image diagnostic. Omit for a run-scoped one.",
    ),
    frame: int = Query(default=0, ge=0, description="Which cell of a `grid` payload."),
) -> Response:
    """Render one stored diagnostic array as a PNG.

    **Addressed through the index, never through `entry.path`.** The client names a
    `(key, image_id)` pair and this resolves it against the index the model wrote; there
    is no request that can name a file. That is the same rule the image routes follow
    (§11), and it means path traversal is impossible by construction rather than by
    sanitising a query parameter after the fact.

    Colormapped kinds are stretched over the **run-wide** range the writer recorded, so
    every image's student-teacher error is drawn on one scale and two images can be
    compared by eye. An index written before ranges were recorded has none, and each
    array then falls back to its own extremes — visibly worse, and better than refusing.
    """
    experiment, settings = _load(request, experiment_id)
    root = settings.experiment_dir(experiment.id) / "diagnostics"
    index = load_index(root)

    entry = next(
        (item for item in index.entries if item.key == key and item.image_id == image_id),
        None,
    )
    if entry is None:
        scope = "run-scoped" if image_id is None else f"image {image_id}"
        raise HTTPException(
            status_code=404,
            detail=f"experiment {experiment_id} recorded no {scope} diagnostic {key!r}",
        )
    if entry.path is None:
        raise HTTPException(
            status_code=400,
            detail=(
                f"diagnostic {key!r} is of kind {entry.kind.value}, whose payload is "
                "already inline in the index; fetching it here would be a second source "
                "of truth for the same data"
            ),
        )

    target = root / entry.path
    etag = _payload_etag(experiment_id, entry, frame, target)
    if etag is not None and request.headers.get("if-none-match") == etag:
        return Response(status_code=304, headers=_diagnostic_headers(etag))

    try:
        array = np.load(target, allow_pickle=False)
    except (OSError, ValueError) as exc:
        # The artifact directory is deletable by design, so a referenced file that is gone
        # is an expected state rather than corruption — the same 410 a missing source file
        # or a missing anomaly map gets.
        raise HTTPException(
            status_code=410,
            detail=f"the payload for diagnostic {key!r} is no longer readable",
        ) from exc

    recorded = index.ranges.get(key)
    value_range = None if recorded is None else (recorded.low, recorded.high)

    if entry.kind is DiagnosticKind.IMAGE:
        content = render_rgb_image(array)
    elif entry.kind is DiagnosticKind.GRID:
        if frame >= array.shape[0]:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"diagnostic {key!r} has {array.shape[0]} frame(s); there is no frame {frame}"
                ),
            )
        content = render_anomaly_map(
            array[frame], value_range=value_range, alpha_follows_score=False
        )
    else:
        content = render_anomaly_map(array, value_range=value_range, alpha_follows_score=False)

    headers = {} if etag is None else _diagnostic_headers(etag)
    return Response(content=content, media_type="image/png", headers=headers)


def _payload_etag(
    experiment_id: int, entry: DiagnosticEntry, frame: int, target: Path
) -> str | None:
    """A validator over what actually decides the bytes.

    Diagnostics are **not** immutable the way an imported image is: re-running inference
    overwrites an image's error maps in place, so the file's size and modification time
    are part of the identity. Without them a browser would keep showing the previous run's
    picture under the current run's caption.
    """
    try:
        stat = target.stat()
    except OSError:
        return None
    return (
        f'W/"diag-{experiment_id}-{entry.key}-{entry.image_id}-{frame}'
        f'-{stat.st_size}-{stat.st_mtime_ns}"'
    )


def _diagnostic_headers(etag: str) -> dict[str, str]:
    # `no-cache` means "revalidate", not "do not store": the client keeps the bytes and
    # asks whether they are still current, which the ETag answers with a 304.
    return {"ETag": etag, "Cache-Control": "no-cache"}
