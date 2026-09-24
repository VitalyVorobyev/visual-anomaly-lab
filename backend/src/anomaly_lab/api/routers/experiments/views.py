"""The experiment API's read models, and the reads that build them.

Every response and request shape the experiment routes use is defined here, so the four
route modules beside this one (`crud`, `runs`, `results`, `diagnostics`) share one
vocabulary and `compare` can import the same `MetricSummary` and `MapScale`. What the
routes *decide* is in `anomaly_lab.experiments.service`.
"""

from __future__ import annotations

import sqlite3
from enum import StrEnum
from pathlib import Path
from typing import Any

from fastapi import Request
from pydantic import BaseModel, Field

from anomaly_lab.api.routers.jobs import JobSummary, summary_of
from anomaly_lab.config import Settings
from anomaly_lab.db.repositories import datasets as datasets_repo
from anomaly_lab.db.repositories import jobs as jobs_repo
from anomaly_lab.db.repositories import region_profiles as region_profiles_repo
from anomaly_lab.db.repositories import results as results_repo
from anomaly_lab.db.repositories import splits as splits_repo
from anomaly_lab.domain.entities import (
    Experiment,
    ExperimentStatus,
    MetricSet,
    Subset,
    Task,
)
from anomaly_lab.eval.ground_truth import current_digest
from anomaly_lab.eval.threshold import SampleVerdict
from anomaly_lab.experiments import service
from anomaly_lab.experiments.train import MODEL_SUBDIR, TrainingState, read_training_state
from anomaly_lab.media.overlay import read_display_range
from anomaly_lab.models.base import ModelDescription, PortableFormat
from anomaly_lab.models.registry import UnknownModelError, get_model_class
from anomaly_lab.schemas import API_MODEL_CONFIG


class PayloadFormat(StrEnum):
    """Whether a caller wants the picture or the numbers behind it (handbook diagnostics.md)."""

    PNG = "png"
    RAW = "raw"


class ExperimentSort(StrEnum):
    """Stable orders offered by the experiment catalogue."""

    NEWEST = "newest"
    OLDEST = "oldest"
    NAME = "name"


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
    region_profile_id: int
    model_type: str
    task: Task = Field(
        default=Task.ANOMALY,
        description="What the run is asked to do. The method must list it in its capabilities.",
    )
    config: dict[str, Any] = Field(default_factory=dict)
    preprocessing: dict[str, Any] = Field(default_factory=dict)
    evaluation: dict[str, Any] = Field(default_factory=dict)
    channels: list[str] = Field(
        default_factory=list,
        description=(
            "Acquisition channels this run should read, by name. Empty means every "
            "channel the dataset has, which is the only meaningful answer for a "
            "single-view dataset."
        ),
    )
    notes: str | None = None


class MetricSummary(BaseModel):
    model_config = API_MODEL_CONFIG

    subset: Subset
    metrics: dict[str, Any] = Field(default_factory=dict)
    computed_at: str
    ground_truth_digest: str | None = None
    ground_truth_stale: bool = False


class ExperimentSummary(BaseModel):
    model_config = API_MODEL_CONFIG

    id: int
    name: str
    dataset_id: int
    split_id: int
    region_profile_id: int
    region_manifest_sha256: str
    model_type: str
    task: Task = Task.ANOMALY
    channels: list[str] = Field(
        default_factory=list,
        description=(
            "Acquisition channels this run read. Empty means every channel, so a catalogue "
            "row can say 'bright-field only' without a second request."
        ),
    )
    status: ExperimentStatus
    created_at: str
    notes: str | None = None
    headline_roc_auc: float | None = Field(
        default=None,
        description="Sample-level ROC-AUC on test, or on the best subset scored so far.",
    )


class ExperimentDeletionPreview(BaseModel):
    """What an experiment deletion will remove, and what currently blocks it."""

    model_config = API_MODEL_CONFIG

    experiment_id: int
    name: str
    generated_files: int
    generated_bytes: int
    active_jobs: list[JobSummary] = Field(default_factory=list)
    resident_loaded: bool = False
    artifact_location_safe: bool = True
    can_delete: bool
    blocker: str | None = None


class ExperimentDeletionResult(BaseModel):
    """The completed database deletion and best-effort artifact cleanup."""

    model_config = API_MODEL_CONFIG

    deleted: bool
    artifacts_removed: bool
    freed_files: int
    freed_bytes: int
    artifact_error: str | None = None


class MapScale(BaseModel):
    """The numbers a rendered map is drawn against.

    Served as JSON because an `<img>` tag cannot read a response header, and the map
    endpoint exists to be an `img src`. Without these on screen, a map that is genuinely
    cold looks exactly like one that failed to render — which is what score-driven alpha
    does to every low-scoring image (handbook diagnostics.md).
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
    region_profile_name: str | None = None
    metrics: list[MetricSummary] = Field(default_factory=list)
    scored_subsets: list[Subset] = Field(default_factory=list)
    jobs: list[JobSummary] = Field(default_factory=list)
    produces_anomaly_map: bool = True
    produces_diagnostics: bool = False
    supports_resume: bool = False
    """Whether this method can continue a finished run (handbook jobs.md)."""
    portable_formats: list[PortableFormat] = Field(default_factory=list)
    """Verified deployment formats this fitted method can produce (ADR-0034)."""
    training_state: TrainingState | None = None
    """
    How much training the stored checkpoint has actually had.

    `None` when nothing has trained, or when the method has no notion of a step. Rendered
    beside the frozen configuration, because `max_steps` there is the **per-run** budget
    and would otherwise read as the total on a model that has been continued.
    """
    map_range: MapScale | None = None
    """
    The run-wide display range every one of this run's maps is drawn against
    (handbook diagnostics.md).

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


class MapPeak(BaseModel):
    """Where one map's largest value sits, in source-frame pixels.

    The **map's** peak, not the score's location. A stored map has been blurred, upsampled
    from the patch grid and projected back into source coordinates, so its argmax is near
    the cell that produced the score without being the same cell — and a method scoring at
    a percentile below 100 is not reading a single cell at all. It marks the picture on
    screen, which is the only thing a reviewer can check by eye.
    """

    model_config = API_MODEL_CONFIG

    x: int
    y: int


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
    peak: MapPeak | None = None
    """`None` when no map was written, or when a legacy run has not been re-evaluated."""
    localized: bool | None = None
    """
    Whether this image's map peaked inside its annotated region, within `tolerance_px`.

    Threshold-free, so it does not move with the results slider. `null` is not applicable —
    a normal image, a defect with no resolved mask, or a map that could not be read — and
    never a miss.
    """
    tolerance_px: int | None = None
    """
    The radius the verdict was decided by, for this image's own frame.

    Present whenever the frame is known, including when `localized` is `null`, so a viewer
    can draw the tolerance it *would* be judged against. `EvalConfig.localization_tolerance`
    is a fraction of the diagonal, so this differs per image on a mixed-size dataset.
    """
    """
    The source image's pixel dimensions, so a viewer can shape its frame before the
    picture arrives. Without them the canvas is laid out square and reflows on load, or —
    worse — is drawn full-width with the image letterboxed inside it, which spends the
    window on black bars on exactly the screen that exists to show a photograph.
    """
    map_scale: MapScale | None = None
    """This image's own extremes. `None` when the map file could not be read."""


class DiagnoseRequest(BaseModel):
    model_config = API_MODEL_CONFIG

    image_id: int = Field(description="The image to diagnose. Must belong to this split.")


class DiagnoseResponse(BaseModel):
    model_config = API_MODEL_CONFIG

    keys: list[str] = Field(description="The diagnostic keys recorded for this image.")
    elapsed_ms: float
    warm: bool = Field(
        description=(
            "False when this request had to load the model first, which is what makes it "
            "take seconds. True once a resident is serving."
        )
    )


def headline(metric_sets: list[MetricSummary] | list[MetricSet]) -> float | None:
    """Test if there is one, otherwise whatever was scored — never a blend of subsets."""
    by_subset = {found.subset: found.metrics for found in metric_sets}
    for subset in (Subset.TEST, Subset.VAL, Subset.TRAIN):
        metrics = by_subset.get(subset)
        if metrics and metrics.get("sample_roc_auc") is not None:
            return float(metrics["sample_roc_auc"])
    return None


def metric_summaries(conn: sqlite3.Connection, experiment_id: int) -> list[MetricSummary]:
    summaries: list[MetricSummary] = []
    for found in results_repo.list_metric_sets(conn, experiment_id):
        current = current_digest(conn, experiment_id, found.subset)
        summaries.append(
            MetricSummary(
                subset=found.subset,
                metrics=found.metrics,
                computed_at=found.computed_at,
                ground_truth_digest=found.ground_truth_digest,
                ground_truth_stale=found.ground_truth_digest != current,
            )
        )
    return summaries


def summary(conn: sqlite3.Connection, experiment: Experiment) -> ExperimentSummary:
    return ExperimentSummary(
        id=experiment.id,
        name=experiment.name,
        dataset_id=experiment.dataset_id,
        split_id=experiment.split_id,
        region_profile_id=experiment.region_profile_id,
        region_manifest_sha256=experiment.region_manifest_sha256,
        model_type=experiment.model_type,
        task=experiment.task,
        channels=experiment.channels,
        status=experiment.status,
        created_at=experiment.created_at,
        notes=experiment.notes,
        headline_roc_auc=headline(results_repo.list_metric_sets(conn, experiment.id)),
    )


def detail(conn: sqlite3.Connection, experiment: Experiment) -> ExperimentDetail:
    dataset = datasets_repo.get_dataset(conn, experiment.dataset_id)
    split = splits_repo.get_split(conn, experiment.split_id)
    profile = region_profiles_repo.get_profile(conn, experiment.region_profile_id)
    metrics = metric_summaries(conn, experiment.id)

    try:
        capabilities = get_model_class(experiment.model_type).capabilities()
        produces_map = capabilities.produces_anomaly_map
        produces_diagnostics = capabilities.produces_diagnostics
        supports_resume = capabilities.supports_resume
        portable_formats = capabilities.portable_formats
    except UnknownModelError:
        # A method can be removed from the registry while its experiments remain. The
        # record stays readable; only the capability-driven affordances go away.
        produces_map = False
        produces_diagnostics = False
        supports_resume = False
        portable_formats = []

    return ExperimentDetail(
        **summary(conn, experiment).model_dump(),
        config=experiment.model_config_,
        preprocessing=experiment.preprocessing_config,
        evaluation=experiment.eval_config,
        artifact_dir=experiment.artifact_dir,
        dataset_name=dataset.name if dataset else None,
        split_name=split.name if split else None,
        region_profile_name=profile.name if profile else None,
        metrics=metrics,
        scored_subsets=results_repo.scored_subsets(conn, experiment.id),
        jobs=[summary_of(job) for job in jobs_repo.list_jobs_for_experiment(conn, experiment.id)],
        produces_anomaly_map=produces_map,
        produces_diagnostics=produces_diagnostics,
        supports_resume=supports_resume,
        portable_formats=portable_formats,
        # Read from a JSON sidecar, never from the checkpoint: this process has no torch,
        # by design, and a `.pt` is unreadable without it.
        training_state=read_training_state(Path(experiment.artifact_dir) / MODEL_SUBDIR),
        map_range=run_map_range(experiment.artifact_dir),
    )


def run_map_range(artifact_dir: str) -> MapScale | None:
    """The run-wide range from `maps/range.json`, or `None` before anything is scored."""
    found = read_display_range(Path(artifact_dir) / "maps")
    return None if found is None else MapScale(low=found[0], high=found[1])


def load(request: Request, experiment_id: int) -> tuple[Experiment, Settings]:
    """The experiment a route names, with the settings it was read under."""
    settings: Settings = request.app.state.settings
    return service.load_experiment(settings, experiment_id), settings
