"""Several runs read against each other, under one protocol (ADR-0028).

Nothing here recomputes a metric and nothing re-runs inference: the threshold-independent
numbers are the ones a job already stored (ADR-0011), and the threshold-dependent ones are
the same read over a few hundred floats the single-run threshold route does.

The one thing this route decides that no other route does is **the operating point**. Score
units do not survive a change of method, so a single numeric threshold across runs would
print N true confusion matrices at operating points nobody chose. Instead one *rule* is
applied per run to that run's own distribution, and every resolved value is returned beside
the sentence that explains it — the caller is expected to show both.

Its own prefix rather than a path under `/api/experiments`, because `/{experiment_id}`
there would shadow it unless the declaration order happened to be right, and a route whose
correctness depends on the order two `include_router` calls appear in is a trap.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, Field

from anomaly_lab.api.routers.experiments.views import MapScale, MetricSummary
from anomaly_lab.config import Settings
from anomaly_lab.db.connection import connection
from anomaly_lab.db.repositories import datasets as datasets_repo
from anomaly_lab.db.repositories import experiments as experiments_repo
from anomaly_lab.db.repositories import jobs as jobs_repo
from anomaly_lab.db.repositories import results as results_repo
from anomaly_lab.db.repositories import splits as splits_repo
from anomaly_lab.db.repositories.results import ScoredSample
from anomaly_lab.domain.entities import (
    Experiment,
    ExperimentStatus,
    JobKind,
    JobStatus,
    Label,
    Subset,
    Task,
)
from anomaly_lab.eval.compare import (
    OperatingPoint,
    OperatingThreshold,
    agreement,
    resolve_threshold,
)
from anomaly_lab.eval.detection import CUT_KEYS
from anomaly_lab.eval.evaluators import evaluator_for
from anomaly_lab.eval.segmentation import sample_outcomes
from anomaly_lab.eval.threshold import ConfusionCounts, report
from anomaly_lab.media.overlay import read_display_range
from anomaly_lab.schemas import API_MODEL_CONFIG

router = APIRouter(prefix="/api/compare", tags=["compare"])

# Six columns of numbers is already a wide table; past that it stops being read, which is
# the only thing a comparison screen is for. The runs are capped and the samples are not:
# the standing rule bounds what costs more than it is worth, and every disagreeing sample
# is worth exactly what it costs.
MAX_RUNS = 6

DEFAULT_RECALL_TARGET = 0.95


class ComparedRun(BaseModel):
    """One experiment as a column of the comparison.

    `metrics` is the stored metric set for the chosen subset, verbatim and open-ended, for
    the same reason `MetricSummary` is: what is computable depends on the data, and a fixed
    set of columns would have to invent the ones that are missing.
    """

    model_config = API_MODEL_CONFIG

    id: int
    name: str
    model_type: str
    status: ExperimentStatus
    created_at: str
    scored: bool = Field(description="Whether this run has results for the chosen subset.")
    metrics: dict[str, Any] = Field(default_factory=dict)
    ground_truth_stale: bool = False
    map_range: MapScale | None = Field(
        default=None,
        description=(
            "This run's own display range. Never reconciled with another run's — what "
            "transfers across runs is a fraction of the range, not a value (ADR-0028)."
        ),
    )
    threshold: float | None = Field(
        default=None,
        description=(
            "The operating point in **this run's** units, or null when the rule cannot apply."
        ),
    )
    threshold_rationale: str = Field(
        default="",
        description="How that number was arrived at, to be printed beside it.",
    )
    confusion: ConfusionCounts | None = None
    precision: float | None = None
    recall: float | None = None
    f1: float | None = None
    accuracy: float | None = None
    config: dict[str, Any] = Field(default_factory=dict)
    preprocessing: dict[str, Any] = Field(default_factory=dict)
    evaluation: dict[str, Any] = Field(default_factory=dict)


class ComparedSample(BaseModel):
    """One sample as every run judged it, index-aligned with `ComparisonReport.runs`.

    Aligned lists rather than a map keyed by experiment id: the screen draws one column per
    run in the order it asked for them, and the alignment is the response's contract.
    """

    model_config = API_MODEL_CONFIG

    sample_id: int
    group_key: str
    external_id: str
    label: Label
    scores: list[float | None]
    predicted: list[bool | None]
    outcomes: list[str | None] = Field(
        description="tp, fp, tn, fn, or 'unlabeled' — null where a run has no verdict."
    )
    agree: bool = Field(
        description="Whether every run that could judge this sample predicted the same thing."
    )


class ComparisonReport(BaseModel):
    """N runs on one split, at one operating-point rule."""

    model_config = API_MODEL_CONFIG

    dataset_id: int
    dataset_name: str | None = None
    split_id: int
    split_name: str | None = None
    subset: Subset | None = Field(
        default=None,
        description="The subset actually compared, which may not be the one requested.",
    )
    subsets: list[Subset] = Field(
        default_factory=list,
        description="Every subset at least one of these runs has scored.",
    )
    operating_point: OperatingPoint
    recall_target: float
    runs: list[ComparedRun] = Field(default_factory=list)
    samples: list[ComparedSample] = Field(default_factory=list)
    warnings: list[str] = Field(
        default_factory=list,
        description=(
            "Reasons to read the table with care. A comparison across datasets or splits "
            "is refused outright; these are the differences that are legitimate but change "
            "what the numbers mean."
        ),
    )


@router.get("", summary="Several experiments side by side, at one operating-point rule")
def compare_experiments(
    request: Request,
    ids: Annotated[
        list[int],
        Query(description="Experiment ids, in the order the columns should appear."),
    ],
    subset: Subset | None = Query(
        default=None,
        description="Omitted means the most test-like subset every selected run has scored.",
    ),
    at: OperatingPoint = Query(
        default=OperatingPoint.F1,
        description="The rule applied to each run's own scores to choose its threshold.",
    ),
    recall_target: float = Query(
        default=DEFAULT_RECALL_TARGET,
        gt=0.0,
        le=1.0,
        description="Only used by the `recall` rule.",
    ),
) -> ComparisonReport:
    settings: Settings = request.app.state.settings
    with connection(settings.db_path) as conn:
        experiments = _selected(conn, ids)
        first = experiments[0]

        scored_by_run = [results_repo.scored_subsets(conn, run.id) for run in experiments]
        subsets = [value for value in Subset if any(value in found for found in scored_by_run)]
        chosen = subset if subset is not None else _default_subset(scored_by_run)

        samples_by_run = [
            results_repo.list_scored_samples(conn, run.id, subset=chosen) for run in experiments
        ]
        thresholds = [
            resolve_threshold(samples, at, recall_target=recall_target)
            for samples in samples_by_run
        ]
        metrics_by_run = [_metrics_for(conn, run, chosen) for run in experiments]

        runs = [
            _compared_run(run, samples, operating, metrics)
            for run, samples, operating, metrics in zip(
                experiments, samples_by_run, thresholds, metrics_by_run, strict=True
            )
        ]
        warnings = _warnings(conn, experiments, samples_by_run, chosen)
        stale_names = [run.name for run in runs if run.ground_truth_stale]
        if stale_names:
            warnings.append(
                f"Ground truth changed after metrics were computed for {', '.join(stale_names)}. "
                "Recompute those runs before comparing their stored metrics or curves."
            )

        dataset = datasets_repo.get_dataset(conn, first.dataset_id)
        split = splits_repo.get_split(conn, first.split_id)

    rows = [
        ComparedSample(
            sample_id=row.sample_id,
            group_key=row.group_key,
            external_id=row.external_id,
            label=row.label,
            scores=row.scores,
            predicted=row.predicted,
            outcomes=row.outcomes,
            agree=row.agree,
        )
        for row in agreement(samples_by_run, [found.value for found in thresholds])
    ]
    return ComparisonReport(
        dataset_id=first.dataset_id,
        dataset_name=dataset.name if dataset else None,
        split_id=first.split_id,
        split_name=split.name if split else None,
        subset=chosen,
        subsets=subsets,
        operating_point=at,
        recall_target=recall_target,
        runs=runs,
        samples=[ComparedSample(**vars(row)) for row in rows],
        warnings=warnings,
    )


def _selected(conn: sqlite3.Connection, ids: list[int]) -> list[Experiment]:
    """The requested experiments, or the reason they cannot be compared.

    A different split is a different question, so it is refused rather than warned: the
    numbers are computed over different samples and putting them in adjacent columns is the
    error the column layout invites. Preprocessing is the opposite case — a legitimate
    experiment whose result needs a caveat — and it is a warning below.
    """
    if len(ids) < 2:
        raise HTTPException(status_code=422, detail="a comparison needs at least two experiments")
    if len(set(ids)) != len(ids):
        raise HTTPException(status_code=422, detail="the same experiment was selected twice")
    if len(ids) > MAX_RUNS:
        raise HTTPException(
            status_code=422,
            detail=f"at most {MAX_RUNS} experiments can be compared at once",
        )

    experiments: list[Experiment] = []
    for experiment_id in ids:
        found = experiments_repo.get_experiment(conn, experiment_id)
        if found is None:
            raise HTTPException(status_code=404, detail=f"no experiment with id {experiment_id}")
        experiments.append(found)

    for experiment in experiments:
        # Every comparison here is read at thresholds against normal/defect labels. A
        # few-shot segmentation run is measured against its class truth instead (ADR-0040),
        # and is compared by `compare_few_shot` below.
        if experiment.task is not Task.ANOMALY:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"'{experiment.name}' is a {experiment.task.value} run; this comparison "
                    "reads anomaly runs — few-shot runs are compared at /api/compare/few-shot "
                    "and detection runs at /api/compare/detection"
                ),
            )

    first = experiments[0]
    for other in experiments[1:]:
        if other.dataset_id != first.dataset_id:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"'{other.name}' is on a different dataset from '{first.name}'; "
                    "runs on different data are not comparable"
                ),
            )
        if other.split_id != first.split_id:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"'{other.name}' uses a different split from '{first.name}'; "
                    "the numbers would be computed over different samples"
                ),
            )
    return experiments


def _default_subset(scored_by_run: list[list[Subset]]) -> Subset | None:
    """The most test-like subset **every** run has scored, falling back to any of them.

    Pooling every subset — which is what `subset=None` means on the single-run routes —
    would be a poor default here: a table headed by no subset at all reads as one protocol
    while mixing training and held-out data.
    """
    if not scored_by_run:  # pragma: no cover - the route requires two runs
        return None
    for candidate in (Subset.TEST, Subset.VAL, Subset.TRAIN):
        if all(candidate in found for found in scored_by_run):
            return candidate
    scored = [value for value in Subset if any(value in found for found in scored_by_run)]
    return scored[-1] if scored else None


def _metrics_for(
    conn: sqlite3.Connection,
    experiment: Experiment,
    subset: Subset | None,
) -> MetricSummary | None:
    evaluator = evaluator_for(experiment.task)
    for found in results_repo.list_metric_sets(conn, experiment.id):
        if found.subset is subset:
            return MetricSummary(
                subset=found.subset,
                metrics=found.metrics,
                computed_at=found.computed_at,
                ground_truth_digest=found.ground_truth_digest,
                ground_truth_stale=(
                    found.ground_truth_digest
                    != evaluator.current_digest(conn, experiment, found.subset)
                ),
            )
    return None


def _compared_run(
    experiment: Experiment,
    samples: list[ScoredSample],
    operating: OperatingThreshold,
    metrics: MetricSummary | None,
) -> ComparedRun:
    at_threshold = None if operating.value is None else report(samples, operating.value)
    return ComparedRun(
        id=experiment.id,
        name=experiment.name,
        model_type=experiment.model_type,
        status=experiment.status,
        created_at=experiment.created_at,
        scored=bool(samples),
        metrics=metrics.metrics if metrics else {},
        ground_truth_stale=metrics.ground_truth_stale if metrics else False,
        map_range=_map_range(Path(experiment.artifact_dir)),
        threshold=operating.value,
        threshold_rationale=operating.rationale,
        confusion=at_threshold.confusion if at_threshold else None,
        precision=at_threshold.precision if at_threshold else None,
        recall=at_threshold.recall if at_threshold else None,
        f1=at_threshold.f1 if at_threshold else None,
        accuracy=at_threshold.accuracy if at_threshold else None,
        config=experiment.model_config_,
        preprocessing=experiment.preprocessing_config,
        evaluation=experiment.eval_config,
    )


def _map_range(artifact_dir: Path) -> MapScale | None:
    found = read_display_range(artifact_dir / "maps")
    return None if found is None else MapScale(low=found[0], high=found[1])


def _warnings(
    conn: sqlite3.Connection,
    experiments: list[Experiment],
    samples_by_run: list[list[ScoredSample]],
    subset: Subset | None,
) -> list[str]:
    """Everything legitimate that still changes what the table means.

    Preprocessing first and in the strongest words available: two runs shown different
    pixels produce a difference in AUROC that is partly a measurement of the resize, which
    is the exact failure the shared preprocessing bridge exists to prevent.
    """
    notes: list[str] = []

    region_profiles = {run.region_profile_id for run in experiments}
    if len(region_profiles) > 1:
        notes.append(
            "These runs use different prepared-region revisions. Differences below may "
            "measure localisation or spatial preparation as well as the methods."
        )

    differing = _differing_keys([run.preprocessing_config for run in experiments])
    if differing:
        notes.append(
            f"These runs were shown different pixels — {', '.join(differing)} differ. "
            "A difference in the metrics below is then partly a measurement of the "
            "preprocessing, not of the methods."
        )

    differing = _differing_keys([run.eval_config for run in experiments])
    if differing:
        notes.append(
            f"The evaluation configuration differs — {', '.join(differing)}. "
            "The stored scores are the same; how they were read into these numbers is not."
        )

    unscored = [
        run.name for run, samples in zip(experiments, samples_by_run, strict=True) if not samples
    ]
    if unscored:
        where = "any subset" if subset is None else f"the {subset.value} subset"
        notes.append(
            f"{', '.join(unscored)} has no results for {where}, so its column is empty. "
            "Score it to fill it in."
        )

    for run in experiments:
        if _trained_after_scoring(conn, run):
            notes.append(
                f"'{run.name}' finished training after it was last scored, so the numbers "
                "here describe an older checkpoint. Re-run Score & evaluate."
            )

    return notes


def _differing_keys(configs: list[dict[str, Any]]) -> list[str]:
    """Which keys are not the same across every run, including keys one run omits."""
    keys = sorted({key for config in configs for key in config})
    return [key for key in keys if len({repr(config.get(key)) for config in configs}) > 1]


def _trained_after_scoring(conn: sqlite3.Connection, experiment: Experiment) -> bool:
    """Whether the stored scores predate the checkpoint they are attributed to.

    Only `train` jobs count. An `infer` job writes the metric sets *before* it marks itself
    finished, so comparing against every job kind would report every experiment as stale.
    Timestamps are both `strftime('%Y-%m-%dT%H:%M:%fZ')`, so they order lexicographically.
    """
    scored_at = [found.computed_at for found in results_repo.list_metric_sets(conn, experiment.id)]
    if not scored_at:
        return False
    trained_at = [
        job.finished_at
        for job in jobs_repo.list_jobs_for_experiment(conn, experiment.id)
        if job.kind is JobKind.TRAIN
        and job.status is JobStatus.SUCCEEDED
        and job.finished_at is not None
    ]
    return bool(trained_at) and max(trained_at) > max(scored_at)


# ---------------------------------------------------------------------------- few-shot


class FewShotRun(BaseModel):
    """One few-shot segmentation run as a column: its reference draw and its test metrics."""

    model_config = API_MODEL_CONFIG

    id: int
    name: str
    model_type: str
    status: ExperimentStatus
    split_id: int
    split_name: str | None = None
    references: int = Field(description="Samples in the split's `train` subset: the shot count.")
    seed: int | None = Field(
        default=None, description="The draw's seed, for a `few_shot` split; null for `manual`."
    )
    region_profile_id: int
    scored: bool
    metrics: dict[str, Any] = Field(default_factory=dict)
    ground_truth_stale: bool = False


class FewShotSample(BaseModel):
    """A query every run scored, with each run's outcome and IoU, index-aligned with `runs`."""

    model_config = API_MODEL_CONFIG

    sample_id: int
    group_key: str
    external_id: str
    outcomes: list[str]
    ious: list[float | None]
    agree: bool


class FewShotComparison(BaseModel):
    """N few-shot runs of one class, which may learn from different reference draws."""

    model_config = API_MODEL_CONFIG

    dataset_id: int
    dataset_name: str | None = None
    target_label: str
    runs: list[FewShotRun] = Field(default_factory=list)
    samples: list[FewShotSample] = Field(
        default_factory=list,
        description=(
            "The test queries every run scored — a run's own references are not queries, so "
            "samples one draw learned from are left out of everyone's rows."
        ),
    )
    warnings: list[str] = Field(default_factory=list)


def _few_shot_selected(conn: sqlite3.Connection, ids: list[int]) -> list[Experiment]:
    """Few-shot runs of one dataset and one class; their splits may differ, by design."""
    if len(ids) < 2:
        raise HTTPException(status_code=422, detail="a comparison needs at least two experiments")
    if len(set(ids)) != len(ids):
        raise HTTPException(status_code=422, detail="the same experiment was selected twice")
    if len(ids) > MAX_RUNS:
        raise HTTPException(
            status_code=422, detail=f"at most {MAX_RUNS} experiments can be compared at once"
        )
    experiments: list[Experiment] = []
    for experiment_id in ids:
        found = experiments_repo.get_experiment(conn, experiment_id)
        if found is None:
            raise HTTPException(status_code=404, detail=f"no experiment with id {experiment_id}")
        if found.task is not Task.FEW_SHOT_SEGMENTATION:
            raise HTTPException(
                status_code=422,
                detail=f"'{found.name}' is a {found.task.value} run, not a few-shot one",
            )
        experiments.append(found)
    first = experiments[0]
    for other in experiments[1:]:
        if other.dataset_id != first.dataset_id:
            raise HTTPException(
                status_code=422,
                detail=f"'{other.name}' is on a different dataset from '{first.name}'",
            )
        if other.target_label != first.target_label:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"'{other.name}' segments {other.target_label!r}, not "
                    f"{first.target_label!r}; runs of different classes are different questions"
                ),
            )
    return experiments


@router.get("/few-shot", summary="Few-shot segmentation runs of one class side by side")
def compare_few_shot(
    request: Request,
    ids: Annotated[
        list[int],
        Query(description="Experiment ids, in the order the columns should appear."),
    ],
) -> FewShotComparison:
    """Runs of one class, including runs that learned from different references (ADR-0040).

    The question is how the answer moves with the references — how many, and which — so the
    split is allowed to differ where the anomaly comparison refuses it. What stays fixed is
    the class, and what is compared is `test`: the stored metrics, and each shared query's
    outcome under each run.
    """
    settings: Settings = request.app.state.settings
    with connection(settings.db_path) as conn:
        experiments = _few_shot_selected(conn, ids)
        first = experiments[0]
        runs: list[FewShotRun] = []
        outcomes = []
        for run in experiments:
            split = splits_repo.get_split(conn, run.split_id)
            scored = Subset.TEST in results_repo.scored_subsets(conn, run.id)
            summary = _metrics_for(conn, run, Subset.TEST) if scored else None
            runs.append(
                FewShotRun(
                    id=run.id,
                    name=run.name,
                    model_type=run.model_type,
                    status=run.status,
                    split_id=run.split_id,
                    split_name=split.name if split else None,
                    references=len(splits_repo.list_sample_ids(conn, run.split_id, Subset.TRAIN)),
                    seed=split.seed if split and split.strategy == "few_shot" else None,
                    region_profile_id=run.region_profile_id,
                    scored=scored,
                    metrics=summary.metrics if summary else {},
                    ground_truth_stale=summary.ground_truth_stale if summary else False,
                )
            )
            outcomes.append(
                {
                    verdict.sample_id: verdict
                    for verdict in sample_outcomes(conn, run, Subset.TEST).samples
                }
                if scored
                else {}
            )
        dataset = datasets_repo.get_dataset(conn, first.dataset_id)

    shared = set.intersection(*(set(found) for found in outcomes)) if outcomes else set()
    samples: list[FewShotSample] = []
    for sample_id in sorted(shared):
        row = [found[sample_id] for found in outcomes]
        names = [verdict.outcome for verdict in row]
        samples.append(
            FewShotSample(
                sample_id=sample_id,
                group_key=row[0].group_key,
                external_id=row[0].external_id,
                outcomes=names,
                ious=[verdict.iou for verdict in row],
                agree=len(set(names)) == 1,
            )
        )

    warnings: list[str] = []
    if len({run.region_profile_id for run in experiments}) > 1:
        warnings.append(
            "These runs read different region profiles, so their pixels differ as well as "
            "their references."
        )
    unscored = [run.name for run in runs if not run.scored]
    if unscored:
        warnings.append(f"Not scored on test yet: {', '.join(unscored)}.")
    stale = [run.name for run in runs if run.ground_truth_stale]
    if stale:
        warnings.append(
            f"Ground truth changed after metrics were computed for {', '.join(stale)}. "
            "Recompute those runs before comparing them."
        )
    return FewShotComparison(
        dataset_id=first.dataset_id,
        dataset_name=dataset.name if dataset else None,
        target_label=first.target_label or "",
        runs=runs,
        samples=samples,
        warnings=warnings,
    )


# ---------------------------------------------------------------------------- detection


class DetectionRun(BaseModel):
    """One object detection run as a column: its threshold-free metrics on one subset."""

    model_config = API_MODEL_CONFIG

    id: int
    name: str
    model_type: str
    status: ExperimentStatus
    scored: bool = Field(description="Whether this run has results for the chosen subset.")
    metrics: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "The stored metric set, less everything read at the run's own confidence cut: "
            "only the AP family, recall and counts cross runs (ADR-0028)."
        ),
    )
    ground_truth_stale: bool = False


class DetectionComparison(BaseModel):
    """N object detection runs of one split and one class list, threshold-free."""

    model_config = API_MODEL_CONFIG

    dataset_id: int
    dataset_name: str | None = None
    split_id: int
    split_name: str | None = None
    subset: Subset | None = None
    subsets: list[Subset] = Field(default_factory=list)
    classes: list[str] = Field(default_factory=list)
    runs: list[DetectionRun] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


def _detection_selected(conn: sqlite3.Connection, ids: list[int]) -> list[Experiment]:
    """Detection runs of one dataset, one split and one class list, or why not."""
    if len(ids) < 2:
        raise HTTPException(status_code=422, detail="a comparison needs at least two experiments")
    if len(set(ids)) != len(ids):
        raise HTTPException(status_code=422, detail="the same experiment was selected twice")
    if len(ids) > MAX_RUNS:
        raise HTTPException(
            status_code=422, detail=f"at most {MAX_RUNS} experiments can be compared at once"
        )
    experiments: list[Experiment] = []
    for experiment_id in ids:
        found = experiments_repo.get_experiment(conn, experiment_id)
        if found is None:
            raise HTTPException(status_code=404, detail=f"no experiment with id {experiment_id}")
        if found.task is not Task.OBJECT_DETECTION:
            raise HTTPException(
                status_code=422,
                detail=f"'{found.name}' is a {found.task.value} run, not a detection one",
            )
        experiments.append(found)
    first = experiments[0]
    for other in experiments[1:]:
        if other.dataset_id != first.dataset_id or other.split_id != first.split_id:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"'{other.name}' is on a different dataset or split from '{first.name}'; "
                    "the numbers would be computed over different samples"
                ),
            )
        if list(other.classes) != list(first.classes):
            raise HTTPException(
                status_code=422,
                detail=(
                    f"'{other.name}' pins different classes from '{first.name}'; its AP is "
                    "averaged over a different set"
                ),
            )
    return experiments


@router.get("/detection", summary="Object detection runs of one split side by side")
def compare_detection(
    request: Request,
    ids: Annotated[
        list[int],
        Query(description="Experiment ids, in the order the columns should appear."),
    ],
    subset: Subset | None = Query(
        default=None,
        description="Omitted means the most test-like subset every selected run has scored.",
    ),
) -> DetectionComparison:
    """Detection runs of one split and class list, on their threshold-free metrics alone.

    A run's verdicts are read at its own confidence cut, and a confidence means nothing
    outside its run (ADR-0028), so nothing cut crosses a column: each run's AP family,
    recall and box counts are the stored ones, and the cut and what it achieves stay on
    that run's own Overview.
    """
    settings: Settings = request.app.state.settings
    with connection(settings.db_path) as conn:
        experiments = _detection_selected(conn, ids)
        first = experiments[0]
        scored_by_run = [results_repo.scored_subsets(conn, run.id) for run in experiments]
        subsets = [value for value in Subset if any(value in found for found in scored_by_run)]
        chosen = subset if subset is not None else _default_subset(scored_by_run)
        samples_by_run = [
            results_repo.list_scored_samples(conn, run.id, subset=chosen) for run in experiments
        ]
        runs: list[DetectionRun] = []
        for run, samples in zip(experiments, samples_by_run, strict=True):
            summary = _metrics_for(conn, run, chosen)
            runs.append(
                DetectionRun(
                    id=run.id,
                    name=run.name,
                    model_type=run.model_type,
                    status=run.status,
                    scored=bool(samples),
                    metrics={
                        key: value
                        for key, value in (summary.metrics if summary else {}).items()
                        if key not in CUT_KEYS
                    },
                    ground_truth_stale=summary.ground_truth_stale if summary else False,
                )
            )
        warnings = _warnings(conn, experiments, samples_by_run, chosen)
        stale = [run.name for run in runs if run.ground_truth_stale]
        if stale:
            warnings.append(
                f"Ground truth changed after metrics were computed for {', '.join(stale)}. "
                "Recompute those runs before comparing them."
            )
        dataset = datasets_repo.get_dataset(conn, first.dataset_id)
        split = splits_repo.get_split(conn, first.split_id)
    return DetectionComparison(
        dataset_id=first.dataset_id,
        dataset_name=dataset.name if dataset else None,
        split_id=first.split_id,
        split_name=split.name if split else None,
        subset=chosen,
        subsets=subsets,
        classes=list(first.classes),
        runs=runs,
        warnings=warnings,
    )
