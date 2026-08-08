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

from anomaly_lab.api.routers.experiments import MapScale, MetricSummary
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
)
from anomaly_lab.eval.compare import (
    OperatingPoint,
    OperatingThreshold,
    agreement,
    resolve_threshold,
)
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
        metrics_by_run = [_metrics_for(conn, run.id, chosen) for run in experiments]

        runs = [
            _compared_run(run, samples, operating, metrics)
            for run, samples, operating, metrics in zip(
                experiments, samples_by_run, thresholds, metrics_by_run, strict=True
            )
        ]
        warnings = _warnings(conn, experiments, samples_by_run, chosen)

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
    experiment_id: int,
    subset: Subset | None,
) -> MetricSummary | None:
    for found in results_repo.list_metric_sets(conn, experiment_id):
        if found.subset is subset:
            return MetricSummary(
                subset=found.subset, metrics=found.metrics, computed_at=found.computed_at
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
