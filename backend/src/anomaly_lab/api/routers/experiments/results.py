"""Reading a scored run: rankings, thresholds, curves, previews and what it left on disk.

Threshold-dependent numbers are computed per request rather than stored, so the slider is
a filter over a few hundred floats and never a database write (ADR-0011).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from fastapi import APIRouter, Query, Request

from anomaly_lab.api.routers.experiments.views import (
    ArtifactFile,
    ArtifactGroup,
    ArtifactListing,
    Curve,
    CurveSet,
    ImageScore,
    MapPeak,
    MapScale,
    ResultsPage,
    SamplePreview,
    load,
)
from anomaly_lab.db.connection import connection
from anomaly_lab.db.repositories import annotations as annotations_repo
from anomaly_lab.db.repositories import results as results_repo
from anomaly_lab.domain.entities import Label, Subset
from anomaly_lab.eval.localization import tolerance_px
from anomaly_lab.eval.metrics import pr_curve, roc_curve
from anomaly_lab.eval.runner import EvalConfig
from anomaly_lab.eval.threshold import ThresholdReport, classify, report, suggest_threshold
from anomaly_lab.models.base import evenly_spaced

router = APIRouter(prefix="/api/experiments", tags=["experiments"])

# A ROC curve has one point per distinct score, so a large test set produces more points
# than a chart has pixels. Capped, and the cap is reported rather than applied silently.
CURVE_POINT_LIMIT = 2000


@router.get("/{experiment_id}/results", summary="Ranked samples for one subset")
def get_results(
    request: Request,
    experiment_id: int,
    subset: Subset | None = Query(default=None),
) -> ResultsPage:
    experiment, settings = load(request, experiment_id)
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
    experiment, settings = load(request, experiment_id)
    with connection(settings.db_path) as conn:
        samples = results_repo.list_scored_samples(conn, experiment.id, subset=subset)
    return report(samples, value, include_samples=True)


@router.get("/{experiment_id}/artifacts", summary="What this run left on disk")
def get_artifacts(request: Request, experiment_id: int) -> ArtifactListing:
    """Everything under the experiment's directory, grouped and sized.

    A *listing*, not a download and not a mount. Serving the artifact directory
    statically was ruled out (handbook diagnostics.md), one stated reason being that it
    exposes the checkpoints; nothing here changes that. What it fixes is the other half of the
    problem, which is that a run could spend eleven minutes producing a 31 MB checkpoint
    and then not say where it was — the path was in `ExperimentDetail` all along and no
    screen showed it.

    Opening the directory is the desktop shell's job (handbook frontend.md), and a browser gets the
    path as text, which is a different affordance rather than a broken one.
    """
    experiment, _ = load(request, experiment_id)
    root = Path(experiment.artifact_dir)

    groups = [
        _artifact_group(root, "model", "Trained weights"),
        _artifact_group(root, "maps", "Anomaly maps", summarize_from=32),
        _artifact_group(root, "diagnostics", "Diagnostics", summarize_from=32),
        _artifact_group(root, "exports", "Portable exports"),
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
    experiment, settings = load(request, experiment_id)
    with connection(settings.db_path) as conn:
        scored = results_repo.list_scored_images(conn, experiment.id, subset=subset)
        masks = annotations_repo.resolve_ground_truth_masks(
            conn, [image.image_id for image in scored]
        )

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
    histograms and discards them by design (handbook evaluation.md), so drawing that
    curve would mean
    re-reading every anomaly map — the expensive pass this layer exists to avoid.
    """
    experiment, settings = load(request, experiment_id)
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
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return None
    return MapScale(low=float(finite.min()), high=float(finite.max()))


@router.get("/{experiment_id}/samples/{sample_id}/images", summary="Per-image scores of a sample")
def get_sample_images(request: Request, experiment_id: int, sample_id: int) -> list[ImageScore]:
    """What the result viewer needs to draw one part across its channels."""
    experiment, settings = load(request, experiment_id)
    with connection(settings.db_path) as conn:
        scored = [
            image
            for image in results_repo.list_scored_images(conn, experiment.id)
            if image.sample_id == sample_id
        ]
        masks = annotations_repo.resolve_ground_truth_masks(
            conn, [image.image_id for image in scored]
        )

    # Read from the frozen `eval_config`, not from the current default: the verdict on the
    # row was decided under the run's own tolerance, and printing a different radius beside
    # it would describe a test that was never performed.
    tolerance = EvalConfig.model_validate(experiment.eval_config).localization_tolerance

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
            peak=(
                MapPeak(x=image.peak_x, y=image.peak_y)
                if image.peak_x is not None and image.peak_y is not None
                else None
            ),
            localized=image.localized,
            tolerance_px=(
                tolerance_px(image.width, image.height, tolerance)
                if image.width > 0 and image.height > 0
                else None
            ),
        )
        for image in scored
    ]
