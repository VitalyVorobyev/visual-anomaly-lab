"""Reading a scored run: rankings, thresholds, curves, previews and what it left on disk.

Threshold-dependent numbers are computed per request rather than stored, so the slider is
a filter over a few hundred floats and never a database write (ADR-0011).
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from fastapi import APIRouter, HTTPException, Query, Request, Response

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
from anomaly_lab.config import Settings
from anomaly_lab.db.connection import connection
from anomaly_lab.db.repositories import annotations as annotations_repo
from anomaly_lab.db.repositories import results as results_repo
from anomaly_lab.domain.entities import Experiment, Label, Subset, Task
from anomaly_lab.errors import ConflictError
from anomaly_lab.eval import detection, semantic
from anomaly_lab.eval.detection import DetectionOutcomes, ImageBoxes
from anomaly_lab.eval.localization import tolerance_px
from anomaly_lab.eval.metrics import pr_curve, roc_curve
from anomaly_lab.eval.runner import EvalConfig
from anomaly_lab.eval.segmentation import SegmentationOutcomes, sample_outcomes
from anomaly_lab.eval.threshold import ThresholdReport, classify, report, suggest_threshold
from anomaly_lab.media.overlay import (
    fit_label_plane,
    parse_label_colours,
    render_box_map,
    render_label_map,
)
from anomaly_lab.media.values import encode_plane
from anomaly_lab.models.base import evenly_spaced

router = APIRouter(prefix="/api/experiments", tags=["experiments"])

# A ROC curve has one point per distinct score, so a large test set produces more points
# than a chart has pixels. Capped, and the cap is reported rather than applied silently.
CURVE_POINT_LIMIT = 2000

# A drawn label map lies on a gallery tile, over the `thumb` tier, so it is no larger than
# that tier's long edge — whatever the image's size, the response is bounded.
LABEL_MAP_LONG_EDGE = 256


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


@router.get(
    "/{experiment_id}/segmentation-outcomes",
    summary="Each sample's segmentation outcome, ranked by score",
)
def get_segmentation_outcomes(
    request: Request,
    experiment_id: int,
    subset: Subset | None = Query(default=None),
) -> SegmentationOutcomes:
    """What a segmentation run did to each sample, for the gallery and the sample page.

    The segmentation counterpart of the threshold report, computed per request from what
    the run stored. A few-shot run is read against its class under the evaluator's rule
    (ADR-0040); a supervised run's label maps against every pinned class, with one more
    outcome, `false_class` (ADR-0039). An anomaly run is refused: its outcomes are the
    threshold report's.
    """
    experiment, settings = load(request, experiment_id)
    if experiment.task not in (Task.FEW_SHOT_SEGMENTATION, Task.SEMANTIC_SEGMENTATION):
        raise ConflictError(
            f"experiment {experiment.id} is a {experiment.task.value} run; it has no "
            "segmentation outcomes"
        )
    with connection(settings.db_path) as conn:
        if experiment.task is Task.SEMANTIC_SEGMENTATION:
            return semantic.sample_outcomes(conn, experiment, subset)
        return sample_outcomes(conn, experiment, subset)


@router.get(
    "/{experiment_id}/images/{image_id}/labels",
    summary="One image's label map, predicted or true, as class indices",
    response_class=Response,
    responses={200: {"content": {"application/octet-stream": {}}}},
)
def read_label_plane(
    request: Request,
    experiment_id: int,
    image_id: int,
    truth: bool = Query(
        default=False,
        description=(
            "The image's truth over the run's pinned classes instead of the method's label "
            "map. A pixel of a class the run does not know is NaN."
        ),
    ),
) -> Response:
    """A supervised segmentation run's class per pixel, for the sample page to draw.

    The value-plane format (handbook diagnostics.md): 0 is background and `i + 1` is the
    run's `classes[i]`. Served as indices rather than a picture because a class's colour is
    the interface's, from the design system's palette; a large frame arrives decimated by an
    integer stride, so every value sent is a class the method or the annotator gave.
    """
    experiment, settings = load(request, experiment_id)
    if experiment.task is not Task.SEMANTIC_SEGMENTATION:
        raise ConflictError(
            f"experiment {experiment.id} is a {experiment.task.value} run; it wrote no label maps"
        )
    with connection(settings.db_path) as conn:
        scored = results_repo.get_image_result(conn, experiment.id, image_id) is not None
        plane = semantic.label_plane(conn, experiment, image_id, truth=truth) if scored else None
    if plane is None:
        what = "no truth for every pinned class" if truth else "no label map"
        raise HTTPException(
            status_code=404, detail=f"image {image_id} has {what} in experiment {experiment.id}"
        )
    return Response(
        content=encode_plane(plane),
        media_type="application/octet-stream",
        # Revalidated: re-running inference or completing an annotation changes the answer.
        headers={"Cache-Control": "no-cache"},
    )


@router.get(
    "/{experiment_id}/images/{image_id}/label-map",
    summary="One image's label map, predicted or true, drawn for a gallery tile",
    response_class=Response,
    responses={
        200: {"content": {"image/png": {}}},
        304: {"description": "The client's copy is current."},
    },
)
def read_label_map_image(
    request: Request,
    experiment_id: int,
    image_id: int,
    colours: str = Query(
        pattern=r"^#?[0-9a-fA-F]{6}(,#?[0-9a-fA-F]{6})*$",
        description=(
            "One hex colour per pinned class, in the run's order: class `i` is drawn in the "
            "`i`-th. The client's palette, so no class colour is kept on this side."
        ),
    ),
    truth: bool = Query(
        default=False,
        description=(
            "Draw the image's truth over the run's pinned classes — dashed border only — "
            "instead of the method's map, which is filled with a solid border."
        ),
    ),
) -> Response:
    """The `labels` plane above as a picture, at the thumbnail tier's size.

    A gallery tile cannot afford a value plane per tile painted in the browser, so this
    draws the same map with the same rule as the sample page's `LabelLayer`. The colours
    come from the client rather than from a table here: the palette has one home, the
    design system, and a server copy would be a second one to keep in step. The size is
    bounded by the thumbnail's long edge, never by the image, and the plane is decimated by
    an integer stride before it is painted, so every border is traced at the size it is
    drawn and every pixel is a class somebody gave.

    Revalidated rather than immutable: re-running inference or completing an annotation
    changes the answer. The `ETag` is the drawn plane's digest, so an unchanged map is a
    304 without being encoded.
    """
    experiment, settings = load(request, experiment_id)
    if experiment.task is not Task.SEMANTIC_SEGMENTATION:
        raise ConflictError(
            f"experiment {experiment.id} is a {experiment.task.value} run; it wrote no label maps"
        )
    palette = parse_label_colours(colours)
    classes = experiment.classes
    if len(palette) < len(classes):
        raise HTTPException(
            status_code=422,
            detail=f"{len(classes)} pinned classes need {len(classes)} colours, got {len(palette)}",
        )
    with connection(settings.db_path) as conn:
        scored = results_repo.get_image_result(conn, experiment.id, image_id) is not None
        plane = semantic.label_plane(conn, experiment, image_id, truth=truth) if scored else None
    if plane is None:
        what = "no truth for every pinned class" if truth else "no label map"
        raise HTTPException(
            status_code=404, detail=f"image {image_id} has {what} in experiment {experiment.id}"
        )

    drawn = fit_label_plane(plane, LABEL_MAP_LONG_EDGE)
    digest = hashlib.sha256(np.ascontiguousarray(drawn, dtype="<f4").tobytes()).hexdigest()
    etag = f'W/"label-map-{experiment.id}-{image_id}-{int(truth)}-{digest[:16]}-{colours.lower()}"'
    headers = {"ETag": etag, "Cache-Control": "no-cache"}
    if request.headers.get("if-none-match") == etag:
        return Response(status_code=304, headers=headers)
    return Response(
        content=render_label_map(drawn, palette, truth=truth),
        media_type="image/png",
        headers=headers,
    )


def _detection_run(request: Request, experiment_id: int) -> tuple[Experiment, Settings]:
    experiment, settings = load(request, experiment_id)
    if experiment.task is not Task.OBJECT_DETECTION:
        raise ConflictError(
            f"experiment {experiment.id} is a {experiment.task.value} run; it wrote no boxes"
        )
    return experiment, settings


def _boxes_of(request: Request, experiment_id: int, image_id: int) -> ImageBoxes:
    experiment, settings = _detection_run(request, experiment_id)
    with connection(settings.db_path) as conn:
        image = next(
            (
                found
                for found in results_repo.list_scored_images(conn, experiment.id)
                if found.image_id == image_id
            ),
            None,
        )
        boxes = None if image is None else detection.image_boxes(conn, experiment, image)
    if boxes is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"image {image_id} has neither stored detections nor truth for every pinned "
                f"class in experiment {experiment.id}"
            ),
        )
    return boxes


@router.get(
    "/{experiment_id}/images/{image_id}/boxes",
    summary="One image's detections and true boxes, matched at the subset's cut",
)
def read_image_boxes(request: Request, experiment_id: int, image_id: int) -> ImageBoxes:
    """An object detection run's boxes for one image, for the sample page to draw.

    Both lists are in the source frame, in pixel-edge coordinates. Each detection says
    whether the subset's confidence cut keeps it and whether, kept, it matched a truth box of
    its class at IoU 0.5; each truth box says whether it was found. The cut is the one the
    evaluator stored for the image's subset, printed with its rule (ADR-0028), so the page
    draws the same verdicts the gallery counts. Bounded by construction: at most
    `MAX_INSTANCES_PER_IMAGE` detections are stored an image. A list is null when there is no
    such answer — no detections written, or truth that does not answer for every class.
    """
    return _boxes_of(request, experiment_id, image_id)


@router.get(
    "/{experiment_id}/images/{image_id}/box-map",
    summary="One image's kept detections and true boxes, drawn for a gallery tile",
    response_class=Response,
    responses={
        200: {"content": {"image/svg+xml": {}}},
        304: {"description": "The client's copy is current."},
    },
)
def read_box_map_image(
    request: Request,
    experiment_id: int,
    image_id: int,
    colours: str = Query(
        pattern=r"^#?[0-9a-fA-F]{6}(,#?[0-9a-fA-F]{6}){2}$",
        description=(
            "Three hex colours: a match, a false positive and a missed truth box. The "
            "client's tokens, so no tone is kept on this side."
        ),
    ),
    predictions: bool = Query(default=True, description="Draw the kept detections."),
    truth: bool = Query(default=True, description="Draw the true boxes."),
) -> Response:
    """The `boxes` above as a picture: kept detections solid, truth dashed, each in its tone.

    A gallery tile cannot afford a request for box data and a drawing of its own per tile,
    so this draws the same verdicts the sample page does into one SVG at the source's size.
    Detections below the cut and truth of a class the run does not pin are left out. The
    `ETag` is the drawing's digest, so an unchanged picture is a 304.
    """
    match, false_positive, missed = parse_label_colours(colours)
    boxes = _boxes_of(request, experiment_id, image_id)
    drawn: list[tuple[Sequence[float], tuple[int, int, int], bool]] = [
        (item.box, match if item.found else missed, True)
        for item in boxes.truth or []
        if truth and item.class_index is not None
    ]
    drawn += [
        (item.box, match if item.matched else false_positive, False)
        for item in boxes.predictions or []
        if predictions and item.kept
    ]
    content = render_box_map(boxes.width, boxes.height, drawn)
    digest = hashlib.sha256(content).hexdigest()[:16]
    etag = f'W/"box-map-{experiment_id}-{image_id}-{digest}"'
    headers = {"ETag": etag, "Cache-Control": "no-cache"}
    if request.headers.get("if-none-match") == etag:
        return Response(status_code=304, headers=headers)
    return Response(content=content, media_type="image/svg+xml", headers=headers)


@router.get(
    "/{experiment_id}/detection-outcomes",
    summary="Each sample's detection outcome at its subset's cut, ranked by score",
)
def get_detection_outcomes(
    request: Request,
    experiment_id: int,
    subset: Subset | None = Query(default=None),
) -> DetectionOutcomes:
    """What a detection run did to each sample, for the gallery and the sample page.

    The detection counterpart of the threshold report, computed per request from the stored
    detections at the cut the evaluator resolved for each subset and printed beside it
    (ADR-0028). Every other task is refused: its outcomes are another report's.
    """
    experiment, settings = _detection_run(request, experiment_id)
    with connection(settings.db_path) as conn:
        return detection.sample_outcomes(conn, experiment, subset)


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
