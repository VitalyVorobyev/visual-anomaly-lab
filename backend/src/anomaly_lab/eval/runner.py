"""Computing an experiment's metrics from what is already stored.

The evaluation layer is **model-independent by construction** (handbook evaluation.md): its
inputs are
`ImageResult.score`, `Sample.label`, `SplitAssignment.subset` and — new in M3 — the
`Mask` rows and the float32 maps on disk. It never imports a model module and never
re-runs inference, which is the precondition for the comparison view to mean anything.

It writes three things back: the sample rows, the metric sets, and each image's map peak
and localization verdict. All three are threshold-free, so persisting them leaves the
evaluation layer's line where it was: what moves with the slider is still computed on
demand and still stored nowhere.

Re-running this on a finished experiment is safe and cheap: it reads persisted scores and
rewrites the metric sets. That is what makes changing the aggregation mode a re-read
rather than a re-train.
"""

from __future__ import annotations

import math
import sqlite3
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from pydantic import BaseModel, Field
from pydantic.config import JsonDict

from anomaly_lab.db.repositories import results as results_repo
from anomaly_lab.db.repositories.annotations import GroundTruthMask
from anomaly_lab.db.repositories.results import ScoredImage, ScoredSample
from anomaly_lab.domain.entities import (
    Aggregation,
    ChannelNormalization,
    Experiment,
    Label,
    Subset,
    Task,
)
from anomaly_lab.eval.aggregate import build_sample_results
from anomaly_lab.eval.ground_truth import digest as ground_truth_digest
from anomaly_lab.eval.ground_truth import resolved_masks
from anomaly_lab.eval.localization import hits, peak_of, tolerance_px
from anomaly_lab.eval.metrics import average_precision, roc_auc, timing_summary
from anomaly_lab.eval.pixel import DEFAULT_BINS, PixelAccumulator
from anomaly_lab.map_files import read_map
from anomaly_lab.media.decode import UnreadableImageError
from anomaly_lab.models.preprocessing import load_mask
from anomaly_lab.schemas import API_MODEL_CONFIG

# A field only the anomaly evaluator reads (see `EvalConfig`).
_ANOMALY_ONLY: JsonDict = {"x-tasks": [Task.ANOMALY.value]}


class EvalConfig(BaseModel):
    """How an experiment's stored scores are read. The only config that may be revisited.

    Every task's sample rows are aggregated here (`rebuild_sample_results`), so the two
    aggregation fields apply to all of them; the rest are read by the anomaly evaluator
    alone, and say so with `x-tasks`, which the create form reads to hide a field that
    would change nothing.
    """

    model_config = API_MODEL_CONFIG

    aggregation: Aggregation = Field(
        default=Aggregation.MAX,
        description=(
            "How a part's per-channel scores become one score. 'max' treats a defect "
            "visible under any single view as a defective part."
        ),
    )
    channel_normalization: ChannelNormalization = Field(
        default=ChannelNormalization.NONE,
        description=(
            "Put a part's per-channel scores on one scale before combining them. Without "
            "it, 'max' picks whichever illumination the method scores highest overall "
            "rather than the one showing a defect. 'robust_z' centres each channel on its "
            "median; 'rank' is scale-free but keeps only the ordering, so a dramatic "
            "outlier and a marginal one score the same."
        ),
    )
    pixel_metrics: bool = Field(
        default=True,
        description="Compute pixel ROC-AUC and AU-PRO where ground-truth masks exist.",
        json_schema_extra=_ANOMALY_ONLY,
    )
    pixel_bins: int = Field(
        default=DEFAULT_BINS,
        json_schema_extra=_ANOMALY_ONLY,
        ge=256,
        le=1 << 20,
        description="Score-histogram resolution for the pixel curves.",
    )
    localization_tolerance: float = Field(
        default=0.02,
        json_schema_extra=_ANOMALY_ONLY,
        ge=0.0,
        le=0.25,
        description=(
            "How far the map's peak may sit from the annotated region and still count as "
            "localized, as a fraction of the image diagonal. Resolved to a pixel radius per "
            "image and reported beside the counts. A fraction rather than a pixel count "
            "because a map's real resolution is its patch stride, which scales with the "
            "frame."
        ),
    )


def _resize_map(array: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Bilinearly resample an anomaly map onto `(height, width)`.

    Maps are produced at the preprocessing resolution and masks live at the source
    resolution. Upsampling the map — rather than downsampling the mask — is what the
    reference implementations do, and it avoids the question of what a downsampled label
    means. Memory stays bounded because exactly one image is in flight.
    """
    height, width = shape
    if array.shape == shape:
        return array
    resized = Image.fromarray(np.asarray(array, dtype=np.float32), mode="F").resize(
        (width, height), Image.Resampling.BILINEAR
    )
    return np.asarray(resized, dtype=np.float32)


def _pixel_metrics(
    images: Sequence[ScoredImage],
    masks: dict[int, GroundTruthMask],
    *,
    bins: int,
) -> dict[str, Any] | None:
    """Pixel ROC-AUC and AU-PRO over one subset, at constant memory.

    **Which images count is a protocol decision, not a detail.** Ground truth is attached
    only to defective images, so evaluating over those alone would draw the negative
    pixels exclusively from inside defective images and report a false-positive rate that
    no published number is comparable to. Normal images are therefore included with an
    all-zero mask, which is the convention MVTec and VisA results are computed under.

    A defective image with no mask is *skipped and counted*: its defect pixels are
    somewhere, and guessing they are nowhere would quietly reward a model for missing it.
    """
    scored_with_maps = [image for image in images if image.map_path]
    if not scored_with_maps:
        return None

    if not masks:
        return None

    # (image, its map file, its mask file or None for an all-normal image).
    evaluable: list[tuple[ScoredImage, str, str | None]] = []
    skipped_unannotated = 0
    for image in scored_with_maps:
        if image.map_path is None:  # pragma: no cover - filtered above, narrows the type
            continue
        truth = masks.get(image.image_id)
        if truth is not None:
            evaluable.append((image, image.map_path, truth.path))
        elif image.label is Label.NORMAL:
            evaluable.append((image, image.map_path, None))
        else:
            skipped_unannotated += 1

    # One cheap pass for the histogram range. Reading each map twice costs a few hundred
    # milliseconds and buys binning that adapts to the model's actual score scale, which
    # a fixed [0, 1] assumption would get wrong for every method that is not normalized.
    minimum, maximum = math.inf, -math.inf
    readable: list[tuple[ScoredImage, str, str | None]] = []
    missing_maps = 0
    for image, map_path, mask_path in evaluable:
        try:
            array = read_map(map_path)
        except (OSError, ValueError):
            missing_maps += 1
            continue
        readable.append((image, map_path, mask_path))
        finite = array[np.isfinite(array)]
        if finite.size:
            minimum = min(minimum, float(finite.min()))
            maximum = max(maximum, float(finite.max()))

    if not readable:
        return None
    if not math.isfinite(minimum):
        # Every map is explicitly uncovered. There is still a valid result: no pixel
        # received evidence, so every pixel is scored at one shared floor.
        minimum, maximum = 0.0, 1.0

    accumulator = PixelAccumulator(vmin=minimum, vmax=maximum, bins=bins)
    unreadable_masks = 0
    for image, map_path, mask_path in readable:
        array = read_map(map_path)
        if mask_path is None:
            shape = (image.height, image.width)
            mask = np.zeros(shape, dtype=bool)
        else:
            try:
                mask = load_mask(Path(mask_path))
            except UnreadableImageError:
                unreadable_masks += 1
                continue
            shape = mask.shape
        accumulator.add(_resize_map(array, shape), mask)

    summary = accumulator.summary()
    summary["skipped_unannotated_defects"] = skipped_unannotated
    summary["skipped_missing_maps"] = missing_maps
    summary["skipped_unreadable_masks"] = unreadable_masks
    return dict(summary)


def ensure_peaks(
    conn: sqlite3.Connection,
    experiment_id: int,
    images: Sequence[ScoredImage],
) -> dict[int, tuple[int, int]]:
    """Every scored map's peak, computing and persisting the ones not recorded yet.

    A run records its peaks as it writes each map, so this is usually a no-op. A row with no
    peak beside a map gets one here — one `read_map` per map, once, after which the column
    is filled for good. Idempotent by
    construction: a row that already has a peak is never re-read, so the cost of the second
    call is a dictionary comprehension.

    A map that cannot be read is skipped rather than reported as absent evidence. It is
    already counted by the pixel metrics as a missing map, and inventing a coordinate for it
    would put a marker on a picture that does not exist.
    """
    resolved: dict[int, tuple[int, int]] = {}
    for image in images:
        if image.peak_x is not None and image.peak_y is not None:
            resolved[image.image_id] = (image.peak_x, image.peak_y)

    computed: dict[int, tuple[int, int]] = {}
    for image in images:
        if image.map_path is None or image.image_id in resolved:
            continue
        try:
            array = read_map(image.map_path)
        except (OSError, ValueError):
            continue
        peak = peak_of(array)
        if peak is not None:
            computed[image.image_id] = peak

    results_repo.update_image_peaks(conn, experiment_id, computed)
    return {**resolved, **computed}


def localization_verdicts(
    images: Sequence[ScoredImage],
    peaks: Mapping[int, tuple[int, int]],
    masks: Mapping[int, GroundTruthMask],
    tolerance: float,
) -> dict[int, bool | None]:
    """Per-image `localized`, for every scored image — `None` where it does not apply.

    Which images qualify is the whole policy: a **defective** image, with a resolved ground-
    truth mask, whose map produced a peak. A normal image has no region for the peak to be
    inside of, so "localized" is not false for it, it is meaningless; an unannotated defect
    is the case pixel metrics already refuse to guess at, and this refuses in the same way.

    Every image gets an entry, hits and non-applicable alike, because the caller writes the
    whole dictionary: a verdict that has *become* inapplicable — the annotation was deleted,
    the label was corrected — must be cleared, not left behind.

    The peak is stored in the map's frame, which projection makes the source frame, and the
    mask is read at its own resolution. Where a published mask disagrees with the source
    size the peak is scaled rather than the mask resampled: the argmax of a bilinearly
    resized map is at the same place as the argmax of the original, so scaling the one
    coordinate is exact where resampling the label map would invent labels.
    """
    verdicts: dict[int, bool | None] = {image.image_id: None for image in images}

    for image in images:
        truth = masks.get(image.image_id)
        peak = peaks.get(image.image_id)
        if image.label is not Label.DEFECT or truth is None or peak is None:
            continue
        try:
            mask = load_mask(Path(truth.path))
        except UnreadableImageError:
            continue
        if mask.ndim != 2 or mask.size == 0 or image.width <= 0 or image.height <= 0:
            continue

        mask_height, mask_width = mask.shape
        x = min(mask_width - 1, peak[0] * mask_width // image.width)
        y = min(mask_height - 1, peak[1] * mask_height // image.height)
        radius = tolerance_px(mask_width, mask_height, tolerance)
        verdicts[image.image_id] = hits(mask, (x, y), radius)

    return verdicts


def _localization_metrics(
    images: Sequence[ScoredImage],
    samples: Sequence[ScoredSample],
    *,
    tolerance: float,
) -> dict[str, Any] | None:
    """The subset's peak-on-target counts, or `None` when nothing was checked.

    Absent rather than zeroed, exactly as `_pixel_metrics` is absent for a dataset with no
    masks. A block reading `0 of 0` on a run whose defects are unannotated says "the method
    never localized anything", which is a claim about the method made out of the absence of
    ground truth.
    """
    defect_images = [image for image in images if image.label is Label.DEFECT]
    tested_images = [image for image in defect_images if image.localized is not None]
    defect_samples = [sample for sample in samples if sample.label is Label.DEFECT]
    tested_samples = [sample for sample in defect_samples if sample.localized is not None]

    if not tested_images and not tested_samples:
        return None

    # One number only when every scored image in the subset shares a frame; a mixed-size
    # subset has a different radius per image and a single figure would name none of them.
    frames = {(image.width, image.height) for image in images}
    shared = frames.pop() if len(frames) == 1 else None

    return {
        "tolerance_fraction": tolerance,
        "tolerance_pixels": (
            None if shared is None else tolerance_px(shared[0], shared[1], tolerance)
        ),
        "defect_images_with_truth": len(tested_images),
        "peak_on_target_images": sum(1 for image in tested_images if image.localized),
        "defect_samples_with_truth": len(tested_samples),
        "localized_samples": sum(1 for sample in tested_samples if sample.localized),
        "unannotated_defect_samples": len(defect_samples) - len(tested_samples),
    }


def _label_array(labels: Sequence[Label]) -> np.ndarray:
    return np.array([label is Label.DEFECT for label in labels], dtype=bool)


def _subset_metrics(
    images: Sequence[ScoredImage],
    samples: Sequence[ScoredSample],
    config: EvalConfig,
    masks: dict[int, GroundTruthMask],
) -> dict[str, Any]:
    labelled_samples = [s for s in samples if s.label is not Label.UNLABELED]
    labelled_images = [i for i in images if i.label is not Label.UNLABELED]

    sample_labels = _label_array([s.label for s in labelled_samples])
    sample_scores = np.array([s.agg_score for s in labelled_samples], dtype=np.float64)
    image_labels = _label_array([i.label for i in labelled_images])
    image_scores = np.array([i.score for i in labelled_images], dtype=np.float64)

    metrics: dict[str, Any] = {
        "aggregation": config.aggregation.value,
        # Beside the aggregation, because it is the other half of the same decision and a
        # sample-level number is uninterpretable without both. `image_roc_auc` below stays
        # on **raw** scores deliberately: evaluation keeps it as the measure that isolates
        # model quality from how the channels were combined, and normalization is part of
        # combining them.
        "channel_normalization": config.channel_normalization.value,
        "samples": {
            "total": len(samples),
            "normal": sum(1 for s in samples if s.label is Label.NORMAL),
            "defect": sum(1 for s in samples if s.label is Label.DEFECT),
            "unlabeled": len(samples) - len(labelled_samples),
        },
        "images": {"total": len(images), "labelled": len(labelled_images)},
        "sample_roc_auc": roc_auc(sample_labels, sample_scores),
        "sample_average_precision": average_precision(sample_labels, sample_scores),
        "image_roc_auc": roc_auc(image_labels, image_scores),
        "image_average_precision": average_precision(image_labels, image_scores),
        "timing": timing_summary(np.array([i.inference_ms for i in images], dtype=np.float64)),
    }

    if config.pixel_metrics:
        pixel = _pixel_metrics(images, masks, bins=config.pixel_bins)
        if pixel is not None:
            metrics["pixel"] = pixel

    localization = _localization_metrics(images, samples, tolerance=config.localization_tolerance)
    if localization is not None:
        metrics["localization"] = localization

    return metrics


def _scored_with_truth(
    conn: sqlite3.Connection,
    experiment: Experiment,
) -> tuple[list[ScoredImage], dict[int, GroundTruthMask]]:
    """Every scored image of a run, and the ground truth resolved for it — once.

    Hoisted out of the per-subset loop it used to sit in. `resolved_masks` is keyed by image
    id, so a superset is exactly as correct per subset as a per-subset resolution was, and
    the byte verification that makes a changed pinned file a named failure now happens once
    per run instead of once per subset.
    """
    images = results_repo.list_scored_images(conn, experiment.id)
    return images, resolved_masks(conn, images, verify_bytes=True)


def _evaluate_with_digests(
    conn: sqlite3.Connection,
    experiment: Experiment,
    images: Sequence[ScoredImage],
    masks: dict[int, GroundTruthMask],
) -> tuple[dict[Subset, dict[str, Any]], dict[Subset, str]]:
    config = EvalConfig.model_validate(experiment.eval_config)
    computed: dict[Subset, dict[str, Any]] = {}
    digests: dict[Subset, str] = {}
    for subset in results_repo.scored_subsets(conn, experiment.id):
        in_subset = [image for image in images if image.subset is subset]
        samples = results_repo.list_scored_samples(conn, experiment.id, subset=subset)
        if not samples:
            continue
        computed[subset] = _subset_metrics(in_subset, samples, config, masks)
        digests[subset] = ground_truth_digest(in_subset, masks)
    return computed, digests


def evaluate_experiment(
    conn: sqlite3.Connection,
    experiment: Experiment,
) -> dict[Subset, dict[str, Any]]:
    """Compute every subset's metrics from stored scores. Does not store metrics.

    Reads the localization verdicts as persisted rather than recomputing them, which is what
    "does not store" has to mean: a caller asking what the database currently says must not
    be the caller that changes it.
    """
    images, masks = _scored_with_truth(conn, experiment)
    computed, _ = _evaluate_with_digests(conn, experiment, images, masks)
    return computed


def rebuild_sample_results(
    conn: sqlite3.Connection,
    experiment: Experiment,
    images: Sequence[ScoredImage] | None = None,
) -> int:
    """Re-derive every sample score from the stored per-image scores.

    Owned here rather than by the `infer` handler, because it is not part of running a
    model — it is the first half of reading the results, and `eval_config` is the one
    configuration that may be reinterpreted without re-running anything. Keeping it in the
    handler is what made `reevaluate` a half-truth: it refreshed the metrics from
    `sample_result` while leaving `sample_result` itself at the aggregation of the original
    run, so a changed aggregation appeared to apply and did not.

    `images` is accepted so a caller that has just refreshed them does not read them twice;
    the sample's localization verdict is carried up from those rows, so it follows the
    aggregation being applied here rather than the one the last run used.
    """
    config = EvalConfig.model_validate(experiment.eval_config)
    scored = results_repo.list_scored_images(conn, experiment.id) if images is None else images
    rows = build_sample_results(
        experiment.id, scored, config.aggregation, config.channel_normalization
    )
    return results_repo.replace_sample_results(conn, experiment.id, rows)


def evaluate_and_store(
    conn: sqlite3.Connection,
    experiment: Experiment,
) -> dict[Subset, dict[str, Any]]:
    """Compute and persist. The `infer` job's last act, and re-runnable on its own.

    The order is the contract. Ground truth is resolved once for the whole run; peaks are
    backfilled for any map that has none; the localization verdicts are computed against
    that truth and written per image; sample rows are rebuilt from the refreshed image rows,
    so a changed aggregation moves both the score and the verdict that explains it; and only
    then are the metrics read. Doing this in any other order gives a metric set describing
    an intermediate state of its own inputs.

    Everything written here is threshold-free, which is why persisting it does not contradict
    the evaluation layer: the slider still recomputes every count it moves.
    """
    config = EvalConfig.model_validate(experiment.eval_config)
    images, masks = _scored_with_truth(conn, experiment)

    peaks = ensure_peaks(conn, experiment.id, images)
    verdicts = localization_verdicts(images, peaks, masks, config.localization_tolerance)
    results_repo.update_image_localization(conn, experiment.id, verdicts)

    # Re-read rather than patch in memory: the rows the metrics are computed from are then
    # the rows on disk, and a write that failed to land shows up as a wrong number here
    # rather than as a right one over a stale database.
    images = results_repo.list_scored_images(conn, experiment.id)
    rebuild_sample_results(conn, experiment, images)

    computed, digests = _evaluate_with_digests(conn, experiment, images, masks)
    results_repo.replace_metric_sets(conn, experiment.id, computed, ground_truth_digests=digests)
    return computed
