"""Computing an experiment's metrics from what is already stored.

The evaluation layer is **model-independent by construction** (ADR-0011): its inputs are
`ImageResult.score`, `Sample.label`, `SplitAssignment.subset` and — new in M3 — the
`Mask` rows and the float32 maps on disk. It never imports a model module and never
re-runs inference, which is the precondition for the comparison view to mean anything.

Re-running this on a finished experiment is safe and cheap: it reads persisted scores and
rewrites the metric sets. That is what makes changing the aggregation mode a re-read
rather than a re-train.
"""

from __future__ import annotations

import math
import sqlite3
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from pydantic import BaseModel, Field

from anomaly_lab.db.repositories import results as results_repo
from anomaly_lab.db.repositories.annotations import GroundTruthMask
from anomaly_lab.db.repositories.results import ScoredImage, ScoredSample
from anomaly_lab.domain.entities import Aggregation, Experiment, Label, Subset
from anomaly_lab.eval.ground_truth import digest as ground_truth_digest
from anomaly_lab.eval.ground_truth import resolved_masks
from anomaly_lab.eval.metrics import average_precision, roc_auc, timing_summary
from anomaly_lab.eval.pixel import DEFAULT_BINS, PixelAccumulator
from anomaly_lab.media.decode import UnreadableImageError
from anomaly_lab.models.preprocessing import load_mask
from anomaly_lab.schemas import API_MODEL_CONFIG


class EvalConfig(BaseModel):
    """How an experiment's stored scores are read. The only config that may be revisited."""

    model_config = API_MODEL_CONFIG

    aggregation: Aggregation = Field(
        default=Aggregation.MAX,
        description=(
            "How a part's per-channel scores become one score. 'max' treats a defect "
            "visible under any single view as a defective part."
        ),
    )
    pixel_metrics: bool = Field(
        default=True,
        description="Compute pixel ROC-AUC and AU-PRO where ground-truth masks exist.",
    )
    pixel_bins: int = Field(
        default=DEFAULT_BINS,
        ge=256,
        le=1 << 20,
        description="Score-histogram resolution for the pixel curves.",
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
            array = np.load(map_path, allow_pickle=False)
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
        array = np.load(map_path, allow_pickle=False)
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

    return metrics


def _evaluate_with_digests(
    conn: sqlite3.Connection,
    experiment: Experiment,
) -> tuple[dict[Subset, dict[str, Any]], dict[Subset, str]]:
    config = EvalConfig.model_validate(experiment.eval_config)
    computed: dict[Subset, dict[str, Any]] = {}
    digests: dict[Subset, str] = {}
    for subset in results_repo.scored_subsets(conn, experiment.id):
        images = results_repo.list_scored_images(conn, experiment.id, subset=subset)
        samples = results_repo.list_scored_samples(conn, experiment.id, subset=subset)
        if not samples:
            continue
        masks = resolved_masks(conn, images, verify_bytes=True)
        computed[subset] = _subset_metrics(images, samples, config, masks)
        digests[subset] = ground_truth_digest(images, masks)
    return computed, digests


def evaluate_experiment(
    conn: sqlite3.Connection,
    experiment: Experiment,
) -> dict[Subset, dict[str, Any]]:
    """Compute every subset's metrics from stored scores. Does not store metrics."""
    computed, _ = _evaluate_with_digests(conn, experiment)
    return computed


def evaluate_and_store(
    conn: sqlite3.Connection,
    experiment: Experiment,
) -> dict[Subset, dict[str, Any]]:
    """Compute and persist. The `infer` job's last act, and re-runnable on its own."""
    computed, digests = _evaluate_with_digests(conn, experiment)
    results_repo.replace_metric_sets(conn, experiment.id, computed, ground_truth_digests=digests)
    return computed
