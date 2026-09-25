"""Metrics for a targeted segmentation task: one class, foreground against background
(ADR-0040).

The inputs are what is already stored — each image's presence score, its foreground map
or the mask the method wrote, and the class truth resolved per image — so this module,
like the anomaly runner, never imports a model and never re-runs one.

- **Per image, not per sample.** A sample-level rule for classes on a multi-channel part
  has not been decided, so every number here counts images, and says so in its name.
- **Unlabelled images are excluded and counted.** An image whose truth does not answer for
  the class is neither a hit nor a miss (`images.unlabeled`).
- **One threshold rule, printed.** A method that writes its own mask is read as written.
  Otherwise the map is its foreground probability and is cut at one fixed rule, whose name
  and value are stored beside the numbers (ADR-0028). Everything else is threshold-free,
  and pixel average precision is the threshold-free reading of the map itself: the stored
  foreground probability ranked against the class truth, over present and absent images.
- **Constant memory in pixels.** Pixel quantities are pooled counts, and the probability
  ranking is folded into the anomaly evaluator's fixed-bin histograms (`eval.pixel`); only
  the per-image presence scores and latencies are kept, which is linear in images.
- **A metric that cannot be computed is `None`.** A subset with no absent image has no
  false-positive rate on absent images.
"""

from __future__ import annotations

import hashlib
import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from pydantic import BaseModel, Field

from anomaly_lab.annotations.class_truth import ClassTruth, load_class_mask, resolve_class_truth
from anomaly_lab.db.repositories import results as results_repo
from anomaly_lab.db.repositories.results import ScoredImage
from anomaly_lab.domain.entities import Experiment, Subset
from anomaly_lab.eval.metrics import roc_auc, timing_summary
from anomaly_lab.eval.pixel import PixelAccumulator
from anomaly_lab.eval.threshold import SampleVerdict
from anomaly_lab.map_files import read_map
from anomaly_lab.schemas import API_MODEL_CONFIG

THRESHOLD_RULE = "foreground probability >= 0.5"
THRESHOLD = 0.5
BOUNDARY_TOLERANCE_PX = 2
"""How far a predicted boundary pixel may be from the true boundary and still match."""
SMALL_REGION_FRACTION = 0.01
"""A present image whose region covers at most this share of the frame is a small region."""


class SegmentationEvalError(ValueError):
    """A stored prediction cannot be read against its truth."""


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    """Binary dilation by a (2r+1)-square, separably, without scipy."""
    if radius <= 0:
        return mask.copy()
    out = mask.copy()
    for axis in (0, 1):
        grown = out.copy()
        for shift in range(1, radius + 1):
            grown |= _shifted(out, shift, axis) | _shifted(out, -shift, axis)
        out = grown
    return out


def _shifted(mask: np.ndarray, shift: int, axis: int) -> np.ndarray:
    result = np.zeros_like(mask)
    if axis == 0:
        if shift > 0:
            result[shift:] = mask[:-shift]
        else:
            result[:shift] = mask[-shift:]
    elif shift > 0:
        result[:, shift:] = mask[:, :-shift]
    else:
        result[:, :shift] = mask[:, -shift:]
    return result


def boundary(mask: np.ndarray) -> np.ndarray:
    """Foreground pixels with a background 4- or 8-neighbour, or on the frame's edge."""
    padded = np.pad(mask, 1, constant_values=False)
    interior = ~_dilate(~padded, 1)[1:-1, 1:-1]
    return np.asarray(mask & ~interior, dtype=bool)


@dataclass
class SegmentationAccumulator:
    """Pooled counts for one subset. Add images, then read `metrics`."""

    true_positive: int = 0
    false_positive: int = 0
    false_negative: int = 0
    predicted_boundary: int = 0
    predicted_boundary_matched: int = 0
    true_boundary: int = 0
    true_boundary_matched: int = 0
    present: int = 0
    present_found: int = 0
    absent: int = 0
    absent_flagged: int = 0
    small: int = 0
    small_found: int = 0
    unlabeled: int = 0
    without_prediction: int = 0
    map_images: int = 0
    pixels: PixelAccumulator = field(default_factory=lambda: PixelAccumulator(vmin=0.0, vmax=1.0))
    scores: list[float] = field(default_factory=list)
    presence: list[bool] = field(default_factory=list)
    milliseconds: list[float] = field(default_factory=list)

    def add(
        self,
        truth: np.ndarray,
        predicted: np.ndarray,
        *,
        score: float,
        inference_ms: float,
        probability: np.ndarray | None = None,
    ) -> None:
        """Fold one answered image in; `probability` is its stored map, when there is one."""
        if truth.shape != predicted.shape:
            msg = f"prediction of shape {predicted.shape} against truth of shape {truth.shape}"
            raise SegmentationEvalError(msg)
        if probability is not None:
            if probability.shape != truth.shape:
                msg = f"map of shape {probability.shape} against truth of shape {truth.shape}"
                raise SegmentationEvalError(msg)
            # No regions: AU-PRO is an anomaly metric, and skipping it skips the union-find
            # over what may be most of the frame.
            self.pixels.add(probability, truth, regions=())
            self.map_images += 1
        is_present = bool(truth.any())
        self.true_positive += int(np.count_nonzero(truth & predicted))
        self.false_positive += int(np.count_nonzero(~truth & predicted))
        self.false_negative += int(np.count_nonzero(truth & ~predicted))

        true_edge, predicted_edge = boundary(truth), boundary(predicted)
        self.true_boundary += int(np.count_nonzero(true_edge))
        self.predicted_boundary += int(np.count_nonzero(predicted_edge))
        near_true = _dilate(true_edge, BOUNDARY_TOLERANCE_PX)
        near_predicted = _dilate(predicted_edge, BOUNDARY_TOLERANCE_PX)
        self.predicted_boundary_matched += int(np.count_nonzero(predicted_edge & near_true))
        self.true_boundary_matched += int(np.count_nonzero(true_edge & near_predicted))

        if is_present:
            found = bool((truth & predicted).any())
            self.present += 1
            self.present_found += found
            if truth.mean() <= SMALL_REGION_FRACTION:
                self.small += 1
                self.small_found += found
        else:
            self.absent += 1
            self.absent_flagged += bool(predicted.any())
        self.scores.append(score)
        self.presence.append(is_present)
        self.milliseconds.append(inference_ms)

    def metrics(self) -> dict[str, Any]:
        union = self.true_positive + self.false_positive + self.false_negative
        precision = _ratio(self.predicted_boundary_matched, self.predicted_boundary)
        recall = _ratio(self.true_boundary_matched, self.true_boundary)
        return {
            "threshold_rule": THRESHOLD_RULE,
            "threshold": THRESHOLD,
            "images": {
                "present": self.present,
                "absent": self.absent,
                "unlabeled": self.unlabeled,
                "without_prediction": self.without_prediction,
            },
            "foreground_iou": _ratio(self.true_positive, union),
            "foreground_dice": _ratio(2 * self.true_positive, union + self.true_positive),
            "boundary_f1": (
                None
                if precision is None or recall is None or precision + recall == 0
                else 2 * precision * recall / (precision + recall)
            ),
            "boundary_tolerance_px": BOUNDARY_TOLERANCE_PX,
            "pixel_average_precision": self.pixels.average_precision(),
            "pixel_roc_auc": self.pixels.roc_auc(),
            "pixel_map_images": self.map_images,
            "image_present_recall": _ratio(self.present_found, self.present),
            "image_absent_false_positive_rate": _ratio(self.absent_flagged, self.absent),
            "image_small_region_recall": _ratio(self.small_found, self.small),
            "small_region_fraction": SMALL_REGION_FRACTION,
            "image_presence_roc_auc": roc_auc(
                np.array(self.presence, dtype=np.int64), np.array(self.scores, dtype=np.float64)
            ),
            "timing": timing_summary(np.array(self.milliseconds, dtype=np.float64)),
        }


def _ratio(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else numerator / denominator


def probability_map(image: ScoredImage) -> np.ndarray | None:
    """The stored foreground probability, or `None` when the method wrote no map.

    Pixels a region crop left uncovered are `NaN`; they read as background everywhere.
    """
    if image.map_path is None or not Path(image.map_path).is_file():
        return None
    return read_map(image.map_path)


def predicted_mask(
    image: ScoredImage, maps_dir: Path, probability: np.ndarray | None = None
) -> np.ndarray | None:
    """The method's own mask if it wrote one, else its map cut at the rule; `None` if neither.

    `probability` is the image's map when the caller has already loaded it. Pixels a region
    crop left uncovered (`NaN` in the map) are background.
    """
    written = maps_dir / f"{image.image_id}.mask.png"
    if written.is_file():
        with Image.open(written) as opened:
            return np.asarray(opened.convert("L")) > 0
    if probability is None:
        probability = probability_map(image)
    if probability is None:
        return None
    return np.asarray(np.nan_to_num(probability, nan=0.0) >= THRESHOLD, dtype=bool)


def _truths(
    conn: sqlite3.Connection, experiment: Experiment, images: Sequence[ScoredImage]
) -> dict[int, ClassTruth]:
    if experiment.target_label is None:
        msg = f"experiment {experiment.id} has no target class to evaluate"
        raise SegmentationEvalError(msg)
    return resolve_class_truth(
        conn, experiment.dataset_id, [image.image_id for image in images], experiment.target_label
    )


def digest(
    experiment: Experiment, images: Sequence[ScoredImage], truths: dict[int, ClassTruth]
) -> str:
    """What the metrics were computed against: the class, and each image's pinned answer."""
    lines = [f"class-truth-v1:{experiment.target_label}"]
    for image in sorted(images, key=lambda item: item.image_id):
        truth = truths.get(image.image_id)
        lines.append(f"image:{image.image_id}:{truth.identity if truth else 'unlabeled'}")
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()


def evaluate(
    conn: sqlite3.Connection, experiment: Experiment
) -> tuple[dict[Subset, dict[str, Any]], dict[Subset, str]]:
    """Every scored subset's metrics and ground-truth digest. Stores nothing."""
    images = results_repo.list_scored_images(conn, experiment.id)
    truths = _truths(conn, experiment, images)
    maps_dir = Path(experiment.artifact_dir) / "maps"
    computed: dict[Subset, dict[str, Any]] = {}
    digests: dict[Subset, str] = {}
    for subset in results_repo.scored_subsets(conn, experiment.id):
        in_subset = [image for image in images if image.subset is subset]
        accumulator = SegmentationAccumulator()
        for image in in_subset:
            truth = truths.get(image.image_id)
            if truth is None:
                accumulator.unlabeled += 1
                continue
            probability = probability_map(image)
            predicted = predicted_mask(image, maps_dir, probability)
            if predicted is None:
                accumulator.without_prediction += 1
                continue
            region = load_class_mask(truth)
            try:
                accumulator.add(
                    region,
                    predicted,
                    score=image.score,
                    inference_ms=image.inference_ms,
                    probability=probability,
                )
            except SegmentationEvalError as exc:
                raise SegmentationEvalError(f"image {image.image_id}: {exc}") from exc
        computed[subset] = accumulator.metrics()
        digests[subset] = digest(experiment, in_subset, truths)
    return computed, digests


def current_digest(conn: sqlite3.Connection, experiment: Experiment, subset: Subset) -> str:
    """The digest a subset's metrics would carry now; reads metadata only, never pixels."""
    images = results_repo.list_scored_images(conn, experiment.id, subset=subset)
    return digest(experiment, images, _truths(conn, experiment, images))


# ---------------------------------------------------------------------------- per sample

LOW_IOU_BELOW = 0.5
"""A present sample found with less overlap than this is a `low_iou` outcome, not a `hit`."""
MISTAKES = ("miss", "false_presence", "low_iou")


class SegmentationVerdict(SampleVerdict):
    """One sample as the few-shot gallery shows it: its outcome, and its overlap."""

    iou: float | None = Field(
        default=None,
        description="Intersection over union over the sample's answered images; null when absent.",
    )


class SegmentationOutcomes(BaseModel):
    """Every scored sample of a subset, classified by what its prediction did to its truth."""

    model_config = API_MODEL_CONFIG

    threshold_rule: str = THRESHOLD_RULE
    low_iou_below: float = LOW_IOU_BELOW
    samples: list[SegmentationVerdict] = Field(default_factory=list)


def _outcome(present: bool, predicted: bool, iou: float | None) -> str:
    if present:
        if iou is None or iou == 0.0:
            return "miss"
        return "hit" if iou >= LOW_IOU_BELOW else "low_iou"
    return "false_presence" if predicted else "correct_absence"


def sample_outcomes(
    conn: sqlite3.Connection, experiment: Experiment, subset: Subset | None
) -> SegmentationOutcomes:
    """Per sample: `hit`, `low_iou`, `miss`, `false_presence`, `correct_absence` or `unlabeled`.

    Computed on request from the stored maps, like the anomaly threshold report, and ranked by
    presence score. A sample's images are pooled: it is present when any answered image
    shows the class, and its IoU is over all of its answered images. Images whose truth does
    not answer are left out of the pool; a sample with none answered is `unlabeled`.
    """
    images = results_repo.list_scored_images(conn, experiment.id, subset=subset)
    samples = results_repo.list_scored_samples(conn, experiment.id, subset=subset)
    truths = _truths(conn, experiment, images)
    maps_dir = Path(experiment.artifact_dir) / "maps"

    pooled: dict[int, list[int]] = {}
    for image in images:
        truth = truths.get(image.image_id)
        predicted = predicted_mask(image, maps_dir)
        if truth is None or predicted is None:
            continue
        region = load_class_mask(truth)
        counts = pooled.setdefault(image.sample_id, [0, 0, 0, 0])
        counts[0] += int(np.count_nonzero(region & predicted))
        counts[1] += int(np.count_nonzero(~region & predicted))
        counts[2] += int(np.count_nonzero(region & ~predicted))
        counts[3] += int(bool(region.any()))

    verdicts: list[SegmentationVerdict] = []
    for sample in samples:
        found = pooled.get(sample.sample_id)
        if found is None:
            outcome, iou, predicted_any = "unlabeled", None, False
        else:
            hit, extra, missed, shows = found
            present = shows > 0
            iou = hit / (hit + extra + missed) if present else None
            predicted_any = hit + extra > 0
            outcome = _outcome(present, predicted_any, iou)
        verdicts.append(
            SegmentationVerdict(
                sample_id=sample.sample_id,
                group_key=sample.group_key,
                external_id=sample.external_id,
                label=sample.label,
                notes=sample.notes,
                score=sample.agg_score,
                predicted_defect=predicted_any,
                outcome=outcome,
                iou=iou,
            )
        )
    verdicts.sort(key=lambda verdict: verdict.score, reverse=True)
    return SegmentationOutcomes(samples=verdicts)
