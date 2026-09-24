"""Metrics for supervised semantic segmentation: every pinned class, per pixel (ADR-0039).

The inputs are what is already stored — each image's label map as the method wrote it, and
its truth resolved over the run's pinned classes — so this module, like the anomaly runner,
never imports a model and never re-runs one.

- **One confusion matrix per subset, in constant memory.** Rows are the true class, columns
  the predicted one, background first. Every image adds its pixel counts and is dropped;
  nothing per pixel is kept across images. Every metric is read off the matrix.
- **Per image, not per sample.** A sample-level rule for classes on a multi-channel part
  has not been decided (ADR-0039), so every count here is of images.
- **Unlabelled images are excluded and counted.** An image whose truth does not answer for
  every pinned class is neither right nor wrong (`images.unlabeled`), and a pixel drawn in a
  class the run was not created with is ignored and counted (`ignored_pixels`).
- **Nothing is thresholded.** A label map is the method's own decision, read as written;
  there is no cut to resolve per run, so ADR-0028 holds by construction.
- **A metric that cannot be computed is `None`.** A class absent from both truth and
  prediction in a subset has no IoU; one absent from truth has no accuracy.
- **Background is a class of the matrix, not of the means.** Pixel accuracy and
  frequency-weighted IoU count it; mean IoU and mean class accuracy average the annotation
  classes, because a background that covers most of every frame would otherwise carry the
  mean. Its own IoU is reported beside them.
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

from anomaly_lab.annotations.class_truth import (
    LabelTruth,
    load_label_map,
    resolve_label_truth,
)
from anomaly_lab.db.repositories import results as results_repo
from anomaly_lab.db.repositories.results import ScoredImage
from anomaly_lab.domain.entities import Experiment, Subset
from anomaly_lab.eval.metrics import timing_summary
from anomaly_lab.eval.segmentation import LOW_IOU_BELOW, SegmentationOutcomes, SegmentationVerdict
from anomaly_lab.models.base import IGNORE_INDEX

BACKGROUND = "background"
LABEL_MAP_SUFFIX = ".labels.png"
"""Where `InferContext.write_label_map` puts an image's label map, beside its anomaly map.
Named here as well because the evaluation layer never imports a model module."""


class SemanticEvalError(ValueError):
    """A stored label map cannot be read against its truth."""


@dataclass
class ConfusionAccumulator:
    """One subset's pooled confusion matrix. Add images, then read `metrics`."""

    classes: tuple[str, ...]
    counts: np.ndarray = field(init=False)
    labelled: int = 0
    unlabeled: int = 0
    without_prediction: int = 0
    ignored_pixels: int = 0
    milliseconds: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        size = len(self.classes) + 1
        self.counts = np.zeros((size, size), dtype=np.int64)

    def add(self, truth: np.ndarray, predicted: np.ndarray, *, inference_ms: float) -> None:
        if truth.shape != predicted.shape:
            msg = f"label map of shape {predicted.shape} against truth of shape {truth.shape}"
            raise SemanticEvalError(msg)
        size = len(self.classes) + 1
        if predicted.size and int(predicted.max()) >= size:
            msg = f"label map holds index {int(predicted.max())}; the run has {size - 1} classes"
            raise SemanticEvalError(msg)
        valid = truth != IGNORE_INDEX
        pairs = truth[valid].astype(np.int64) * size + predicted[valid].astype(np.int64)
        self.counts += np.bincount(pairs, minlength=size * size).reshape(size, size)
        self.ignored_pixels += int(truth.size - np.count_nonzero(valid))
        self.labelled += 1
        self.milliseconds.append(inference_ms)

    def metrics(self) -> dict[str, Any]:
        counts = self.counts
        hits = np.diag(counts)
        truth_totals = counts.sum(axis=1)
        predicted_totals = counts.sum(axis=0)
        unions = truth_totals + predicted_totals - hits
        total = int(counts.sum())

        iou = [_ratio(int(hits[i]), int(unions[i])) for i in range(len(hits))]
        accuracy = [_ratio(int(hits[i]), int(truth_totals[i])) for i in range(len(hits))]
        named_iou = dict(zip(self.classes, iou[1:], strict=True))
        named_accuracy = dict(zip(self.classes, accuracy[1:], strict=True))
        frequency_weighted = (
            None
            if total == 0
            else sum(
                int(truth_totals[i]) * value
                for i, value in enumerate(iou)
                if value is not None and truth_totals[i] > 0
            )
            / total
        )
        return {
            "classes": list(self.classes),
            "images": {
                "labelled": self.labelled,
                "unlabeled": self.unlabeled,
                "without_prediction": self.without_prediction,
            },
            "mean_iou": _mean(named_iou.values()),
            "per_class_iou": named_iou,
            "background_iou": iou[0],
            "pixel_accuracy": _ratio(int(hits.sum()), total),
            "mean_class_accuracy": _mean(named_accuracy.values()),
            "per_class_accuracy": named_accuracy,
            "frequency_weighted_iou": frequency_weighted,
            "confusion": {
                "classes": [BACKGROUND, *self.classes],
                "counts": counts.tolist(),
            },
            "ignored_pixels": self.ignored_pixels,
            "timing": timing_summary(np.array(self.milliseconds, dtype=np.float64)),
        }


def _ratio(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else numerator / denominator


def _mean(values: Any) -> float | None:
    known = [value for value in values if value is not None]
    return None if not known else float(sum(known) / len(known))


def label_map_path(maps_dir: Path, image_id: int) -> Path:
    """Where an image's predicted label map is, whether or not it was written."""
    return maps_dir / f"{image_id}{LABEL_MAP_SUFFIX}"


def predicted_labels(image: ScoredImage, maps_dir: Path) -> np.ndarray | None:
    """The label map the method wrote for an image, or `None` if it wrote none."""
    return read_label_map(label_map_path(maps_dir, image.image_id))


def read_label_map(path: Path) -> np.ndarray | None:
    """A stored label map as `uint8` class indices, or `None` if there is none."""
    if not path.is_file():
        return None
    with Image.open(path) as opened:
        return np.asarray(opened.convert("L"))


def _classes(experiment: Experiment) -> tuple[str, ...]:
    if not experiment.classes:
        raise SemanticEvalError(f"experiment {experiment.id} pins no classes to evaluate")
    return tuple(experiment.classes)


def _truths(
    conn: sqlite3.Connection, experiment: Experiment, images: Sequence[ScoredImage]
) -> dict[int, LabelTruth]:
    return _truths_for(conn, experiment, [image.image_id for image in images])


def _truths_for(
    conn: sqlite3.Connection, experiment: Experiment, image_ids: Sequence[int]
) -> dict[int, LabelTruth]:
    return resolve_label_truth(conn, experiment.dataset_id, image_ids, _classes(experiment))


def digest(
    experiment: Experiment, images: Sequence[ScoredImage], truths: dict[int, LabelTruth]
) -> str:
    """What the metrics were computed against: the classes, and each image's pinned answer."""
    lines = [f"label-truth-v1:{','.join(_classes(experiment))}"]
    for image in sorted(images, key=lambda item: item.image_id):
        truth = truths.get(image.image_id)
        lines.append(f"image:{image.image_id}:{truth.identity if truth else 'unlabeled'}")
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()


def evaluate(
    conn: sqlite3.Connection, experiment: Experiment
) -> tuple[dict[Subset, dict[str, Any]], dict[Subset, str]]:
    """Every scored subset's metrics and ground-truth digest. Stores nothing."""
    classes = _classes(experiment)
    images = results_repo.list_scored_images(conn, experiment.id)
    truths = _truths(conn, experiment, images)
    maps_dir = Path(experiment.artifact_dir) / "maps"
    computed: dict[Subset, dict[str, Any]] = {}
    digests: dict[Subset, str] = {}
    for subset in results_repo.scored_subsets(conn, experiment.id):
        in_subset = [image for image in images if image.subset is subset]
        accumulator = ConfusionAccumulator(classes)
        for image in in_subset:
            truth = truths.get(image.image_id)
            if truth is None:
                accumulator.unlabeled += 1
                continue
            predicted = predicted_labels(image, maps_dir)
            if predicted is None:
                accumulator.without_prediction += 1
                continue
            try:
                accumulator.add(
                    load_label_map(truth, classes), predicted, inference_ms=image.inference_ms
                )
            except SemanticEvalError as exc:
                raise SemanticEvalError(f"image {image.image_id}: {exc}") from exc
        computed[subset] = accumulator.metrics()
        digests[subset] = digest(experiment, in_subset, truths)
    return computed, digests


def current_digest(conn: sqlite3.Connection, experiment: Experiment, subset: Subset) -> str:
    """The digest a subset's metrics would carry now; reads metadata only, never pixels."""
    images = results_repo.list_scored_images(conn, experiment.id, subset=subset)
    return digest(experiment, images, _truths(conn, experiment, images))


# ---------------------------------------------------------------------------- per sample

LABEL_MAP_RULE = "the method's label map, as written"
"""What decided each pixel's class: nothing is cut, so the rule printed is that there is none."""


def _sample_outcome(counts: np.ndarray) -> tuple[str, float | None, bool]:
    """One sample's outcome, mean IoU and whether it predicted any class, from its matrix.

    `counts` is the sample's pooled confusion matrix, background first. The classes that
    matter are the ones it shows (`truth`) and the ones it was given (`predicted`):

    - shows none: `correct_absence` if nothing was predicted, `false_presence` otherwise;
    - a class it shows was not found anywhere on it: `miss`;
    - a class it does not show was predicted on it: `false_class`;
    - otherwise the mean IoU over its classes decides `hit` or `low_iou`.

    The mean IoU is over the classes shown or predicted, and `None` when it shows none — the
    same absence the few-shot verdict has.
    """
    hits = np.diag(counts)[1:]
    truth = counts.sum(axis=1)[1:]
    predicted = counts.sum(axis=0)[1:]
    shown = truth > 0
    given = predicted > 0
    predicted_any = bool(given.any())
    if not shown.any():
        return ("false_presence" if predicted_any else "correct_absence"), None, predicted_any
    involved = shown | given
    unions = truth + predicted - hits
    iou = float(np.mean(hits[involved] / unions[involved]))
    if bool((hits[shown] == 0).any()):
        return "miss", iou, predicted_any
    if bool((given & ~shown).any()):
        return "false_class", iou, predicted_any
    return ("hit" if iou >= LOW_IOU_BELOW else "low_iou"), iou, predicted_any


def sample_outcomes(
    conn: sqlite3.Connection, experiment: Experiment, subset: Subset | None
) -> SegmentationOutcomes:
    """Per sample: `hit`, `low_iou`, `miss`, `false_class`, `false_presence`,
    `correct_absence` or `unlabeled`, ranked by score.

    Computed on request from the stored label maps, like the few-shot report. A sample's
    labelled images are pooled into one small confusion matrix — `(classes + 1)²` counts,
    never pixels — so its verdict is over every image that answers for every pinned class;
    one with none is `unlabeled`. Nothing is thresholded: the label map is the method's own
    decision.
    """
    classes = _classes(experiment)
    size = len(classes) + 1
    images = results_repo.list_scored_images(conn, experiment.id, subset=subset)
    samples = results_repo.list_scored_samples(conn, experiment.id, subset=subset)
    truths = _truths(conn, experiment, images)
    maps_dir = Path(experiment.artifact_dir) / "maps"

    pooled: dict[int, np.ndarray] = {}
    for image in images:
        truth = truths.get(image.image_id)
        predicted = None if truth is None else predicted_labels(image, maps_dir)
        if truth is None or predicted is None:
            continue
        accumulator = ConfusionAccumulator(classes)
        try:
            accumulator.add(load_label_map(truth, classes), predicted, inference_ms=0.0)
        except SemanticEvalError as exc:
            raise SemanticEvalError(f"image {image.image_id}: {exc}") from exc
        found = pooled.setdefault(image.sample_id, np.zeros((size, size), dtype=np.int64))
        found += accumulator.counts

    verdicts: list[SegmentationVerdict] = []
    for sample in samples:
        counts = pooled.get(sample.sample_id)
        if counts is None:
            outcome, iou, predicted_any = "unlabeled", None, False
        else:
            outcome, iou, predicted_any = _sample_outcome(counts)
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
    return SegmentationOutcomes(threshold_rule=LABEL_MAP_RULE, samples=verdicts)


def label_plane(
    conn: sqlite3.Connection, experiment: Experiment, image_id: int, *, truth: bool
) -> np.ndarray | None:
    """One image's label map for drawing, as float32 class indices in the source frame.

    `truth=False` is what the method wrote; `truth=True` is the image's truth over the run's
    pinned classes, with a pixel of a class the run does not know as NaN — ignored, not
    background. `None` when there is no such map: no prediction, or truth that does not
    answer for every pinned class.
    """
    if truth:
        found = _truths_for(conn, experiment, [image_id]).get(image_id)
        if found is None:
            return None
        labels = load_label_map(found, _classes(experiment)).astype(np.float32)
        labels[labels == IGNORE_INDEX] = np.nan
        return labels
    predicted = read_label_map(label_map_path(Path(experiment.artifact_dir) / "maps", image_id))
    return None if predicted is None else predicted.astype(np.float32)
