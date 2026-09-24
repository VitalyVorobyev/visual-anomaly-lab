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


def predicted_labels(image: ScoredImage, maps_dir: Path) -> np.ndarray | None:
    """The label map the method wrote for an image, or `None` if it wrote none."""
    path = maps_dir / f"{image.image_id}{LABEL_MAP_SUFFIX}"
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
    return resolve_label_truth(
        conn, experiment.dataset_id, [image.image_id for image in images], _classes(experiment)
    )


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
