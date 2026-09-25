"""Metrics for object detection: every pinned class, as boxes, by COCO's protocol (ADR-0039).

The inputs are what is already stored — each image's detections as the method wrote them
(`maps/<id>.instances.json`, source frame) and its object instances resolved over the run's
pinned classes — so this module, like every evaluator, never imports a model and never
re-runs one.

- **COCO's matching and AP.** Per image and class, detections most confident first are each
  matched to the unmatched truth box of highest IoU, if that IoU reaches the threshold. That
  is done at each of ten IoU thresholds, 0.50 to 0.95. AP is the 101-point interpolated area
  under the precision envelope; the headline `ap` averages it over the thresholds and then
  over the classes with truth.
- **Bounded memory.** Each class keeps one confidence and one row of ten matched flags per
  detection, and a count of its truth boxes. That is linear in detections, which the write
  seam caps at `MAX_INSTANCES_PER_IMAGE` an image, and never in pixels.
- **Per image, not per sample.** A sample-level rule for boxes on a multi-channel part has
  not been decided (ADR-0039), so every count here is of images.
- **Unlabelled images are excluded and counted** (`images.unlabeled`), and a truth box of a
  class the run was not created with is ignored and counted (`ignored_instances`).
- **No confidence is cut.** AP and recall read a run's detections in its own confidence order,
  so nothing here needs a per-run rule, and ADR-0028 holds by construction.
- **A metric that cannot be computed is `None`.** A class with no truth box in a subset has no
  AP and no recall — a detection of it is a false positive that no recall can be measured
  against. A class with truth and no detection has AP 0, which is measured, not missing.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from anomaly_lab.annotations.class_truth import BoxTruth, load_boxes, resolve_box_truth
from anomaly_lab.db.repositories import results as results_repo
from anomaly_lab.db.repositories.results import ScoredImage
from anomaly_lab.domain.entities import Experiment, Subset
from anomaly_lab.eval.metrics import timing_summary
from anomaly_lab.models.base import MAX_INSTANCES_PER_IMAGE, PredictedInstance, TargetBox

IOU_THRESHOLDS: tuple[float, ...] = tuple(round(0.5 + 0.05 * step, 2) for step in range(10))
"""COCO's ten IoU thresholds. AP50 is the first and AP75 the sixth."""

RECALL_POINTS = np.linspace(0.0, 1.0, 101)
"""Where the precision envelope is read — COCO's 101 recall points."""

INSTANCES_SUFFIX = ".instances.json"
"""Where `InferContext.write_instances` puts an image's detections, beside its anomaly map.
Named here as well because the evaluation layer never imports a model module."""


class DetectionEvalError(ValueError):
    """A stored detection file cannot be read against its truth."""


def box_iou(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """IoU of every box in `left` `(n, 4)` against every box in `right` `(m, 4)`, as `(n, m)`.

    Boxes are pixel-edge `(x0, y0, x1, y1)`, so a box's area is `(x1 - x0) * (y1 - y0)`.
    """
    if len(left) == 0 or len(right) == 0:
        return np.zeros((len(left), len(right)), dtype=np.float64)
    x0 = np.maximum(left[:, None, 0], right[None, :, 0])
    y0 = np.maximum(left[:, None, 1], right[None, :, 1])
    x1 = np.minimum(left[:, None, 2], right[None, :, 2])
    y1 = np.minimum(left[:, None, 3], right[None, :, 3])
    overlap = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    area_left = (left[:, 2] - left[:, 0]) * (left[:, 3] - left[:, 1])
    area_right = (right[:, 2] - right[:, 0]) * (right[:, 3] - right[:, 1])
    union = area_left[:, None] + area_right[None, :] - overlap
    return np.where(union > 0, overlap / np.where(union > 0, union, 1.0), 0.0)


def match(truth: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    """Which detections are true positives at each IoU threshold, `(n, thresholds)` bool.

    `predicted` is one image's boxes of one class, most confident first; each takes the
    unmatched truth box it overlaps most, if that overlap reaches the threshold.
    """
    matched = np.zeros((len(predicted), len(IOU_THRESHOLDS)), dtype=bool)
    if len(truth) == 0 or len(predicted) == 0:
        return matched
    overlaps = box_iou(predicted, truth)
    for column, threshold in enumerate(IOU_THRESHOLDS):
        taken = np.zeros(len(truth), dtype=bool)
        for row in range(len(predicted)):
            candidates = np.where(taken, -1.0, overlaps[row])
            best = int(np.argmax(candidates))
            if candidates[best] >= threshold:
                taken[best] = True
                matched[row, column] = True
    return matched


def average_precision(confidence: np.ndarray, matched: np.ndarray, truth: int) -> float | None:
    """COCO's 101-point interpolated AP for one class at one threshold; `None` with no truth."""
    if truth == 0:
        return None
    if len(confidence) == 0:
        return 0.0
    order = np.argsort(-confidence, kind="mergesort")
    hits = matched[order]
    true_positives = np.cumsum(hits)
    false_positives = np.cumsum(~hits)
    recall = true_positives / truth
    precision = true_positives / (true_positives + false_positives)
    envelope = np.maximum.accumulate(precision[::-1])[::-1]
    at = np.searchsorted(recall, RECALL_POINTS, side="left")
    sampled = np.zeros(len(RECALL_POINTS), dtype=np.float64)
    reached = at < len(envelope)
    sampled[reached] = envelope[at[reached]]
    return float(sampled.mean())


@dataclass
class _ClassTally:
    truth: int = 0
    confidence: list[float] = field(default_factory=list)
    matched: list[np.ndarray] = field(default_factory=list)


@dataclass
class DetectionAccumulator:
    """One subset's detections and truth, per class. Add images, then read `metrics`."""

    classes: tuple[str, ...]
    labelled: int = 0
    unlabeled: int = 0
    without_prediction: int = 0
    ignored_instances: int = 0
    milliseconds: list[float] = field(default_factory=list)
    _tallies: dict[str, _ClassTally] = field(init=False)

    def __post_init__(self) -> None:
        self._tallies = {key: _ClassTally() for key in self.classes}

    def add(
        self,
        truth: Sequence[TargetBox],
        predicted: Sequence[PredictedInstance],
        *,
        inference_ms: float,
    ) -> None:
        if len(predicted) > MAX_INSTANCES_PER_IMAGE:
            msg = (
                f"{len(predicted)} detections of one image; at most "
                f"{MAX_INSTANCES_PER_IMAGE} are counted"
            )
            raise DetectionEvalError(msg)
        for instance in predicted:
            if instance.label_key not in self._tallies:
                msg = f"a detection of {instance.label_key!r}, which the run does not pin"
                raise DetectionEvalError(msg)
        for key, tally in self._tallies.items():
            boxes = np.array([item.box for item in truth if item.label_key == key], dtype=float)
            found = sorted(
                (item for item in predicted if item.label_key == key),
                key=lambda item: -item.confidence,
            )
            guesses = np.array([item.box for item in found], dtype=float).reshape(-1, 4)
            tally.truth += len(boxes)
            tally.confidence.extend(item.confidence for item in found)
            tally.matched.extend(match(boxes.reshape(-1, 4), guesses))
        self.ignored_instances += sum(1 for item in truth if item.label_key not in self._tallies)
        self.labelled += 1
        self.milliseconds.append(inference_ms)

    def metrics(self) -> dict[str, Any]:
        per_threshold: dict[str, list[float | None]] = {}
        recall: dict[str, list[float | None]] = {}
        for key, tally in self._tallies.items():
            confidence = np.array(tally.confidence, dtype=np.float64)
            matched = (
                np.stack(tally.matched)
                if tally.matched
                else np.zeros((0, len(IOU_THRESHOLDS)), dtype=bool)
            )
            per_threshold[key] = [
                average_precision(confidence, matched[:, column], tally.truth)
                for column in range(len(IOU_THRESHOLDS))
            ]
            recall[key] = [
                None if tally.truth == 0 else float(matched[:, column].sum()) / tally.truth
                for column in range(len(IOU_THRESHOLDS))
            ]
        per_class_ap = {key: _mean(values) for key, values in per_threshold.items()}
        per_class_ap50 = {key: values[0] for key, values in per_threshold.items()}
        per_class_ap75 = {key: values[5] for key, values in per_threshold.items()}
        per_class_recall = {key: _mean(values) for key, values in recall.items()}
        per_class_recall50 = {key: values[0] for key, values in recall.items()}
        return {
            "classes": list(self.classes),
            "iou_thresholds": list(IOU_THRESHOLDS),
            "images": {
                "labelled": self.labelled,
                "unlabeled": self.unlabeled,
                "without_prediction": self.without_prediction,
            },
            "ap": _mean(per_class_ap.values()),
            "ap50": _mean(per_class_ap50.values()),
            "ap75": _mean(per_class_ap75.values()),
            "recall": _mean(per_class_recall.values()),
            "recall50": _mean(per_class_recall50.values()),
            "per_class_ap": per_class_ap,
            "per_class_ap50": per_class_ap50,
            "per_class_ap75": per_class_ap75,
            "per_class_recall": per_class_recall,
            "per_class_recall50": per_class_recall50,
            "truth_instances": {key: tally.truth for key, tally in self._tallies.items()},
            "predicted_instances": {
                key: len(tally.confidence) for key, tally in self._tallies.items()
            },
            "ignored_instances": self.ignored_instances,
            "timing": timing_summary(np.array(self.milliseconds, dtype=np.float64)),
        }


def _mean(values: Any) -> float | None:
    known = [value for value in values if value is not None]
    return None if not known else float(sum(known) / len(known))


def instances_path(maps_dir: Path, image_id: int) -> Path:
    """Where an image's detections are, whether or not they were written."""
    return maps_dir / f"{image_id}{INSTANCES_SUFFIX}"


def read_instances(path: Path) -> list[PredictedInstance] | None:
    """A stored detection file, or `None` if there is none."""
    if not path.is_file():
        return None
    stored = json.loads(path.read_text(encoding="utf-8"))
    found: list[PredictedInstance] = []
    for entry in stored["instances"]:
        x0, y0, x1, y1 = (float(value) for value in entry["box"])
        found.append(
            PredictedInstance(str(entry["label_key"]), (x0, y0, x1, y1), float(entry["confidence"]))
        )
    return found


def _classes(experiment: Experiment) -> tuple[str, ...]:
    if not experiment.classes:
        raise DetectionEvalError(f"experiment {experiment.id} pins no classes to evaluate")
    return tuple(experiment.classes)


def _truths(
    conn: sqlite3.Connection, experiment: Experiment, images: Sequence[ScoredImage]
) -> dict[int, BoxTruth]:
    return resolve_box_truth(
        conn, experiment.dataset_id, [image.image_id for image in images], _classes(experiment)
    )


def digest(
    experiment: Experiment, images: Sequence[ScoredImage], truths: dict[int, BoxTruth]
) -> str:
    """What the metrics were computed against: the classes, and each image's pinned answer."""
    lines = [f"box-truth-v1:{','.join(_classes(experiment))}"]
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
        accumulator = DetectionAccumulator(classes)
        for image in in_subset:
            truth = truths.get(image.image_id)
            if truth is None:
                accumulator.unlabeled += 1
                continue
            predicted = read_instances(instances_path(maps_dir, image.image_id))
            if predicted is None:
                accumulator.without_prediction += 1
                continue
            try:
                accumulator.add(load_boxes(truth), predicted, inference_ms=image.inference_ms)
            except DetectionEvalError as exc:
                raise DetectionEvalError(f"image {image.image_id}: {exc}") from exc
        computed[subset] = accumulator.metrics()
        digests[subset] = digest(experiment, in_subset, truths)
    return computed, digests


def current_digest(conn: sqlite3.Connection, experiment: Experiment, subset: Subset) -> str:
    """The digest a subset's metrics would carry now; reads metadata only, never pixels."""
    images = results_repo.list_scored_images(conn, experiment.id, subset=subset)
    return digest(experiment, images, _truths(conn, experiment, images))
