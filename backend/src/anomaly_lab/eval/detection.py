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
- **No metric is cut; a verdict is, by one printed rule.** AP and recall read a run's
  detections in its own confidence order and need no cut. A per-sample verdict and a drawn
  box do: each subset resolves one confidence cut by `CUT_RULE` — the confidence that
  maximises F1 at IoU 0.5, every class pooled — and stores it beside the metrics, so the
  screens that draw a verdict read one value and print it (ADR-0028). It is this run's own
  and is never carried to another.
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
from pydantic import BaseModel, Field

from anomaly_lab.annotations.class_truth import BoxTruth, load_boxes, resolve_box_truth
from anomaly_lab.db.repositories import results as results_repo
from anomaly_lab.db.repositories.results import ScoredImage
from anomaly_lab.domain.entities import Experiment, Subset
from anomaly_lab.eval.metrics import timing_summary
from anomaly_lab.eval.threshold import SampleVerdict
from anomaly_lab.models.base import MAX_INSTANCES_PER_IMAGE, PredictedInstance, TargetBox
from anomaly_lab.schemas import API_MODEL_CONFIG

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


def assign(truth: np.ndarray, predicted: np.ndarray, threshold: float) -> list[int | None]:
    """Which truth box each detection takes at one IoU threshold, or `None`.

    `predicted` is one image's boxes of one class, most confident first; each takes the
    unmatched truth box it overlaps most, if that overlap reaches the threshold. Greedy by
    confidence, so the detections above any cut are matched exactly as they would be alone.
    """
    taken_by: list[int | None] = [None] * len(predicted)
    if len(truth) == 0 or len(predicted) == 0:
        return taken_by
    overlaps = box_iou(predicted, truth)
    taken = np.zeros(len(truth), dtype=bool)
    for row in range(len(predicted)):
        candidates = np.where(taken, -1.0, overlaps[row])
        best = int(np.argmax(candidates))
        if candidates[best] >= threshold:
            taken[best] = True
            taken_by[row] = best
    return taken_by


def match(truth: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    """Which detections are true positives at each IoU threshold, `(n, thresholds)` bool."""
    matched = np.zeros((len(predicted), len(IOU_THRESHOLDS)), dtype=bool)
    for column, threshold in enumerate(IOU_THRESHOLDS):
        for row, found in enumerate(assign(truth, predicted, threshold)):
            matched[row, column] = found is not None
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


CUT_IOU = IOU_THRESHOLDS[0]
"""The IoU a verdict matches at: AP50's."""

CUT_RULE = "the confidence that maximises F1 at IoU 0.5 over the subset, every class pooled"
"""How each subset's confidence cut is resolved; printed beside the value it produced."""

UNRESOLVED_RULE = "no cut: the subset has no truth box or no detection, so every detection counts"
"""What is printed when the rule has nothing to maximise over."""


CUT_KEYS = frozenset(
    {"confidence_cut", "cut_rule", "cut_iou", "f1_at_cut", "precision_at_cut", "recall_at_cut"}
)
"""The stored metrics that depend on the run's own cut. A comparison across runs leaves them
out: a confidence means nothing outside its run (ADR-0028)."""


@dataclass(frozen=True)
class ConfidenceCut:
    """One subset's cut and what it achieves there. `value` is `None` when the rule cannot be
    applied — no truth box or no detection — and every stored detection then counts."""

    value: float | None
    f1: float | None = None
    precision: float | None = None
    recall: float | None = None


def resolve_cut(confidence: np.ndarray, matched: np.ndarray, truth: int) -> ConfidenceCut:
    """`CUT_RULE` over pooled detections: their confidences, whether each matched at IoU 0.5,
    and how many truth boxes there are.

    Only a confidence that some detection has is a candidate, and a cut keeps every detection
    at or above it, so tied confidences are kept or dropped together. Among cuts of equal F1
    the highest wins — the fewest detections that do as well.
    """
    if truth == 0 or len(confidence) == 0:
        return ConfidenceCut(None)
    order = np.argsort(-confidence, kind="mergesort")
    ranked = confidence[order]
    true_positives = np.cumsum(matched[order])
    kept = np.arange(1, len(ranked) + 1)
    f1 = 2 * true_positives / (kept + truth)
    ends = np.flatnonzero(np.append(ranked[1:] != ranked[:-1], True))
    best = int(ends[int(np.argmax(f1[ends]))])
    return ConfidenceCut(
        value=float(ranked[best]),
        f1=float(f1[best]),
        precision=float(true_positives[best] / kept[best]),
        recall=float(true_positives[best] / truth),
    )


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

    def cut(self) -> ConfidenceCut:
        """This subset's confidence cut, over every class's detections pooled."""
        tallies = list(self._tallies.values())
        confidence = np.array(
            [value for tally in tallies for value in tally.confidence], dtype=np.float64
        )
        matched = np.array([row[0] for tally in tallies for row in tally.matched], dtype=bool)
        return resolve_cut(confidence, matched, sum(tally.truth for tally in tallies))

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
        cut = self.cut()
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
            "confidence_cut": cut.value,
            "cut_rule": CUT_RULE if cut.value is not None else UNRESOLVED_RULE,
            "cut_iou": CUT_IOU,
            "f1_at_cut": cut.f1,
            "precision_at_cut": cut.precision,
            "recall_at_cut": cut.recall,
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


# ------------------------------------------------------------- per image and per sample


def stored_cuts(conn: sqlite3.Connection, experiment: Experiment) -> dict[Subset, float | None]:
    """Each evaluated subset's confidence cut, as the evaluator stored it.

    Read rather than recomputed, so the gallery's verdicts and the sample page's boxes are
    drawn at the very value Overview prints. A subset with no metric set has no cut.
    """
    cuts: dict[Subset, float | None] = {}
    for found in results_repo.list_metric_sets(conn, experiment.id):
        value = found.metrics.get("confidence_cut")
        cuts[found.subset] = float(value) if isinstance(value, int | float) else None
    return cuts


def rule_for(cut: float | None) -> str:
    """The sentence printed beside a cut."""
    if cut is None:
        return UNRESOLVED_RULE
    return f"confidence ≥ {cut:.4f}, {CUT_RULE}"


@dataclass(frozen=True)
class ImageMatch:
    """One image's boxes at a cut: which detections are kept and matched, which truth found.

    Index-aligned with the lists it was built from. A truth box of a class the run does not
    pin is never matched and never missed; the reader counts it apart.
    """

    kept: list[bool]
    matched: list[bool]
    found: list[bool]
    pinned: list[bool]

    @property
    def true_positives(self) -> int:
        return sum(1 for kept, hit in zip(self.kept, self.matched, strict=True) if kept and hit)

    @property
    def false_positives(self) -> int:
        return sum(1 for kept, hit in zip(self.kept, self.matched, strict=True) if kept and not hit)

    @property
    def missed(self) -> int:
        return sum(
            1 for pinned, found in zip(self.pinned, self.found, strict=True) if pinned and not found
        )

    @property
    def truth(self) -> int:
        return sum(self.pinned)


def match_image(
    truth: Sequence[TargetBox],
    predicted: Sequence[PredictedInstance],
    classes: Sequence[str],
    cut: float | None,
) -> ImageMatch:
    """Match one image's kept detections to its truth at `CUT_IOU`, class by class.

    A detection is kept when its confidence reaches the cut, or always when there is none.
    The matching is the evaluator's own greedy rule (`assign`), over the kept detections.
    """
    kept = [cut is None or item.confidence >= cut for item in predicted]
    matched = [False] * len(predicted)
    found = [False] * len(truth)
    pinned = [item.label_key in classes for item in truth]
    for key in classes:
        rows = sorted(
            (
                index
                for index, item in enumerate(predicted)
                if kept[index] and item.label_key == key
            ),
            key=lambda index: -predicted[index].confidence,
        )
        columns = [index for index, item in enumerate(truth) if item.label_key == key]
        boxes = np.array([truth[index].box for index in columns], dtype=float).reshape(-1, 4)
        guesses = np.array([predicted[index].box for index in rows], dtype=float).reshape(-1, 4)
        for row, taken in zip(rows, assign(boxes, guesses, CUT_IOU), strict=True):
            if taken is not None:
                matched[row] = True
                found[columns[taken]] = True
    return ImageMatch(kept=kept, matched=matched, found=found, pinned=pinned)


class PredictedBox(BaseModel):
    """One stored detection, in the source frame, with what the cut made of it."""

    model_config = API_MODEL_CONFIG

    label_key: str
    class_index: int = Field(description="The class's position in the run's pinned classes.")
    box: list[float] = Field(description="Pixel-edge `[x0, y0, x1, y1]`.")
    confidence: float
    kept: bool = Field(description="Whether the confidence reaches the subset's cut.")
    matched: bool = Field(description="Kept, and matched to a truth box of its class at IoU 0.5.")


class TrueBox(BaseModel):
    """One true object instance, in the source frame."""

    model_config = API_MODEL_CONFIG

    label_key: str
    class_index: int | None = Field(
        description="The class's position in the run's pinned classes; null for one it doesn't pin."
    )
    box: list[float] = Field(description="Pixel-edge `[x0, y0, x1, y1]`.")
    found: bool = Field(description="Matched by a kept detection of its class at IoU 0.5.")


class ImageBoxes(BaseModel):
    """One image's stored detections and its true boxes, matched at the subset's cut."""

    model_config = API_MODEL_CONFIG

    image_id: int
    subset: Subset | None
    width: int
    height: int
    classes: list[str]
    iou_threshold: float = CUT_IOU
    confidence_cut: float | None = Field(
        description="The subset's cut, as the evaluator resolved it; null when none could be."
    )
    threshold_rule: str
    predictions: list[PredictedBox] | None = Field(
        description=(
            f"At most {MAX_INSTANCES_PER_IMAGE}, most confident first; null when the method "
            "wrote none for this image."
        )
    )
    truth: list[TrueBox] | None = Field(
        description="Null when the image's truth does not answer for every pinned class."
    )


def image_boxes(
    conn: sqlite3.Connection, experiment: Experiment, image: ScoredImage
) -> ImageBoxes | None:
    """What the sample page draws for one scored image; `None` when there is nothing to draw —
    no stored detections and no truth."""
    classes = _classes(experiment)
    predicted = read_instances(
        instances_path(Path(experiment.artifact_dir) / "maps", image.image_id)
    )
    resolved = _truths(conn, experiment, [image]).get(image.image_id)
    if predicted is None and resolved is None:
        return None
    truth = None if resolved is None else load_boxes(resolved)
    stored = predicted or []
    for item in stored:
        if item.label_key not in classes:
            msg = f"image {image.image_id}: a detection of {item.label_key!r}, not a pinned class"
            raise DetectionEvalError(msg)
    cut = None if image.subset is None else stored_cuts(conn, experiment).get(image.subset)
    found = match_image(truth or [], stored, classes, cut)
    ordered = sorted(range(len(stored)), key=lambda index: -stored[index].confidence)
    predictions = [
        PredictedBox(
            label_key=stored[index].label_key,
            class_index=classes.index(stored[index].label_key),
            box=list(stored[index].box),
            confidence=stored[index].confidence,
            kept=found.kept[index],
            matched=found.matched[index],
        )
        for index in ordered
    ]
    true_boxes = [
        TrueBox(
            label_key=item.label_key,
            class_index=classes.index(item.label_key) if item.label_key in classes else None,
            box=list(item.box),
            found=found.found[index],
        )
        for index, item in enumerate(truth or [])
    ]
    return ImageBoxes(
        image_id=image.image_id,
        subset=image.subset,
        width=image.width,
        height=image.height,
        classes=list(classes),
        confidence_cut=cut,
        threshold_rule=rule_for(cut),
        predictions=None if predicted is None else predictions,
        truth=None if truth is None else true_boxes,
    )


MISTAKES = ("miss", "false_presence", "mixed")


class DetectionVerdict(SampleVerdict):
    """One sample as the detection gallery shows it, at its subset's cut."""

    matched: int = Field(
        default=0, description="Kept detections matched to truth, over its images."
    )
    missed: int = Field(
        default=0, description="Truth boxes of a pinned class no kept detection found."
    )
    false_positives: int = Field(default=0, description="Kept detections that matched nothing.")


class DetectionOutcomes(BaseModel):
    """Every scored sample of a subset, classified by what its kept boxes did to its truth."""

    model_config = API_MODEL_CONFIG

    threshold_rule: str
    confidence_cut: float | None = Field(
        description="The subset's cut; null when none could be resolved, or over several subsets."
    )
    iou_threshold: float = CUT_IOU
    samples: list[DetectionVerdict] = Field(default_factory=list)


def _outcome(truth: int, missed: int, false_positives: int) -> str:
    if truth == 0:
        return "false_presence" if false_positives else "correct_absence"
    if missed and false_positives:
        return "mixed"
    if missed:
        return "miss"
    return "false_presence" if false_positives else "hit"


def sample_outcomes(
    conn: sqlite3.Connection, experiment: Experiment, subset: Subset | None
) -> DetectionOutcomes:
    """Per sample: `hit`, `miss`, `false_presence`, `mixed`, `correct_absence` or `unlabeled`.

    Computed on request from the stored detections at each image's subset cut, and ranked by
    score. A sample's images that answer for every pinned class and have detections are
    pooled into three counts — never a box list — so the memory is linear in samples:

    - no truth box: `false_presence` if a kept detection is left, else `correct_absence`;
    - a truth box missed and a kept detection that matched nothing: `mixed`;
    - only a miss: `miss`; only an unmatched detection: `false_presence`; neither: `hit`.

    A sample with no such image is `unlabeled`.
    """
    classes = _classes(experiment)
    images = results_repo.list_scored_images(conn, experiment.id, subset=subset)
    samples = results_repo.list_scored_samples(conn, experiment.id, subset=subset)
    truths = _truths(conn, experiment, images)
    cuts = stored_cuts(conn, experiment)
    maps_dir = Path(experiment.artifact_dir) / "maps"

    pooled: dict[int, list[int]] = {}
    for image in images:
        truth = truths.get(image.image_id)
        predicted = (
            None if truth is None else read_instances(instances_path(maps_dir, image.image_id))
        )
        if truth is None or predicted is None:
            continue
        cut = None if image.subset is None else cuts.get(image.subset)
        found = match_image(load_boxes(truth), predicted, classes, cut)
        counts = pooled.setdefault(image.sample_id, [0, 0, 0, 0])
        counts[0] += found.truth
        counts[1] += found.true_positives
        counts[2] += found.missed
        counts[3] += found.false_positives

    verdicts: list[DetectionVerdict] = []
    for sample in samples:
        tally = pooled.get(sample.sample_id)
        truth_count, matched, missed, false_positives = tally or (0, 0, 0, 0)
        verdicts.append(
            DetectionVerdict(
                sample_id=sample.sample_id,
                group_key=sample.group_key,
                external_id=sample.external_id,
                label=sample.label,
                notes=sample.notes,
                score=sample.agg_score,
                predicted_defect=bool(matched or false_positives),
                outcome="unlabeled"
                if tally is None
                else _outcome(truth_count, missed, false_positives),
                matched=matched,
                missed=missed,
                false_positives=false_positives,
            )
        )
    verdicts.sort(key=lambda verdict: verdict.score, reverse=True)
    in_play = {image.subset for image in images}
    only = next(iter(in_play)) if len(in_play) == 1 else None
    cut = None if only is None else cuts.get(only)
    return DetectionOutcomes(
        threshold_rule=rule_for(cut) if only is not None else "each subset's own cut, " + CUT_RULE,
        confidence_cut=cut,
        samples=verdicts,
    )
