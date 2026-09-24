"""`color_detector`: the torch-free floor for object detection (ADR-0039).

It is `color_classifier` read as boxes. Training paints each truth box's interior with its
class and everything outside every box as background, and fits `color_classifier`'s colour
Gaussians to that; a smaller box is painted over a larger one it overlaps. At inference each
pixel takes the class with the highest smoothed posterior, and **every 8-connected component
of a class is one detection**: its box is the component's tight bounding box, and its
confidence the mean posterior of its class over the component. Components smaller than
`min_area` are dropped, and at most `max_detections` of the most confident are kept. The
image's score is its highest confidence, which is what every detection method's score means.

A box is not an outline, so a class's colour model also learns whatever background its
boxes enclose; a loose box teaches a muddier colour. It knows nothing but colour, so it is a
floor, not a candidate: two touching objects of one class are one detection, and a learned
detector that does not beat it has learned nothing about shape. What it proves is the slice
— boxed truth in, boxes out, AP read — in the torch-free CI job.

numpy and Pillow only. Connected components are the evaluation layer's union-find
(`eval/pixel.py`), a Python loop over foreground pixels: quick on sparse objects, slow when a
class covers most of the frame.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

import numpy as np
from pydantic import BaseModel, Field

from anomaly_lab.domain.entities import Task
from anomaly_lab.eval.pixel import connected_regions
from anomaly_lab.models.base import (
    MAX_INSTANCES_PER_IMAGE,
    AnomalyModel,
    Capabilities,
    Device,
    ImageRecord,
    InferContext,
    PredictedInstance,
    Prediction,
    TargetBox,
    TrainContext,
)
from anomaly_lab.models.color_classifier import ColorClassifierConfig, ColorClassifierModel
from anomaly_lab.models.preprocessing import load_array

CLASSES_FILENAME = "color_detector.json"


class ColorDetectorConfig(ColorClassifierConfig):
    min_area: int = Field(
        default=4,
        ge=1,
        le=1_000_000,
        description=(
            "A component of a class smaller than this many prepared pixels is not a detection."
        ),
    )
    max_detections: int = Field(
        default=MAX_INSTANCES_PER_IMAGE,
        ge=1,
        le=MAX_INSTANCES_PER_IMAGE,
        description="At most this many detections of one image, the most confident kept.",
    )


def paint_boxes(
    boxes: Sequence[TargetBox], classes: Sequence[str], height: int, width: int
) -> np.ndarray:
    """A label map of box interiors: class `classes[i]` is `i + 1`, outside every box is 0.

    A pixel is inside a box when its centre is; larger boxes are painted first, so where two
    overlap the smaller one keeps its class.
    """
    position = {key: index + 1 for index, key in enumerate(classes)}
    labels = np.zeros((height, width), dtype=np.uint8)
    for target in sorted(
        boxes, key=lambda item: -(item.box[2] - item.box[0]) * (item.box[3] - item.box[1])
    ):
        x0, y0, x1, y1 = target.box
        left, right = max(0, math.ceil(x0 - 0.5)), min(width, math.ceil(x1 - 0.5))
        top, bottom = max(0, math.ceil(y0 - 0.5)), min(height, math.ceil(y1 - 0.5))
        if right > left and bottom > top:
            labels[top:bottom, left:right] = position[target.label_key]
    return labels


class _PaintedBoxes:
    """The run's box targets as the label targets `color_classifier` fits on."""

    def __init__(self, ctx: TrainContext) -> None:
        if ctx.box_targets is None:
            msg = "color_detector detects annotated classes and needs box targets"
            raise RuntimeError(msg)
        self._targets = ctx.box_targets
        self._size = (ctx.preprocessing.height, ctx.preprocessing.width)

    @property
    def classes(self) -> tuple[str, ...]:
        return self._targets.classes

    def labels(self, image_id: int) -> np.ndarray:
        return paint_boxes(self._targets.boxes(image_id), self.classes, *self._size)


class ColorDetectorModel(AnomalyModel):
    """One colour Gaussian per class, fitted on box interiors, and its components boxed."""

    title = "Colour detector (detection floor)"
    summary = (
        "Fits one colour model to the inside of each class's boxes and one to everything "
        "outside them, then boxes each connected region of a class. CPU, seconds, no torch."
    )

    def __init__(self, config: ColorDetectorConfig) -> None:
        super().__init__(config)
        self.config = config
        self._pixels = ColorClassifierModel(config)
        self._classes: tuple[str, ...] = ()

    @classmethod
    def config_model(cls) -> type[BaseModel]:
        return ColorDetectorConfig

    @classmethod
    def capabilities(cls) -> Capabilities:
        return Capabilities(
            tasks=[Task.OBJECT_DETECTION],
            requires_training=True,
            produces_anomaly_map=True,
            preferred_device=Device.CPU,
        )

    def fit(self, train: Sequence[ImageRecord], ctx: TrainContext) -> None:
        painted = _PaintedBoxes(ctx)
        self._pixels.fit(train, replace(ctx, box_targets=None, label_targets=painted))
        self._classes = painted.classes

    def detect(self, array: np.ndarray) -> tuple[list[PredictedInstance], np.ndarray]:
        """One prepared image's detections, most confident first, and its foreground map."""
        labels, posterior, foreground = self._pixels.classify(array)
        width = labels.shape[1]
        found: list[PredictedInstance] = []
        for plane, index in enumerate(self._pixels.label_indices):
            if index == 0:
                continue
            confidence = posterior[plane].reshape(-1)
            for region in connected_regions(labels == index):
                if region.size < self.config.min_area:
                    continue
                ys, xs = np.divmod(region, width)
                found.append(
                    PredictedInstance(
                        label_key=self._classes[index - 1],
                        box=(
                            float(xs.min()),
                            float(ys.min()),
                            float(xs.max()) + 1.0,
                            float(ys.max()) + 1.0,
                        ),
                        confidence=float(confidence[region].mean()),
                    )
                )
        found.sort(key=lambda instance: -instance.confidence)
        return found[: self.config.max_detections], foreground

    def predict(self, images: Sequence[ImageRecord], ctx: InferContext) -> list[Prediction]:
        if not self._classes:
            raise RuntimeError("color_detector was asked to predict before it was fitted")
        predictions: list[Prediction] = []
        for position, record in enumerate(images):
            ctx.raise_if_cancelled()
            started = time.perf_counter()
            found, foreground = self.detect(load_array(record.path, ctx.preprocessing))
            map_path = ctx.write_map(record.image_id, foreground)
            boxes_path = ctx.write_instances(record.image_id, found, classes=self._classes)
            predictions.append(
                Prediction(
                    image_id=record.image_id,
                    score=found[0].confidence if found else 0.0,
                    anomaly_map=map_path,
                    instances=boxes_path,
                    inference_ms=(time.perf_counter() - started) * 1000.0,
                )
            )
            ctx.progress((position + 1) / len(images), f"detected {position + 1}/{len(images)}")
        return predictions

    def save(self, artifact_dir: Path) -> None:
        if not self._classes:
            raise RuntimeError("color_detector has nothing to save; it was never fitted")
        self._pixels.save(artifact_dir)
        (artifact_dir / CLASSES_FILENAME).write_text(
            json.dumps({"classes": list(self._classes)}), encoding="utf-8"
        )

    def load(self, artifact_dir: Path) -> None:
        self._pixels.load(artifact_dir)
        stored = json.loads((artifact_dir / CLASSES_FILENAME).read_text(encoding="utf-8"))
        self._classes = tuple(str(key) for key in stored["classes"])
