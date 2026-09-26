"""`dino_linear_det`: `dino_linear_seg`'s head read as boxes, for object detection (ADR-0039).

The first deep detector, and built the way the floor is built on its segmenter: `color_detector`
is `color_classifier` read as boxes, and this is `dino_linear_seg` read as boxes. Training
paints each truth box's interior with its class and everything outside every box as background
(`color_detector.PaintedBoxes`, a pixel inside when its centre is, a smaller box over a larger
one) and fits `dino_linear_seg`'s softmax head on the shared frozen-DINO patch features to that
— its bounded pixel plan, per-class sampling, seeded CPU fit and held-out per-class logit bias
unchanged. At inference each pixel takes the argmax of the head's probabilities
(`DinoLinearSegModel.probabilities`, exactly what the segmenter draws its label map from), and
**every 8-connected component of a class is one detection** (`color_detector.component_boxes`):
its tight box, and as confidence the mean probability of its class over it.

Why this and not a box-regression head with non-maximum suppression: it is the one deep
detector that adds no trained part the workbench has not already measured. The head cleared
its public segmentation gate, the decoding is the floor's to the line, so the public detection
gate reads one difference — DINO features against colour — and not a second loss, a regression
target and an NMS threshold at once. What it gives up is the floor's weakness too: two touching
objects of one class are one detection, and a box is only as tight as the component the
upsampled logits draw. A detector that regresses boxes is the answer to that, if the gate says
the features are worth it.

`held_out_iou` fits each class's constant for the IoU of the painted box interiors, which is
the pixel-level shadow of a box's IoU; AP depends only on how detections rank, and the mean
class probability over a component ranks a confident, well-separated region above a faint one.

The anomaly map is the probability of anything but background; the image's score is its most
confident detection, as for every detection method.
"""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

import numpy as np
from pydantic import BaseModel, Field

from anomaly_lab.domain.entities import Task
from anomaly_lab.models.base import (
    MAX_INSTANCES_PER_IMAGE,
    AnomalyModel,
    Availability,
    Capabilities,
    Device,
    ImageRecord,
    InferContext,
    PredictedInstance,
    Prediction,
    TrainContext,
    module_available,
)
from anomaly_lab.models.color_detector import PaintedBoxes, component_boxes
from anomaly_lab.models.dino_linear_seg import DinoLinearSegConfig, DinoLinearSegModel
from anomaly_lab.models.preprocessing import PreprocessingConfig

CLASSES_FILENAME = "dino_linear_det.json"


class DinoLinearDetConfig(DinoLinearSegConfig):
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


def decode(
    probability: np.ndarray,
    classes: Sequence[str],
    *,
    min_area: int,
    max_detections: int,
) -> list[PredictedInstance]:
    """One image's detections from its `(C, h, w)` class probabilities, background first.

    The argmax labels every pixel, and each class's components are boxed by
    `component_boxes` with that class's probability as their confidence. Pure numpy.
    """
    labels = np.argmax(probability, axis=0)
    planes = {index: probability[index] for index in range(1, probability.shape[0])}
    return component_boxes(
        labels, planes, classes, min_area=min_area, max_detections=max_detections
    )


class DinoLinearDetModel(AnomalyModel):
    """A softmax head on frozen DINO patch features, fitted on box interiors, components boxed."""

    title = "DINO linear head (detection)"
    summary = (
        "Paints each class's boxes as its pixels, fits the DINO linear segmentation head on a "
        "bounded sample of them, then boxes each connected region of a class; a region's "
        "confidence is the mean probability of its class."
    )

    def __init__(self, config: DinoLinearDetConfig) -> None:
        super().__init__(config)
        self.config = config
        self._head = DinoLinearSegModel(config, method="dino_linear_det")
        self._classes: tuple[str, ...] = ()

    @classmethod
    def config_model(cls) -> type[BaseModel]:
        return DinoLinearDetConfig

    @classmethod
    def native_size(cls, config: BaseModel) -> tuple[int, int]:
        """The segmentation head's frame: the VisA detection gate ran at 448x448."""
        if not isinstance(config, DinoLinearDetConfig):
            raise TypeError(f"expected DinoLinearDetConfig, got {type(config).__name__}")
        return DinoLinearSegModel.native_size(config)

    @classmethod
    def size_multiple(cls, config: BaseModel) -> int:
        if not isinstance(config, DinoLinearDetConfig):
            raise TypeError(f"expected DinoLinearDetConfig, got {type(config).__name__}")
        return DinoLinearSegModel.size_multiple(config)

    @classmethod
    def check_input(cls, config: BaseModel, preprocessing: PreprocessingConfig) -> None:
        if not isinstance(config, DinoLinearDetConfig):
            raise TypeError(f"expected DinoLinearDetConfig, got {type(config).__name__}")
        DinoLinearSegModel.check_input(config, preprocessing)

    @classmethod
    def capabilities(cls) -> Capabilities:
        return Capabilities(
            tasks=[Task.OBJECT_DETECTION],
            requires_training=True,
            produces_anomaly_map=True,
            preferred_device=Device.MPS,
        )

    @classmethod
    def availability(cls) -> Availability:
        return module_available("torch", "dl", "the DINO linear detection head")

    def fit(self, train: Sequence[ImageRecord], ctx: TrainContext) -> None:
        painted = PaintedBoxes(ctx, "dino_linear_det")
        ctx.log("box targets: each box's interior is its class, outside every box background")
        self._head.fit(train, replace(ctx, box_targets=None, label_targets=painted))
        self._classes = painted.classes

    def detect(
        self, record: ImageRecord, ctx: InferContext
    ) -> tuple[list[PredictedInstance], np.ndarray]:
        """One image's detections, most confident first, and its foreground probability."""
        probability = self._head.probabilities(record, ctx)
        found = decode(
            probability,
            self._classes,
            min_area=self.config.min_area,
            max_detections=self.config.max_detections,
        )
        return found, 1.0 - probability[0]

    def predict(self, images: Sequence[ImageRecord], ctx: InferContext) -> list[Prediction]:
        if not self._classes:
            raise RuntimeError(
                "dino_linear_det was asked to predict before it was fitted or loaded"
            )
        predictions: list[Prediction] = []
        for position, record in enumerate(images):
            ctx.raise_if_cancelled()
            started = time.perf_counter()
            found, foreground = self.detect(record, ctx)
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
            raise RuntimeError("dino_linear_det has nothing to save; it was never fitted")
        self._head.save(artifact_dir)
        # Written beside itself, then renamed over, so it lands whole or not at all.
        path = artifact_dir / CLASSES_FILENAME
        temporary = artifact_dir / f"{CLASSES_FILENAME}.tmp"
        try:
            temporary.write_text(json.dumps({"classes": list(self._classes)}), encoding="utf-8")
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)

    def load(self, artifact_dir: Path) -> None:
        self._head.load(artifact_dir)
        stored = json.loads((artifact_dir / CLASSES_FILENAME).read_text(encoding="utf-8"))
        self._classes = tuple(str(key) for key in stored["classes"])
