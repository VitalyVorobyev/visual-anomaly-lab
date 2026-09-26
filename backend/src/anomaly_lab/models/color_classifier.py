"""`color_classifier`: the torch-free floor for supervised semantic segmentation (ADR-0039).

Every pinned class, and background, gets one Gaussian over the colour of its training pixels.
A pixel's class is the one with the highest posterior under equal priors, after each class's
posterior is smoothed. The label map is that argmax; the anomaly map beside it is the
probability of anything but background; the image's score is the share of it assigned to a
class, which is what every supervised segmentation method's score means.

It knows nothing but colour, so it is a floor, not a candidate: two classes of one colour
cannot be told apart, and a learned method that does not beat it has learned nothing about
shape or texture. What it proves is the slice — annotated images in, label maps out,
confusion matrix read — in the torch-free CI job.

numpy and Pillow only, like `color_prototype`, whose colour model it shares.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from pydantic import BaseModel, Field

from anomaly_lab.domain.entities import Task
from anomaly_lab.models.base import (
    IGNORE_INDEX,
    AnomalyModel,
    Capabilities,
    Device,
    ImageRecord,
    InferContext,
    Prediction,
    TrainContext,
    evenly_spaced,
)
from anomaly_lab.models.color_prototype import ColorSpace, _Gaussian, srgb_to_lab
from anomaly_lab.models.preprocessing import load_array
from anomaly_lab.models.score_map import gaussian_blur
from anomaly_lab.schemas import API_MODEL_CONFIG

MODEL_FILENAME = "color_classifier.npz"


class ColorClassifierConfig(BaseModel):
    model_config = API_MODEL_CONFIG

    color_space: ColorSpace = Field(
        default=ColorSpace.LAB,
        description=(
            "Where the colour models live. Lab separates lightness from hue, so a class under "
            "uneven lighting stays one cluster. A grey image is modelled on intensity either way."
        ),
    )
    max_pixels_per_class: int = Field(
        default=100_000,
        ge=100,
        le=5_000_000,
        description=(
            "At most this many training pixels fit each class's model, sampled evenly across "
            "the training images. The fit is linear in pixels and its value is not."
        ),
    )
    smoothing_sigma: float = Field(
        default=1.0,
        ge=0.0,
        le=16.0,
        description=(
            "Gaussian smoothing of each class's posterior before the argmax, in prepared "
            "pixels. 0 is none."
        ),
    )


class ColorClassifierModel(AnomalyModel):
    """One colour Gaussian per class and one for background, from annotated images."""

    title = "Colour classifier (segmentation floor)"
    summary = (
        "Fits one colour model to each annotated class and one to background, then gives each "
        "pixel the class with the highest posterior. CPU, seconds, no torch."
    )

    def __init__(self, config: ColorClassifierConfig) -> None:
        super().__init__(config)
        self.config = config
        self._indices: list[int] = []
        """The label index each fitted Gaussian stands for, background (0) included."""
        self._models: list[_Gaussian] = []
        self._classes = 0

    @classmethod
    def config_model(cls) -> type[BaseModel]:
        return ColorClassifierConfig

    @classmethod
    def native_size(cls, config: BaseModel) -> tuple[int, int]:
        """448 px square: the supervised-segmentation gates ran this floor at 448x448.

        See docs/measurements.md.
        """
        return (448, 448)

    @classmethod
    def capabilities(cls) -> Capabilities:
        return Capabilities(
            tasks=[Task.SEMANTIC_SEGMENTATION],
            requires_training=True,
            produces_anomaly_map=True,
            preferred_device=Device.CPU,
        )

    def _features(self, array: np.ndarray) -> np.ndarray:
        """`(H, W, C)` in `[0, 1]` to `(H*W, D)` colour features."""
        if array.shape[-1] == 3 and self.config.color_space is ColorSpace.LAB:
            array = srgb_to_lab(array.astype(np.float64))
        return array.reshape(-1, array.shape[-1]).astype(np.float64)

    def fit(self, train: Sequence[ImageRecord], ctx: TrainContext) -> None:
        targets = ctx.label_targets
        if targets is None:
            msg = "color_classifier segments annotated classes and needs label targets"
            raise RuntimeError(msg)
        names = ("background", *targets.classes)
        size = len(names)

        # Two passes so memory stays bounded by the cap, not by the training set: count each
        # class's pixels first, choose the evenly spaced ones across the whole pool, then
        # read only those.
        per_image = np.zeros((len(train), size), dtype=np.int64)
        for position, record in enumerate(train):
            ctx.raise_if_cancelled()
            labels = targets.labels(record.image_id).reshape(-1)
            per_image[position] = np.bincount(labels[labels != IGNORE_INDEX], minlength=size)[:size]
            ctx.progress(0.3 * (position + 1) / len(train), f"counted {position + 1}/{len(train)}")

        totals = per_image.sum(axis=0)
        chosen: list[np.ndarray] = []
        for index in range(size):
            picked = np.asarray(
                evenly_spaced(int(totals[index]), self.config.max_pixels_per_class), dtype=np.int64
            )
            chosen.append(picked)
            if totals[index] == 0:
                ctx.log(
                    f"no training pixel of {names[index]!r}; it is not modelled and is never "
                    "predicted",
                    level="warning",
                )
            elif len(picked) < totals[index]:
                ctx.log(
                    f"fitting {names[index]!r} on {len(picked)} of {int(totals[index])} training "
                    f"pixels, sampled evenly (max_pixels_per_class="
                    f"{self.config.max_pixels_per_class})"
                )
            else:
                ctx.log(f"fitting {names[index]!r} on all {int(totals[index])} training pixels")
        if not totals[1:].any():
            msg = f"the training images hold no pixel of any class of {', '.join(names[1:])}"
            raise RuntimeError(msg)

        starts = np.vstack([np.zeros((1, size), dtype=np.int64), np.cumsum(per_image, axis=0)])
        samples: list[list[np.ndarray]] = [[] for _ in range(size)]
        for position, record in enumerate(train):
            ctx.raise_if_cancelled()
            if not per_image[position].any():
                continue
            pixels = self._features(load_array(record.path, ctx.preprocessing))
            labels = targets.labels(record.image_id).reshape(-1)
            for index in range(size):
                low, high = starts[position, index], starts[position + 1, index]
                wanted = chosen[index][(chosen[index] >= low) & (chosen[index] < high)] - low
                if len(wanted):
                    samples[index].append(pixels[np.flatnonzero(labels == index)[wanted]])
            ctx.progress(
                0.3 + 0.7 * (position + 1) / len(train), f"read {position + 1}/{len(train)}"
            )

        self._classes = size - 1
        self._indices = [index for index in range(size) if totals[index] > 0]
        self._models = [_Gaussian.fit(np.concatenate(samples[index])) for index in self._indices]
        for index in self._indices:
            ctx.metric(f"pixels_{names[index]}", len(chosen[index]))
        ctx.progress(1.0, "fitted")

    @property
    def label_indices(self) -> list[int]:
        """The label index each plane of `classify`'s posterior stands for, background (0)
        included when it had training pixels."""
        return list(self._indices)

    def classify(self, array: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """One prepared image's label map, its smoothed posterior and its foreground
        probability.

        The posterior is `(len(label_indices), H, W)`; the label map is its argmax as label
        indices; the foreground is the probability of anything but background.
        """
        if not self._models:
            raise RuntimeError("color_classifier was asked to predict before it was fitted")
        pixels = self._features(array)
        log_likelihood = np.stack([model.log_likelihood(pixels) for model in self._models])
        log_likelihood -= log_likelihood.max(axis=0, keepdims=True)
        posterior = np.exp(log_likelihood)
        posterior /= posterior.sum(axis=0, keepdims=True)
        posterior = posterior.reshape(len(self._models), *array.shape[:2])
        if self.config.smoothing_sigma > 0:
            posterior = np.stack(
                [gaussian_blur(plane, self.config.smoothing_sigma) for plane in posterior]
            )
        indices = np.asarray(self._indices, dtype=np.uint8)
        labels = indices[np.argmax(posterior, axis=0)]
        background = self._indices.index(0) if 0 in self._indices else None
        foreground = (
            1.0 - posterior[background]
            if background is not None
            else np.ones(array.shape[:2], dtype=np.float64)
        )
        return labels, posterior, foreground

    def predict(self, images: Sequence[ImageRecord], ctx: InferContext) -> list[Prediction]:
        if not self._models:
            raise RuntimeError("color_classifier was asked to predict before it was fitted")
        predictions: list[Prediction] = []
        for position, record in enumerate(images):
            ctx.raise_if_cancelled()
            started = time.perf_counter()
            array = load_array(record.path, ctx.preprocessing)
            labels, _, foreground = self.classify(array)
            map_path = ctx.write_map(record.image_id, foreground)
            label_path = ctx.write_label_map(record.image_id, labels, classes=self._classes)
            predictions.append(
                Prediction(
                    image_id=record.image_id,
                    score=float(np.mean(labels > 0)),
                    anomaly_map=map_path,
                    label_map=label_path,
                    inference_ms=(time.perf_counter() - started) * 1000.0,
                )
            )
            ctx.progress((position + 1) / len(images), f"segmented {position + 1}/{len(images)}")
        return predictions

    def save(self, artifact_dir: Path) -> None:
        if not self._models:
            raise RuntimeError("color_classifier has nothing to save; it was never fitted")
        np.savez_compressed(
            artifact_dir / MODEL_FILENAME,
            classes=np.array([self._classes]),
            indices=np.array(self._indices, dtype=np.int64),
            means=np.stack([model.mean for model in self._models]),
            precisions=np.stack([model.precision for model in self._models]),
            log_norms=np.array([model.log_norm for model in self._models]),
        )

    def load(self, artifact_dir: Path) -> None:
        with np.load(artifact_dir / MODEL_FILENAME, allow_pickle=False) as stored:
            self._classes = int(stored["classes"][0])
            self._indices = [int(value) for value in stored["indices"]]
            self._models = [
                _Gaussian(mean, precision, float(log_norm))
                for mean, precision, log_norm in zip(
                    stored["means"], stored["precisions"], stored["log_norms"], strict=True
                )
            ]
