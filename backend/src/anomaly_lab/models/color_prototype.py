"""`color_prototype`: the torch-free floor for few-shot segmentation (ADR-0040).

The references' masks split their pixels into the class and everything else. One Gaussian
is fitted to the colour of each, and a query pixel's foreground probability is the
posterior of the class under equal priors. The map is that probability, smoothed; the
image's presence score is its high percentile.

It knows nothing but colour, so it is a floor, not a candidate: a class that shares its
colour with the background cannot be told apart, and a deep method that does not beat it
has learned nothing about shape or texture. What it proves is the slice — references in,
masks and presence out, evaluated — in the torch-free CI job.

`calibration` at `leave_one_out` rescales the posterior on the references (`calibration.py`):
equal priors put the class's 0.5 far too low for a class that covers a sliver of the frame.

numpy and Pillow only, like `pixel_reference`.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import numpy as np
from pydantic import BaseModel, Field

from anomaly_lab.domain.entities import Task
from anomaly_lab.models.base import (
    AnomalyModel,
    Capabilities,
    Device,
    ImageRecord,
    InferContext,
    Prediction,
    TrainContext,
    evenly_spaced,
)
from anomaly_lab.models.calibration import IDENTITY, Calibration, PlattScale, leave_one_out
from anomaly_lab.models.preprocessing import load_array
from anomaly_lab.models.score_map import gaussian_blur
from anomaly_lab.schemas import API_MODEL_CONFIG

PROTOTYPE_FILENAME = "color_prototype.npz"
# Keeps a covariance invertible when a class is one flat colour.
COVARIANCE_RIDGE = 1e-4


class ColorSpace(StrEnum):
    LAB = "lab"
    """CIE L*a*b*: distances roughly track perceived colour difference."""
    RGB = "rgb"


class ColorPrototypeConfig(BaseModel):
    model_config = API_MODEL_CONFIG

    color_space: ColorSpace = Field(
        json_schema_extra={"x-primary": True},
        default=ColorSpace.LAB,
        description=(
            "Where the two colour models live. Lab separates lightness from hue, so a class "
            "under uneven lighting stays one cluster. A grey image is modelled on intensity "
            "either way."
        ),
    )
    max_pixels_per_class: int = Field(
        default=100_000,
        ge=100,
        le=5_000_000,
        description=(
            "At most this many reference pixels fit each model, sampled evenly across the "
            "references. The fit is linear in pixels and its value is not."
        ),
    )
    smoothing_sigma: float = Field(
        default=1.0,
        ge=0.0,
        le=16.0,
        description="Gaussian smoothing of the probability map, in prepared pixels. 0 is none.",
    )
    calibration: Calibration = Field(
        json_schema_extra={"x-primary": True},
        default=Calibration.NONE,
        description=(
            "'leave_one_out' rescales the foreground probability on the references: each is "
            "scored by colour models of the others, and a Platt scale fitted to those pixels "
            "is applied to every map and presence score, so the 0.5 cut means the same on "
            "every class. One reference cannot be left out and stays unscaled."
        ),
    )
    presence_percentile: float = Field(
        default=99.9,
        ge=50.0,
        le=100.0,
        description=(
            "The presence score is this percentile of the probability map, so one noisy "
            "pixel does not decide it."
        ),
    )


@dataclass(frozen=True)
class _Gaussian:
    mean: np.ndarray
    precision: np.ndarray
    log_norm: float

    @classmethod
    def fit(cls, samples: np.ndarray) -> _Gaussian:
        mean = samples.mean(axis=0)
        centred = samples - mean
        covariance = centred.T @ centred / max(len(samples) - 1, 1)
        covariance += COVARIANCE_RIDGE * np.eye(covariance.shape[0])
        _, log_det = np.linalg.slogdet(covariance)
        return cls(mean=mean, precision=np.linalg.inv(covariance), log_norm=-0.5 * log_det)

    def log_likelihood(self, pixels: np.ndarray) -> np.ndarray:
        centred = pixels - self.mean
        distance = np.einsum("ni,ij,nj->n", centred, self.precision, centred)
        return np.asarray(self.log_norm - 0.5 * distance, dtype=np.float64)


def srgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """`(..., 3)` sRGB in `[0, 1]` to CIE L*a*b* under D65."""
    linear = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    to_xyz = np.array(
        [[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]]
    )
    xyz = linear @ to_xyz.T / np.array([0.95047, 1.0, 1.08883])
    delta = 6 / 29
    f = np.where(xyz > delta**3, np.cbrt(xyz), xyz / (3 * delta**2) + 4 / 29)
    return np.stack(
        [116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])],
        axis=-1,
    )


class ColorPrototypeModel(AnomalyModel):
    """A foreground and a background colour Gaussian, from the references' masks."""

    title = "Colour prototype (few-shot floor)"
    summary = (
        "Fits one colour model to the class and one to everything else in the references, "
        "then paints each query pixel with the class's posterior. CPU, seconds, no torch."
    )

    def __init__(self, config: ColorPrototypeConfig) -> None:
        super().__init__(config)
        self.config = config
        self._foreground: _Gaussian | None = None
        self._background: _Gaussian | None = None
        self._calibration: PlattScale = IDENTITY

    @classmethod
    def config_model(cls) -> type[BaseModel]:
        return ColorPrototypeConfig

    @classmethod
    def native_size(cls, config: BaseModel) -> tuple[int, int]:
        """448 px square: the frame the few-shot gates ran this floor at (docs/measurements.md)."""
        return (448, 448)

    @classmethod
    def capabilities(cls) -> Capabilities:
        return Capabilities(
            tasks=[Task.FEW_SHOT_SEGMENTATION],
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
        targets = ctx.targets
        if targets is None:
            msg = "color_prototype segments a class and needs references with masks"
            raise RuntimeError(msg)

        arrays: list[np.ndarray] = []
        foreground: list[np.ndarray] = []
        background: list[np.ndarray] = []
        for position, record in enumerate(train):
            ctx.raise_if_cancelled()
            array = load_array(record.path, ctx.preprocessing)
            pixels = self._features(array)
            mask = targets.mask(record.image_id).reshape(-1)
            arrays.append(array)
            foreground.append(pixels[mask])
            background.append(pixels[~mask])
            ctx.progress(0.8 * (position + 1) / len(train), f"read {position + 1}/{len(train)}")

        self._foreground = self._fit_class("the class", foreground, targets.label_key, ctx)
        self._background = self._fit_class("the background", background, targets.label_key, ctx)
        self._calibration = IDENTITY
        if self.config.calibration is Calibration.LEAVE_ONE_OUT:
            ctx.progress(0.9, "calibrating on the references")

            def held_out(fold: int) -> tuple[np.ndarray, np.ndarray] | None:
                label = targets.label_key
                fg_parts = [part for i, part in enumerate(foreground) if i != fold]
                bg_parts = [part for i, part in enumerate(background) if i != fold]
                if not sum(map(len, fg_parts)) or not sum(map(len, bg_parts)):
                    return None
                fg = self._fit_class("the class", fg_parts, label, None)
                bg = self._fit_class("the background", bg_parts, label, None)
                return self._probability(arrays[fold], fg, bg), targets.mask(train[fold].image_id)

            self._calibration = leave_one_out(
                len(train), held_out, ctx.log, cancelled=ctx.raise_if_cancelled
            )
            ctx.metric("calibration_slope", self._calibration.slope)
            ctx.metric("calibration_bias", self._calibration.bias)
        ctx.progress(1.0, "fitted")

    def _fit_class(
        self, name: str, parts: list[np.ndarray], label_key: str, ctx: TrainContext | None
    ) -> _Gaussian:
        """One colour Gaussian; `ctx` is `None` in a quiet leave-one-out fold."""
        pooled = np.concatenate(parts) if parts else np.empty((0, 1))
        if len(pooled) == 0:
            msg = f"the references hold no pixel of {name} for {label_key!r}"
            raise RuntimeError(msg)
        chosen = evenly_spaced(len(pooled), self.config.max_pixels_per_class)
        if ctx is not None:
            if len(chosen) < len(pooled):
                ctx.log(
                    f"fitting {name} on {len(chosen)} of {len(pooled)} reference pixels, sampled "
                    f"evenly (max_pixels_per_class={self.config.max_pixels_per_class})"
                )
            else:
                ctx.log(f"fitting {name} on all {len(pooled)} reference pixels")
            kind = "foreground" if name == "the class" else "background"
            ctx.metric(f"pixels_{kind}", len(chosen))
        return _Gaussian.fit(pooled[np.asarray(chosen, dtype=np.int64)])

    def _probability(
        self, array: np.ndarray, foreground: _Gaussian, background: _Gaussian
    ) -> np.ndarray:
        """The class's posterior under equal priors, smoothed; unscaled."""
        pixels = self._features(array)
        margin = foreground.log_likelihood(pixels) - background.log_likelihood(pixels)
        probability = (1.0 / (1.0 + np.exp(-np.clip(margin, -60.0, 60.0)))).reshape(array.shape[:2])
        if self.config.smoothing_sigma > 0:
            probability = gaussian_blur(probability, self.config.smoothing_sigma)
        return np.asarray(probability, dtype=np.float64)

    def predict(self, images: Sequence[ImageRecord], ctx: InferContext) -> list[Prediction]:
        if self._foreground is None or self._background is None:
            raise RuntimeError("color_prototype was asked to predict before it was fitted")
        predictions: list[Prediction] = []
        for index, record in enumerate(images):
            ctx.raise_if_cancelled()
            started = time.perf_counter()
            array = load_array(record.path, ctx.preprocessing)
            unscaled = self._probability(array, self._foreground, self._background)
            probability = self._calibration.apply(unscaled)
            # The percentile of the unscaled map, then scaled: interpolating between two
            # pixels does not commute with the scale, and this keeps the ranking exact.
            score = self._calibration.apply_score(
                float(np.percentile(unscaled, self.config.presence_percentile))
            )
            map_path = ctx.write_map(record.image_id, probability)
            predictions.append(
                Prediction(
                    image_id=record.image_id,
                    score=score,
                    anomaly_map=map_path,
                    inference_ms=(time.perf_counter() - started) * 1000.0,
                )
            )
            ctx.progress((index + 1) / len(images), f"segmented {index + 1}/{len(images)}")
        return predictions

    def save(self, artifact_dir: Path) -> None:
        if self._foreground is None or self._background is None:
            raise RuntimeError("color_prototype has nothing to save; it was never fitted")
        payload = {
            f"{name}_{part}": getattr(gaussian, part)
            for name, gaussian in (("fg", self._foreground), ("bg", self._background))
            for part in ("mean", "precision")
        }
        payload["log_norm"] = np.array([self._foreground.log_norm, self._background.log_norm])
        payload["calibration"] = self._calibration.to_array()
        np.savez_compressed(artifact_dir / PROTOTYPE_FILENAME, **payload)

    def load(self, artifact_dir: Path) -> None:
        with np.load(artifact_dir / PROTOTYPE_FILENAME, allow_pickle=False) as stored:
            fg_norm, bg_norm = (float(value) for value in stored["log_norm"])
            self._foreground = _Gaussian(stored["fg_mean"], stored["fg_precision"], fg_norm)
            self._background = _Gaussian(stored["bg_mean"], stored["bg_precision"], bg_norm)
            self._calibration = PlattScale.from_array(
                stored["calibration"] if "calibration" in stored.files else None
            )
