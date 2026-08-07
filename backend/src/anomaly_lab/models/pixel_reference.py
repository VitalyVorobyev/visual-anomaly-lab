"""`pixel_reference` — the dataset-agnostic floor baseline.

Per-pixel median and MAD over the training normals, then a robust z-map per test image,
smoothed, reduced to a high percentile. numpy and Pillow only: it trains in seconds, runs
anywhere, and needs no optional dependency group.

Its job is to be **beaten**. A deep method reporting 0.94 AUROC means one thing if this
scores 0.55 and quite another if it scores 0.93 — and without a floor there is no way to
tell which. It is also what makes the results path testable before torch is involved.

It is deliberately the geometry-free core of the classical baseline in ADR-0010: that
method is this one with a circle fit and a polar unwrap in front. If `classical_circular`
is ever revived (optional M8), it is built on top of this rather than beside it.

**What it assumes:** that the images are roughly registered — pixel (i, j) means the same
thing across the dataset. That holds for VisA and for most inspection captures, and fails
badly for anything freely posed. When it fails it fails visibly, with a map that lights up
on every edge, which is more useful than a baseline that quietly does nothing.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from pydantic import BaseModel, Field

from anomaly_lab.models.base import (
    AnomalyModel,
    Capabilities,
    Device,
    ImageRecord,
    InferContext,
    Prediction,
    TrainContext,
)
from anomaly_lab.models.diagnostics import DiagnosticKind
from anomaly_lab.models.preprocessing import load_array
from anomaly_lab.schemas import API_MODEL_CONFIG

# MAD times this estimates the standard deviation of a normal distribution, which is what
# makes the resulting z-map comparable to a familiar scale.
MAD_TO_SIGMA = 1.4826

REFERENCE_FILENAME = "reference.npz"


class PixelReferenceConfig(BaseModel):
    """Hyperparameters. Every field here becomes a control on the experiment form."""

    model_config = API_MODEL_CONFIG

    max_reference_images: int = Field(
        default=128,
        ge=4,
        le=2048,
        description=(
            "How many training normals build the reference. A per-pixel median needs "
            "every image in memory at once, so this bounds it; images are sampled evenly "
            "across the training set rather than taking the first N."
        ),
    )
    smoothing_sigma: float = Field(
        default=4.0,
        ge=0.0,
        le=32.0,
        description=(
            "Gaussian blur applied to the z-map, in pixels. Suppresses isolated noisy "
            "pixels so the score reflects a region rather than one outlier."
        ),
    )
    score_percentile: float = Field(
        default=99.5,
        ge=50.0,
        le=100.0,
        description=(
            "Which percentile of the smoothed z-map becomes the image score. The maximum "
            "(100) is the most sensitive and the least stable."
        ),
    )
    mad_floor: float = Field(
        default=1e-3,
        gt=0.0,
        description=(
            "Lower bound on the per-pixel scale. Without it, a pixel that is identical "
            "in every training image divides by zero and dominates every map."
        ),
    )


def gaussian_blur(image: np.ndarray, sigma: float) -> np.ndarray:
    """Separable Gaussian blur with edge padding, in numpy.

    Hand-rolled rather than pulled from scipy: this is the only filtering the baseline
    needs, and the point of `pixel_reference` is that it runs with no optional
    dependencies at all.
    """
    if sigma <= 0:
        return image
    radius = max(1, round(3.0 * sigma))
    offsets = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-(offsets**2) / (2.0 * sigma * sigma))
    kernel /= kernel.sum()

    padded = np.pad(image, ((radius, radius), (0, 0)), mode="edge")
    columns = np.apply_along_axis(lambda row: np.convolve(row, kernel, mode="valid"), 0, padded)
    padded = np.pad(columns, ((0, 0), (radius, radius)), mode="edge")
    return np.apply_along_axis(lambda row: np.convolve(row, kernel, mode="valid"), 1, padded)


def _evenly_spaced(count: int, limit: int) -> list[int]:
    """Indices sampled across the whole range, not the first `limit` of it.

    Datasets arrive in acquisition order often enough that the first 128 images are one
    production batch. Taking a stride keeps the reference representative of the set.
    """
    if count <= limit:
        return list(range(count))
    return [round(index) for index in np.linspace(0, count - 1, limit)]


class PixelReferenceModel(AnomalyModel):
    """Per-pixel robust reference statistics, compared pixelwise."""

    title = "Pixel reference (baseline)"
    summary = (
        "Per-pixel median and MAD over the training normals, then a smoothed robust "
        "z-map. Trains in seconds on CPU and gives every deep result a floor to beat."
    )

    def __init__(self, config: PixelReferenceConfig) -> None:
        super().__init__(config)
        self.config = config
        self._median: np.ndarray | None = None
        self._scale: np.ndarray | None = None

    @classmethod
    def config_model(cls) -> type[BaseModel]:
        return PixelReferenceConfig

    @classmethod
    def capabilities(cls) -> Capabilities:
        return Capabilities(
            requires_training=True,
            produces_anomaly_map=True,
            produces_diagnostics=True,
            channel_aware=False,
            dataset_specific=False,
            preferred_device=Device.CPU,
        )

    def fit(self, train: Sequence[ImageRecord], ctx: TrainContext) -> None:
        chosen = _evenly_spaced(len(train), self.config.max_reference_images)
        if len(chosen) < len(train):
            ctx.log(
                f"building the reference from {len(chosen)} of {len(train)} training "
                f"images, sampled evenly (max_reference_images={self.config.max_reference_images})"
            )
        else:
            ctx.log(f"building the reference from all {len(chosen)} training images")

        stack = np.empty(
            (
                len(chosen),
                ctx.preprocessing.height,
                ctx.preprocessing.width,
                ctx.preprocessing.channels,
            ),
            dtype=np.float32,
        )
        for position, index in enumerate(chosen):
            ctx.raise_if_cancelled()
            stack[position] = load_array(train[index].path, ctx.preprocessing)
            ctx.progress(0.8 * (position + 1) / len(chosen), f"read {position + 1}/{len(chosen)}")

        ctx.progress(0.85, "computing the per-pixel median")
        median = np.median(stack, axis=0).astype(np.float32)

        ctx.progress(0.95, "computing the per-pixel deviation")
        deviation = np.median(np.abs(stack - median), axis=0).astype(np.float32)
        scale = np.maximum(deviation * MAD_TO_SIGMA, self.config.mad_floor).astype(np.float32)

        self._median = median
        self._scale = scale

        ctx.metric("reference_images", float(len(chosen)))
        ctx.metric("median_scale", float(np.median(scale)))
        ctx.log(
            f"per-pixel scale: median {np.median(scale):.4f}, "
            f"{float((scale <= self.config.mad_floor).mean()) * 100:.1f}% of pixels at the floor"
        )

        ctx.emit_diagnostic(
            "reference_median",
            "Reference median",
            DiagnosticKind.IMAGE if median.shape[-1] == 3 else DiagnosticKind.MAP,
            median if median.shape[-1] == 3 else median[:, :, 0],
            description="What this model considers a normal part to look like.",
        )
        ctx.emit_diagnostic(
            "reference_scale",
            "Per-pixel deviation",
            DiagnosticKind.MAP,
            scale.max(axis=2),
            description=(
                "How much each pixel varies across normal images. Bright regions "
                "tolerate more variation before being called anomalous."
            ),
        )

    def predict(self, images: Sequence[ImageRecord], ctx: InferContext) -> list[Prediction]:
        if self._median is None or self._scale is None:
            msg = "pixel_reference was asked to predict before it was fitted or loaded"
            raise RuntimeError(msg)

        predictions: list[Prediction] = []
        for index, record in enumerate(images):
            ctx.raise_if_cancelled()
            started = time.perf_counter()

            array = load_array(record.path, ctx.preprocessing)
            deviation = np.abs(array - self._median) / self._scale
            # Across channels, not averaged: a defect that shows under one illumination
            # is a defect, which is the same reasoning the evaluation layer applies one
            # level up when it aggregates a sample's images (ADR-0011).
            raw = deviation.max(axis=2)
            smoothed = gaussian_blur(raw.astype(np.float64), self.config.smoothing_sigma)
            score = float(np.percentile(smoothed, self.config.score_percentile))

            map_path = ctx.write_map(record.image_id, smoothed)
            elapsed_ms = (time.perf_counter() - started) * 1000.0

            ctx.emit_diagnostic(
                "z_map_raw",
                "Unsmoothed z-map",
                DiagnosticKind.MAP,
                raw.astype(np.float32),
                image_id=record.image_id,
                description="Per-pixel deviation before smoothing — noisier, more literal.",
            )

            predictions.append(
                Prediction(
                    image_id=record.image_id,
                    score=score,
                    anomaly_map=map_path,
                    inference_ms=elapsed_ms,
                )
            )
            ctx.progress((index + 1) / len(images), f"scored {index + 1}/{len(images)}")

        return predictions

    def save(self, artifact_dir: Path) -> None:
        if self._median is None or self._scale is None:
            msg = "pixel_reference has nothing to save; it was never fitted"
            raise RuntimeError(msg)
        np.savez_compressed(
            artifact_dir / REFERENCE_FILENAME,
            median=self._median,
            scale=self._scale,
        )

    def load(self, artifact_dir: Path) -> None:
        with np.load(artifact_dir / REFERENCE_FILENAME) as stored:
            self._median = stored["median"]
            self._scale = stored["scale"]
