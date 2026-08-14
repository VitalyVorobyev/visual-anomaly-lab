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
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import numpy as np
from pydantic import BaseModel, Field

from anomaly_lab.deployment.protocol import OnnxGraphContract
from anomaly_lab.deployment.schema import PercentileReducer
from anomaly_lab.models.base import (
    AnomalyModel,
    Capabilities,
    Device,
    ImageRecord,
    InferContext,
    PortableFormat,
    Prediction,
    TrainContext,
    evenly_spaced,
)
from anomaly_lab.models.diagnostics import DiagnosticKind
from anomaly_lab.models.preprocessing import PreprocessingConfig, load_array
from anomaly_lab.schemas import API_MODEL_CONFIG

# MAD times this estimates the standard deviation of a normal distribution, which is what
# makes the resulting z-map comparable to a familiar scale.
MAD_TO_SIGMA = 1.4826

REFERENCE_FILENAME = "reference.npz"


UNASSIGNED = ""
"""Reference key for images belonging to no channel — the same spelling the import commit
already uses for "no channel" (`datasets/commit.py`). A channel name is never empty."""


class ReferenceScope(StrEnum):
    """What one reference is built over."""

    DATASET = "dataset"
    """One reference for the whole training set."""

    CHANNEL = "channel"
    """One reference per acquisition channel."""


class PixelReferenceConfig(BaseModel):
    """Hyperparameters. Every field here becomes a control on the experiment form."""

    model_config = API_MODEL_CONFIG

    reference_scope: ReferenceScope = Field(
        default=ReferenceScope.CHANNEL,
        description=(
            "What each per-pixel reference is built over. On a dataset whose samples are "
            "photographed under several illuminations, one pooled reference is the median "
            "of images that look nothing like each other, and the deviation it measures is "
            "mostly the difference between channels. 'dataset' is the old behaviour and is "
            "correct for single-view data, where the two are identical anyway."
        ),
    )
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


@dataclass(frozen=True)
class _Reference:
    """One fitted per-pixel reference: what normal looks like, and how much it varies."""

    median: np.ndarray
    scale: np.ndarray


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
        self._references: dict[str, _Reference] = {}

    @classmethod
    def config_model(cls) -> type[BaseModel]:
        return PixelReferenceConfig

    @classmethod
    def capabilities(cls) -> Capabilities:
        return Capabilities(
            requires_training=True,
            produces_anomaly_map=True,
            produces_diagnostics=True,
            # Reads `ImageRecord.channel` to partition its own reference, and still returns
            # one `Prediction` per input image. Nothing outside this module changes.
            channel_aware=True,
            dataset_specific=False,
            portable_formats=[PortableFormat.ONNX],
            preferred_device=Device.CPU,
        )

    def _key(self, record: ImageRecord) -> str:
        """Which reference this image is measured against."""
        if self.config.reference_scope is ReferenceScope.DATASET:
            return UNASSIGNED
        return record.channel or UNASSIGNED

    @property
    def _single(self) -> _Reference:
        """The one reference, for callers that can only hold one (the ONNX graph)."""
        if len(self._references) != 1:
            msg = (
                f"pixel_reference holds {len(self._references)} references; this operation "
                "needs exactly one"
            )
            raise RuntimeError(msg)
        return next(iter(self._references.values()))

    def fit(self, train: Sequence[ImageRecord], ctx: TrainContext) -> None:
        groups: dict[str, list[ImageRecord]] = {}
        for record in train:
            groups.setdefault(self._key(record), []).append(record)

        if self.config.reference_scope is ReferenceScope.CHANNEL and len(groups) > 1:
            listed = ", ".join(
                f"{name or 'unassigned'} ({len(records)})"
                for name, records in sorted(groups.items())
            )
            ctx.log(f"fitting one reference per channel: {listed}")

        # The cap applies per group, because each reference is a separate estimator and
        # splitting one budget across three illuminations would make each of them worse
        # for no reason.
        self._references = {}
        for index, (name, records) in enumerate(sorted(groups.items())):
            self._references[name] = self._fit_one(
                name, records, ctx, group_index=index, groups=len(groups)
            )

    def _fit_one(
        self,
        name: str,
        records: Sequence[ImageRecord],
        ctx: TrainContext,
        *,
        group_index: int,
        groups: int,
    ) -> _Reference:
        label = f" for channel {name}" if name and groups > 1 else ""
        chosen = evenly_spaced(len(records), self.config.max_reference_images)
        if len(chosen) < len(records):
            ctx.log(
                f"building the reference{label} from {len(chosen)} of {len(records)} training "
                f"images, sampled evenly (max_reference_images={self.config.max_reference_images})"
            )
        else:
            ctx.log(f"building the reference{label} from all {len(chosen)} training images")

        stack = np.empty(
            (
                len(chosen),
                ctx.preprocessing.height,
                ctx.preprocessing.width,
                ctx.preprocessing.channels,
            ),
            dtype=np.float32,
        )
        span = 1.0 / groups
        base = group_index * span
        for position, index in enumerate(chosen):
            ctx.raise_if_cancelled()
            stack[position] = load_array(records[index].path, ctx.preprocessing)
            ctx.progress(
                base + span * 0.8 * (position + 1) / len(chosen),
                f"read {position + 1}/{len(chosen)}{label}",
            )

        ctx.progress(base + span * 0.85, f"computing the per-pixel median{label}")
        median = np.median(stack, axis=0).astype(np.float32)

        ctx.progress(base + span * 0.95, f"computing the per-pixel deviation{label}")
        deviation = np.median(np.abs(stack - median), axis=0).astype(np.float32)
        scale = np.maximum(deviation * MAD_TO_SIGMA, self.config.mad_floor).astype(np.float32)

        suffix = f"_{name}" if name and groups > 1 else ""
        ctx.metric(f"reference_images{suffix}", float(len(chosen)))
        ctx.metric(f"median_scale{suffix}", float(np.median(scale)))
        ctx.log(
            f"per-pixel scale{label}: median {np.median(scale):.4f}, "
            f"{float((scale <= self.config.mad_floor).mean()) * 100:.1f}% of pixels at the floor"
        )

        # Keyed per channel, so N references produce N entries. The diagnostics index
        # renders by `kind` and never by method name, so this needs no UI change at all —
        # which is the check that the plugin boundary held.
        ctx.emit_diagnostic(
            f"reference_median{suffix}",
            f"Reference median{label}",
            DiagnosticKind.IMAGE if median.shape[-1] == 3 else DiagnosticKind.MAP,
            median if median.shape[-1] == 3 else median[:, :, 0],
            description="What this model considers a normal part to look like.",
        )
        ctx.emit_diagnostic(
            f"reference_scale{suffix}",
            f"Per-pixel deviation{label}",
            DiagnosticKind.MAP,
            scale.max(axis=2),
            description=(
                "How much each pixel varies across normal images. Bright regions "
                "tolerate more variation before being called anomalous."
            ),
        )
        return _Reference(median=median, scale=scale)

    def predict(self, images: Sequence[ImageRecord], ctx: InferContext) -> list[Prediction]:
        if not self._references:
            msg = "pixel_reference was asked to predict before it was fitted or loaded"
            raise RuntimeError(msg)

        predictions: list[Prediction] = []
        for index, record in enumerate(images):
            ctx.raise_if_cancelled()
            started = time.perf_counter()

            key = self._key(record)
            reference = self._references.get(key)
            if reference is None:
                # No silent fall back to another channel's reference: scoring a dark-field
                # image against the bright-field normal would produce a plausible-looking
                # map of nothing but the difference between two illuminations.
                available = ", ".join(sorted(name or "unassigned" for name in self._references))
                msg = (
                    f"pixel_reference has no reference for channel "
                    f"{record.channel or 'unassigned'}; it was fitted on {available}. "
                    "Train on the channels this run scores, or select them on the experiment."
                )
                raise RuntimeError(msg)

            array = load_array(record.path, ctx.preprocessing)
            deviation = np.abs(array - reference.median) / reference.scale
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
        if not self._references:
            msg = "pixel_reference has nothing to save; it was never fitted"
            raise RuntimeError(msg)
        names = sorted(self._references)
        # A plain unicode array for the names, so the whole file still loads with
        # `allow_pickle=False`. Arrays are keyed by index rather than by channel name,
        # because a channel name is free text and `npz` keys are filenames inside a zip.
        payload: dict[str, np.ndarray] = {"names": np.array(names, dtype=np.str_)}
        for index, name in enumerate(names):
            payload[f"median_{index}"] = self._references[name].median
            payload[f"scale_{index}"] = self._references[name].scale
        # numpy's stub types the second positional as `allow_pickle`; the call is the
        # documented keyword-arrays form.
        np.savez_compressed(artifact_dir / REFERENCE_FILENAME, **payload)  # type: ignore[arg-type]

    def load(self, artifact_dir: Path) -> None:
        with np.load(artifact_dir / REFERENCE_FILENAME, allow_pickle=False) as stored:
            if "names" not in stored:
                # A checkpoint written before references were keyed. One reference, and it
                # applied to everything — which is exactly `dataset` scope.
                self._references = {
                    UNASSIGNED: _Reference(median=stored["median"], scale=stored["scale"])
                }
                return
            self._references = {
                str(name): _Reference(
                    median=stored[f"median_{index}"], scale=stored[f"scale_{index}"]
                )
                for index, name in enumerate(stored["names"])
            }

    def export_onnx(
        self,
        destination: Path,
        preprocessing: PreprocessingConfig,
    ) -> OnnxGraphContract:
        """Write the complete robust z-map computation as a static ONNX graph.

        The graph owns method-specific normalization, channel reduction and smoothing.
        Decode/colour conversion and the percentile score remain explicit bundle
        contracts, so the Rust consumer performs exactly the same host operations.
        """
        if not self._references:
            raise RuntimeError("pixel_reference has no fitted reference to export")
        if len(self._references) > 1:
            # A static single-input graph holds one reference tensor. Refused inside the
            # plugin rather than pre-checked in a route, because a route that branched on
            # a method's own config would be exactly the leak ADR-0007 forbids.
            named = ", ".join(sorted(name or "unassigned" for name in self._references))
            msg = (
                f"pixel_reference fitted {len(self._references)} references ({named}) under "
                "reference_scope=channel, and one ONNX graph cannot carry more than one. "
                "Export a run whose channel selection leaves a single channel, or refit "
                "with reference_scope=dataset."
            )
            raise ValueError(msg)
        reference = self._single
        if reference.median.shape != (
            preprocessing.height,
            preprocessing.width,
            preprocessing.channels,
        ):
            raise ValueError(
                "stored pixel reference shape does not match the experiment preprocessing contract"
            )

        # Lazy even though ONNX is a normal application dependency: reading the method
        # catalogue must stay a cheap pydantic operation (ADR-0007).
        import onnx
        from onnx import TensorProto, helper, numpy_helper

        opset = 18
        input_name = "image"
        output_name = "anomaly_map"
        initializers = [
            numpy_helper.from_array(
                np.ascontiguousarray(reference.median.transpose(2, 0, 1)[np.newaxis]),
                name="reference_median",
            ),
            numpy_helper.from_array(
                np.ascontiguousarray(reference.scale.transpose(2, 0, 1)[np.newaxis]),
                name="reference_scale",
            ),
            numpy_helper.from_array(np.asarray([1], dtype=np.int64), name="channel_axis"),
            numpy_helper.from_array(np.asarray([1], dtype=np.int64), name="map_axis"),
        ]
        nodes = [
            helper.make_node("Sub", [input_name, "reference_median"], ["difference"]),
            helper.make_node("Abs", ["difference"], ["absolute_difference"]),
            helper.make_node("Div", ["absolute_difference", "reference_scale"], ["channel_z"]),
            helper.make_node("ReduceMax", ["channel_z", "channel_axis"], ["raw_map"], keepdims=0),
            helper.make_node("Unsqueeze", ["raw_map", "map_axis"], ["map_nchw"]),
        ]

        sigma = self.config.smoothing_sigma
        graph_output = "map_nchw"
        if sigma > 0:
            radius = max(1, round(3.0 * sigma))
            offsets = np.arange(-radius, radius + 1, dtype=np.float64)
            kernel = np.exp(-(offsets**2) / (2.0 * sigma * sigma))
            kernel = np.asarray(kernel / kernel.sum(), dtype=np.float32)
            initializers.extend(
                [
                    numpy_helper.from_array(
                        kernel.reshape(1, 1, kernel.size, 1), name="vertical_kernel"
                    ),
                    numpy_helper.from_array(
                        kernel.reshape(1, 1, 1, kernel.size), name="horizontal_kernel"
                    ),
                    numpy_helper.from_array(
                        np.asarray([0, 0, radius, 0, 0, 0, radius, 0], dtype=np.int64),
                        name="vertical_pads",
                    ),
                    numpy_helper.from_array(
                        np.asarray([0, 0, 0, radius, 0, 0, 0, radius], dtype=np.int64),
                        name="horizontal_pads",
                    ),
                ]
            )
            nodes.extend(
                [
                    helper.make_node(
                        "Pad", ["map_nchw", "vertical_pads"], ["vertical_padded"], mode="edge"
                    ),
                    helper.make_node(
                        "Conv", ["vertical_padded", "vertical_kernel"], ["vertical_blur"]
                    ),
                    helper.make_node(
                        "Pad",
                        ["vertical_blur", "horizontal_pads"],
                        ["horizontal_padded"],
                        mode="edge",
                    ),
                    helper.make_node(
                        "Conv", ["horizontal_padded", "horizontal_kernel"], [output_name]
                    ),
                ]
            )
            graph_output = output_name
        else:
            nodes.append(helper.make_node("Identity", ["map_nchw"], [output_name]))
            graph_output = output_name

        graph = helper.make_graph(
            nodes,
            "visual-anomaly-lab pixel_reference",
            [
                helper.make_tensor_value_info(
                    input_name,
                    TensorProto.FLOAT,
                    [1, preprocessing.channels, preprocessing.height, preprocessing.width],
                )
            ],
            [
                helper.make_tensor_value_info(
                    graph_output,
                    TensorProto.FLOAT,
                    [1, 1, preprocessing.height, preprocessing.width],
                )
            ],
            initializer=initializers,
        )
        model = helper.make_model(
            graph,
            opset_imports=[helper.make_opsetid("", opset)],
            producer_name="visual-anomaly-lab",
            producer_version="0.1.0",
        )
        onnx.checker.check_model(model)
        destination.parent.mkdir(parents=True, exist_ok=True)
        onnx.save_model(model, destination)
        return OnnxGraphContract(
            opset=opset,
            input_name=input_name,
            output_name=output_name,
            score=PercentileReducer(percentile=self.config.score_percentile),
        )

    def portable_reference(self, input_nchw: np.ndarray) -> tuple[np.ndarray, float]:
        """The fitted Python path over the tensor used by deployment parity."""
        if not self._references:
            raise RuntimeError("pixel_reference has no fitted reference for parity")
        reference = self._single
        if input_nchw.shape != (1, reference.median.shape[2], *reference.median.shape[:2]):
            raise ValueError("portable parity input shape does not match the stored reference")
        array = np.ascontiguousarray(input_nchw[0].transpose(1, 2, 0), dtype=np.float32)
        deviation = np.abs(array - reference.median) / reference.scale
        raw = deviation.max(axis=2)
        anomaly_map = gaussian_blur(raw.astype(np.float64), self.config.smoothing_sigma).astype(
            np.float32
        )
        return anomaly_map, float(np.percentile(anomaly_map, self.config.score_percentile))
