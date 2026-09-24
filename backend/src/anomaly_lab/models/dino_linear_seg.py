"""`dino_linear_seg`: a linear head on frozen DINO patch features, for semantic segmentation.

The first deep method of supervised segmentation (ADR-0039), and deliberately the simplest one
that can learn texture and context rather than colour: the shared frozen encoder
(`dino_backbone.image_patch_features`) gives every patch a feature, and one softmax classifier —
a 1x1 convolution over the patch grid — maps it to background and the pinned classes.

**It is trained at pixels, and predicts at pixels, with one interpolation between them.** The
head is linear, so the logits of a bilinearly interpolated feature are the bilinear
interpolation of the logits. Training reads each sampled pixel's feature by interpolating the
patch grid at that pixel (`pixel_features`, which reproduces the resize `refine.upsample` does),
and prediction upsamples the patch-grid logits with `refine.upsample` before the argmax — so the
loss is taken on exactly the function the label map is drawn from, and a pixel's label is never
the label of the patch it happens to sit in.

- **The pixel sample is bounded before anything is encoded** (`plan_pixels`): images first,
  evenly spaced, then at most `pixels_per_image` labelled pixels from each, and the float32
  footprint is logged and refused above a ceiling. Pixels marked `IGNORE_INDEX` are never
  sampled.
- **Each class in an image gets a share of that image's pixels** (`sample_pixels`,
  `pixel_sampling = per_class`): the budget is split equally among the classes present, a class
  with fewer pixels than its share gives the rest back to the others, and each class's pixels
  are evenly spaced over its own raster order. Sampled evenly over the whole image instead
  (`raster`), a defect covering a fraction of a percent of the frame gets a pixel or two.
- **The head is fitted on the CPU**, with AdamW on shuffled minibatches, from a seeded
  generator that draws both its initial weights and the batch order — nothing reads torch's
  global stream, so a seed is the whole answer. It is small enough that the accelerator would
  buy nothing, and the CPU makes the fit reproducible bit for bit.
- **`class_balancing`** weights the cross-entropy by inverse class frequency over the sample.
  Per-class sampling balances the classes *within* an image; the weights balance what is left
  *across* images, since an image without a class still contributes background.
- A class with no training pixel is never predicted, as in `color_classifier`: the fitted head
  would still hold a row for it, trained only to lose.

The label map is the argmax; the anomaly map beside it is the probability of anything but
background; the image's score is the share of pixels given a class, as for every supervised
segmentation method.
"""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from anomaly_lab.domain.entities import Task
from anomaly_lab.models.base import (
    IGNORE_INDEX,
    AnomalyModel,
    Availability,
    Capabilities,
    Device,
    ImageRecord,
    InferContext,
    Prediction,
    TrainContext,
    evenly_spaced,
    module_available,
)
from anomaly_lab.models.dino_backbone import (
    BACKBONES,
    DinoBackbone,
    FeatureLayers,
    FrozenEncoder,
    image_patch_features,
    patch_grid,
    validate_prepared_size,
)
from anomaly_lab.models.preprocessing import PreprocessingConfig, load_array
from anomaly_lab.models.refine import (
    GUIDED_EPS,
    GUIDED_RADIUS,
    Refinement,
    guided_filter,
    upsample,
)
from anomaly_lab.schemas import API_MODEL_CONFIG

STATE_FILENAME = "dino_linear_seg.npz"
META_FILENAME = "dino_linear_seg.json"
MAX_FEATURE_BYTES = 4 * 1024**3
"""The largest training sample a fit may hold in memory, as float32 features.

Four GiB, which a laptop survives beside the encoder. The plan is refused above it by name,
rather than left to find out halfway through the encoding pass."""


class ClassBalancing(StrEnum):
    NONE = "none"
    """Every sampled pixel counts once."""
    INVERSE_FREQUENCY = "inverse_frequency"
    """Each class's pixels weigh in inverse proportion to how many of them were sampled."""


class PixelSampling(StrEnum):
    PER_CLASS = "per_class"
    """Each class present in an image gets an equal share of the image's pixel budget."""
    RASTER = "raster"
    """The budget is spaced evenly over the image's labelled pixels, whatever their class."""


class DinoLinearSegConfig(BaseModel):
    model_config = API_MODEL_CONFIG

    backbone: DinoBackbone = Field(
        default=DinoBackbone.DINOV2_VIT_B14,
        description=(
            "Frozen encoder. The DINOv3 entries are licence-gated (an approved HF_TOKEN must "
            "be in the environment), so the default is the ungated DINOv2 ViT-B/14."
        ),
    )
    layers: FeatureLayers = Field(
        default=FeatureLayers.LAST,
        description=(
            "Which blocks the patch features are read from, each normalised, then joined. "
            "More blocks widen the feature and the memory the training sample takes."
        ),
    )
    pixels_per_image: int = Field(
        default=1024,
        ge=16,
        le=65_536,
        description=(
            "At most this many labelled pixels are sampled from each training image. "
            "Neighbouring pixels share a patch, so more buys little."
        ),
    )
    pixel_sampling: PixelSampling = Field(
        default=PixelSampling.PER_CLASS,
        description=(
            "'per_class' splits each image's pixels equally among the classes present in it, "
            "so a small defect is sampled as often as the background around it; 'raster' "
            "spaces them evenly over the image, where a small class gets almost none."
        ),
    )
    max_training_pixels: int = Field(
        default=131_072,
        ge=1_000,
        le=2_000_000,
        description=(
            "At most this many pixels train the head in total. Images are sampled evenly first, "
            "then pixels within each, and the plan and its memory are logged before encoding."
        ),
    )
    epochs: int = Field(
        default=10, ge=1, le=500, description="Passes of the head over the sampled pixels."
    )
    batch_size: int = Field(
        default=1024, ge=16, le=65_536, description="Pixels per optimisation step."
    )
    learning_rate: float = Field(
        default=1e-3, gt=0.0, le=1.0, description="AdamW learning rate of the head."
    )
    weight_decay: float = Field(
        default=1e-4, ge=0.0, le=1.0, description="AdamW weight decay of the head."
    )
    class_balancing: ClassBalancing = Field(
        default=ClassBalancing.INVERSE_FREQUENCY,
        description=(
            "'inverse_frequency' weights each class's pixels by the inverse of how often it was "
            "sampled, so a small class is not drowned by background; 'none' counts every pixel "
            "once."
        ),
    )
    refine: Refinement = Field(
        default=Refinement.BILINEAR,
        description=(
            "How the patch-grid logits reach pixels: 'bilinear', which is the function the head "
            "was trained on, or 'guided', which also moves each class's edges onto the image's."
        ),
    )
    pretrained_backbone: bool = Field(
        default=True,
        description="Use the published weights. Off gives a seeded random ViT, for hermetic tests.",
    )
    allow_downloads: bool = Field(
        default=True,
        description="Permit fetching the encoder weights; off makes a missing encoder an error.",
    )
    seed: int = Field(
        default=0,
        description=(
            "Seeds the head's initial weights, the order of its minibatches and — with "
            "pretrained_backbone off — the encoder."
        ),
    )


# ------------------------------------------------------------------------ the plan


@dataclass(frozen=True)
class PixelPlan:
    """What the training sample will hold, resolved before a single image is encoded.

    Torch-free and pure, so the arithmetic that decides whether the fit fits is checked by the
    CI job without the `dl` extra. Images are capped first and pixels within each surviving
    image second, as every plan here composes its caps (methods.md, "Memory planning").
    """

    images_available: int
    images_used: int
    pixels_per_image: int
    feature_dim: int

    @property
    def max_pixels(self) -> int:
        """The most pixels the sample can hold; fewer when images have fewer labelled pixels."""
        return self.images_used * self.pixels_per_image

    @property
    def feature_bytes(self) -> int:
        return self.max_pixels * self.feature_dim * 4

    def describe(self) -> str:
        dropped = self.images_available - self.images_used
        images = (
            f"{self.images_used} of {self.images_available} training images, sampled evenly"
            if dropped
            else f"all {self.images_available} training images"
        )
        return (
            f"pixel plan: {images}; at most {self.pixels_per_image} labelled pixels from each, "
            f"so at most {self.max_pixels} pixels of {self.feature_dim} features "
            f"({self.feature_bytes / 1024**2:.1f} MiB as float32)"
        )


def plan_pixels(
    images: int, *, pixels_per_image: int, max_training_pixels: int, feature_dim: int
) -> PixelPlan:
    """Bound the training sample: images first, then pixels per image, under one total.

    An image contributes at least one pixel, so a training set larger than the total keeps an
    evenly spaced `max_training_pixels` of its images. Refused, by name and with the knobs that
    shrink it, when the features would not fit in `MAX_FEATURE_BYTES`.
    """
    if images < 1:
        raise ValueError("the head needs at least one training image")
    used = min(images, max_training_pixels)
    per_image = max(1, min(pixels_per_image, max_training_pixels // used))
    plan = PixelPlan(
        images_available=images,
        images_used=used,
        pixels_per_image=per_image,
        feature_dim=feature_dim,
    )
    if plan.feature_bytes > MAX_FEATURE_BYTES:
        msg = (
            f"{plan.describe()} would exceed the {MAX_FEATURE_BYTES / 1024**3:.0f} GiB the "
            "training sample may take; lower max_training_pixels, or read fewer layers"
        )
        raise ValueError(msg)
    return plan


def allocate_pixels(counts: np.ndarray, budget: int) -> np.ndarray:
    """`(C,)` pixels to take from each class of one image, from its `(C,)` labelled counts.

    Water-filling: the classes present split `budget` equally, visited from the smallest, and a
    class with fewer pixels than its share takes all it has and leaves the rest to the classes
    after it. The total never exceeds `budget`, and reaches it whenever the image holds that
    many labelled pixels.
    """
    counts = np.asarray(counts, dtype=np.int64)
    taken = np.zeros_like(counts)
    present = [int(index) for index in np.argsort(counts, kind="stable") if counts[index] > 0]
    remaining = int(budget)
    for position, index in enumerate(present):
        share = -(-remaining // (len(present) - position))
        taken[index] = min(int(counts[index]), share)
        remaining -= int(taken[index])
    return taken


def sample_pixels(
    truth: np.ndarray, classes: int, budget: int, sampling: PixelSampling
) -> np.ndarray:
    """Sorted flat indices of at most `budget` labelled pixels of one flat label map.

    A pixel is labelled when it is neither `IGNORE_INDEX` nor outside the `classes` modelled.
    `raster` spaces the budget evenly over them; `per_class` gives each class its
    `allocate_pixels` share, evenly spaced over that class's own pixels. Deterministic.
    """
    labelled = (truth != IGNORE_INDEX) & (truth >= 0) & (truth < classes)
    if sampling is PixelSampling.RASTER:
        indices = np.flatnonzero(labelled)
        return indices[np.asarray(evenly_spaced(len(indices), budget), dtype=np.int64)]
    counts = np.bincount(truth[labelled], minlength=classes)[:classes]
    picked: list[np.ndarray] = []
    for index, take in enumerate(allocate_pixels(counts, budget)):
        if take:
            members = np.flatnonzero(truth == index)
            picked.append(members[np.asarray(evenly_spaced(len(members), int(take)))])
    return np.sort(np.concatenate(picked)) if picked else np.zeros(0, dtype=np.int64)


def feature_dim(backbone: DinoBackbone, layers: FeatureLayers) -> int:
    """Width of one patch feature: the encoder's width times the blocks read."""
    return BACKBONES[backbone].embedding_dim * len(layers.indices)


# ------------------------------------------------------------------------ the arithmetic


def class_weights(counts: np.ndarray, balancing: ClassBalancing) -> np.ndarray:
    """`(C,)` loss weight per class from its sampled pixel count.

    `inverse_frequency` gives a class `N / (C_present * n_c)`, so every class that was sampled
    carries the same total weight; a class that was not sampled weighs nothing either way.
    """
    counts = np.asarray(counts, dtype=np.float64)
    present = counts > 0
    if balancing is ClassBalancing.NONE:
        return present.astype(np.float32)
    weights = np.zeros_like(counts)
    weights[present] = counts.sum() / (present.sum() * counts[present])
    return weights.astype(np.float32)


def _axis(positions: np.ndarray, source: int, target: int) -> tuple[np.ndarray, ...]:
    """Pillow's bilinear upsampling along one axis: the two neighbours and the far weight."""
    centre = (positions + 0.5) * (source / target) - 0.5
    low = np.floor(centre)
    weight = (centre - low).astype(np.float32)
    first = np.clip(low, 0, source - 1).astype(np.int64)
    second = np.clip(low + 1, 0, source - 1).astype(np.int64)
    return first, second, weight


def pixel_features(
    grid_features: np.ndarray, pixels: np.ndarray, grid: tuple[int, int], size: tuple[int, int]
) -> np.ndarray:
    """`(n, D)` features at flat pixel indices of a `size` (`(w, h)`) frame.

    The patch grid's `(rows * cols, D)` features, interpolated at each pixel exactly as
    `refine.upsample` resizes a grid — pixel centres aligned, edges clamped — so that a linear
    head trained on these rows and applied to the grid before upsampling is one function.
    """
    rows, cols = grid
    width, height = size
    ys, xs = np.divmod(np.asarray(pixels, dtype=np.int64), width)
    y0, y1, wy = _axis(ys, rows, height)
    x0, x1, wx = _axis(xs, cols, width)
    table = grid_features.reshape(rows, cols, -1)
    top = table[y0, x0] * (1 - wx)[:, None] + table[y0, x1] * wx[:, None]
    bottom = table[y1, x0] * (1 - wx)[:, None] + table[y1, x1] * wx[:, None]
    return np.asarray(top * (1 - wy)[:, None] + bottom * wy[:, None], dtype=np.float32)


def _softmax(logits: np.ndarray, axis: int) -> np.ndarray:
    shifted = logits - logits.max(axis=axis, keepdims=True)
    weights = np.exp(shifted)
    return np.asarray(weights / weights.sum(axis=axis, keepdims=True), dtype=np.float32)


# ------------------------------------------------------------------------ the plugin


class DinoLinearSegModel(AnomalyModel):
    """A softmax head on frozen DINO patch features, trained on labelled pixels."""

    title = "DINO linear head (segmentation)"
    summary = (
        "A per-pixel softmax classifier on frozen DINO patch features, fitted on a bounded "
        "sample of annotated pixels; logits are upsampled to the image before the argmax."
    )

    def __init__(self, config: DinoLinearSegConfig) -> None:
        super().__init__(config)
        self.config = config
        self._encoder = FrozenEncoder(
            config.backbone,
            pretrained=config.pretrained_backbone,
            allow_downloads=config.allow_downloads,
            seed=config.seed,
            method="dino_linear_seg",
        )
        self._state: dict[str, np.ndarray] = {}

    @classmethod
    def config_model(cls) -> type[BaseModel]:
        return DinoLinearSegConfig

    @classmethod
    def check_input(cls, config: BaseModel, preprocessing: PreprocessingConfig) -> None:
        if not isinstance(config, DinoLinearSegConfig):
            raise TypeError(f"expected DinoLinearSegConfig, got {type(config).__name__}")
        validate_prepared_size(config.backbone, preprocessing.width, preprocessing.height)

    @classmethod
    def capabilities(cls) -> Capabilities:
        return Capabilities(
            tasks=[Task.SEMANTIC_SEGMENTATION],
            requires_training=True,
            produces_anomaly_map=True,
            preferred_device=Device.MPS,
        )

    @classmethod
    def availability(cls) -> Availability:
        return module_available("torch", "dl", "the DINO linear segmentation head")

    def _grid_features(self, record: ImageRecord, ctx: TrainContext | InferContext) -> np.ndarray:
        """`(rows * cols, D)` unit features of one image, on the CPU."""
        encoder = self._encoder.model(ctx.device.value, ctx.cache_dir)
        features: Any = image_patch_features(
            encoder, [record], ctx.preprocessing, self.config.layers.indices, ctx.device.value
        )
        return np.asarray(features.numpy()[0], dtype=np.float32)

    # ------------------------------------------------------------------ fit

    def fit(self, train: Sequence[ImageRecord], ctx: TrainContext) -> None:
        targets = ctx.label_targets
        if targets is None:
            msg = "dino_linear_seg segments annotated classes and needs label targets"
            raise RuntimeError(msg)
        width, height = ctx.preprocessing.width, ctx.preprocessing.height
        grid = patch_grid(self.config.backbone, width, height)
        names = ("background", *targets.classes)
        size = len(names)

        plan = plan_pixels(
            len(train),
            pixels_per_image=self.config.pixels_per_image,
            max_training_pixels=self.config.max_training_pixels,
            feature_dim=feature_dim(self.config.backbone, self.config.layers),
        )
        ctx.log(plan.describe())
        chosen = [train[index] for index in evenly_spaced(len(train), plan.images_used)]

        ctx.log(f"pixel sampling: {self.config.pixel_sampling.value}")

        features: list[np.ndarray] = []
        labels: list[np.ndarray] = []
        available = np.zeros(size, dtype=np.int64)
        for position, record in enumerate(chosen):
            ctx.raise_if_cancelled()
            truth = targets.labels(record.image_id).reshape(-1)
            labelled = truth[(truth != IGNORE_INDEX) & (truth >= 0) & (truth < size)]
            available += np.bincount(labelled, minlength=size)[:size]
            picked = sample_pixels(truth, size, plan.pixels_per_image, self.config.pixel_sampling)
            if len(picked):
                grid_features = self._grid_features(record, ctx)
                features.append(pixel_features(grid_features, picked, grid, (width, height)))
                labels.append(truth[picked].astype(np.int64))
            ctx.progress(
                0.6 * (position + 1) / len(chosen), f"encoded {position + 1}/{len(chosen)}"
            )

        sampled_labels = np.concatenate(labels) if labels else np.zeros(0, dtype=np.int64)
        counts = np.bincount(sampled_labels, minlength=size)[:size]
        for index, name in enumerate(names):
            if counts[index] == 0:
                ctx.log(
                    f"no training pixel of {name!r}; it is not learned and is never predicted",
                    level="warning",
                )
            ctx.metric(f"pixels_{name}", float(counts[index]))
        if not counts[1:].any():
            msg = f"the training images hold no pixel of any class of {', '.join(names[1:])}"
            raise RuntimeError(msg)
        ctx.log(
            f"training the head on {len(sampled_labels)} pixels: "
            + ", ".join(
                f"{name} {int(count)} of {int(total)}"
                for name, count, total in zip(names, counts, available, strict=True)
            )
        )

        weights, bias = self._train_head(
            np.concatenate(features),
            sampled_labels,
            class_weights(counts, self.config.class_balancing),
            ctx,
        )
        self._state = {
            "weights": weights,
            "bias": bias,
            "present": counts > 0,
            "classes": np.array([size - 1], dtype=np.int64),
        }
        ctx.progress(1.0, "head fitted")

    def _train_head(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        loss_weights: np.ndarray,
        ctx: TrainContext,
    ) -> tuple[np.ndarray, np.ndarray]:
        """AdamW on the softmax head, on the CPU, from one seeded generator."""
        import torch

        generator = torch.Generator().manual_seed(self.config.seed)
        inputs = torch.from_numpy(features)
        targets = torch.from_numpy(labels)
        classes = len(loss_weights)
        weight = torch.nn.Parameter(
            0.01 * torch.randn(classes, inputs.shape[1], generator=generator)
        )
        bias = torch.nn.Parameter(torch.zeros(classes))
        optimizer = torch.optim.AdamW(
            [weight, bias], lr=self.config.learning_rate, weight_decay=self.config.weight_decay
        )
        criterion = torch.nn.CrossEntropyLoss(
            weight=torch.from_numpy(loss_weights), ignore_index=IGNORE_INDEX
        )
        count = len(targets)
        batches = -(-count // self.config.batch_size)
        for epoch in range(self.config.epochs):
            order = torch.randperm(count, generator=generator)
            total = 0.0
            for start in range(0, count, self.config.batch_size):
                ctx.raise_if_cancelled()
                batch = order[start : start + self.config.batch_size]
                loss = criterion(inputs[batch] @ weight.T + bias, targets[batch])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total += float(loss.detach())
            ctx.metric("loss", total / batches, step=epoch + 1)
            ctx.progress(
                0.6 + 0.4 * (epoch + 1) / self.config.epochs,
                f"epoch {epoch + 1}/{self.config.epochs}",
            )
        return (
            weight.detach().numpy().astype(np.float32),
            bias.detach().numpy().astype(np.float32),
        )

    # ------------------------------------------------------------------ predict

    def predict(self, images: Sequence[ImageRecord], ctx: InferContext) -> list[Prediction]:
        if "weights" not in self._state:
            raise RuntimeError(
                "dino_linear_seg was asked to predict before it was fitted or loaded"
            )
        width, height = ctx.preprocessing.width, ctx.preprocessing.height
        rows, cols = patch_grid(self.config.backbone, width, height)
        classes = int(self._state["classes"][0])
        absent = ~self._state["present"]
        predictions: list[Prediction] = []
        for position, record in enumerate(images):
            ctx.raise_if_cancelled()
            started = time.perf_counter()
            patch_logits = self._grid_features(record, ctx) @ self._state["weights"].T
            patch_logits += self._state["bias"]
            logits = np.stack(
                [upsample(plane.reshape(rows, cols), (width, height)) for plane in patch_logits.T]
            )
            logits[absent] = -np.inf
            probability = _softmax(logits, axis=0)
            if self.config.refine is Refinement.GUIDED:
                image = load_array(record.path, ctx.preprocessing)
                grey = image.mean(axis=2) if image.ndim == 3 else image
                probability = np.stack(
                    [
                        np.clip(guided_filter(plane, grey, GUIDED_RADIUS, GUIDED_EPS), 0.0, 1.0)
                        for plane in probability
                    ]
                ).astype(np.float32)
                probability[absent] = 0.0
            labels = np.argmax(probability, axis=0)
            map_path = ctx.write_map(record.image_id, 1.0 - probability[0])
            label_path = ctx.write_label_map(record.image_id, labels, classes=classes)
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

    # ------------------------------------------------------------------ persistence

    def save(self, artifact_dir: Path) -> None:
        if "weights" not in self._state or self._encoder.fingerprint is None:
            raise RuntimeError("dino_linear_seg has nothing to save; it was never fitted")
        meta = {
            "backbone": self.config.backbone.value,
            "layers": self.config.layers.value,
            "fingerprint": self._encoder.fingerprint,
        }
        # Each file lands whole or not at all: written beside itself, then renamed over.
        state = artifact_dir / STATE_FILENAME
        state_tmp = artifact_dir / f"{STATE_FILENAME}.tmp.npz"
        meta_path = artifact_dir / META_FILENAME
        meta_tmp = artifact_dir / f"{META_FILENAME}.tmp"
        try:
            # numpy's stub types the second positional as `allow_pickle`; this is the keyword form.
            np.savez_compressed(state_tmp, **self._state)  # type: ignore[arg-type]
            meta_tmp.write_text(json.dumps(meta), encoding="utf-8")
            state_tmp.replace(state)
            meta_tmp.replace(meta_path)
        finally:
            state_tmp.unlink(missing_ok=True)
            meta_tmp.unlink(missing_ok=True)

    def load(self, artifact_dir: Path) -> None:
        meta = json.loads((artifact_dir / META_FILENAME).read_text(encoding="utf-8"))
        fitted = (meta["backbone"], meta["layers"])
        asked = (self.config.backbone.value, self.config.layers.value)
        if fitted != asked:
            msg = (
                f"this dino_linear_seg checkpoint was fitted with {fitted[0]} ({fitted[1]}), "
                f"not {asked[0]} ({asked[1]})"
            )
            raise RuntimeError(msg)
        self._encoder.expect(str(meta["fingerprint"]))
        with np.load(artifact_dir / STATE_FILENAME, allow_pickle=False) as stored:
            self._state = {key: stored[key] for key in stored.files}
