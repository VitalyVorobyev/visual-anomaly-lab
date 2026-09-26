"""`fss_dino`: a reproduction of FSSDINO (arXiv 2602.07550) for one class (ADR-0040).

Training-free. The references' last-block patch features are split by their masks into
the class and the background. Each side gets `k` prototypes by cosine k-means and a Gram
matrix. A query patch is compared with every prototype (cosine) and with each Gram matrix
(energy). Each class's maps are upsampled and combined as `mean * max`, and a pixel goes to
the higher class. The arithmetic is `models/prototypes.py`; this module drives the encoder.

**What is reproduced and what is not.** The paper segments every class of a taxonomy at
once with an argmax; here the classes are the target and its background, which is the same
rule with two classes. The paper's encoder is DINOv3 ViT-B/16 at 512 px. That encoder is
licence-gated, so the default here is the ungated DINOv2 ViT-B/14 and DINOv3 is one field
away. The prepared size is the experiment's. FSSDINO has no presence score; this one is a
high percentile of the foreground probability.

The method writes its argmax as its own mask (`InferContext.write_mask`). Its map is the
foreground's share of the two combined scores, which is at least 0.5 exactly where the
argmax picks the class, so the evaluator's rule reads it the same way. With `calibration`
at `leave_one_out` the map is rescaled on the references (`calibration.py`) and the mask is
the rescaled map at 0.5 — the argmax moved to where the references say the class begins.
"""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from anomaly_lab.domain.entities import Task
from anomaly_lab.models.base import (
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
from anomaly_lab.models.calibration import IDENTITY, Calibration, PlattScale, leave_one_out
from anomaly_lab.models.dino_backbone import (
    BACKBONES,
    DinoBackbone,
    FrozenEncoder,
    image_patch_features,
    native_frame,
    patch_grid,
    patch_multiple,
    validate_prepared_size,
)
from anomaly_lab.models.preprocessing import PreprocessingConfig
from anomaly_lab.models.prototypes import (
    class_maps,
    combined_score,
    foreground_probability,
    gram_matrix,
    patch_coverage,
    spherical_kmeans,
    split_patches,
)
from anomaly_lab.schemas import API_MODEL_CONFIG

STATE_FILENAME = "fss_dino.npz"
META_FILENAME = "fss_dino.json"
SIDES = ("foreground", "background")
MASK_CUT = 0.5
"""Where the calibrated map becomes the mask: the cut at which the unscaled map is the argmax."""


class FssDinoConfig(BaseModel):
    model_config = API_MODEL_CONFIG

    backbone: DinoBackbone = Field(
        json_schema_extra={"x-primary": True},
        default=DinoBackbone.DINOV2_VIT_B14,
        description=(
            "Frozen encoder; its last block is read. FSSDINO used DINOv3 ViT-B/16, which is "
            "licence-gated (an approved HF_TOKEN must be in the environment), so the default "
            "is the ungated DINOv2 ViT-B/14."
        ),
    )
    prototypes_per_class: int = Field(
        json_schema_extra={"x-primary": True},
        default=5,
        ge=1,
        le=64,
        description="Cosine k-means prototypes for the class and for the background. FSSDINO: 5.",
    )
    gram: bool = Field(
        default=True,
        description="Add each side's Gram-matrix energy map, FSSDINO's channel-correlation signal.",
    )
    kmeans_iterations: int = Field(
        default=20, ge=1, le=200, description="Iterations of the seeded cosine k-means."
    )
    max_features_per_class: int = Field(
        default=20_000,
        ge=100,
        le=500_000,
        description=(
            "At most this many reference patches per side feed k-means and the Gram matrix, "
            "sampled evenly. Linear in references; the value is not."
        ),
    )
    calibration: Calibration = Field(
        json_schema_extra={"x-primary": True},
        default=Calibration.NONE,
        description=(
            "'leave_one_out' rescales the foreground probability on the references: each is "
            "scored by prototypes of the others, and a Platt scale fitted to those pixels is "
            "applied to every map, mask and presence score, so the 0.5 cut means the same on "
            "every class. One reference cannot be left out and stays unscaled."
        ),
    )
    presence_percentile: float = Field(
        default=99.9,
        ge=50.0,
        le=100.0,
        description="The presence score is this percentile of the foreground probability.",
    )
    pretrained_backbone: bool = Field(
        default=True,
        description=(
            "Use the published self-supervised weights. Off gives a seeded random ViT, which "
            "is what the hermetic tests use."
        ),
    )
    allow_downloads: bool = Field(
        default=True,
        description="Permit fetching the encoder weights; off makes a missing encoder an error.",
    )
    seed: int = Field(
        default=0,
        description="Seeds k-means and, with pretrained_backbone off, the encoder initialisation.",
    )


class FssDinoModel(AnomalyModel):
    """FSSDINO's prototypes and Gram energy over a frozen DINO, for one class."""

    title = "FSSDINO (few-shot)"
    summary = (
        "Cosine prototypes and a Gram matrix for the class and its background, from frozen "
        "DINO patch features of the references. Training-free; a published baseline."
    )

    def __init__(self, config: FssDinoConfig) -> None:
        super().__init__(config)
        self.config = config
        self._prototypes: dict[str, np.ndarray] = {}
        self._grams: dict[str, np.ndarray] = {}
        self._calibration: PlattScale = IDENTITY
        self._encoder = FrozenEncoder(
            config.backbone,
            pretrained=config.pretrained_backbone,
            allow_downloads=config.allow_downloads,
            seed=config.seed,
            method="fss_dino",
        )

    @classmethod
    def config_model(cls) -> type[BaseModel]:
        return FssDinoConfig

    @classmethod
    def native_size(cls, config: BaseModel) -> tuple[int, int]:
        """448 px square: the few-shot gates ran at 448x448 (docs/measurements.md)."""
        if not isinstance(config, FssDinoConfig):
            raise TypeError(f"expected FssDinoConfig, got {type(config).__name__}")
        return native_frame(config.backbone, 448)

    @classmethod
    def size_multiple(cls, config: BaseModel) -> int:
        if not isinstance(config, FssDinoConfig):
            raise TypeError(f"expected FssDinoConfig, got {type(config).__name__}")
        return patch_multiple(config.backbone)

    @classmethod
    def check_input(cls, config: BaseModel, preprocessing: PreprocessingConfig) -> None:
        if not isinstance(config, FssDinoConfig):
            raise TypeError(f"expected FssDinoConfig, got {type(config).__name__}")
        validate_prepared_size(config.backbone, preprocessing.width, preprocessing.height)

    @classmethod
    def capabilities(cls) -> Capabilities:
        return Capabilities(
            tasks=[Task.FEW_SHOT_SEGMENTATION],
            requires_training=True,
            produces_anomaly_map=True,
            preferred_device=Device.MPS,
        )

    @classmethod
    def availability(cls) -> Availability:
        return module_available("torch", "dl", "FSSDINO")

    # ------------------------------------------------------------------ encoder

    def _layer(self) -> tuple[int, ...]:
        return (BACKBONES[self.config.backbone].depth - 1,)

    def _features(self, records: Sequence[ImageRecord], ctx: TrainContext | InferContext) -> Any:
        encoder = self._encoder.model(ctx.device.value, ctx.cache_dir)
        features = image_patch_features(
            encoder, records, ctx.preprocessing, self._layer(), ctx.device.value
        )
        return features.numpy()

    # ------------------------------------------------------------------ fit

    def fit(self, train: Sequence[ImageRecord], ctx: TrainContext) -> None:
        targets = ctx.targets
        if targets is None:
            raise RuntimeError("fss_dino segments a class and needs references with masks")
        width, height = ctx.preprocessing.width, ctx.preprocessing.height
        validate_prepared_size(self.config.backbone, width, height)
        grid = patch_grid(self.config.backbone, width, height)

        # Kept per reference, whole, so a leave-one-out fold can pool every reference but one
        # and score the one it left out.
        encoded: list[np.ndarray] = []
        sides: dict[str, list[np.ndarray]] = {side: [] for side in SIDES}
        for position, record in enumerate(train):
            ctx.raise_if_cancelled()
            features = self._features([record], ctx)[0]
            covered, background = split_patches(patch_coverage(targets.mask(record.image_id), grid))
            encoded.append(features)
            sides["foreground"].append(features[covered])
            sides["background"].append(features[background])
            ctx.progress(0.8 * (position + 1) / len(train), f"encoded {position + 1}/{len(train)}")

        self._prototypes, self._grams = self._build(
            sides, targets.label_key, np.random.default_rng(self.config.seed), ctx
        )
        self._calibration = IDENTITY
        if self.config.calibration is Calibration.LEAVE_ONE_OUT:
            ctx.progress(0.9, "calibrating on the references")
            self._calibration = leave_one_out(
                len(train),
                lambda fold: self._held_out(fold, train, encoded, sides, ctx),
                ctx.log,
                cancelled=ctx.raise_if_cancelled,
            )
            ctx.metric("calibration_slope", self._calibration.slope)
            ctx.metric("calibration_bias", self._calibration.bias)
        ctx.progress(1.0, "prototypes built")

    def _build(
        self,
        sides: dict[str, list[np.ndarray]],
        label_key: str,
        rng: np.random.Generator,
        ctx: TrainContext | None,
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """Prototypes and Gram matrices per side; `ctx` is `None` in a quiet fold."""
        prototypes: dict[str, np.ndarray] = {}
        grams: dict[str, np.ndarray] = {}
        for side in SIDES:
            parts = sides[side]
            pooled = np.concatenate(parts) if parts else np.empty((0, 1), dtype=np.float32)
            if len(pooled) == 0:
                msg = (
                    f"the references hold no {side} patch for {label_key!r}: a patch "
                    "is the class when half of it, or the most of any patch, is covered"
                )
                raise RuntimeError(msg)
            chosen = evenly_spaced(len(pooled), self.config.max_features_per_class)
            if ctx is not None:
                if len(chosen) < len(pooled):
                    ctx.log(
                        f"{side}: {len(chosen)} of {len(pooled)} reference patches, sampled "
                        f"evenly (max_features_per_class={self.config.max_features_per_class})"
                    )
                else:
                    ctx.log(f"{side}: all {len(pooled)} reference patches")
                ctx.metric(f"{side}_patches", float(len(chosen)))
            sample = pooled[np.asarray(chosen, dtype=np.int64)]
            prototypes[side] = spherical_kmeans(
                sample,
                self.config.prototypes_per_class,
                iterations=self.config.kmeans_iterations,
                rng=rng,
            )
            if self.config.gram:
                grams[side] = gram_matrix(sample)
        return prototypes, grams

    def _held_out(
        self,
        fold: int,
        train: Sequence[ImageRecord],
        encoded: list[np.ndarray],
        sides: dict[str, list[np.ndarray]],
        ctx: TrainContext,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Reference `fold`'s map from prototypes of the others, and its truth; `None`
        when the others hold no patch of one side.

        Each fold draws its own k-means stream from the seed, so the final prototypes — built
        first, from the seed alone — are the same with calibration on or off.
        """
        targets = ctx.targets
        if targets is None:
            raise RuntimeError("fss_dino calibrates on references with masks")
        others = {side: [p for i, p in enumerate(sides[side]) if i != fold] for side in SIDES}
        if any(sum(len(part) for part in others[side]) == 0 for side in SIDES):
            return None
        prototypes, grams = self._build(
            others, targets.label_key, np.random.default_rng([self.config.seed, fold + 1]), None
        )
        probability, _ = self._probability(encoded[fold], prototypes, grams, ctx)
        return probability, targets.mask(train[fold].image_id)

    # ------------------------------------------------------------------ predict

    def _probability(
        self,
        query: np.ndarray,
        prototypes: dict[str, np.ndarray],
        grams: dict[str, np.ndarray],
        ctx: TrainContext | InferContext,
    ) -> tuple[np.ndarray, np.ndarray]:
        """The unscaled foreground probability at pixels, and the argmax mask."""
        width, height = ctx.preprocessing.width, ctx.preprocessing.height
        grid = patch_grid(self.config.backbone, width, height)
        scores = {
            side: combined_score(
                class_maps(query, prototypes[side], grams.get(side)), grid, (width, height)
            )
            for side in SIDES
        }
        probability = foreground_probability(scores["foreground"], scores["background"])
        return probability, scores["foreground"] > scores["background"]

    def predict(self, images: Sequence[ImageRecord], ctx: InferContext) -> list[Prediction]:
        if not self._prototypes:
            raise RuntimeError("fss_dino was asked to predict before it was fitted or loaded")
        predictions: list[Prediction] = []
        for index, record in enumerate(images):
            ctx.raise_if_cancelled()
            started = time.perf_counter()
            query = self._features([record], ctx)[0]
            probability, argmax = self._probability(query, self._prototypes, self._grams, ctx)
            # The percentile of the unscaled map, then scaled: interpolating between two
            # pixels does not commute with the scale, and this keeps the ranking exact.
            presence = self._calibration.apply_score(
                float(np.percentile(probability, self.config.presence_percentile))
            )
            if not self._calibration.is_identity:
                probability = self._calibration.apply(probability)
                argmax = probability >= MASK_CUT
            map_path = ctx.write_map(record.image_id, probability)
            ctx.write_mask(record.image_id, argmax)
            predictions.append(
                Prediction(
                    image_id=record.image_id,
                    score=presence,
                    anomaly_map=map_path,
                    inference_ms=(time.perf_counter() - started) * 1000.0,
                )
            )
            ctx.progress((index + 1) / len(images), f"segmented {index + 1}/{len(images)}")
        return predictions

    # ------------------------------------------------------------------ persistence

    def save(self, artifact_dir: Path) -> None:
        if not self._prototypes or self._encoder.fingerprint is None:
            raise RuntimeError("fss_dino has nothing to save; it was never fitted")
        arrays = {f"prototypes_{side}": value for side, value in self._prototypes.items()}
        arrays.update({f"gram_{side}": value for side, value in self._grams.items()})
        arrays["calibration"] = self._calibration.to_array()
        # numpy's stub types the second positional as `allow_pickle`; this is the keyword form.
        np.savez_compressed(artifact_dir / STATE_FILENAME, **arrays)  # type: ignore[arg-type]
        meta = {"backbone": self.config.backbone.value, "fingerprint": self._encoder.fingerprint}
        (artifact_dir / META_FILENAME).write_text(json.dumps(meta), encoding="utf-8")

    def load(self, artifact_dir: Path) -> None:
        meta = json.loads((artifact_dir / META_FILENAME).read_text(encoding="utf-8"))
        if meta["backbone"] != self.config.backbone.value:
            msg = (
                f"this fss_dino checkpoint was fitted with {meta['backbone']}, not "
                f"{self.config.backbone.value}"
            )
            raise RuntimeError(msg)
        self._encoder.expect(str(meta["fingerprint"]))
        with np.load(artifact_dir / STATE_FILENAME, allow_pickle=False) as stored:
            self._prototypes = {
                key.removeprefix("prototypes_"): stored[key]
                for key in stored.files
                if key.startswith("prototypes_")
            }
            self._grams = {
                key.removeprefix("gram_"): stored[key]
                for key in stored.files
                if key.startswith("gram_")
            }
            self._calibration = PlattScale.from_array(
                stored["calibration"] if "calibration" in stored.files else None
            )
