"""`proto_seg`: our few-shot segmenter over a frozen, positionally debiased DINO (ADR-0040).

Training-free by default, and built from the shared blocks so each axis is a field:

- **Debiased features.** Patch features from the chosen blocks, with INSID3's positional
  subspace projected out (`positional.py`), so a reference teaches what the class looks
  like and not where it sat.
- **A hybrid prototype bank.** Each side — the class and its background — keeps its mean
  direction (the global prototype) and `clusters_per_class` cosine k-means prototypes (the
  parts and modes a mean averages away).
- **LSE scoring.** A patch's foreground probability is the class's share of a softmax over
  every prototype at `temperature`: a soft maximum per side, compared.
- **`adaptation`.** `training_free` scores with the bank. `linear_adapt` fits a
  class-balanced logistic probe on the references' debiased patches and scores with it —
  the cheapest adaptation there is, and a measurement of whether the bank leaves anything
  on the table.
- **`refine`.** The patch-grid probability is brought to pixels by `bilinear` or `guided`
  (`refine.py`).
- **Presence.** The mean of the `presence_patches` most confident patches, so a claim has
  to be carried by a region rather than one patch.
- **`calibration`.** `leave_one_out` fits a Platt scale on the references, each scored by a
  bank of the others (`calibration.py`), and applies it to the map and the presence score.
  The bank itself is the one an uncalibrated run builds.
"""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from enum import StrEnum
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
    DinoBackbone,
    FeatureLayers,
    FrozenEncoder,
    image_patch_features,
    native_frame,
    noise_patch_features,
    patch_grid,
    patch_multiple,
    validate_prepared_size,
)
from anomaly_lab.models.positional import INSID3_RANK, debias, positional_basis, resolved_rank
from anomaly_lab.models.preprocessing import PreprocessingConfig, load_array
from anomaly_lab.models.prototypes import (
    fit_linear_probe,
    lse_probability,
    mean_prototype,
    patch_coverage,
    presence_score,
    probe_probability,
    spherical_kmeans,
    split_patches,
)
from anomaly_lab.models.refine import Refinement, refine
from anomaly_lab.schemas import API_MODEL_CONFIG

STATE_FILENAME = "proto_seg.npz"
META_FILENAME = "proto_seg.json"
SIDES = ("foreground", "background")


class Adaptation(StrEnum):
    TRAINING_FREE = "training_free"
    """Score with the prototype bank; nothing is fitted but the bank itself."""
    LINEAR_ADAPT = "linear_adapt"
    """Fit a logistic probe on the references' patches, and score with it."""


class ProtoSegConfig(BaseModel):
    model_config = API_MODEL_CONFIG

    backbone: DinoBackbone = Field(
        json_schema_extra={"x-primary": True},
        default=DinoBackbone.DINOV2_VIT_B14,
        description=(
            "Frozen encoder. The DINOv3 entries are licence-gated (an approved HF_TOKEN must "
            "be in the environment), so the default is the ungated DINOv2 ViT-B/14."
        ),
    )
    layers: FeatureLayers = Field(
        default=FeatureLayers.LAST_TWO,
        description="Which blocks the patch features are read from, each normalised, then joined.",
    )
    positional_debias: bool = Field(
        default=True,
        description=(
            "Project out the positional subspace a noise image reveals (INSID3), so matching "
            "rewards what a patch shows rather than where it is."
        ),
    )
    positional_rank: int = Field(
        default=INSID3_RANK,
        ge=1,
        le=4096,
        description=(
            "Directions removed. INSID3 uses 500 for ViT-L; it is clipped to half of the "
            "smaller of the patch count and the feature width, and the resolved rank is logged."
        ),
    )
    clusters_per_class: int = Field(
        default=8,
        ge=0,
        le=64,
        description=(
            "Cosine k-means prototypes per side, beside the mean prototype. 0 keeps the mean only."
        ),
    )
    temperature: float = Field(
        default=0.1,
        gt=0.0,
        le=1.0,
        description="Softmax temperature over cosine similarities. Lower is a harder maximum.",
    )
    adaptation: Adaptation = Field(
        json_schema_extra={"x-primary": True},
        default=Adaptation.TRAINING_FREE,
        description=(
            "'training_free' scores with the prototype bank. 'linear_adapt' fits a "
            "class-balanced logistic probe on the references' patches and scores with it."
        ),
    )
    refine: Refinement = Field(
        default=Refinement.GUIDED,
        description=(
            "How the patch-grid probability reaches pixels: 'bilinear', or 'guided', which "
            "moves transitions onto the image's own edges."
        ),
    )
    calibration: Calibration = Field(
        json_schema_extra={"x-primary": True},
        default=Calibration.NONE,
        description=(
            "'leave_one_out' rescales the foreground probability on the references: each is "
            "scored by a bank of the others, and a Platt scale fitted to those pixels is "
            "applied to every map and presence score, so the 0.5 cut means the same on every "
            "class. One reference cannot be left out and stays unscaled."
        ),
    )
    presence_patches: int = Field(
        default=4,
        ge=1,
        le=256,
        description="Presence is the mean probability of this many most confident patches.",
    )
    max_features_per_class: int = Field(
        default=20_000,
        ge=100,
        le=500_000,
        description="At most this many reference patches per side, sampled evenly.",
    )
    kmeans_iterations: int = Field(
        default=20, ge=1, le=200, description="Cosine k-means iterations."
    )
    probe_iterations: int = Field(
        default=300, ge=1, le=10_000, description="Gradient steps of the linear probe."
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
            "Seeds k-means, the noise image behind the positional subspace, and — with "
            "pretrained_backbone off — the encoder."
        ),
    )


class ProtoSegModel(AnomalyModel):
    """Debiased DINO features, a hybrid prototype bank, LSE scoring, optional linear adaptation."""

    title = "Prototype segmenter (few-shot, ours)"
    summary = (
        "Positionally debiased DINO patches matched against a mean-plus-cluster prototype "
        "bank per side, or a linear probe; refined to the image's edges."
    )

    def __init__(self, config: ProtoSegConfig) -> None:
        super().__init__(config)
        self.config = config
        self._encoder = FrozenEncoder(
            config.backbone,
            pretrained=config.pretrained_backbone,
            allow_downloads=config.allow_downloads,
            seed=config.seed,
            method="proto_seg",
        )
        self._state: dict[str, np.ndarray] = {}
        self._calibration: PlattScale = IDENTITY

    @classmethod
    def config_model(cls) -> type[BaseModel]:
        return ProtoSegConfig

    @classmethod
    def native_size(cls, config: BaseModel) -> tuple[int, int]:
        """448 px square: the few-shot gates ran at 448x448 (docs/measurements.md)."""
        if not isinstance(config, ProtoSegConfig):
            raise TypeError(f"expected ProtoSegConfig, got {type(config).__name__}")
        return native_frame(config.backbone, 448)

    @classmethod
    def size_multiple(cls, config: BaseModel) -> int:
        if not isinstance(config, ProtoSegConfig):
            raise TypeError(f"expected ProtoSegConfig, got {type(config).__name__}")
        return patch_multiple(config.backbone)

    @classmethod
    def check_input(cls, config: BaseModel, preprocessing: PreprocessingConfig) -> None:
        if not isinstance(config, ProtoSegConfig):
            raise TypeError(f"expected ProtoSegConfig, got {type(config).__name__}")
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
        return module_available("torch", "dl", "the prototype segmenter")

    def _features(self, record: ImageRecord, ctx: TrainContext | InferContext) -> np.ndarray:
        encoder = self._encoder.model(ctx.device.value, ctx.cache_dir)
        features: Any = image_patch_features(
            encoder, [record], ctx.preprocessing, self.config.layers.indices, ctx.device.value
        )
        return debias(features.numpy()[0], self._state["basis"])

    def _basis(self, ctx: TrainContext) -> np.ndarray:
        """`(D, r)`, or `(1, 0)` — no direction removed — when debiasing is off."""
        if not self.config.positional_debias:
            return np.zeros((1, 0), dtype=np.float32)
        rows, cols = patch_grid(
            self.config.backbone, ctx.preprocessing.width, ctx.preprocessing.height
        )
        encoder = self._encoder.model(ctx.device.value, ctx.cache_dir)
        noise = noise_patch_features(
            encoder,
            ctx.preprocessing,
            self.config.layers.indices,
            ctx.device.value,
            seed=self.config.seed,
        )
        rank = resolved_rank(self.config.positional_rank, patches=rows * cols, width=noise.shape[1])
        ctx.log(
            f"positional debias: removing {rank} of {noise.shape[1]} directions "
            f"(asked {self.config.positional_rank}; {rows}x{cols} patches)"
        )
        return positional_basis(noise, self.config.positional_rank)

    # ------------------------------------------------------------------ fit

    def fit(self, train: Sequence[ImageRecord], ctx: TrainContext) -> None:
        targets = ctx.targets
        if targets is None:
            raise RuntimeError("proto_seg segments a class and needs references with masks")
        width, height = ctx.preprocessing.width, ctx.preprocessing.height
        validate_prepared_size(self.config.backbone, width, height)
        grid = patch_grid(self.config.backbone, width, height)

        basis = self._basis(ctx)
        self._state = {"basis": basis}

        # Kept per reference, whole, so a leave-one-out fold can pool every reference but one
        # and score the one it left out.
        encoded: list[np.ndarray] = []
        parts: dict[str, list[np.ndarray]] = {side: [] for side in SIDES}
        for position, record in enumerate(train):
            ctx.raise_if_cancelled()
            features = self._features(record, ctx)
            covered, background = split_patches(patch_coverage(targets.mask(record.image_id), grid))
            encoded.append(features)
            parts["foreground"].append(features[covered])
            parts["background"].append(features[background])
            ctx.progress(0.7 * (position + 1) / len(train), f"encoded {position + 1}/{len(train)}")

        self._state.update(
            self._build(parts, targets.label_key, np.random.default_rng(self.config.seed), ctx)
        )
        self._calibration = IDENTITY
        if self.config.calibration is Calibration.LEAVE_ONE_OUT:
            ctx.progress(0.85, "calibrating on the references")
            self._calibration = leave_one_out(
                len(train),
                lambda fold: self._held_out(fold, train, encoded, parts, ctx),
                ctx.log,
                cancelled=ctx.raise_if_cancelled,
            )
            ctx.metric("calibration_slope", self._calibration.slope)
            ctx.metric("calibration_bias", self._calibration.bias)
        self._state["calibration"] = self._calibration.to_array()
        ctx.progress(1.0, "bank built")

    def _build(
        self,
        parts: dict[str, list[np.ndarray]],
        label_key: str,
        rng: np.random.Generator,
        ctx: TrainContext | None,
    ) -> dict[str, np.ndarray]:
        """The bank (and probe) from per-reference patches; `ctx` is `None` in a quiet fold."""
        samples: dict[str, np.ndarray] = {}
        for side in SIDES:
            pooled = np.concatenate(parts[side])
            if len(pooled) == 0:
                msg = (
                    f"the references hold no {side} patch for {label_key!r}: a patch "
                    "is the class when half of it, or the most of any patch, is covered"
                )
                raise RuntimeError(msg)
            chosen = evenly_spaced(len(pooled), self.config.max_features_per_class)
            if ctx is not None and len(chosen) < len(pooled):
                ctx.log(
                    f"{side}: {len(chosen)} of {len(pooled)} reference patches, sampled evenly "
                    f"(max_features_per_class={self.config.max_features_per_class})"
                )
            samples[side] = pooled[np.asarray(chosen, dtype=np.int64)]
            if ctx is not None:
                ctx.metric(f"{side}_patches", float(len(chosen)))

        state: dict[str, np.ndarray] = {}
        # One cluster count for both sides: the LSE score sums over a side's prototypes, so
        # a side with more of them would be favoured for having more (`lse_probability`).
        clusters = min(self.config.clusters_per_class, *(len(found) for found in samples.values()))
        if ctx is not None and clusters < self.config.clusters_per_class:
            ctx.log(f"{clusters} clusters per side: the smaller side has too few patches for more")
        for side in SIDES:
            bank = [mean_prototype(samples[side])]
            if clusters > 0:
                bank.append(
                    spherical_kmeans(
                        samples[side], clusters, iterations=self.config.kmeans_iterations, rng=rng
                    )
                )
            state[f"bank_{side}"] = np.concatenate(bank)

        if self.config.adaptation is Adaptation.LINEAR_ADAPT:
            features = np.concatenate([samples["foreground"], samples["background"]])
            labels = np.concatenate(
                [np.ones(len(samples["foreground"])), np.zeros(len(samples["background"]))]
            )
            weights, bias = fit_linear_probe(
                features,
                labels,
                l2=1e-3,
                iterations=self.config.probe_iterations,
                learning_rate=1.0,
            )
            state["probe_weights"] = weights
            state["probe_bias"] = np.array([bias], dtype=np.float32)
            if ctx is not None:
                ctx.log(f"linear probe fitted on {len(labels)} reference patches")
        return state

    def _held_out(
        self,
        fold: int,
        train: Sequence[ImageRecord],
        encoded: list[np.ndarray],
        parts: dict[str, list[np.ndarray]],
        ctx: TrainContext,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Reference `fold`'s pixel map from a bank of the others, and its truth; `None`
        when the others hold no patch of one side.

        Each fold draws its own k-means stream from the seed, so the final bank — built
        first, from the seed alone — is the same with calibration on or off.
        """
        targets = ctx.targets
        if targets is None:
            raise RuntimeError("proto_seg calibrates on references with masks")
        others = {side: [p for i, p in enumerate(parts[side]) if i != fold] for side in SIDES}
        if any(sum(len(part) for part in others[side]) == 0 for side in SIDES):
            return None
        state = self._build(
            others, targets.label_key, np.random.default_rng([self.config.seed, fold + 1]), None
        )
        rows, cols = patch_grid(
            self.config.backbone, ctx.preprocessing.width, ctx.preprocessing.height
        )
        record = train[fold]
        patches = self._patch_probability(encoded[fold], state)
        image = load_array(record.path, ctx.preprocessing)
        probability = refine(patches.reshape(rows, cols), image, self.config.refine)
        return probability, targets.mask(record.image_id)

    # ------------------------------------------------------------------ predict

    def _patch_probability(
        self, query: np.ndarray, state: dict[str, np.ndarray] | None = None
    ) -> np.ndarray:
        state = self._state if state is None else state
        if self.config.adaptation is Adaptation.LINEAR_ADAPT:
            return probe_probability(query, state["probe_weights"], float(state["probe_bias"][0]))
        return lse_probability(
            query, state["bank_foreground"], state["bank_background"], self.config.temperature
        )

    def predict(self, images: Sequence[ImageRecord], ctx: InferContext) -> list[Prediction]:
        if "bank_foreground" not in self._state:
            raise RuntimeError("proto_seg was asked to predict before it was fitted or loaded")
        rows, cols = patch_grid(
            self.config.backbone, ctx.preprocessing.width, ctx.preprocessing.height
        )
        predictions: list[Prediction] = []
        for index, record in enumerate(images):
            ctx.raise_if_cancelled()
            started = time.perf_counter()
            patches = self._patch_probability(self._features(record, ctx))
            image = load_array(record.path, ctx.preprocessing)
            probability = self._calibration.apply(
                refine(patches.reshape(rows, cols), image, self.config.refine)
            )
            map_path = ctx.write_map(record.image_id, probability)
            presence = presence_score(patches, self.config.presence_patches)
            predictions.append(
                Prediction(
                    image_id=record.image_id,
                    score=self._calibration.apply_score(presence),
                    anomaly_map=map_path,
                    inference_ms=(time.perf_counter() - started) * 1000.0,
                )
            )
            ctx.progress((index + 1) / len(images), f"segmented {index + 1}/{len(images)}")
        return predictions

    # ------------------------------------------------------------------ persistence

    def save(self, artifact_dir: Path) -> None:
        if "bank_foreground" not in self._state or self._encoder.fingerprint is None:
            raise RuntimeError("proto_seg has nothing to save; it was never fitted")
        # numpy's stub types the second positional as `allow_pickle`; this is the keyword form.
        np.savez_compressed(artifact_dir / STATE_FILENAME, **self._state)  # type: ignore[arg-type]
        meta = {"backbone": self.config.backbone.value, "fingerprint": self._encoder.fingerprint}
        (artifact_dir / META_FILENAME).write_text(json.dumps(meta), encoding="utf-8")

    def load(self, artifact_dir: Path) -> None:
        meta = json.loads((artifact_dir / META_FILENAME).read_text(encoding="utf-8"))
        if meta["backbone"] != self.config.backbone.value:
            msg = (
                f"this proto_seg checkpoint was fitted with {meta['backbone']}, not "
                f"{self.config.backbone.value}"
            )
            raise RuntimeError(msg)
        self._encoder.expect(str(meta["fingerprint"]))
        with np.load(artifact_dir / STATE_FILENAME, allow_pickle=False) as stored:
            self._state = {key: stored[key] for key in stored.files}
        self._calibration = PlattScale.from_array(self._state.get("calibration"))
