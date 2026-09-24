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
argmax picks the class, so the evaluator's rule reads it the same way.
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
from anomaly_lab.models.dino_backbone import (
    BACKBONES,
    DinoBackbone,
    backbone_fingerprint,
    image_patch_features,
    load_backbone,
    patch_grid,
    validate_prepared_size,
)
from anomaly_lab.models.prototypes import (
    class_maps,
    combined_score,
    foreground_probability,
    gram_matrix,
    patch_coverage,
    spherical_kmeans,
)
from anomaly_lab.schemas import API_MODEL_CONFIG

STATE_FILENAME = "fss_dino.npz"
META_FILENAME = "fss_dino.json"


class FssDinoConfig(BaseModel):
    model_config = API_MODEL_CONFIG

    backbone: DinoBackbone = Field(
        default=DinoBackbone.DINOV2_VIT_B14,
        description=(
            "Frozen encoder; its last block is read. FSSDINO used DINOv3 ViT-B/16, which is "
            "licence-gated (an approved HF_TOKEN must be in the environment), so the default "
            "is the ungated DINOv2 ViT-B/14."
        ),
    )
    prototypes_per_class: int = Field(
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
        self._fingerprint: str | None = None
        self._encoder: Any = None

    @classmethod
    def config_model(cls) -> type[BaseModel]:
        return FssDinoConfig

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

    def _encoder_for(self, device: str, cache_dir: Path) -> Any:
        if self._encoder is None:
            encoder = load_backbone(
                self.config.backbone,
                pretrained=self.config.pretrained_backbone,
                allow_downloads=self.config.allow_downloads,
                cache_dir=cache_dir,
                seed=self.config.seed,
                method="fss_dino",
            )
            fingerprint = backbone_fingerprint(encoder)
            if self._fingerprint is not None and fingerprint != self._fingerprint:
                msg = (
                    "the encoder fss_dino was fitted with has changed since: its weights no "
                    "longer match the stored fingerprint. Refit the run."
                )
                raise RuntimeError(msg)
            self._fingerprint = fingerprint
            self._encoder = encoder.to(device)
        return self._encoder

    def _layer(self) -> tuple[int, ...]:
        return (BACKBONES[self.config.backbone].depth - 1,)

    def _features(self, records: Sequence[ImageRecord], ctx: TrainContext | InferContext) -> Any:
        encoder = self._encoder_for(ctx.device.value, ctx.cache_dir)
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

        sides: dict[str, list[np.ndarray]] = {"foreground": [], "background": []}
        for position, record in enumerate(train):
            ctx.raise_if_cancelled()
            features = self._features([record], ctx)[0]
            covered = patch_coverage(targets.mask(record.image_id), grid) >= 0.5
            sides["foreground"].append(features[covered])
            sides["background"].append(features[~covered])
            ctx.progress(0.8 * (position + 1) / len(train), f"encoded {position + 1}/{len(train)}")

        rng = np.random.default_rng(self.config.seed)
        for side, parts in sides.items():
            pooled = np.concatenate(parts) if parts else np.empty((0, 1), dtype=np.float32)
            if len(pooled) == 0:
                msg = (
                    f"the references hold no {side} patch for {targets.label_key!r}: a patch "
                    "is the class when at least half of it is covered by the mask"
                )
                raise RuntimeError(msg)
            chosen = evenly_spaced(len(pooled), self.config.max_features_per_class)
            if len(chosen) < len(pooled):
                ctx.log(
                    f"{side}: {len(chosen)} of {len(pooled)} reference patches, sampled evenly "
                    f"(max_features_per_class={self.config.max_features_per_class})"
                )
            else:
                ctx.log(f"{side}: all {len(pooled)} reference patches")
            sample = pooled[np.asarray(chosen, dtype=np.int64)]
            self._prototypes[side] = spherical_kmeans(
                sample,
                self.config.prototypes_per_class,
                iterations=self.config.kmeans_iterations,
                rng=rng,
            )
            if self.config.gram:
                self._grams[side] = gram_matrix(sample)
            ctx.metric(f"{side}_patches", float(len(chosen)))
        ctx.progress(1.0, "prototypes built")

    # ------------------------------------------------------------------ predict

    def predict(self, images: Sequence[ImageRecord], ctx: InferContext) -> list[Prediction]:
        if not self._prototypes:
            raise RuntimeError("fss_dino was asked to predict before it was fitted or loaded")
        width, height = ctx.preprocessing.width, ctx.preprocessing.height
        grid = patch_grid(self.config.backbone, width, height)
        predictions: list[Prediction] = []
        for index, record in enumerate(images):
            ctx.raise_if_cancelled()
            started = time.perf_counter()
            query = self._features([record], ctx)[0]
            scores = {
                side: combined_score(
                    class_maps(query, self._prototypes[side], self._grams.get(side)),
                    grid,
                    (width, height),
                )
                for side in ("foreground", "background")
            }
            probability = foreground_probability(scores["foreground"], scores["background"])
            map_path = ctx.write_map(record.image_id, probability)
            ctx.write_mask(record.image_id, scores["foreground"] > scores["background"])
            predictions.append(
                Prediction(
                    image_id=record.image_id,
                    score=float(np.percentile(probability, self.config.presence_percentile)),
                    anomaly_map=map_path,
                    inference_ms=(time.perf_counter() - started) * 1000.0,
                )
            )
            ctx.progress((index + 1) / len(images), f"segmented {index + 1}/{len(images)}")
        return predictions

    # ------------------------------------------------------------------ persistence

    def save(self, artifact_dir: Path) -> None:
        if not self._prototypes or self._fingerprint is None:
            raise RuntimeError("fss_dino has nothing to save; it was never fitted")
        arrays = {f"prototypes_{side}": value for side, value in self._prototypes.items()}
        arrays.update({f"gram_{side}": value for side, value in self._grams.items()})
        # numpy's stub types the second positional as `allow_pickle`; this is the keyword form.
        np.savez_compressed(artifact_dir / STATE_FILENAME, **arrays)  # type: ignore[arg-type]
        meta = {"backbone": self.config.backbone.value, "fingerprint": self._fingerprint}
        (artifact_dir / META_FILENAME).write_text(json.dumps(meta), encoding="utf-8")

    def load(self, artifact_dir: Path) -> None:
        meta = json.loads((artifact_dir / META_FILENAME).read_text(encoding="utf-8"))
        if meta["backbone"] != self.config.backbone.value:
            msg = (
                f"this fss_dino checkpoint was fitted with {meta['backbone']}, not "
                f"{self.config.backbone.value}"
            )
            raise RuntimeError(msg)
        self._fingerprint = str(meta["fingerprint"])
        self._encoder = None
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
