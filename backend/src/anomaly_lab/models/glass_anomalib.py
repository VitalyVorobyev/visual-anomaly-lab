"""`glass_anomalib` — learned anomaly synthesis through anomalib.

GLASS trains a projection and discriminator against two synthetic anomaly families:
Perlin regions in image space and mined Gaussian perturbations in feature space.  The
wrapper keeps anomalib's model, synthesis, loss and map rule, while owning the finite,
cancellable pass, deterministic image order, bounded centre refreshes, exact continuation
state and external-backbone contract.

The optional DTD texture corpus is deliberately absent from the first plugin.  Built-in
Perlin synthesis is complete and requires no second download; a paired public-data gate
must demonstrate that the 5640-image corpus adds value before it becomes managed state.
Heavy dependencies stay inside functions so the method registry remains torch-free.
"""

from __future__ import annotations

import contextlib
import time
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

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
from anomaly_lab.models.model_assets import fingerprint_state, huggingface_environment
from anomaly_lab.models.preprocessing import IMAGENET_MEAN, IMAGENET_STD, load_array, to_chw
from anomaly_lab.schemas import API_MODEL_CONFIG

STATE_FILENAME = "glass.pt"
CHECKPOINT_FORMAT = 1
BACKBONE = "wide_resnet50_2"
LAYERS = ("layer2", "layer3")
LOSS_LOG_EVERY = 25
ESTIMATED_CHECKPOINT_BYTES = 50_000_000


class SynthesisAnchor(StrEnum):
    """Reference used when truncating mined feature perturbations."""

    SAMPLE = "sample"
    CENTRE = "centre"


class GlassConfig(BaseModel):
    """Stable experiment controls rendered from JSON Schema."""

    model_config = API_MODEL_CONFIG

    max_steps: int = Field(
        default=5_000,
        ge=1,
        le=100_000,
        description=(
            "Batch-1 synthesis updates. At 288 px, 5000 measured about 17 minutes of "
            "model update time on MPS, plus bounded centre refreshes and image loading."
        ),
    )
    center_images: int = Field(
        default=128,
        ge=1,
        le=4_096,
        description=(
            "Normals sampled evenly for each feature-centre refresh. This bounds the "
            "otherwise full-dataset pass and reports how many images were omitted."
        ),
    )
    center_refresh_steps: int = Field(
        default=392,
        ge=1,
        le=100_000,
        description=(
            "Recompute the projected normal-feature centre at this fixed step interval. "
            "The schedule is absolute, so interrupted and uninterrupted runs agree."
        ),
    )
    mining_steps: int = Field(
        default=20,
        ge=0,
        le=100,
        description=(
            "Inner gradient-ascent steps for global synthetic anomalies. 20 is the "
            "published setting and measured Mac-credible; 0 disables ascent."
        ),
    )
    synthesis_anchor: SynthesisAnchor = Field(
        default=SynthesisAnchor.SAMPLE,
        description=(
            "Anchor for truncating mined feature perturbations. 'sample' is one fixed, "
            "dataset-agnostic default; 'centre' is a generic ablation, never selected by "
            "dataset or category name."
        ),
    )
    learning_rate: float = Field(
        default=1e-4,
        gt=0.0,
        le=1e-2,
        description=(
            "Adam learning rate for the projection; the discriminator uses twice this value."
        ),
    )
    allow_downloads: bool = Field(
        default=True,
        description=(
            "Permit fetching the public ImageNet WRN-50 weights on the first run. Turn "
            "off to require the exact weights in the app-managed model cache."
        ),
    )
    seed: int = Field(
        default=0,
        description=(
            "Seeds initialization, image order, Perlin synthesis and Gaussian feature "
            "perturbations. Same data, configuration and seed define one experiment."
        ),
    )


@dataclass(frozen=True)
class TrainingPlan:
    """Bounded work announced before a tensor is allocated."""

    images_available: int
    center_images: int
    center_passes: int
    steps: int
    width: int
    height: int
    mining_steps: int

    @property
    def center_images_dropped(self) -> int:
        return self.images_available - self.center_images

    def describe(self) -> str:
        return (
            f"GLASS plan: {self.steps:,} batch-1 updates at {self.width}x{self.height}; "
            f"{self.center_passes} centre passes x {self.center_images}/"
            f"{self.images_available} evenly spaced normals; {self.mining_steps}-step "
            f"mining; resumable checkpoint about "
            f"{ESTIMATED_CHECKPOINT_BYTES / 1e6:.0f} MB"
        )


def plan_training(config: GlassConfig, train_count: int, width: int, height: int) -> TrainingPlan:
    """Pure, torch-free resource plan for the API process and ordinary CI."""
    if train_count < 1:
        raise ValueError("GLASS needs at least one normal training image")
    if width < 64 or height < 64:
        raise ValueError(
            f"GLASS needs prepared width and height of at least 64 pixels; got {width}x{height}"
        )
    return TrainingPlan(
        images_available=train_count,
        center_images=min(train_count, config.center_images),
        center_passes=1 + (config.max_steps - 1) // config.center_refresh_steps,
        steps=config.max_steps,
        width=width,
        height=height,
        mining_steps=config.mining_steps,
    )


def _normalised_chw(record: ImageRecord, ctx: TrainContext | InferContext) -> np.ndarray:
    array = to_chw(load_array(record.path, ctx.preprocessing))
    if array.shape[0] == 1:
        array = np.repeat(array, 3, axis=0)
    if array.shape[0] != 3:
        raise ValueError(f"GLASS expects one or three channels; got {array.shape[0]}")
    mean = np.asarray(IMAGENET_MEAN, dtype=np.float32)[:, None, None]
    std = np.asarray(IMAGENET_STD, dtype=np.float32)[:, None, None]
    return np.ascontiguousarray((array - mean) / std, dtype=np.float32)


class GlassAnomalibModel(AnomalyModel):
    """GLASS arithmetic with a workbench-owned bounded pass."""

    title = "GLASS (anomalib)"
    summary = (
        "A frozen WRN-50 feature extractor with learned global and local anomaly "
        "synthesis; a discriminator turns their feature boundary into anomaly maps."
    )

    def __init__(self, config: GlassConfig) -> None:
        super().__init__(config)
        self.config = config
        self._model: Any = None
        self._fingerprint: str | None = None
        self._cache_dir: Path | None = None
        self._completed = 0
        self._optimizer_states: list[dict[str, Any]] | None = None
        self._torch_rng_state: Any = None
        self._mps_rng_state: Any = None
        self._generator: np.random.Generator | None = None
        self._generator_state: Any = None
        self._width = 0
        self._height = 0

    @classmethod
    def config_model(cls) -> type[BaseModel]:
        return GlassConfig

    @classmethod
    def capabilities(cls) -> Capabilities:
        return Capabilities(
            requires_training=True,
            produces_anomaly_map=True,
            produces_diagnostics=False,
            supports_resume=True,
            channel_aware=False,
            dataset_specific=False,
            preferred_device=Device.MPS,
        )

    @classmethod
    def availability(cls) -> Availability:
        return module_available("anomalib", "dl", "GLASS")

    def _build_model(self, device: str, cache_dir: Path) -> Any:
        import torch

        torch.manual_seed(self.config.seed)
        with huggingface_environment(
            cache_dir,
            allow_downloads=self.config.allow_downloads,
            method="GLASS",
            asset=BACKBONE,
        ):
            from anomalib.models import Glass

            module = Glass(
                input_shape=(self._height, self._width),
                anomaly_source_path=None,
                backbone=BACKBONE,
                layers=list(LAYERS),
                learning_rate=self.config.learning_rate,
                step=self.config.mining_steps,
                svd=1 if self.config.synthesis_anchor is SynthesisAnchor.CENTRE else 0,
                mining=self.config.mining_steps > 0,
                pre_processor=False,
                post_processor=False,
                evaluator=False,
                visualizer=False,
            )
        model: Any = module.model.to(device)
        extractor = model.forward_modules["feature_aggregator"].feature_extractor
        self._fingerprint = fingerprint_state(extractor.state_dict())
        self._cache_dir = cache_dir
        return model

    def _optimizers(self, model: Any) -> list[Any]:
        import torch

        return [
            torch.optim.AdamW(model.discriminator.parameters(), lr=self.config.learning_rate * 2),
            torch.optim.Adam(
                model.projection.parameters(),
                lr=self.config.learning_rate,
                weight_decay=1e-5,
            ),
        ]

    def fit(self, train: Sequence[ImageRecord], ctx: TrainContext) -> None:
        plan = plan_training(
            self.config, len(train), ctx.preprocessing.width, ctx.preprocessing.height
        )
        ctx.log(plan.describe())
        if plan.center_images_dropped:
            ctx.log(
                f"centre cap drops {plan.center_images_dropped:,} normals per pass; "
                "the retained records are evenly spaced"
            )
        self._width = ctx.preprocessing.width
        self._height = ctx.preprocessing.height
        ctx.progress(0.01, f"loading {BACKBONE} (downloads on first run only)")
        self._model = self._build_model(ctx.device.value, ctx.cache_dir)
        self._completed = 0
        self._generator = np.random.default_rng(self.config.seed)
        optimizers = self._optimizers(self._model)
        self._run_steps(train, ctx, optimizers, self.config.max_steps)
        self._capture_resume_state(optimizers)
        self._model.eval()

    def completed_steps(self) -> int:
        return self._completed

    def fit_more(
        self,
        train: Sequence[ImageRecord],
        ctx: TrainContext,
        *,
        additional_steps: int,
    ) -> None:
        if self._model is None or self._optimizer_states is None:
            raise RuntimeError("GLASS cannot continue because no resumable checkpoint is loaded")
        plan_training(self.config, len(train), ctx.preprocessing.width, ctx.preprocessing.height)
        if (ctx.preprocessing.width, ctx.preprocessing.height) != (self._width, self._height):
            raise ValueError(
                "GLASS continuation must use the checkpoint's prepared dimensions "
                f"{self._width}x{self._height}"
            )
        import torch

        self._model = self._model.to(ctx.device.value)
        self._model.center = self._model.center.to(ctx.device.value)
        optimizers = self._optimizers(self._model)
        for optimizer, state in zip(optimizers, self._optimizer_states, strict=True):
            optimizer.load_state_dict(state)
        self._restore_rng(torch)
        ctx.log(
            f"continuing GLASS from step {self._completed} for {additional_steps} more; "
            "optimizers, synthesis RNG and image order resume exactly"
        )
        self._run_steps(train, ctx, optimizers, additional_steps)
        self._capture_resume_state(optimizers)
        self._model.eval()

    def _calculate_center(self, train: Sequence[ImageRecord], ctx: TrainContext) -> None:
        import torch

        if self._model is None:
            raise RuntimeError("GLASS training state was not initialised")
        records = [
            train[index]
            for index in evenly_spaced(len(train), min(len(train), self.config.center_images))
        ]
        self._model.eval()
        total: Any = None
        with torch.no_grad():
            for record in records:
                image = torch.from_numpy(_normalised_chw(record, ctx)[None]).to(ctx.device.value)
                embeddings = self._model.generate_embeddings(image, evaluation=True)[0]
                projected = self._model.projection(embeddings)
                if isinstance(projected, (tuple, list)):
                    projected = projected[0]
                total = projected if total is None else total + projected
        if total is None:
            raise ValueError("GLASS cannot calculate a feature centre from an empty set")
        self._model.center = total / len(records)
        self._model.train()

    def _run_steps(
        self,
        train: Sequence[ImageRecord],
        ctx: TrainContext,
        optimizers: list[Any],
        steps: int,
    ) -> None:
        import torch

        if self._model is None or self._generator is None:
            raise RuntimeError("GLASS training state was not initialised")
        self._model.train()
        started = time.perf_counter()
        start_step = self._completed
        total_steps = start_step + steps
        for index in range(steps):
            ctx.raise_if_cancelled()
            step = self._completed
            if step % self.config.center_refresh_steps == 0:
                ctx.progress(index / max(steps, 1), f"refreshing feature centre at step {step}")
                self._calculate_center(train, ctx)

            record = train[int(self._generator.integers(len(train)))]
            image = torch.from_numpy(_normalised_chw(record, ctx)[None]).to(ctx.device.value)
            for optimizer in optimizers:
                optimizer.zero_grad(set_to_none=True)
            losses = self._model(image)
            loss = losses[-1]
            loss.backward()
            for optimizer in reversed(optimizers):
                optimizer.step()
            self._completed += 1

            if index % LOSS_LOG_EVERY == 0 or index == steps - 1:
                value = float(loss.detach().cpu())
                ctx.metric("loss_total", value, step=step)
                ctx.progress(
                    (index + 1) / max(steps, 1),
                    f"step {self._completed}/{total_steps}, loss {value:.4f}",
                )
        elapsed = time.perf_counter() - started
        ctx.log(
            f"trained {steps} GLASS steps in {elapsed:.1f}s "
            f"({elapsed / max(steps, 1) * 1000:.0f} ms/step on {ctx.device.value}); "
            f"{self._completed} completed in total"
        )

    def _capture_resume_state(self, optimizers: list[Any]) -> None:
        import torch

        self._optimizer_states = [optimizer.state_dict() for optimizer in optimizers]
        self._torch_rng_state = torch.get_rng_state()
        self._generator_state = (
            None if self._generator is None else self._generator.bit_generator.state
        )
        self._mps_rng_state = None
        if hasattr(torch, "mps") and hasattr(torch.mps, "get_rng_state"):
            with contextlib.suppress(Exception):
                self._mps_rng_state = torch.mps.get_rng_state()

    def _restore_rng(self, torch_module: Any) -> None:
        if self._torch_rng_state is not None:
            torch_module.set_rng_state(self._torch_rng_state)
        if self._mps_rng_state is not None and hasattr(torch_module.mps, "set_rng_state"):
            with contextlib.suppress(Exception):
                torch_module.mps.set_rng_state(self._mps_rng_state)
        if self._generator_state is not None:
            self._generator = np.random.default_rng()
            self._generator.bit_generator.state = self._generator_state
        else:
            self._generator = np.random.default_rng(self.config.seed)

    def predict(self, images: Sequence[ImageRecord], ctx: InferContext) -> list[Prediction]:
        import torch

        if self._model is None:
            raise RuntimeError("GLASS was asked to predict before it was fitted or loaded")
        if (ctx.preprocessing.width, ctx.preprocessing.height) != (self._width, self._height):
            raise ValueError(
                f"GLASS was fitted at {self._width}x{self._height}; inference requested "
                f"{ctx.preprocessing.width}x{ctx.preprocessing.height}"
            )
        self._model = self._model.to(ctx.device.value).eval()
        predictions: list[Prediction] = []
        with torch.inference_mode():
            for index, record in enumerate(images):
                ctx.raise_if_cancelled()
                image = torch.from_numpy(_normalised_chw(record, ctx)[None]).to(ctx.device.value)
                started = time.perf_counter()
                output = self._model(image)
                if ctx.device is Device.MPS:
                    torch.mps.synchronize()
                elapsed_ms = (time.perf_counter() - started) * 1000
                anomaly_map = output.anomaly_map[0].detach().cpu().numpy()
                path = ctx.write_map(record.image_id, anomaly_map)
                predictions.append(
                    Prediction(
                        image_id=record.image_id,
                        score=float(output.pred_score[0].detach().cpu()),
                        anomaly_map=path,
                        inference_ms=elapsed_ms,
                    )
                )
                ctx.progress(
                    (index + 1) / max(len(images), 1),
                    f"scored {index + 1}/{len(images)} images",
                )
        return predictions

    def save(self, artifact_dir: Path) -> None:
        import torch

        if self._model is None or self._fingerprint is None or self._cache_dir is None:
            raise RuntimeError("GLASS has nothing to save; it was never fitted")
        torch.save(
            {
                "format": CHECKPOINT_FORMAT,
                "backbone": BACKBONE,
                "layers": LAYERS,
                "backbone_fingerprint": self._fingerprint,
                "width": self._width,
                "height": self._height,
                "projection": self._model.projection.state_dict(),
                "discriminator": self._model.discriminator.state_dict(),
                "center": self._model.center.detach().cpu(),
                "completed_steps": self._completed,
                "optimizers": self._optimizer_states,
                "torch_rng_state": self._torch_rng_state,
                "mps_rng_state": self._mps_rng_state,
                "generator_state": self._generator_state,
                "cache_dir": str(self._cache_dir),
                "versions": {
                    "anomalib": version("anomalib"),
                    "torch": version("torch"),
                    "timm": version("timm"),
                },
            },
            artifact_dir / STATE_FILENAME,
        )

    def load(self, artifact_dir: Path) -> None:
        import torch

        stored = torch.load(artifact_dir / STATE_FILENAME, map_location="cpu", weights_only=False)
        if int(stored.get("format", 0)) != CHECKPOINT_FORMAT:
            raise RuntimeError(
                f"unsupported GLASS checkpoint format {stored.get('format')!r}; "
                f"expected {CHECKPOINT_FORMAT}"
            )
        if stored.get("backbone") != BACKBONE or tuple(stored.get("layers", ())) != LAYERS:
            raise RuntimeError("GLASS checkpoint architecture does not match this plugin")
        self._width = int(stored["width"])
        self._height = int(stored["height"])
        model = self._build_model("cpu", Path(stored["cache_dir"]))
        expected = str(stored["backbone_fingerprint"])
        actual = str(self._fingerprint)
        if actual != expected:
            raise RuntimeError(
                f"GLASS backbone weights changed: checkpoint expects {expected[:12]}, "
                f"but {BACKBONE} now resolves to {actual[:12]}. Restore the original "
                "asset or refit; these results are not comparable."
            )
        model.projection.load_state_dict(stored["projection"])
        model.discriminator.load_state_dict(stored["discriminator"])
        model.center = stored["center"]
        self._model = model.eval()
        self._completed = int(stored["completed_steps"])
        self._optimizer_states = stored.get("optimizers")
        self._torch_rng_state = stored.get("torch_rng_state")
        self._mps_rng_state = stored.get("mps_rng_state")
        self._generator_state = stored.get("generator_state")
