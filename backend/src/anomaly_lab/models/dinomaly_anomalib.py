"""`dinomaly_anomalib` — transformer feature reconstruction through anomalib.

Dinomaly freezes a registered DINOv2 encoder and trains a bottleneck plus transformer
decoder to reconstruct intermediate normal features.  It is the first M11 reference and
represents a different family from EfficientAD's distillation and PatchCore's memory bank.

The wrapper owns the pass, not the algorithm.  Anomalib supplies the model, hard-mining
loss, StableAdamW optimizer and anomaly-map rule.  This module supplies the shared pixel
bridge, finite/cancellable step loop, exact continuation state, asset fingerprint and the
early validation anomalib currently omits.

Two measured invariants are fixed rather than exposed as brittle options:

* the registered DINOv2 patch size is 14, so both prepared dimensions must divide by 14;
* anomalib's default fusion topology indexes eight decoder outputs even though the public
  constructor accepts shallower depths.  This plugin therefore uses depth eight only.

Heavy imports stay inside functions.  Opening the method picker must not import torch.
"""

from __future__ import annotations

import contextlib
import math
import time
from collections.abc import Sequence
from dataclasses import dataclass
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
    module_available,
)
from anomaly_lab.models.model_assets import fingerprint_state, huggingface_environment
from anomaly_lab.models.preprocessing import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    load_array,
    to_chw,
)
from anomaly_lab.schemas import API_MODEL_CONFIG

STATE_FILENAME = "dinomaly.pt"
CHECKPOINT_FORMAT = 1
ENCODER_NAME = "vit_small_patch14_reg4_dinov2"
PATCH_SIZE = 14
DECODER_DEPTH = 8
BASE_LR = 2e-3
FINAL_LR = 2e-4
WARMUP_STEPS = 100
SCHEDULE_STEPS = 5_000
LOSS_LOG_EVERY = 25


@dataclass(frozen=True)
class TrainingPlan:
    """Bounded work announced before a tensor is allocated."""

    images_available: int
    steps: int
    width: int
    height: int
    estimated_checkpoint_bytes: int = 246_000_000

    def describe(self) -> str:
        return (
            f"Dinomaly plan: {self.steps:,} batch-1 steps sampling "
            f"{self.images_available:,} normals at {self.width}x{self.height}; "
            f"small DINOv2 encoder, depth {DECODER_DEPTH}; resumable checkpoint "
            f"about {self.estimated_checkpoint_bytes / 1e6:.0f} MB"
        )


class DinomalyConfig(BaseModel):
    """Stable experiment controls; the method schema renders these without UI code."""

    model_config = API_MODEL_CONFIG

    max_steps: int = Field(
        default=5_000,
        ge=1,
        le=100_000,
        description=(
            "Batch-1 reconstruction updates. The published anomalib recipe uses 5000; "
            "at 392 px this measured about ten minutes on MPS. Continuations retain the "
            "optimizer and continue the same fixed 5000-step warm-cosine schedule."
        ),
    )
    allow_downloads: bool = Field(
        default=True,
        description=(
            "Permit fetching the public DINOv2 encoder weights on the first run. Turn "
            "off to require the exact weights to exist in the app-managed model cache."
        ),
    )
    seed: int = Field(
        default=0,
        description=(
            "Seeds decoder initialisation, dropout and training-image order. Same data, "
            "configuration and seed define one reproducible experiment."
        ),
    )


def plan_training(
    config: DinomalyConfig, train_count: int, width: int, height: int
) -> TrainingPlan:
    """Pure, torch-free resource plan for the API process and ordinary CI."""
    if train_count < 1:
        raise ValueError("Dinomaly needs at least one normal training image")
    validate_prepared_size(width, height)
    return TrainingPlan(
        images_available=train_count,
        steps=config.max_steps,
        width=width,
        height=height,
    )


def validate_prepared_size(width: int, height: int) -> None:
    """Fail at the plugin boundary, before timm's token reshape fails downstream."""
    if width % PATCH_SIZE or height % PATCH_SIZE:
        raise ValueError(
            f"Dinomaly's DINOv2 encoder uses {PATCH_SIZE}-pixel patches, so prepared "
            f"width and height must both be divisible by {PATCH_SIZE}; got {width}x{height}. "
            "Use 196, 280, 392 or another multiple of 14 in the experiment preprocessing."
        )


def learning_rate(step: int) -> float:
    """Anomalib's fixed 100-step warm-up and 5000-step cosine recipe.

    The horizon is deliberately independent of the first run's requested step count.  A
    1000-step run continued for 1000 more therefore follows exactly the same schedule as a
    2000-step run, rather than retroactively moving its cosine endpoint.
    """
    if step < 0:
        raise ValueError(f"training step cannot be negative; got {step}")
    if step < WARMUP_STEPS:
        return float(np.linspace(0.0, BASE_LR, WARMUP_STEPS)[step])
    if step >= SCHEDULE_STEPS:
        return FINAL_LR
    offset = step - WARMUP_STEPS
    span = SCHEDULE_STEPS - WARMUP_STEPS
    return float(FINAL_LR + 0.5 * (BASE_LR - FINAL_LR) * (1 + math.cos(math.pi * offset / span)))


def _normalised_chw(record: ImageRecord, ctx: TrainContext | InferContext) -> np.ndarray:
    """Shared prepared pixels, then the frozen encoder's channel standardisation."""
    array = to_chw(load_array(record.path, ctx.preprocessing))
    if array.shape[0] == 1:
        array = np.repeat(array, 3, axis=0)
    if array.shape[0] != 3:
        raise ValueError(f"Dinomaly expects one or three channels; got {array.shape[0]}")
    mean = np.asarray(IMAGENET_MEAN, dtype=np.float32)[:, None, None]
    std = np.asarray(IMAGENET_STD, dtype=np.float32)[:, None, None]
    return np.ascontiguousarray((array - mean) / std, dtype=np.float32)


class DinomalyAnomalibModel(AnomalyModel):
    """Dinomaly with anomalib arithmetic and a workbench-owned bounded pass."""

    title = "Dinomaly (anomalib)"
    summary = (
        "A frozen DINOv2 encoder with a trainable transformer decoder that reconstructs "
        "normal feature maps; reconstruction disagreement becomes the anomaly map."
    )

    def __init__(self, config: DinomalyConfig) -> None:
        super().__init__(config)
        self.config = config
        self._model: Any = None
        self._fingerprint: str | None = None
        self._cache_dir: Path | None = None
        self._completed = 0
        self._optimizer_state: dict[str, Any] | None = None
        self._torch_rng_state: Any = None
        self._mps_rng_state: Any = None
        self._generator: np.random.Generator | None = None
        self._generator_state: Any = None

    @classmethod
    def config_model(cls) -> type[BaseModel]:
        return DinomalyConfig

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
        return module_available("anomalib", "dl", "Dinomaly")

    def _build_model(self, device: str, cache_dir: Path) -> Any:
        import torch

        torch.manual_seed(self.config.seed)
        with huggingface_environment(
            cache_dir,
            allow_downloads=self.config.allow_downloads,
            method="Dinomaly",
            asset=ENCODER_NAME,
        ):
            from anomalib.models import Dinomaly

            module = Dinomaly(
                encoder_name=ENCODER_NAME,
                decoder_depth=DECODER_DEPTH,
                pre_processor=False,
                post_processor=False,
                evaluator=False,
                visualizer=False,
            )
        model = module.model.to(device)
        self._fingerprint = fingerprint_state(model.encoder.state_dict())
        self._cache_dir = cache_dir
        return model

    @staticmethod
    def _trainable(model: Any) -> list[Any]:
        return [parameter for parameter in model.parameters() if parameter.requires_grad]

    def _optimizer(self, model: Any) -> Any:
        from anomalib.models.image.dinomaly.components import StableAdamW

        return StableAdamW(
            [{"params": self._trainable(model)}],
            lr=BASE_LR,
            betas=(0.9, 0.999),
            weight_decay=1e-4,
            amsgrad=True,
            eps=1e-8,
        )

    def fit(self, train: Sequence[ImageRecord], ctx: TrainContext) -> None:
        plan = plan_training(
            self.config,
            len(train),
            ctx.preprocessing.width,
            ctx.preprocessing.height,
        )
        ctx.log(plan.describe())
        ctx.progress(0.01, f"loading {ENCODER_NAME} (downloads on first run only)")
        self._model = self._build_model(ctx.device.value, ctx.cache_dir)
        self._completed = 0
        self._generator = np.random.default_rng(self.config.seed)
        optimizer = self._optimizer(self._model)
        self._run_steps(train, ctx, optimizer, self.config.max_steps)
        self._capture_resume_state(optimizer)
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
        if self._model is None or self._optimizer_state is None:
            raise RuntimeError("Dinomaly cannot continue because no resumable checkpoint is loaded")
        if not train:
            raise ValueError("Dinomaly needs at least one normal training image")
        validate_prepared_size(ctx.preprocessing.width, ctx.preprocessing.height)

        import torch

        self._model = self._model.to(ctx.device.value)
        optimizer = self._optimizer(self._model)
        optimizer.load_state_dict(self._optimizer_state)
        self._restore_rng(torch)
        ctx.log(
            f"continuing Dinomaly from step {self._completed} for "
            f"{additional_steps} more; optimizer, dropout and image order resume exactly"
        )
        self._run_steps(train, ctx, optimizer, additional_steps)
        self._capture_resume_state(optimizer)
        self._model.eval()

    def _run_steps(
        self,
        train: Sequence[ImageRecord],
        ctx: TrainContext,
        optimizer: Any,
        steps: int,
    ) -> None:
        import torch

        if self._model is None or self._generator is None:
            raise RuntimeError("Dinomaly training state was not initialised")
        self._model.train()
        started = time.perf_counter()
        start_step = self._completed
        total = start_step + steps

        for index in range(steps):
            ctx.raise_if_cancelled()
            step = self._completed
            record = train[int(self._generator.integers(len(train)))]
            image = torch.from_numpy(_normalised_chw(record, ctx)[None]).to(ctx.device.value)
            lr = learning_rate(step)
            for group in optimizer.param_groups:
                group["lr"] = lr

            optimizer.zero_grad(set_to_none=True)
            loss = self._model(image, global_step=step)
            loss.backward()
            optimizer.step()
            self._completed += 1

            if index % LOSS_LOG_EVERY == 0 or index == steps - 1:
                value = float(loss.detach().cpu())
                ctx.metric("loss_total", value, step=step)
                ctx.metric("learning_rate", lr, step=step)
                ctx.progress(
                    (index + 1) / max(steps, 1),
                    f"step {self._completed}/{total}, loss {value:.4f}",
                )

        elapsed = time.perf_counter() - started
        ctx.log(
            f"trained {steps} Dinomaly steps in {elapsed:.1f}s "
            f"({elapsed / max(steps, 1) * 1000:.0f} ms/step on {ctx.device.value}); "
            f"{self._completed} completed in total"
        )

    def _capture_resume_state(self, optimizer: Any) -> None:
        import torch

        self._optimizer_state = optimizer.state_dict()
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
            raise RuntimeError("Dinomaly was asked to predict before it was fitted or loaded")
        validate_prepared_size(ctx.preprocessing.width, ctx.preprocessing.height)
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
                score = float(output.pred_score[0].detach().cpu())
                predictions.append(
                    Prediction(
                        image_id=record.image_id,
                        score=score,
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
            raise RuntimeError("Dinomaly has nothing to save; it was never fitted")
        payload: dict[str, Any] = {
            "format": CHECKPOINT_FORMAT,
            "encoder": ENCODER_NAME,
            "decoder_depth": DECODER_DEPTH,
            "encoder_fingerprint": self._fingerprint,
            "bottleneck": self._model.bottleneck.state_dict(),
            "decoder": self._model.decoder.state_dict(),
            "completed_steps": self._completed,
            "optimizer": self._optimizer_state,
            "torch_rng_state": self._torch_rng_state,
            "mps_rng_state": self._mps_rng_state,
            "generator_state": self._generator_state,
            "cache_dir": str(self._cache_dir),
            "versions": {
                "anomalib": version("anomalib"),
                "torch": version("torch"),
                "timm": version("timm"),
            },
        }
        torch.save(payload, artifact_dir / STATE_FILENAME)

    def load(self, artifact_dir: Path) -> None:
        import torch

        stored = torch.load(
            artifact_dir / STATE_FILENAME,
            map_location="cpu",
            weights_only=False,
        )
        if int(stored.get("format", 0)) != CHECKPOINT_FORMAT:
            raise RuntimeError(
                f"unsupported Dinomaly checkpoint format {stored.get('format')!r}; "
                f"expected {CHECKPOINT_FORMAT}"
            )
        if stored.get("encoder") != ENCODER_NAME or stored.get("decoder_depth") != DECODER_DEPTH:
            raise RuntimeError("Dinomaly checkpoint architecture does not match this plugin")
        cache_dir = Path(stored["cache_dir"])
        model = self._build_model("cpu", cache_dir)
        expected = str(stored["encoder_fingerprint"])
        actual = str(self._fingerprint)
        if actual != expected:
            raise RuntimeError(
                f"Dinomaly encoder weights changed: checkpoint expects {expected[:12]}, "
                f"but {ENCODER_NAME} now resolves to {actual[:12]}. Restore the original "
                "asset or refit; these results are not comparable."
            )
        model.bottleneck.load_state_dict(stored["bottleneck"])
        model.decoder.load_state_dict(stored["decoder"])
        self._model = model.eval()
        self._completed = int(stored["completed_steps"])
        self._optimizer_state = stored.get("optimizer")
        self._torch_rng_state = stored.get("torch_rng_state")
        self._mps_rng_state = stored.get("mps_rng_state")
        self._generator_state = stored.get("generator_state")
