"""`dinomaly_custom` — Dinomaly, ours.

The seventh method, and the second in-house implementation of a method the workbench already
carries through anomalib. `dinomaly_anomalib` is the **baseline this is measured against, not
a specification to match** (**ADR-0029**): a wrapper establishes that a family is worth
having, and an implementation we own is what lets every decision inside it become a field on
a form. If this reaches parity on the VisA gate, the wrapper retires the way
`efficientad_anomalib` did — and until that verdict lands, nothing about the wrapper changes.

**What is genuinely different, and it is not the arithmetic.** The mechanism is ported
faithfully — `dinomaly_nets` says exactly what and cites where each shape comes from. Two
things the wrapper cannot do are the reason this module exists:

  * **The encoder is a field.** `dinomaly_anomalib` hard-codes `vit_small_patch14_reg4_dinov2`
    because that is what anomalib's constructor resolves. Here the encoder comes from the
    shared table in `dino_backbone.py`, so the two DINOv3 entries arrive for free and a
    comparison can ask whether Dinomaly's result is about *Dinomaly* or about DINOv2. The
    default is the encoder anomalib pins, so an untouched run is the like-for-like one.
  * **Decoder depth is a field.** anomalib's constructor accepts a depth and then indexes a
    fixed eight-output fusion topology, so any value but eight fails; the wrapper's docstring
    says so and fixes it at eight. Here the fusion groups are *derived* from the depth
    (`fuse_groups`), so a four-block decoder trains and produces maps.

**No anomalib import appears in this module or in `dinomaly_nets`**, which is the property
that makes a head-to-head meaningful at all: a second reading of one library is not a second
implementation.

**One asymmetry is worth stating rather than hiding.** The wrapper exports ONNX and this does
not yet, so `portable_formats` is empty. Nothing about the graph is hard — it is the parity
gate that has to be written and run — and it is on the backlog. An export offer is made from
the registry before any configuration is read, so claiming a format that has not passed the
generic parity gate would be worse than an absent one.

Heavy imports stay inside functions. This module is imported whenever the method picker opens,
and opening a picker must not cost three seconds of torch.
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
from anomaly_lab.models.diagnostics import DiagnosticKind
from anomaly_lab.models.dino_backbone import (
    BACKBONES,
    BackboneSpec,
    DinoBackbone,
    backbone_fingerprint,
    load_backbone,
    patch_grid,
    validate_prepared_size,
)
from anomaly_lab.models.preprocessing import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    expand_planes,
    load_array,
    to_chw,
)
from anomaly_lab.schemas import API_MODEL_CONFIG

STATE_FILENAME = "dinomaly_custom.pt"

CHECKPOINT_FORMAT = 1
"""There is no format 0. This method owns its checkpoint from the first commit, so it carries
everything a continuation needs and has no legacy shape to tolerate."""

METHOD = "dinomaly_custom"

TARGET_LAYERS: tuple[int, ...] = (2, 3, 4, 5, 6, 7, 8, 9)
"""Which encoder blocks are reconstructed, counted from the input.

The reference's choice for a twelve-block ViT, and the reason it is the middle of the stack
rather than the end: the last blocks are the most semantic and the least localised, which is
what a classifier wants and the opposite of what a pixel-level map needs. Every encoder in
`BACKBONES` is twelve blocks deep, so one tuple serves all five; `encoder_groups` refuses a depth
this list would not fit rather than silently indexing past the end."""

ENCODER_DEPTH = 12
"""Encoder depth `TARGET_LAYERS` is written for. Asserted, never assumed."""

WARMUP_STEPS = 100
"""Linear warm-up before the cosine begins — the reference's, and the wrapper's."""

SCHEDULE_STEPS = 5_000
"""The cosine's horizon, and it is deliberately **not** `max_steps`.

The published recipe's step budget, fixed here for the reason `dinomaly_anomalib` fixes it:
a horizon that followed the configured budget would move under a continuation. A 1000-step
run continued for 1000 more would then have annealed to the floor by step 1000 and spent its
second thousand there, while one uninterrupted 2000-step run annealed across the whole
range — two different experiments that the screen would show as one. With the horizon fixed,
`N` steps plus `M` more is exactly `N + M` uninterrupted, which is what makes
`test_thirty_plus_thirty_is_sixty` a meaningful thing to assert.

It is also what keeps the promotion gate like-for-like: this is the *same function* as the
wrapper's `learning_rate`, not a function that happens to agree at one setting. At the
default `max_steps` of 5000 the two readings coincide anyway; below it, a run simply stops
part-way down the cosine, which is what a shortened budget means."""

FINAL_LR_RATIO = 0.1
"""The cosine's floor, as a fraction of the base rate: 2e-3 anneals to 2e-4."""

LOSS_LOG_EVERY = 25
"""How often the loss and the learning rate reach the job's metric stream."""

BYTES_PER_FLOAT32 = 4


# ------------------------------------------------------------------ pure arithmetic


def fuse_groups(count: int) -> tuple[tuple[int, ...], ...]:
    """Split `count` ordered layer outputs into the two contiguous groups that get compared.

    Dinomaly does not compare layer to layer. It averages the first half of the encoder's
    target layers into one feature map and the second half into another, does the same to the
    decoder's outputs, and takes a cosine distance per group — two maps, averaged. Comparing
    layer to layer would make the decoder's job "reproduce block 4 at position 4", which is a
    much easier and much less useful task than "reproduce what the early half of the encoder
    saw, from a bottleneck".

    anomalib writes this as the literal `[[0, 1, 2, 3], [4, 5, 6, 7]]` for both sides, which
    is why its `decoder_depth` argument silently only works at eight. Deriving the split is
    the whole reason depth is configurable here: at depth 4 the decoder groups are
    `[[0, 1], [2, 3]]` and the encoder's stay `[[0, 1, 2, 3], [4, 5, 6, 7]]`, because the
    encoder always contributes eight target layers whatever the decoder does.

    An odd count puts the extra output in the *second* group, which is the deep-encoder end;
    at even counts — every value anyone is likely to pick — the question does not arise.
    """
    if count < 2:
        msg = f"two fusion groups need at least two layer outputs; got {count}"
        raise ValueError(msg)
    split = count // 2
    return (tuple(range(split)), tuple(range(split, count)))


def learning_rate(
    step: int,
    *,
    base: float,
    horizon: int = SCHEDULE_STEPS,
    warmup: int = WARMUP_STEPS,
) -> float:
    """The reference's warm-cosine schedule: linear to `base`, then cosine to `base / 10`.

    A pure function of the **absolute** step across the whole experiment, over a horizon that
    does not depend on what any one run asked for — see `SCHEDULE_STEPS` for why that is the
    property rather than an oversight. Past the horizon the rate stays at the floor: a run
    continued beyond its recipe's budget keeps training, at the rate the recipe ends on.
    """
    if step < 0:
        msg = f"training step cannot be negative; got {step}"
        raise ValueError(msg)
    final = base * FINAL_LR_RATIO
    if step < warmup:
        # `linspace(0, base, warmup)[step]`, in closed form: the last warm-up step is already
        # at `base`, so the cosine starts from the top rather than from one tick below it.
        return float(base * step / max(warmup - 1, 1))
    if step >= horizon:
        return float(final)
    span = max(horizon - warmup, 1)
    offset = step - warmup
    return float(final + 0.5 * (base - final) * (1 + math.cos(math.pi * offset / span)))


def trainable_parameter_count(embedding_dim: int, decoder_depth: int) -> int:
    """Closed-form parameter count of the bottleneck plus the decoder.

    Exact, and torch-free, so a plan can state the size of the thing about to be allocated
    before torch is imported — and so a `dl` test can assert the formula against the real
    `numel()` sum rather than against another copy of the same arithmetic.

    Per width `d`: the bottleneck is `8d²` (two bias-free projections through `4d`), and each
    decoder block is `12d² + 8d` — `3d² + 3d` for the fused QKV, `d² + d` for its projection,
    `8d²` for the bias-free MLP and `2d` for each of two LayerNorms.
    """
    if embedding_dim < 1 or decoder_depth < 1:
        msg = (
            "a parameter count needs a positive width and depth; got "
            f"embedding_dim={embedding_dim}, decoder_depth={decoder_depth}"
        )
        raise ValueError(msg)
    bottleneck = 8 * embedding_dim * embedding_dim
    block = 12 * embedding_dim * embedding_dim + 8 * embedding_dim
    return bottleneck + decoder_depth * block


# ------------------------------------------------------------------------ the plan


@dataclass(frozen=True)
class TrainingPlan:
    """What a fit is about to cost, computed before a tensor is allocated.

    Pure and torch-free like `dino_memory.MemoryPlan`, and for the same reason: the arithmetic
    that decides whether a run is legal and affordable is checked by the CI job that installs
    *without* the `dl` extra.
    """

    encoder: DinoBackbone
    timm_name: str
    pretrained_encoder: bool
    images_available: int
    steps: int
    batch_size: int
    width: int
    height: int
    grid_rows: int
    grid_cols: int
    patch_size: int
    embedding_dim: int
    num_heads: int
    decoder_depth: int
    encoder_groups: tuple[tuple[int, ...], ...]
    decoder_groups: tuple[tuple[int, ...], ...]
    trainable_parameters: int
    base_learning_rate: float
    final_learning_rate: float
    mined_fraction: float

    @property
    def positions(self) -> int:
        return self.grid_rows * self.grid_cols

    @property
    def images_seen(self) -> int:
        """Image loads the pass performs. Sampled with replacement, so it may exceed the set."""
        return self.steps * self.batch_size

    @property
    def estimated_checkpoint_bytes(self) -> int:
        """Weights plus the three StableAdamW moment buffers amsgrad keeps, in float32.

        The frozen encoder is deliberately absent from that number: it is somebody else's
        artifact, `load_backbone` rebuilds it from the app-managed cache, and a copy per
        experiment would multiply the disk cost of a comparison by the number of runs in it.
        """
        return self.trainable_parameters * BYTES_PER_FLOAT32 * 4

    def describe(self) -> str:
        """One line for the job log, said before the work rather than after it."""
        weights = "pretrained" if self.pretrained_encoder else "seeded random"
        groups = " + ".join(
            "/".join(str(index) for index in group) for group in self.decoder_groups
        )
        return (
            f"dinomaly_custom plan: {self.steps:,} steps at batch {self.batch_size} "
            f"({self.images_seen:,} image loads) sampling {self.images_available:,} normals "
            f"at {self.width}x{self.height}; frozen {weights} {self.encoder.value} "
            f"({self.timm_name}) on a {self.grid_rows}x{self.grid_cols} grid of "
            f"{self.patch_size}-pixel patches, {self.embedding_dim}-wide; decoder "
            f"{self.decoder_depth} blocks x {self.num_heads} heads fused as {groups}; "
            f"{self.trainable_parameters:,} trainable parameters, resumable checkpoint about "
            f"{self.estimated_checkpoint_bytes / 1e6:.0f} MB; learning rate "
            f"{self.base_learning_rate:.1e} to {self.final_learning_rate:.1e} over a fixed "
            f"{SCHEDULE_STEPS:,}-step horizon after a {WARMUP_STEPS}-step warm-up; mining "
            f"narrows to the hardest {self.mined_fraction:.0%} of points"
        )

    def table(self) -> dict[str, list[Any]]:
        """The same numbers as a diagnostic table, so a run can be read rather than grepped."""
        return {
            "columns": ["quantity", "value"],
            "rows": [
                ["encoder", f"{self.encoder.value} ({self.timm_name})"],
                ["encoder weights", "pretrained" if self.pretrained_encoder else "seeded random"],
                ["prepared frame", f"{self.width}x{self.height}"],
                ["patch grid", f"{self.grid_rows}x{self.grid_cols} ({self.positions} patches)"],
                ["embedding width", str(self.embedding_dim)],
                ["target encoder layers", ", ".join(str(index) for index in TARGET_LAYERS)],
                ["decoder", f"{self.decoder_depth} blocks x {self.num_heads} heads"],
                ["encoder fusion groups", str(self.encoder_groups)],
                ["decoder fusion groups", str(self.decoder_groups)],
                ["trainable parameters", f"{self.trainable_parameters:,}"],
                ["steps x batch", f"{self.steps:,} x {self.batch_size}"],
                [
                    "learning rate",
                    f"{self.base_learning_rate:.2e} to {self.final_learning_rate:.2e} "
                    f"over {SCHEDULE_STEPS:,}",
                ],
                ["mined fraction", f"{self.mined_fraction:.0%}"],
            ],
        }


def encoder_groups(spec: BackboneSpec) -> tuple[tuple[int, ...], ...]:
    """The encoder's fusion groups, refusing an encoder depth `TARGET_LAYERS` does not fit."""
    if spec.depth != ENCODER_DEPTH or max(TARGET_LAYERS) >= spec.depth:
        msg = (
            f"{METHOD}'s target layers {TARGET_LAYERS} are written for a {ENCODER_DEPTH}-block "
            f"encoder and {spec.timm_name} has {spec.depth}. Reconstructing the wrong blocks "
            "would run and produce a map that is not this method's."
        )
        raise ValueError(msg)
    return fuse_groups(len(TARGET_LAYERS))


def plan_training(
    config: DinomalyCustomConfig,
    train_count: int,
    width: int,
    height: int,
) -> TrainingPlan:
    """Resolve every number a fit depends on, without touching a file or a tensor."""
    if train_count < 1:
        msg = f"{METHOD} needs at least one normal training image; got {train_count}"
        raise ValueError(msg)
    spec = BACKBONES[config.encoder]
    rows, cols = patch_grid(config.encoder, width, height)
    return TrainingPlan(
        encoder=config.encoder,
        timm_name=spec.timm_name,
        pretrained_encoder=config.pretrained_encoder,
        images_available=train_count,
        steps=config.max_steps,
        batch_size=config.batch_size,
        width=width,
        height=height,
        grid_rows=rows,
        grid_cols=cols,
        patch_size=spec.patch_size,
        embedding_dim=spec.embedding_dim,
        num_heads=spec.num_heads,
        decoder_depth=config.decoder_depth,
        encoder_groups=encoder_groups(spec),
        decoder_groups=fuse_groups(config.decoder_depth),
        trainable_parameters=trainable_parameter_count(spec.embedding_dim, config.decoder_depth),
        base_learning_rate=config.learning_rate,
        final_learning_rate=config.learning_rate * FINAL_LR_RATIO,
        mined_fraction=config.hard_mining_fraction,
    )


# ---------------------------------------------------------------------------- config


class DinomalyCustomConfig(BaseModel):
    """Hyperparameters. Every field here becomes a control on the experiment form."""

    model_config = API_MODEL_CONFIG

    encoder: DinoBackbone = Field(
        default=DinoBackbone.DINOV2_VIT_S14_REG4,
        description=(
            "Frozen encoder the reconstructed features come from. The default is the "
            "registered DINOv2 that anomalib pins for Dinomaly, so an untouched run compares "
            "this implementation against dinomaly_anomalib rather than against a different "
            "backbone. The two DINOv3 entries are licence-gated: access must be requested "
            "from Meta on Hugging Face and an approved HF_TOKEN must already be in the "
            "environment, which is why they are never the default."
        ),
    )
    decoder_depth: int = Field(
        default=8,
        ge=2,
        le=12,
        description=(
            "Transformer blocks in the trainable decoder. The published recipe uses 8. This "
            "is genuinely configurable here and is not in dinomaly_anomalib, whose fusion "
            "topology indexes exactly eight decoder outputs: the two comparison groups are "
            "derived from the depth instead, so a 4-block decoder fuses outputs 0-1 against "
            "2-3 while the encoder's eight target layers keep their own 0-3 / 4-7 split. "
            "Shallower is faster and has less capacity to reconstruct; a checkpoint records "
            "its depth and refuses to load into an experiment asking for another."
        ),
    )
    max_steps: int = Field(
        default=5_000,
        ge=1,
        le=100_000,
        description=(
            "Reconstruction updates. The published recipe uses 5000. This number is also the "
            "The learning-rate schedule has its own fixed 5000-step horizon and does not "
            "follow this number, so 1000 steps plus a 1000-step continuation is exactly the "
            "same experiment as 2000 uninterrupted steps; a shorter budget simply stops "
            "part-way down the cosine."
        ),
    )
    batch_size: int = Field(
        default=1,
        ge=1,
        le=64,
        description=(
            "Images per step, sampled with replacement from the training normals. The "
            "default is 1 to match dinomaly_anomalib exactly, which is what makes the "
            "promotion comparison like-for-like; larger batches are steadier per step and "
            "proportionally more expensive."
        ),
    )
    learning_rate: float = Field(
        default=2e-3,
        gt=0.0,
        le=1.0,
        description=(
            "Peak rate after a 100-step linear warm-up, annealed by cosine to a tenth of it "
            "over max_steps. StableAdamW clips the update rather than the gradient, which is "
            "what makes a rate this high survive the first hundred steps."
        ),
    )
    weight_decay: float = Field(
        default=1e-4,
        ge=0.0,
        le=1.0,
        description="Decoupled weight decay applied to the bottleneck and decoder weights.",
    )
    hard_mining_fraction: float = Field(
        default=0.1,
        gt=0.0,
        le=1.0,
        description=(
            "Fraction of the hardest feature points whose gradient is kept at full strength; "
            "the rest are multiplied by 0.1. This is what stops the decoder from learning to "
            "reconstruct anomalies too, which would flatten the anomaly map. Annealed from 1 "
            "over the first 1000 steps, because mining on an untrained decoder would mine "
            "noise. 1.0 disables mining entirely and is a real ablation."
        ),
    )
    dropout: float = Field(
        default=0.2,
        ge=0.0,
        lt=1.0,
        description=(
            "Dropout inside the bottleneck MLP, applied to its input and after each "
            "projection. The bottleneck is the narrowing that stops the decoder copying its "
            "input, so this is where the regularisation belongs; the decoder blocks have none."
        ),
    )
    map_blur_sigma: float = Field(
        default=0.0,
        ge=0.0,
        le=32.0,
        description=(
            "Optional Gaussian smoothing of the stored anomaly map, in prepared pixels. The "
            "default of 0 leaves the map exactly as dinomaly_anomalib produces it, so pixel "
            "metrics compare implementations rather than post-processing. The image score is "
            "unaffected either way: it has its own fixed smoothing, from the reference."
        ),
    )
    pretrained_encoder: bool = Field(
        default=True,
        description=(
            "Use the published self-supervised weights. Off gives a seeded random ViT, which "
            "needs no download; reconstruction still works against it, because the decoder "
            "learns whatever feature space it is shown. This is what the hermetic tests use, "
            "and it is a real ablation rather than only a test affordance."
        ),
    )
    allow_downloads: bool = Field(
        default=True,
        description=(
            "Permit fetching the encoder weights on the first run. Turn off to make a "
            "missing encoder an error naming the asset and this switch, instead of a silent "
            "reach for the network."
        ),
    )
    seed: int = Field(
        default=0,
        description=(
            "Seeds the decoder and bottleneck initialisation, the bottleneck dropout, the "
            "training image order and — when pretrained_encoder is off — the encoder itself. "
            "Two runs with one seed are one experiment."
        ),
    )


# ---------------------------------------------------------------------------- pixels


def _normalised_chw(record: ImageRecord, ctx: TrainContext | InferContext) -> np.ndarray:
    """The shared prepared pixels, then this backbone's own channel standardisation.

    Two separate acts, on two sides of a seam (`preprocessing.py`): `load_array` is the
    experiment's decision and is identical for every method, while the plane count and the
    ImageNet statistics belong to *this* encoder.
    """
    array = expand_planes(to_chw(load_array(record.path, ctx.preprocessing)), 3)
    mean = np.asarray(IMAGENET_MEAN, dtype=np.float32)[:, None, None]
    std = np.asarray(IMAGENET_STD, dtype=np.float32)[:, None, None]
    return np.ascontiguousarray((array - mean) / std, dtype=np.float32)


# ----------------------------------------------------------------------- the model


class DinomalyCustomModel(AnomalyModel):
    """Dinomaly with a configurable frozen encoder and a decoder whose depth is a field."""

    title = "Dinomaly (ours)"
    summary = (
        "Our implementation of Dinomaly: a frozen DINOv2/DINOv3 encoder with a trainable "
        "bottleneck and linear-attention decoder that reconstructs normal feature maps. "
        "Reconstruction disagreement is the anomaly map."
    )

    def __init__(self, config: DinomalyCustomConfig) -> None:
        super().__init__(config)
        self.config = config
        self._net: Any = None
        self._fingerprint: str | None = None
        self._cache_dir: Path | None = None
        self._fitted_size: tuple[int, int] | None = None
        self._completed = 0
        # Everything below is what a continuation needs (handbook jobs.md), held on the
        # instance because `save` runs after `fit` returns and has no other way to reach it.
        self._optimizer_state: dict[str, Any] | None = None
        self._generator: np.random.Generator | None = None
        self._generator_state: Any = None
        self._torch_rng_state: Any = None
        self._mps_rng_state: Any = None

    @classmethod
    def config_model(cls) -> type[BaseModel]:
        return DinomalyCustomConfig

    @classmethod
    def capabilities(cls) -> Capabilities:
        return Capabilities(
            requires_training=True,
            produces_anomaly_map=True,
            produces_diagnostics=True,
            supports_resume=True,
            # The method sees one image at a time and knows nothing about views. Aggregating
            # a part's channels into one verdict is the evaluation layer's job.
            channel_aware=False,
            dataset_specific=False,
            # Empty, and the asymmetry with dinomaly_anomalib is deliberate rather than
            # forgotten: the export offer is made from the registry before any configuration
            # is read, so a format that has not passed the generic Python-versus-runtime
            # parity gate must not be claimed. See docs/backlog.md.
            portable_formats=[],
            preferred_device=Device.MPS,
        )

    @classmethod
    def availability(cls) -> Availability:
        # torch and timm, not anomalib. That difference is the point of this method existing.
        return module_available("torch", "dl", "Dinomaly (ours)")

    # ------------------------------------------------------------------ construction

    def _build(self, device: str, cache_dir: Path) -> Any:
        """Encoder from the shared table, then a decoder whose init is a function of the seed.

        The two seeds are both load-bearing and they are not the same act. `load_backbone`
        seeds *before* constructing the encoder, because an unpretrained ViT draws its weights
        from torch's global stream (M6's finding). The second seed here covers the bottleneck
        and decoder for the same reason: left alone, `seed` would mean "the same training
        order over different initial weights", which is not one experiment.
        """
        import torch

        from anomaly_lab.models.dinomaly_nets import DinomalyNet

        spec = BACKBONES[self.config.encoder]
        encoder = load_backbone(
            self.config.encoder,
            pretrained=self.config.pretrained_encoder,
            allow_downloads=self.config.allow_downloads,
            cache_dir=cache_dir,
            seed=self.config.seed,
            method=METHOD,
        )
        self._verify_encoder(encoder, spec)
        self._fingerprint = backbone_fingerprint(encoder)
        self._cache_dir = cache_dir

        torch.manual_seed(self.config.seed)
        net = DinomalyNet(
            encoder,
            embedding_dim=spec.embedding_dim,
            num_heads=spec.num_heads,
            num_prefix_tokens=int(getattr(encoder, "num_prefix_tokens", 1)),
            patch_size=spec.patch_size,
            decoder_depth=self.config.decoder_depth,
            target_layers=TARGET_LAYERS,
            encoder_groups=encoder_groups(spec),
            decoder_groups=fuse_groups(self.config.decoder_depth),
            dropout=self.config.dropout,
        )
        return net.to(device)

    def _verify_encoder(self, encoder: Any, spec: BackboneSpec) -> None:
        """The table's claims about this encoder have to be the constructed model's.

        `dino_memory` verifies its planned grid against the first real batch; the same
        discipline, one step earlier. A plan that announced a 384-wide, six-head decoder for a
        model that is neither would build a decoder that runs and reconstructs the wrong
        thing, which nothing downstream can see.
        """
        measured = {
            "embedding width": (int(encoder.num_features), spec.embedding_dim),
            "depth": (len(encoder.blocks), spec.depth),
            "attention heads": (int(encoder.blocks[0].attn.num_heads), spec.num_heads),
            "patch size": (int(encoder.patch_embed.patch_size[0]), spec.patch_size),
        }
        wrong = [
            f"{name} {actual} rather than {expected}"
            for name, (actual, expected) in measured.items()
            if actual != expected
        ]
        if wrong:
            msg = (
                f"the backbone table describes {spec.timm_name} as a {spec.depth}-block, "
                f"{spec.embedding_dim}-wide, {spec.num_heads}-head encoder with "
                f"{spec.patch_size}-pixel patches, and timm produced {', '.join(wrong)}. The "
                "plan's decoder shape and footprints describe a run that is not this one."
            )
            raise RuntimeError(msg)

    # ------------------------------------------------------------------ training

    def fit(self, train: Sequence[ImageRecord], ctx: TrainContext) -> None:
        import torch

        plan = plan_training(
            self.config,
            len(train),
            ctx.preprocessing.width,
            ctx.preprocessing.height,
        )
        # Before the first forward pass, not after it. Everything in it is arithmetic.
        ctx.log(plan.describe())
        ctx.progress(0.01, f"loading {plan.timm_name} (downloads on first run only)")

        self._net = self._build(ctx.device.value, ctx.cache_dir)
        self._fitted_size = (ctx.preprocessing.width, ctx.preprocessing.height)
        self._completed = 0
        self._generator = np.random.default_rng(self.config.seed)

        frozen = sum(parameter.numel() for parameter in self._net.encoder.parameters())
        trainable = sum(parameter.numel() for parameter in self._net.trainable_parameters())
        if trainable != plan.trainable_parameters:
            msg = (
                f"{METHOD} planned {plan.trainable_parameters:,} trainable parameters and "
                f"built {trainable:,}. The closed form in trainable_parameter_count and the "
                "network in dinomaly_nets disagree."
            )
            raise RuntimeError(msg)
        ctx.log(
            f"{frozen:,} frozen encoder parameters, {trainable:,} trainable "
            f"({trainable / max(frozen, 1):.1%} of the encoder's size)"
        )
        ctx.emit_diagnostic(
            "training_plan",
            "Training plan",
            DiagnosticKind.TABLE,
            plan.table(),
            description=(
                "Every number this fit was bounded by, resolved before the encoder was built."
            ),
        )

        optimizer = self._optimizer()
        self._run_steps(train, ctx, optimizer, self.config.max_steps)
        self._capture_resume_state(optimizer, torch)
        self._net.eval()

    def completed_steps(self) -> int:
        """Steps trained so far, across every run (handbook jobs.md)."""
        return self._completed

    def fit_more(
        self,
        train: Sequence[ImageRecord],
        ctx: TrainContext,
        *,
        additional_steps: int,
    ) -> None:
        """Continue a loaded model for `additional_steps` further steps.

        Every refusal here is **by name**. A continuation that quietly restarted the optimizer,
        moved to a different input size or attached a trained decoder to a different encoder's
        feature space would run, and every number afterwards would be plausible and wrong.
        """
        import torch

        if self._net is None:
            msg = f"{METHOD} cannot continue because no fitted model is loaded"
            raise RuntimeError(msg)
        if self._optimizer_state is None:
            msg = (
                f"{METHOD} cannot continue because this checkpoint carries no optimizer "
                "state. A continuation that resets the moments is not a continuation."
            )
            raise RuntimeError(msg)
        if not train:
            msg = f"{METHOD} needs at least one normal training image to continue"
            raise ValueError(msg)

        width, height = ctx.preprocessing.width, ctx.preprocessing.height
        validate_prepared_size(self.config.encoder, width, height)
        if self._fitted_size is not None and self._fitted_size != (width, height):
            fitted_width, fitted_height = self._fitted_size
            msg = (
                f"this {METHOD} checkpoint was trained at {fitted_width}x{fitted_height} and "
                f"the experiment is now configured for {width}x{height}. The decoder learned "
                "to reconstruct one patch grid and would be continued against another; train "
                "from scratch."
            )
            raise RuntimeError(msg)

        self._net = self._net.to(ctx.device.value)
        optimizer = self._optimizer()
        optimizer.load_state_dict(self._optimizer_state)
        self._restore_random_state(torch, ctx)
        ctx.log(
            f"continuing {METHOD} from step {self._completed} for {additional_steps} more, "
            f"on the same fixed {SCHEDULE_STEPS:,}-step schedule; optimizer moments, "
            f"bottleneck dropout and image order all resume where they stopped"
        )
        self._run_steps(train, ctx, optimizer, additional_steps)
        self._capture_resume_state(optimizer, torch)
        self._net.eval()

    def _optimizer(self) -> Any:
        from anomaly_lab.models.dinomaly_nets import StableAdamW

        return StableAdamW(
            [{"params": self._net.trainable_parameters()}],
            lr=self.config.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=self.config.weight_decay,
            amsgrad=True,
        )

    def _batch(self, train: Sequence[ImageRecord], ctx: TrainContext, torch_module: Any) -> Any:
        if self._generator is None:
            msg = f"{METHOD} training state was not initialised"
            raise RuntimeError(msg)
        indices = self._generator.integers(len(train), size=self.config.batch_size)
        stacked = np.stack([_normalised_chw(train[int(index)], ctx) for index in indices])
        return torch_module.from_numpy(stacked).to(ctx.device.value)

    def _run_steps(
        self,
        train: Sequence[ImageRecord],
        ctx: TrainContext,
        optimizer: Any,
        steps: int,
    ) -> None:
        """The training loop, shared by `fit` and `fit_more`.

        **Steps reported to `ctx.metric` are absolute across the experiment's training**, so a
        continued run's curve continues the first rather than starting a second one at zero
        (**handbook jobs.md**). Cancellation is polled every step, which is the practical
        benefit of owning the loop: stopping a run takes about one step, not one epoch.
        """
        import torch

        from anomaly_lab.models.dinomaly_nets import hard_mined_cosine_loss, mined_fraction

        if self._net is None:
            msg = f"{METHOD} training state was not initialised"
            raise RuntimeError(msg)

        self._net.train()
        started = time.perf_counter()
        total = self._completed + steps

        for index in range(steps):
            ctx.raise_if_cancelled()
            step = self._completed
            batch = self._batch(train, ctx, torch)
            rate = learning_rate(step, base=self.config.learning_rate)
            for group in optimizer.param_groups:
                group["lr"] = rate
            fraction = mined_fraction(step, self.config.hard_mining_fraction)

            optimizer.zero_grad(set_to_none=True)
            encoded, decoded = self._net.features(batch)
            # `Any` because `Tensor.backward` is untyped in torch's stubs and torch itself is
            # absent from the environment that type-checks the torch-free boundary.
            loss: Any = hard_mined_cosine_loss(encoded, decoded, step=step, fraction=fraction)
            loss.backward()
            optimizer.step()
            self._completed = step + 1

            if index % LOSS_LOG_EVERY == 0 or index == steps - 1:
                value = float(loss.detach().cpu())
                ctx.metric("loss_total", value, step=step)
                ctx.metric("learning_rate", rate, step=step)
                ctx.metric("mined_fraction", fraction, step=step)
                ctx.progress(
                    (index + 1) / max(steps, 1),
                    f"step {self._completed}/{total}, loss {value:.4f}",
                )

        elapsed = time.perf_counter() - started
        ctx.log(
            f"trained {steps} {METHOD} steps in {elapsed:.1f}s "
            f"({elapsed / max(steps, 1) * 1000:.0f} ms/step on {ctx.device.value}); "
            f"{self._completed} completed in total"
        )

    # ------------------------------------------------------------------ random state

    def _capture_resume_state(self, optimizer: Any, torch_module: Any) -> None:
        """Leave the instance in exactly the state a reloaded one would be in."""
        self._optimizer_state = optimizer.state_dict()
        self._torch_rng_state = torch_module.get_rng_state()
        self._generator_state = (
            None if self._generator is None else self._generator.bit_generator.state
        )
        self._mps_rng_state = None
        if hasattr(torch_module, "mps") and hasattr(torch_module.mps, "get_rng_state"):
            with contextlib.suppress(Exception):
                self._mps_rng_state = torch_module.mps.get_rng_state()

    def _restore_random_state(self, torch_module: Any, ctx: TrainContext) -> None:
        """Put every stream back where the previous leg left it.

        The torch stream is global here rather than a private `Generator`, because dropout
        draws from it and there is no per-module override. That is the same shape
        `dinomaly_anomalib` has, and it means a continuation restores the process-wide state —
        acceptable because a training worker runs one job.
        """
        if self._torch_rng_state is not None:
            torch_module.set_rng_state(self._torch_rng_state)
        else:
            ctx.log(
                "this checkpoint carries no torch random state, so bottleneck dropout "
                "restarts from the seed rather than continuing",
                level="warning",
            )
            torch_module.manual_seed(self.config.seed)
        if self._mps_rng_state is not None and hasattr(torch_module.mps, "set_rng_state"):
            with contextlib.suppress(Exception):
                torch_module.mps.set_rng_state(self._mps_rng_state)
        if self._generator_state is not None:
            self._generator = np.random.default_rng()
            self._generator.bit_generator.state = self._generator_state
        else:
            ctx.log(
                "this checkpoint carries no image-order state, so the training image "
                "sequence restarts from the seed",
                level="warning",
            )
            self._generator = np.random.default_rng(self.config.seed)

    # ------------------------------------------------------------------ inference

    def predict(self, images: Sequence[ImageRecord], ctx: InferContext) -> list[Prediction]:
        import torch

        from anomaly_lab.models.dinomaly_nets import gaussian_blur, group_maps, image_score

        if self._net is None:
            msg = f"{METHOD} was asked to predict before it was fitted or loaded"
            raise RuntimeError(msg)
        validate_prepared_size(
            self.config.encoder, ctx.preprocessing.width, ctx.preprocessing.height
        )

        net = self._net.to(ctx.device.value)
        net.eval()
        self._net = net
        size = (ctx.preprocessing.height, ctx.preprocessing.width)
        predictions: list[Prediction] = []

        with torch.inference_mode():
            for index, record in enumerate(images):
                ctx.raise_if_cancelled()
                started = time.perf_counter()
                batch = torch.from_numpy(_normalised_chw(record, ctx)[None]).to(ctx.device.value)

                encoded, decoded = net.features(batch)
                maps = group_maps(encoded, decoded, size)
                combined = torch.cat(maps, dim=1).mean(dim=1, keepdim=True)
                score = float(image_score(combined)[0])
                stored = combined
                if self.config.map_blur_sigma > 0:
                    radius = max(1, round(3.0 * self.config.map_blur_sigma))
                    stored = gaussian_blur(
                        combined,
                        kernel_size=2 * radius + 1,
                        sigma=self.config.map_blur_sigma,
                    )
                if ctx.device is Device.MPS:
                    torch.mps.synchronize()
                elapsed_ms = (time.perf_counter() - started) * 1000.0

                map_path = ctx.write_map(record.image_id, stored[0, 0].cpu().numpy())
                # The two groups apart, which is the diagnostic that makes Dinomaly legible:
                # the shallow-encoder group localises and the deep one recognises, and the
                # stored map is their mean.
                for group, values in enumerate(maps):
                    ctx.emit_diagnostic(
                        f"map_group_{group}",
                        f"Reconstruction error, group {group}",
                        DiagnosticKind.MAP,
                        values[0, 0].cpu().numpy().astype(np.float32),
                        image_id=record.image_id,
                        description=(
                            "Cosine distance between encoder layers "
                            f"{net.encoder_groups[group]} and decoder outputs "
                            f"{net.decoder_groups[group]}, before the two groups are "
                            "averaged into the stored map."
                        ),
                    )

                predictions.append(
                    Prediction(
                        image_id=record.image_id,
                        score=score,
                        anomaly_map=map_path,
                        inference_ms=elapsed_ms,
                    )
                )
                ctx.progress(
                    (index + 1) / max(len(images), 1),
                    f"scored {index + 1}/{len(images)} images",
                )
        return predictions

    # ------------------------------------------------------------------ persistence

    def save(self, artifact_dir: Path) -> None:
        """Write everything a *continuation* needs, not only what inference needs.

        The encoder's weights are deliberately absent — see the note on
        `TrainingPlan.estimated_checkpoint_bytes`. What is stored instead is the fingerprint,
        which turns "the weights might have changed" from an unanswerable worry into a check.
        """
        import torch

        if self._net is None or self._fingerprint is None or self._cache_dir is None:
            msg = f"{METHOD} has nothing to save; it was never fitted"
            raise RuntimeError(msg)

        spec = BACKBONES[self.config.encoder]
        payload: dict[str, Any] = {
            "format": CHECKPOINT_FORMAT,
            "encoder": self.config.encoder.value,
            "timm_name": spec.timm_name,
            "pretrained_encoder": self.config.pretrained_encoder,
            "encoder_fingerprint": self._fingerprint,
            "decoder_depth": self.config.decoder_depth,
            "embedding_dim": spec.embedding_dim,
            "target_layers": list(TARGET_LAYERS),
            "trainable": self._net.trainable_state(),
            "completed_steps": self._completed,
            "fitted_size": list(self._fitted_size) if self._fitted_size else None,
            # The schedule has no state of its own: its position *is* completed_steps, read
            # against the fixed SCHEDULE_STEPS horizon. Nothing to store and nothing to drift.
            "optimizer": self._optimizer_state,
            "torch_rng_state": self._torch_rng_state,
            "mps_rng_state": self._mps_rng_state,
            "generator_state": self._generator_state,
            "cache_dir": str(self._cache_dir),
            "versions": {"torch": version("torch"), "timm": version("timm")},
        }
        torch.save(payload, artifact_dir / STATE_FILENAME)

    def load(self, artifact_dir: Path) -> None:
        """Rebuild the encoder, restore the trainable weights, refuse anything that mismatches."""
        import torch

        stored = torch.load(artifact_dir / STATE_FILENAME, map_location="cpu", weights_only=False)

        found = int(stored.get("format", 0))
        if found != CHECKPOINT_FORMAT:
            msg = (
                f"this {METHOD} checkpoint declares format {found}, and this build reads "
                f"format {CHECKPOINT_FORMAT}. The file was written by a different version of "
                "the plugin and its layout is not this one; train the experiment again."
            )
            raise RuntimeError(msg)

        recorded_encoder = str(stored.get("encoder", ""))
        if recorded_encoder != self.config.encoder.value:
            msg = (
                f"this {METHOD} checkpoint was fitted on encoder {recorded_encoder!r} and the "
                f"experiment is configured for {self.config.encoder.value!r}. The decoder "
                "learned to reconstruct one feature space and cannot be read in another; "
                "refit, or open the experiment this checkpoint belongs to."
            )
            raise RuntimeError(msg)

        recorded_depth = int(stored.get("decoder_depth", 0))
        if recorded_depth != self.config.decoder_depth:
            msg = (
                f"this {METHOD} checkpoint holds a {recorded_depth}-block decoder and the "
                f"experiment asks for {self.config.decoder_depth}. Depth changes both the "
                "weight shapes and which outputs are fused against which encoder layers; "
                "refit at the depth you want."
            )
            raise RuntimeError(msg)

        size = stored.get("fitted_size")
        if not size:
            msg = (
                f"this {METHOD} checkpoint records no prepared size, so nothing can check "
                "that it is being scored on the grid it was trained on; train the experiment "
                "again."
            )
            raise RuntimeError(msg)
        fitted_size = (int(size[0]), int(size[1]))

        cache_dir = Path(stored["cache_dir"])
        net = self._build("cpu", cache_dir)
        expected = str(stored["encoder_fingerprint"])
        actual = str(self._fingerprint)
        if expected != actual:
            msg = (
                f"the encoder weights for {recorded_encoder!r} are not the ones this "
                f"experiment was fitted against: the checkpoint records {expected[:12]} and "
                f"the weights now resolving are {actual[:12]}. The decoder was trained in the "
                "old feature space, so its reconstructions of the new one mean nothing. Refit "
                "the experiment, or restore the original weights."
            )
            raise RuntimeError(msg)

        net.load_trainable_state(stored["trainable"])
        self._net = net.eval()
        self._fitted_size = fitted_size
        self._completed = int(stored.get("completed_steps", 0))
        self._optimizer_state = stored.get("optimizer")
        self._torch_rng_state = stored.get("torch_rng_state")
        self._mps_rng_state = stored.get("mps_rng_state")
        self._generator_state = stored.get("generator_state")
