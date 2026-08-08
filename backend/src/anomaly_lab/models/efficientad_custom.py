"""`efficientad_custom` — EfficientAD (arXiv:2303.14535), ours.

The second implementation of one method, behind the same interface as the first
(**ADR-0007**, **ADR-0008**). `efficientad_anomalib` wraps Intel's library and is the
**baseline this is measured against, not the specification it must match** (**ADR-0029**):
everything that decides a number here is ours, and improving on the baseline on evidence is
the point of having built the workbench.

The pieces live next door and are imported inside functions, so the registry stays lazy and
opening the method picker costs nothing: `efficientad_nets` holds the architecture and the
arithmetic, `efficientad_assets` fetches the teacher and the penalty set. Nothing in this
module's import path is torch.

**Every improvement is a configuration field.** Options are free here — the method picker
and every form are generated from this file's JSON Schema — so each one is an ablation the
workbench itself can run and put beside its own baseline on the comparison screen
(**ADR-0028**). Their defaults reproduce the published *algorithm*, so an untouched run is
the verified core that `test_efficientad_equivalence.py` pins, and a change is one field.

**One default is no longer the wrapper's, and it is the most important one.**
`teacher_source` defaults to `nelson1425` rather than to the asset anomalib ships, because
over three seeds on `candle` that teacher measured 0.889 sample ROC-AUC against 0.769, and
0.914 AU-PRO against 0.539 — an effect larger than any other in this milestone and far
outside either implementation's seed spread.

**New runs do not use the anomalib teacher at all.** It remains a value of `teacher_source`
only so the runs already recorded against it stay reproducible — an experiment's
configuration is the record of what it did, and removing the value would make five of them
unloadable. The consequence to keep in view: an implementation-versus-implementation
head-to-head against the wrapper would have to pin it, so **that comparison is now a
deliberate exercise rather than something a default produces**. ADR-0029 still makes the
wrapper the baseline; it is a baseline this method has left behind rather than one it is
tracked against run by run.

**What differs from the wrapper, and why:**

  * **Guards.** The autoencoder's *encoder* has an 8x8 kernel after five stride-2
    convolutions, so 256 px is a hard floor. Below it the wrapper fails inside `conv2d`
    with a message about a padded input size, several layers from anything actionable.
  * **Calibration can hold out its own normals.** EfficientAD fits its score normalization
    on held-out data, and VisA's official one-class protocol has no `val` subset, so the
    wrapper falls back to the training normals and says so. `calibration_holdout` carves a
    slice out of training instead. Off by default, because it is a hypothesis until it has
    been measured.
  * **The quantile fit is a bounded reservoir**, not every map held in memory. `torch.quantile`
    refuses more than 2**24 elements — a hard failure on MPS, not a slow path — so a fit over
    900 normals at 256 px could not be exact whatever we did. Sampling is the honest form of
    an estimate the reference also makes, and it is what lets `quantile_images` be a *time*
    budget rather than a memory one.
  * **Every random decision comes from a generator we own**, so a seed means something, and
    the training order, the augmentation and the penalty order all resume exactly.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

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
from anomaly_lab.models.diagnostics import DiagnosticKind
from anomaly_lab.models.introspect import ModuleRecord, build_tree, collect
from anomaly_lab.models.preprocessing import load_array, to_chw
from anomaly_lab.schemas import API_MODEL_CONFIG

STATE_FILENAME = "efficientad_custom.pt"

CHECKPOINT_FORMAT = 1
"""There is no format 0. This method owns its checkpoint from the first commit, so it can
carry everything a continuation needs (**ADR-0025**) without a legacy shape to refuse."""

DEFAULT_MAX_STEPS = 4000
"""The paper trains for 70000, about two and a half hours here at the measured 123 ms/step.
This default matches the wrapper's so a head-to-head is like-for-like at the same budget,
and it costs accuracy against the published number — which the field description says."""

_LOSS_LOG_EVERY = 20
_QUANTILE_LOW = 0.9
_QUANTILE_HIGH = 0.995
_LR_GAMMA = 0.1
"""The paper's single tenfold drop. Named because `_build_scheduler` applies it twice —
once through `StepLR` and once in the closed form that positions a resumed run."""


class EfficientAdCustomConfig(BaseModel):
    model_config = API_MODEL_CONFIG

    model_size: Literal["small", "medium"] = Field(
        default="small",
        description="PDN capacity. 'medium' is more accurate and roughly twice the cost.",
    )
    max_steps: int = Field(
        default=DEFAULT_MAX_STEPS,
        ge=10,
        le=200_000,
        description=(
            "Training steps, at batch size 1. The paper uses 70000; the default here is "
            "far lower so a first run finishes in minutes, and will cost some accuracy."
        ),
    )
    learning_rate: float = Field(default=1e-4, gt=0.0, le=1.0, description="Adam learning rate.")
    weight_decay: float = Field(default=1e-5, ge=0.0, le=1.0, description="Adam weight decay.")
    seed: int = Field(
        default=0,
        description=(
            "Seeds every random decision: training image order, augmentation, penalty "
            "order and the calibration sample. Two runs with one seed are one experiment."
        ),
    )
    pretrained_teacher: bool = Field(
        default=True,
        description=(
            "Use the published teacher distilled from ImageNet. Turning it off starts from "
            "a seeded random teacher, which needs no download and is a real question: the "
            "distillation mechanism works with any fixed feature extractor."
        ),
    )
    teacher_source: Literal["nelson1425", "distilled", "anomalib"] = Field(
        default="nelson1425",
        description=(
            "Which teacher to distil the student against. The published ones are different "
            "networks, not two copies of one: same architecture, weights differing tensor "
            "by tensor. 'nelson1425' measured far better — 0.889 against 0.769 sample "
            "ROC-AUC over three seeds on candle, with AU-PRO 0.914 against 0.539 — and is "
            "what new runs use. 'distilled' is one this workbench produced; name it in "
            "distilled_teacher. 'anomalib' is kept only so the runs recorded against it "
            "stay reproducible, and is listed last for that reason."
        ),
    )
    distilled_teacher: str = Field(
        default="",
        description=(
            "Name of a teacher produced by a distill job, when teacher_source is "
            "'distilled'. Its recorded architecture and channel count are checked against "
            "this experiment before it is loaded."
        ),
    )
    allow_downloads: bool = Field(
        default=True,
        description=(
            "Permit fetching the pretrained teacher and the ImageNette penalty set into "
            "the shared model cache. Turn off to make a missing asset an error instead."
        ),
    )
    stats_batch_size: int = Field(
        default=8,
        ge=1,
        le=64,
        description=(
            "Batch size for the teacher-statistics and calibration passes only. Training "
            "itself is fixed at 1, which EfficientAD requires."
        ),
    )
    use_penalty: bool = Field(
        default=True,
        description=(
            "Push the student toward zero on unrelated natural images, so it cannot learn "
            "the teacher's function everywhere and lose its capacity to be surprised. The "
            "paper measures this at +0.4 AU-ROC; off, no penalty set is needed."
        ),
    )
    hard_quantile: float = Field(
        default=0.999,
        ge=0.5,
        lt=1.0,
        description=(
            "Hard feature loss: backpropagate only through the student's worst elements, "
            "above this quantile. The paper measures it at +1.0 AU-ROC."
        ),
    )
    calibration_holdout: float = Field(
        default=0.0,
        ge=0.0,
        le=0.5,
        description=(
            "Fraction of training normals held out to fit the score normalization, used "
            "only when the split has no val subset. Quantiles fitted on images the student "
            "has memorized understate its spread and tilt the combined map; 0 keeps the "
            "published behaviour."
        ),
    )
    quantile_images: int = Field(
        default=512,
        ge=8,
        le=4096,
        description=(
            "How many normals fit the score-normalization quantiles, sampled evenly across "
            "the set rather than taking the first N."
        ),
    )
    quantile_pixel_budget: int = Field(
        default=1 << 22,
        ge=1 << 14,
        le=1 << 24,
        description=(
            "Pixels sampled per branch for the quantile fit. torch.quantile refuses more "
            "than 2**24 elements, so an exact fit over a full training set is impossible "
            "on any backend; this bounds the estimate explicitly."
        ),
    )
    student_teacher_weight: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description=(
            "Blend of the two branch maps. 0.5 is the published even mix; higher favours "
            "the local student-teacher branch over the global autoencoder one."
        ),
    )
    score_reduction: Literal["max", "top_k_mean"] = Field(
        default="max",
        description=(
            "How the combined map becomes one score. 'max' is the paper. 'top_k_mean' "
            "averages the hottest pixels, which is less sensitive to a lone outlier."
        ),
    )
    score_top_k: int = Field(
        default=64,
        ge=1,
        le=65_536,
        description="Pixels averaged when score_reduction is 'top_k_mean'.",
    )


class EfficientAdCustomModel(AnomalyModel):
    """EfficientAD, implemented here rather than wrapped."""

    title = "EfficientAD (ours)"
    summary = (
        "Our implementation of student-teacher distillation with an autoencoder branch, "
        "measured against the anomalib version rather than copied from it."
    )

    def __init__(self, config: EfficientAdCustomConfig) -> None:
        super().__init__(config)
        self.config = config
        self._net: Any = None
        self._completed = 0
        self._fitted_size: tuple[int, int] | None = None
        self._teacher: str | None = None
        """Which teacher this student was distilled against; None if it predates recording."""
        # Everything below is what a continuation needs (ADR-0025), held on the instance
        # because `save` runs after `fit` returns and has no other way to reach it.
        self._optimizer_state: Any = None
        self._scheduler_state: Any = None
        self._generator_state: Any = None
        self._torch_rng_state: Any = None
        self._penalty_state: dict[str, Any] | None = None
        self._generator: Any = None
        self._torch_generator: Any = None

    @classmethod
    def config_model(cls) -> type[BaseModel]:
        return EfficientAdCustomConfig

    @classmethod
    def capabilities(cls) -> Capabilities:
        return Capabilities(
            requires_training=True,
            produces_anomaly_map=True,
            produces_diagnostics=True,
            supports_resume=True,
            channel_aware=False,
            dataset_specific=False,
            preferred_device=Device.MPS,
        )

    @classmethod
    def availability(cls) -> Availability:
        # torch, not anomalib. That difference is the point of this method existing.
        return module_available("torch", "dl", "EfficientAD (ours)")

    # ---------------------------------------------------------------- guards

    def _check_preprocessing(self, ctx: TrainContext | InferContext) -> None:
        """Refuse a configuration this architecture cannot run, by name.

        The wrapper does not, and the resulting failure is a `RuntimeError` from `conv2d`
        naming a padded input size — true, and useless to anyone who has not read the
        autoencoder. Both dimensions are checked: the encoder collapses each axis
        independently, so a 256x128 input fails on one of them only.
        """
        from anomaly_lab.models.efficientad_nets import INPUT_MULTIPLE, MINIMUM_INPUT

        width, height = ctx.preprocessing.width, ctx.preprocessing.height
        if min(width, height) < MINIMUM_INPUT:
            msg = (
                f"EfficientAD needs at least {MINIMUM_INPUT}x{MINIMUM_INPUT} input and this "
                f"experiment is configured for {width}x{height}. Its autoencoder reduces "
                f"the image {2**5}x and then applies an 8x8 kernel, so anything smaller has "
                "nothing left to encode. Raise the preprocessing size on the experiment."
            )
            raise ValueError(msg)
        if width % INPUT_MULTIPLE or height % INPUT_MULTIPLE:
            msg = (
                f"EfficientAD needs an input that is a multiple of {INPUT_MULTIPLE} in both "
                f"dimensions and this experiment is configured for {width}x{height}. The "
                "autoencoder's upsample ladder is written in those terms, so another size "
                "lands its reconstruction on a different grid from the student's output."
            )
            raise ValueError(msg)

    # ---------------------------------------------------------------- setup

    def _build(self, ctx: TrainContext | InferContext) -> Any:
        """A net whose initial weights are a function of the seed.

        Weight initialisation draws from torch's **global** stream, which our own
        generators do not cover. Left alone, `seed` would mean "the same training order and
        the same augmentations over different initial weights" — which is not one
        experiment, and would put noise under every measurement this milestone exists to
        make. It matters twice over with `pretrained_teacher` off, where the teacher itself
        is random and would otherwise differ between two runs of the same configuration.
        """
        import torch

        from anomaly_lab.models.efficientad_nets import EfficientAdNet

        torch.manual_seed(self.config.seed)
        net = EfficientAdNet(size=self.config.model_size)
        return net.to(ctx.device.value)

    def _teacher_identity(self) -> str:
        """What the teacher was, as one comparable string.

        `pretrained_teacher=False` seeds a random teacher from `seed`, so the seed is part
        of the identity there and only there — two random teachers from different seeds are
        as different as two published ones.
        """
        if not self.config.pretrained_teacher:
            return f"random:{self.config.seed}"
        if self.config.teacher_source == "distilled":
            return f"distilled:{self.config.distilled_teacher}"
        return f"pretrained:{self.config.teacher_source}"

    def _load_distilled_teacher(self, net: Any, ctx: TrainContext, torch_module: Any) -> None:
        """A teacher this workbench produced, checked against what it says it is.

        The manifest is the point of the check. A distilled teacher is a `.pth` that will
        load into any PDN of matching shape, and a *wrong* one loads just as quietly as a
        right one — different width, different output channels, distilled onto a different
        grid or under a different preprocessing. Each of those is recorded, and each is
        refused here by name rather than discovered as a bad number three hours later.
        """
        from anomaly_lab.models.efficientad_nets import load_pdn_weights
        from anomaly_lab.models.teacher_distill import (
            WEIGHTS_FILENAME,
            load_manifest,
            teacher_dir,
        )

        name = self.config.distilled_teacher
        if not name:
            msg = (
                "teacher_source is 'distilled' but distilled_teacher names nothing. "
                "Set it to the name a distill job was given."
            )
            raise RuntimeError(msg)

        manifest = load_manifest(ctx.cache_dir, name)
        expectations = (
            ("model_size", manifest.get("model_size"), self.config.model_size),
            ("out_channels", manifest.get("out_channels"), 384),
            ("normalization", manifest.get("preprocessing", {}).get("normalization"), "imagenet"),
        )
        for field, recorded, wanted in expectations:
            if recorded != wanted:
                msg = (
                    f"the distilled teacher {name!r} records {field}={recorded!r} and this "
                    f"experiment needs {wanted!r}. Distil one that matches, or change the "
                    "experiment."
                )
                raise RuntimeError(msg)

        path = teacher_dir(ctx.cache_dir, name) / WEIGHTS_FILENAME
        load_pdn_weights(
            net.teacher, torch_module.load(path, map_location="cpu", weights_only=True)
        )
        net.teacher.to(ctx.device.value)
        ctx.log(
            f"distilled teacher {name!r} loaded from {path}: "
            f"{manifest.get('source')} over {manifest.get('corpus_images')} "
            f"{manifest.get('corpus')} images for {manifest.get('steps')} steps"
        )

    def _load_teacher(self, net: Any, ctx: TrainContext) -> None:
        """Put the published weights into the teacher, or say that we did not."""
        import torch

        if not self.config.pretrained_teacher:
            ctx.log(
                "pretrained_teacher is off, so the teacher is a seeded random feature "
                "extractor. Distillation still works against it — the student still fails "
                "on what it has not seen — but the published numbers assume the real one.",
                level="warning",
            )
            return

        source = self.config.teacher_source
        if source == "distilled":
            self._load_distilled_teacher(net, ctx, torch)
            return

        from anomaly_lab.models.efficientad_assets import teacher_weights
        from anomaly_lab.models.efficientad_nets import load_pdn_weights

        path = teacher_weights(
            ctx.cache_dir,
            self.config.model_size,
            source=source,
            allow_downloads=self.config.allow_downloads,
            reporter=ctx.reporter,
        )
        # Not `load_state_dict` directly: the two sources key their files differently, and
        # which one this is has to be readable at the call site rather than buried.
        load_pdn_weights(net.teacher, torch.load(path, map_location="cpu", weights_only=True))
        net.teacher.to(ctx.device.value)
        ctx.log(f"pretrained teacher loaded from {path} (source '{source}')")

    def _penalty_stream(self, ctx: TrainContext) -> Any:
        from anomaly_lab.models.efficientad_assets import PenaltyStream, penalty_images

        if not self.config.use_penalty:
            ctx.log("use_penalty is off, so no penalty set is needed and none is fetched")
            return None
        files = penalty_images(
            ctx.cache_dir, allow_downloads=self.config.allow_downloads, reporter=ctx.reporter
        )
        ctx.log(f"{len(files)} penalty images available")
        return PenaltyStream(
            files, (ctx.preprocessing.width, ctx.preprocessing.height), self._generator
        )

    def _seed(self, torch_module: Any) -> None:
        self._generator = np.random.default_rng(self.config.seed)
        self._torch_generator = torch_module.Generator().manual_seed(self.config.seed)

    def _image(self, record: ImageRecord, ctx: Any, torch_module: Any) -> Any:
        array = to_chw(load_array(record.path, ctx.preprocessing))[np.newaxis]
        return torch_module.from_numpy(array).to(ctx.device.value)

    # ---------------------------------------------------------------- training

    def fit(self, train: Sequence[ImageRecord], ctx: TrainContext) -> None:
        import torch

        self._check_preprocessing(ctx)
        self._seed(torch)
        net = self._build(ctx)
        self._net = net
        self._completed = 0
        self._fitted_size = (ctx.preprocessing.width, ctx.preprocessing.height)

        self._load_teacher(net, ctx)
        penalty = self._penalty_stream(ctx)
        ctx.raise_if_cancelled()

        fitting, calibration, origin = self._partition(train, ctx)
        self._fit_teacher_statistics(net, fitting, ctx, torch)
        ctx.raise_if_cancelled()

        self._emit_architecture(ctx, net, torch)
        self._emit_teacher_view(ctx, net, fitting, torch)

        total = self.config.max_steps
        optimizer = self._build_optimizer(net, torch)
        scheduler = self._build_scheduler(optimizer, total, torch)

        ctx.log(f"training for {total} steps at batch size 1")
        self._run_steps(net, optimizer, scheduler, penalty, fitting, ctx, steps=total, total=total)

        self._fit_quantiles(net, calibration, ctx, torch, origin=origin)
        net.eval()
        self._capture_resume_state(optimizer, scheduler, penalty, torch)

    def completed_steps(self) -> int:
        """Steps trained so far, across every run (ADR-0025)."""
        return self._completed

    def fit_more(
        self,
        train: Sequence[ImageRecord],
        ctx: TrainContext,
        *,
        additional_steps: int,
    ) -> None:
        """Continue a loaded model for `additional_steps` further steps."""
        import torch

        if self._net is None:
            msg = "efficientad_custom cannot continue: no model is loaded"
            raise RuntimeError(msg)
        self._check_preprocessing(ctx)
        if self._fitted_size is not None and self._fitted_size != (
            ctx.preprocessing.width,
            ctx.preprocessing.height,
        ):
            # The quantiles and the teacher statistics were fitted at one resolution and do
            # not transfer to another. Continuing anyway would produce maps that are wrong
            # in a way nothing downstream can see.
            width, height = self._fitted_size
            msg = (
                f"this checkpoint was trained at {width}x{height} and the experiment is now "
                f"configured for {ctx.preprocessing.width}x{ctx.preprocessing.height}. Its "
                "normalization does not transfer between input sizes; train from scratch."
            )
            raise RuntimeError(msg)

        # A continuation reloads the teacher from *configuration*, and does not refit the
        # teacher channel statistics — those were fitted in `fit` against whichever teacher
        # was loaded then. Continuing under a different one therefore gives a student that
        # spent every previous step imitating another network, new teacher weights, and the
        # old teacher's normalization. It would train, the loss would look plausible, and
        # every number afterwards would be wrong. Distillation is *against* a teacher; the
        # teacher is not a detail of the run, it is what the run means.
        wanted = self._teacher_identity()
        if self._teacher is not None and self._teacher != wanted:
            msg = (
                f"this checkpoint was distilled against teacher '{self._teacher}' and the "
                f"experiment now asks for '{wanted}'. A student cannot change teacher "
                "mid-training: its weights, and the teacher statistics its maps are "
                "normalized by, both belong to the old one. Train from scratch."
            )
            raise RuntimeError(msg)
        if self._teacher is None:
            ctx.log(
                f"this checkpoint predates teacher recording, so continuing against "
                f"'{wanted}' is unverified — it is only safe because an experiment's "
                "configuration is frozen at creation",
                level="warning",
            )

        self._seed(torch)
        self._restore_random_state(torch, ctx)
        net = self._net.to(ctx.device.value)
        self._load_teacher(net, ctx)
        penalty = self._penalty_stream(ctx)
        if penalty is not None and self._penalty_state is not None:
            penalty.restore(self._penalty_state, ctx.reporter)
        ctx.raise_if_cancelled()

        fitting, calibration, origin = self._partition(train, ctx)

        done = self._completed
        total = done + additional_steps
        optimizer = self._build_optimizer(net, torch)
        if self._optimizer_state is not None:
            optimizer.load_state_dict(self._optimizer_state)
        # Recomputed against the **new** total, which is what makes 4000 + 4000 close to
        # what one 8000-step run would have done — and it means the learning rate returns
        # to base at the resume point, since the drop moves with it. That is only true
        # because `_build_scheduler` sets the rate rather than letting the restored
        # optimizer's decayed one stand; see its docstring. Printed, not hidden.
        scheduler = self._build_scheduler(optimizer, total, torch, last_epoch=done - 1)

        ctx.log(
            f"continuing from step {done} for {additional_steps} more, to {total}; "
            f"the learning rate resumes at {optimizer.param_groups[0]['lr']:.2e} and drops "
            f"tenfold at step {int(0.95 * total)}"
        )
        self._run_steps(
            net, optimizer, scheduler, penalty, fitting, ctx, steps=additional_steps, total=total
        )
        self._fit_quantiles(net, calibration, ctx, torch, origin=origin)
        net.eval()
        self._capture_resume_state(optimizer, scheduler, penalty, torch)

    def _partition(
        self, train: Sequence[ImageRecord], ctx: TrainContext
    ) -> tuple[Sequence[ImageRecord], Sequence[ImageRecord], str]:
        """Split the normals into what the student learns from and what calibrates it.

        EfficientAD calibrates its score normalization on **held-out** normals. VisA's
        official one-class protocol has no `val` subset, so there are none, and the
        reference's only option is the training normals themselves.

        That is a real weakening, and it is worth being precise about *what* it weakens,
        because the obvious reading is wrong. The normalization looks like a display
        convenience a threshold-free metric would be immune to. Writing the score out:

            s = c + max_p [ w_st * map_st[p] + w_ae * map_stae[p] ]

        with ``w_st = 0.05 / (qb_st - qa_st)`` and ``w_ae = 0.05 / (qb_ae - qa_ae)``. A
        shared scale and the offset are monotone and cannot move a ranking — but the two
        weights come from *different* quantile pairs, so what the fit really decides is
        their **ratio**, the relative weight of the two branches before the max. Change
        that and images reorder, which is precisely what ROC-AUC measures.

        `calibration_holdout` is the response: take the calibration set out of training
        instead. It costs a few training images and it is off by default, because it is a
        hypothesis until it has been measured on the same split as everything else.
        """
        if ctx.val:
            return train, ctx.val, f"{len(ctx.val)} held-out normals"

        holdout = self.config.calibration_holdout
        wanted = int(len(train) * holdout)
        if holdout > 0 and wanted >= 1 and wanted < len(train):
            # Evenly spaced, never the first N: datasets arrive in acquisition order often
            # enough that a prefix is one batch, one lighting condition, one shift.
            indices = set(evenly_spaced(len(train), wanted))
            calibration = [record for index, record in enumerate(train) if index in indices]
            fitting = [record for index, record in enumerate(train) if index not in indices]
            ctx.log(
                f"this split has no val subset, so {len(calibration)} of {len(train)} "
                f"training normals were held out to fit the score normalization "
                f"(calibration_holdout={holdout}); the student trains on {len(fitting)}"
            )
            return fitting, calibration, f"{len(calibration)} normals held out of training"

        ctx.log(
            "this split has no val subset, so the score-normalization quantiles are fitted "
            "on the training normals themselves. The normalization is optimistic as a "
            "result, and because the two branch weights come from different quantile pairs "
            "their ratio — and so the ranking — is affected. Set calibration_holdout to "
            "carve a held-out slice out of training instead.",
            level="warning",
        )
        return train, train, "the training normals themselves"

    def _build_optimizer(self, net: Any, torch_module: Any) -> Any:
        return torch_module.optim.Adam(
            list(net.student.parameters()) + list(net.ae.parameters()),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

    def _build_scheduler(
        self, optimizer: Any, total: int, torch_module: Any, *, last_epoch: int = -1
    ) -> Any:
        """The paper's schedule: one tenfold drop near the end of training.

        Resuming makes this harder than it looks, and getting it wrong is invisible.
        `StepLR.get_lr` is *multiplicative on the group's current learning rate*, and
        `Adam.load_state_dict` restores the rate the previous leg ended on — which, since
        every leg anneals over its own last 5%, is the decayed one. Left to itself each
        continuation would start where the last drop left it and then drop again: measured
        at 1e-5 instead of 1e-4 on the first resume, reaching 1e-9 by the fifth. So the rate
        is *computed* from the schedule's closed form at the resume point rather than
        inherited from the restored optimizer.

        `last_epoch` other than -1 also makes PyTorch read `initial_lr` off every param
        group, which `Adam` does not set — so it is seeded here rather than raising a
        `KeyError` a long way from its cause.
        """
        step_size = max(1, int(0.95 * total))
        if last_epoch >= 0:
            for group in optimizer.param_groups:
                group["initial_lr"] = self.config.learning_rate
                group["lr"] = self.config.learning_rate * _LR_GAMMA ** (last_epoch // step_size)
        return torch_module.optim.lr_scheduler.StepLR(
            optimizer, step_size=step_size, gamma=_LR_GAMMA, last_epoch=last_epoch
        )

    def _run_steps(
        self,
        net: Any,
        optimizer: Any,
        scheduler: Any,
        penalty: Any,
        train: Sequence[ImageRecord],
        ctx: TrainContext,
        *,
        steps: int,
        total: int,
    ) -> None:
        """The training loop, shared by `fit` and `fit_more`.

        **Steps reported to `ctx.metric` are absolute across the experiment's training**, so
        a continued run's curve is a continuation of the first rather than a second curve
        starting at zero — no stitching in the chart and no extra log reads (**ADR-0020**).
        """
        import torch

        from anomaly_lab.models.efficientad_nets import augment

        net.train()
        started = time.perf_counter()
        blank = None

        for index in range(steps):
            # Checked every step rather than every epoch, which is the practical benefit of
            # owning the loop: cancelling a training run stops it in about 100 ms.
            ctx.raise_if_cancelled()
            step = self._completed

            record = train[int(self._generator.integers(len(train)))]
            image = self._image(record, ctx, torch)
            if penalty is None:
                if blank is None:
                    blank = torch.zeros_like(image)
                penalty_batch = blank
            else:
                penalty_batch = torch.from_numpy(penalty.next()[np.newaxis]).to(ctx.device.value)

            augmented = augment(image, self._torch_generator)

            optimizer.zero_grad()
            _, distance = net.student_teacher_distance(image)
            loss_st, loss_ae, loss_stae = net.compute_losses(
                penalty_batch,
                augmented,
                distance,
                hard_quantile=self.config.hard_quantile,
                use_penalty=self.config.use_penalty,
            )
            loss = loss_st + loss_ae + loss_stae
            loss.backward()
            optimizer.step()
            scheduler.step()
            self._completed = step + 1

            if index % _LOSS_LOG_EVERY == 0 or index == steps - 1:
                # Scalar series reuse the `metric` event the job protocol already has
                # (ADR-0009). No new channel, no protocol change.
                ctx.metric("loss_st", float(loss_st.item()), step=step)
                ctx.metric("loss_ae", float(loss_ae.item()), step=step)
                ctx.metric("loss_stae", float(loss_stae.item()), step=step)
                ctx.metric("loss_total", float(loss.item()), step=step)
                ctx.metric("learning_rate", float(scheduler.get_last_lr()[0]), step=step)
                ctx.progress(
                    0.1 + 0.8 * (index + 1) / steps,
                    f"step {self._completed}/{total}, loss {loss.item():.4f}",
                )

        elapsed = time.perf_counter() - started
        ctx.log(
            f"trained {steps} steps in {elapsed:.0f}s "
            f"({elapsed / max(steps, 1) * 1000:.0f} ms/step on {ctx.device.value}); "
            f"{self._completed} completed in total"
        )

    # ---------------------------------------------------------------- calibration

    def _fit_teacher_statistics(
        self, net: Any, records: Sequence[ImageRecord], ctx: TrainContext, torch_module: Any
    ) -> None:
        """Channel mean and deviation of the teacher's output over the training normals.

        Accumulated in **float64 on the CPU** over per-batch float32 reductions. The
        reference accumulates sums and sums-of-squares in float32 across the whole set and
        then takes `sqrt(E[x^2] - E[x]^2)`, which can go slightly negative from
        cancellation and produce a `NaN` deviation that silently poisons every score in the
        run. Moving 384 numbers per batch to the CPU costs nothing and removes that.

        MPS has no float64, which is why the conversion happens after the reduction rather
        than before it.
        """
        ctx.progress(0.05, "fitting teacher channel statistics")
        chosen = list(records)
        total: Any = None
        total_sq: Any = None
        count = 0
        batch_size = self.config.stats_batch_size

        with torch_module.no_grad():
            for start in range(0, len(chosen), batch_size):
                ctx.raise_if_cancelled()
                chunk = chosen[start : start + batch_size]
                stacked = np.stack(
                    [to_chw(load_array(record.path, ctx.preprocessing)) for record in chunk]
                )
                batch = torch_module.from_numpy(stacked).to(ctx.device.value)
                features = net.teacher(batch)
                sums = features.sum(dim=(0, 2, 3)).detach().cpu().to(torch_module.float64)
                squares = (features**2).sum(dim=(0, 2, 3)).detach().cpu().to(torch_module.float64)
                total = sums if total is None else total + sums
                total_sq = squares if total_sq is None else total_sq + squares
                count += features.shape[0] * features.shape[2] * features.shape[3]

        if total is None or count == 0:
            msg = "efficientad_custom cannot fit teacher statistics on an empty training set"
            raise RuntimeError(msg)

        mean = total / count
        variance = (total_sq / count) - mean**2
        std = variance.clamp_min(0.0).sqrt()
        floored = net.set_teacher_statistics(
            mean.to(torch_module.float32)[None, :, None, None],
            std.to(torch_module.float32)[None, :, None, None],
        )
        ctx.log(f"teacher statistics fitted over {len(chosen)} images ({count} positions)")
        if floored:
            ctx.log(
                f"{floored} teacher channel(s) had no variation across the training set and "
                "their deviation was floored; those channels contribute nothing to the score",
                level="warning",
            )

    def _fit_quantiles(
        self,
        net: Any,
        calibration: Sequence[ImageRecord],
        ctx: TrainContext,
        torch_module: Any,
        *,
        origin: str,
    ) -> None:
        """Fit the two branch quantile pairs from a bounded pixel reservoir."""
        ctx.progress(0.92, "fitting score-normalization quantiles")
        chosen = evenly_spaced(len(calibration), self.config.quantile_images)
        records = [calibration[index] for index in chosen]
        if len(records) < len(calibration):
            ctx.log(
                f"fitting quantiles on {len(records)} of {len(calibration)} images from "
                f"{origin}, sampled evenly (quantile_images={self.config.quantile_images})"
            )
        else:
            ctx.log(f"fitting score-normalization quantiles on {origin}")

        per_image = max(1, self.config.quantile_pixel_budget // max(len(records), 1))
        # Indices are drawn on the CPU and moved to the device, so the sample is the same
        # on MPS and on CPU and a seed means the same thing on either.
        picker = torch_module.Generator().manual_seed(self.config.seed + 1)
        samples: dict[str, list[Any]] = {"st": [], "ae": []}

        net.eval()
        with torch_module.no_grad():
            for record in records:
                ctx.raise_if_cancelled()
                image = self._image(record, ctx, torch_module)
                branch_maps = net.maps(image, normalize=False)
                for name, branch in zip(("st", "ae"), branch_maps, strict=True):
                    flat = branch.flatten()
                    take = min(per_image, flat.numel())
                    index = torch_module.randint(
                        0, flat.numel(), (take,), generator=picker, dtype=torch_module.long
                    )
                    samples[name].append(flat[index.to(flat.device)].cpu())

        quantiles = {}
        for name, collected in samples.items():
            pooled = torch_module.cat(collected)
            quantiles[f"qa_{name}"] = torch_module.quantile(pooled, _QUANTILE_LOW)
            quantiles[f"qb_{name}"] = torch_module.quantile(pooled, _QUANTILE_HIGH)
        degenerate = net.set_quantiles(quantiles)
        if degenerate:
            ctx.log(
                f"the {', '.join(degenerate)} branch produced a near-constant map over the "
                "calibration images, so its normalization spread was floored. Its scores "
                "carry almost no information about this dataset.",
                level="warning",
            )
        self._emit_normalization(ctx, quantiles, origin=origin, images=len(records))

    # ---------------------------------------------------------------- diagnostics

    def _emit_normalization(
        self, ctx: TrainContext, quantiles: dict[str, Any], *, origin: str, images: int
    ) -> None:
        """Record what the calibration decided, as a table rather than a chart.

        Four numbers, not a series, so the training charts are the wrong shape for them —
        but they are the most consequential four the run produces after the weights, and
        the `fitted on` row is the difference `calibration_holdout` exists to make.
        """
        rows = [[name, f"{float(value):.6g}"] for name, value in sorted(quantiles.items())]
        ctx.emit_diagnostic(
            "score_normalization",
            "Score normalization",
            DiagnosticKind.TABLE,
            {
                "columns": ["quantity", "value"],
                "rows": [*rows, ["fitted on", origin], ["calibration images", str(images)]],
            },
            description=(
                "The map-normalization quantiles, and what they were fitted on. The two "
                "branch weights derive from different quantile pairs, so their ratio — "
                "and therefore the ranking — depends on this fit."
            ),
        )

    def _emit_architecture(self, ctx: TrainContext, net: Any, torch_module: Any) -> None:
        """Capture the real architecture from a dry forward pass, not from a diagram.

        Every module is recorded through the shared helper in `models/introspect.py`, so
        this method contributes only *which roots to walk* and *how they are wired*
        (**ADR-0024**). That this needed no change to the helper is the evidence for M6's
        third exit criterion.
        """
        if not ctx.diagnostics.enabled:
            return

        probe = torch_module.zeros(
            1, 3, ctx.preprocessing.height, ctx.preprocessing.width, device=ctx.device.value
        )
        records: list[ModuleRecord] = []
        for name in ("teacher", "student", "ae"):
            records.extend(collect(getattr(net, name), probe, prefix=f"{name}."))

        payload = build_tree(records)
        # The wiring between branches is the one thing hooks cannot see: it lives in the
        # losses, not in any module's forward. Stated by the code that knows it.
        payload["edges"] = [
            {"source": "teacher", "target": "student", "label": "distillation"},
            {"source": "student", "target": "ae", "label": "reconstruction"},
        ]
        payload["total_parameters"] = int(sum(p.numel() for p in net.parameters()))

        ctx.emit_diagnostic(
            "architecture",
            "Model architecture",
            DiagnosticKind.GRAPH,
            payload,
            description=(
                "Every module's real input and output shapes, read from a dry forward "
                "pass. Functional operations are not modules and so do not appear."
            ),
        )

    def _emit_teacher_view(
        self, ctx: TrainContext, net: Any, train: Sequence[ImageRecord], torch_module: Any
    ) -> None:
        """What the teacher sees on one training image."""
        if not ctx.diagnostics.enabled or not train:
            return

        image = self._image(train[0], ctx, torch_module)
        with torch_module.no_grad():
            features = net.teacher(image)[0].detach().cpu().numpy()

        ctx.emit_diagnostic(
            "teacher_features_pca",
            "Teacher features (PCA composite)",
            DiagnosticKind.IMAGE,
            _pca_to_rgb(features),
            description="Three leading principal components of the teacher's 384 channels.",
        )
        ctx.emit_diagnostic(
            "teacher_features_grid",
            "Teacher features (first 16 channels)",
            DiagnosticKind.GRID,
            features[:16],
            description="Individual teacher feature channels, as small multiples.",
        )
        ctx.emit_diagnostic(
            "teacher_magnitude",
            "Teacher feature magnitude",
            DiagnosticKind.MAP,
            np.linalg.norm(features, axis=0).astype(np.float32),
            description="Where the teacher responds most strongly on a normal image.",
        )

    # ---------------------------------------------------------------- inference

    def predict(self, images: Sequence[ImageRecord], ctx: InferContext) -> list[Prediction]:
        import torch

        from anomaly_lab.models.efficientad_nets import reduce_score

        if self._net is None:
            msg = "efficientad_custom was asked to predict before it was fitted or loaded"
            raise RuntimeError(msg)
        self._check_preprocessing(ctx)

        # `load` restores onto the CPU so a checkpoint can be read on a machine without the
        # device it was trained on; the move happens here.
        net = self._net.to(ctx.device.value)
        self._net = net
        net.eval()
        weight = self.config.student_teacher_weight
        predictions: list[Prediction] = []

        for index, record in enumerate(images):
            ctx.raise_if_cancelled()
            started = time.perf_counter()
            image = self._image(record, ctx, torch)

            with torch.no_grad():
                map_st, map_stae = net.maps(image, normalize=True)
                combined = weight * map_st + (1.0 - weight) * map_stae
                score = reduce_score(
                    combined,
                    reduction=self.config.score_reduction,
                    top_k=self.config.score_top_k,
                )

            map_path = ctx.write_map(record.image_id, combined[0].cpu().numpy())
            elapsed_ms = (time.perf_counter() - started) * 1000.0

            # The two branches apart, which is the diagnostic that makes EfficientAD
            # legible: a student-teacher hit and an autoencoder hit mean different things
            # about what went wrong, and the stored map averages them together.
            ctx.emit_diagnostic(
                "map_student_teacher",
                "Student-teacher error",
                DiagnosticKind.MAP,
                map_st[0, 0].cpu().numpy().astype(np.float32),
                image_id=record.image_id,
                description="Where the student failed to reproduce the teacher's features.",
            )
            ctx.emit_diagnostic(
                "map_autoencoder",
                "Autoencoder-student error",
                DiagnosticKind.MAP,
                map_stae[0, 0].cpu().numpy().astype(np.float32),
                image_id=record.image_id,
                description="Where the autoencoder and the student disagree — the global branch.",
            )

            predictions.append(
                Prediction(
                    image_id=record.image_id,
                    score=float(score[0].cpu()),
                    anomaly_map=map_path,
                    inference_ms=elapsed_ms,
                )
            )
            ctx.progress((index + 1) / len(images), f"scored {index + 1}/{len(images)}")

        return predictions

    # ---------------------------------------------------------------- persistence

    def _capture_resume_state(
        self, optimizer: Any, scheduler: Any, penalty: Any, torch_module: Any
    ) -> None:
        """Leave the instance in exactly the state a reloaded one would be in."""
        self._optimizer_state = optimizer.state_dict()
        self._scheduler_state = scheduler.state_dict()
        self._generator_state = self._generator.bit_generator.state
        self._torch_rng_state = self._torch_generator.get_state()
        self._penalty_state = None if penalty is None else penalty.state()

    def _restore_random_state(self, torch_module: Any, ctx: TrainContext) -> None:
        """Put both streams back where the previous run left them."""
        if self._generator_state is not None:
            generator = np.random.default_rng()
            generator.bit_generator.state = self._generator_state
            self._generator = generator
        if self._torch_rng_state is not None:
            self._torch_generator.set_state(self._torch_rng_state)
        if self._generator_state is None:
            ctx.log(
                "this checkpoint carries no image-order state, so the training image "
                "sequence restarts from the seed",
                level="warning",
            )

    def save(self, artifact_dir: Path) -> None:
        """Write everything a *continuation* needs, not only what inference needs.

        There is deliberately no option to skip the training state: an option would make
        "can I continue this run?" depend on a flag chosen before anyone knew the answer.
        """
        import torch

        if self._net is None:
            msg = "efficientad_custom has nothing to save; it was never fitted"
            raise RuntimeError(msg)

        payload: dict[str, Any] = {
            "format": CHECKPOINT_FORMAT,
            "model_size": self.config.model_size,
            "state_dict": self._net.state_dict(),
            "completed_steps": self._completed,
            "fitted_size": list(self._fitted_size) if self._fitted_size else None,
            # Which teacher this student was distilled against. Recorded because a
            # continuation reloads the teacher from configuration and does *not* refit the
            # teacher statistics — see `fit_more`'s refusal.
            "teacher": self._teacher_identity(),
            "optimizer": self._optimizer_state,
            "scheduler": self._scheduler_state,
            "generator_state": self._generator_state,
            "torch_generator_state": self._torch_rng_state,
            "penalty_state": self._penalty_state,
        }
        torch.save(payload, artifact_dir / STATE_FILENAME)

    def load(self, artifact_dir: Path) -> None:
        """Restore a fitted model and everything a continuation would need.

        An unrecognised format is refused **by name** rather than tolerated: a checkpoint
        written by a future version may hold the same keys with different meanings, and a
        silently mis-read one produces plausible, wrong scores.
        """
        import torch

        from anomaly_lab.models.efficientad_nets import EfficientAdNet

        stored = torch.load(artifact_dir / STATE_FILENAME, map_location="cpu", weights_only=False)
        found = int(stored.get("format", 0))
        if found != CHECKPOINT_FORMAT:
            msg = (
                f"this efficientad_custom checkpoint declares format {found}, and this "
                f"build reads format {CHECKPOINT_FORMAT}. Train the experiment again."
            )
            raise RuntimeError(msg)

        net = EfficientAdNet(size=stored.get("model_size", self.config.model_size))
        net.load_state_dict(stored["state_dict"])
        self._net = net

        self._completed = int(stored.get("completed_steps", 0))
        size = stored.get("fitted_size")
        self._fitted_size = (int(size[0]), int(size[1])) if size else None
        recorded = stored.get("teacher")
        self._teacher = str(recorded) if recorded else None
        self._optimizer_state = stored.get("optimizer")
        self._scheduler_state = stored.get("scheduler")
        self._generator_state = stored.get("generator_state")
        self._torch_rng_state = stored.get("torch_generator_state")
        self._penalty_state = stored.get("penalty_state")


def _pca_to_rgb(features: np.ndarray) -> np.ndarray:
    """Reduce a `(C, H, W)` feature map to `(H, W, 3)` in [0, 1] by PCA.

    384 teacher channels cannot be looked at directly. The three leading components carry
    what varies spatially, which is what a reader wants to see; the result is a false-colour
    image and is labelled as one in the UI.

    Deliberately duplicated from `efficientad_anomalib` rather than shared. It is twelve
    lines, and hoisting it into a common module to serve exactly two plugins is the
    premature generality the plugin boundary exists to avoid — a third caller would change
    that, and this comment is here so the decision is visible rather than inferred.
    """
    channels, height, width = features.shape
    flat = features.reshape(channels, height * width).T.astype(np.float64)
    centred = flat - flat.mean(axis=0, keepdims=True)
    # SVD rather than an explicit covariance eigendecomposition: 384x384 is small, but the
    # covariance route squares the condition number for no benefit here.
    _, _, components = np.linalg.svd(centred, full_matrices=False)
    projected = (centred @ components[:3].T).reshape(height, width, 3)
    low = projected.min(axis=(0, 1), keepdims=True)
    high = projected.max(axis=(0, 1), keepdims=True)
    return ((projected - low) / np.maximum(high - low, 1e-9)).astype(np.float32)
