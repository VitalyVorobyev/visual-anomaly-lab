"""`efficientad_anomalib` — EfficientAD (arXiv:2303.14535) through Intel's anomalib.

**What comes from anomalib:** the PDN teacher and student, the autoencoder branch, the
three loss terms, the anomaly-map computation, the quantile normalization, the pretrained
teacher weights, and the exact routines that fit the teacher channel statistics and the
map-normalization quantiles. Everything that decides a number is anomalib's.

**What is ours:** the training loop, the data feeding, and the reporting.

That split is a deviation from the plan, which said "a Lightning callback", and the reason
is worth recording. `EfficientAd.on_train_start` reads `self.trainer.datamodule` — for the
batch size, for the training dataloader, and to compute teacher statistics. Using the
Lightning path therefore means adopting an `AnomalibDataModule`, and with it anomalib's
own preprocessing. That would break the one property that makes a comparison mean
anything: **every method in this workbench sees identical pixels** (`preprocessing.py`).
Driving the torch module directly keeps the preprocessing bridge real, and it is also
what makes cancellation land within one training step instead of one epoch.

The cost is stated plainly: this loop can drift from anomalib's if theirs changes, and a
gap against a published number would then have two candidate explanations rather than one.
That is why the statistics and quantile code is *called* rather than reimplemented — the
parts most likely to silently change an AUROC are the parts we do not own.

**Two network downloads are required, and neither is hidden.** The pretrained teacher
(40 MB) and the ImageNette penalty set (~1.5 GB) are fetched on the first training run,
with a log line before each. `allow_downloads=false` turns them into an error naming the
asset instead of a silent fetch, because a tool that claims to be local-only should not
quietly reach for the network.

They do not land in the same place, and that is anomalib's call rather than ours: the
penalty set goes to `ctx.cache_dir` because the path is a constructor argument, while
`prepare_pretrained_model` resolves the teacher through anomalib's own platform cache
(`~/Library/Caches/anomalib` on macOS). Overriding that would mean reimplementing the
download rather than calling it, which is the trade this module refuses everywhere else —
so the teacher's location is documented instead. Deleting our data directory therefore
leaves a 40 MB teacher behind; deleting anomalib's cache is a separate act.
"""

from __future__ import annotations

import contextlib
import shutil
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

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
from anomaly_lab.models.feature_view import pca_to_rgb
from anomaly_lab.models.introspect import ModuleRecord, build_tree, collect
from anomaly_lab.models.preprocessing import load_array, to_chw
from anomaly_lab.schemas import API_MODEL_CONFIG

if TYPE_CHECKING:  # pragma: no cover - import cost is the whole point of deferring it
    import torch

STATE_FILENAME = "efficientad.pt"

CHECKPOINT_FORMAT = 2
"""Format 1 was weights and `model_size` alone. Format 2 adds what a continuation needs:
optimizer moments, LR-schedule state, the absolute step counter and both RNG streams
(ADR-0025). A format-1 checkpoint still loads and still infers; it cannot be continued."""
PENALTY_SUBDIR = "imagenette"

# The paper trains for 70000 steps. That is roughly two and a half hours on this Mac at
# the measured 123 ms/step, which is not a default anyone will wait through the first
# time. The default here trades a known amount of accuracy for a runnable first result,
# and says so in the field description rather than in a comment nobody reads.
DEFAULT_MAX_STEPS = 4000

_LOSS_LOG_EVERY = 20
_LR_GAMMA = 0.1
"""The single tenfold drop. Named because `_build_scheduler` applies it twice — once
through `StepLR` and once in the closed form that positions a resumed run."""


class EfficientAdConfig(BaseModel):
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
            "Batch size for the teacher-statistics and quantile passes only. Training "
            "itself is fixed at 1, which EfficientAD requires."
        ),
    )
    quantile_images: int = Field(
        default=128,
        ge=8,
        le=4096,
        description=(
            "How many normals fit the score-normalization quantiles. anomalib holds every "
            "map in memory for this, so the whole training set would cost gigabytes; "
            "images are sampled evenly rather than taking the first N."
        ),
    )
    seed: int = Field(default=0, description="Seeds the training sample order.")


@dataclass
class _Batch:
    """The two attributes anomalib's statistics routines read off a batch."""

    image: torch.Tensor
    gt_label: torch.Tensor


class _BatchStream:
    """A re-iterable, lazy stream of batches over image records.

    Re-iterable because anomalib's statistics helpers each walk it once and are called
    more than once; lazy because holding 900 preprocessed images would cost most of a
    gigabyte to avoid re-reading files that are in the page cache anyway.
    """

    def __init__(
        self,
        records: Sequence[ImageRecord],
        preprocessing: Any,
        batch_size: int,
        torch_module: Any,
    ) -> None:
        self._records = records
        self._preprocessing = preprocessing
        self._batch_size = batch_size
        self._torch = torch_module

    def __len__(self) -> int:
        return (len(self._records) + self._batch_size - 1) // self._batch_size

    def __iter__(self) -> Iterator[_Batch]:
        for start in range(0, len(self._records), self._batch_size):
            chunk = self._records[start : start + self._batch_size]
            stacked = np.stack(
                [to_chw(load_array(record.path, self._preprocessing)) for record in chunk]
            )
            yield _Batch(
                image=self._torch.from_numpy(stacked),
                # Every record handed to these routines is a training or held-out
                # *normal*, so the label is zero by construction. anomalib's quantile
                # pass filters on it; saying so explicitly keeps that filter meaningful.
                gt_label=self._torch.zeros(len(chunk), dtype=self._torch.long),
            )


class EfficientAdAnomalibModel(AnomalyModel):
    """EfficientAD, with anomalib supplying the model and this class supplying the loop."""

    title = "EfficientAD (anomalib)"
    summary = (
        "Student-teacher distillation with an autoencoder branch, from Intel's anomalib. "
        "Trains on normals only and produces well-localized anomaly maps."
    )

    def __init__(self, config: EfficientAdConfig) -> None:
        super().__init__(config)
        self.config = config
        self._module: Any = None
        # Everything below is what makes a run continuable (ADR-0025). Held on the
        # instance rather than reconstructed, because `save` is called by the job handler
        # after `fit` returns and has no other way to reach the optimizer.
        self._completed = 0
        self._generator: Any = None
        self._optimizer_state: Any = None
        self._scheduler_state: Any = None
        self._torch_rng_state: Any = None
        self._generator_state: Any = None

    @classmethod
    def config_model(cls) -> type[BaseModel]:
        return EfficientAdConfig

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
        return module_available("anomalib", "dl", "EfficientAD")

    # ---------------------------------------------------------------- construction

    def _build_module(self, ctx: Any) -> Any:
        """Instantiate anomalib's Lightning module as a *component*, never to fit with.

        It is the tidiest handle on the pretrained-teacher download, the penalty-set
        loader and the two statistics routines, all of which are methods on it. Lightning
        itself is never started.
        """
        from anomalib.models import EfficientAd

        penalty_dir = ctx.cache_dir / PENALTY_SUBDIR
        module = EfficientAd(
            imagenet_dir=penalty_dir,
            model_size=self.config.model_size,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        return module.to(ctx.device.value)

    def _ensure_teacher(self, module: Any, ctx: Any) -> None:
        if not self.config.allow_downloads:
            _refuse_download_if_missing()
        # anomalib's downloader writes its progress bar to stderr, which the job log
        # captures but the progress bar cannot show. Without a message here the UI sits
        # at 0% for minutes on the first run, which is indistinguishable from a hang.
        ctx.progress(0.01, "fetching the pretrained teacher (40 MB, first run only)")
        ctx.log("loading the pretrained EfficientAD teacher (downloads on first run)")
        module.prepare_pretrained_model()
        ctx.log("pretrained teacher loaded")

    def _ensure_penalty_set(self, module: Any, ctx: Any) -> None:
        penalty_dir = ctx.cache_dir / PENALTY_SUBDIR

        # anomalib decides whether to download by asking whether the *directory* exists.
        # A 1.5 GB download interrupted halfway leaves the directory there holding a
        # partial tarball, so the next run skips the download and then fails inside
        # `ImageFolder` on an empty tree — a confusing error a long way from its cause.
        # An incomplete download is cleared here so the retry is simply a retry.
        if penalty_dir.is_dir() and not _penalty_set_is_extracted(penalty_dir):
            ctx.log(
                f"the penalty set at {penalty_dir} is incomplete — an interrupted "
                "download — and is being discarded so it can be fetched again",
                level="warning",
            )
            shutil.rmtree(penalty_dir, ignore_errors=True)

        if not penalty_dir.is_dir():
            if not self.config.allow_downloads:
                msg = (
                    f"EfficientAD needs the ImageNette penalty set at {penalty_dir} and "
                    "allow_downloads is off. Download imagenette2.tgz from "
                    "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz and "
                    "extract it there, or turn allow_downloads back on."
                )
                raise RuntimeError(msg)
            # Several minutes on a normal connection, and it happens exactly once per
            # machine. Saying so beats a progress bar that has not moved.
            ctx.progress(
                0.02,
                "downloading the ImageNette penalty set (~1.5 GB, first run only) — "
                "several minutes, then training starts",
            )
            ctx.log(f"downloading the ImageNette penalty set (~1.5 GB) into {penalty_dir}")
        module.prepare_imagenette_data((ctx.preprocessing.height, ctx.preprocessing.width))
        ctx.log("penalty set ready")

    # ---------------------------------------------------------------- training

    def fit(self, train: Sequence[ImageRecord], ctx: TrainContext) -> None:
        import torch

        torch.manual_seed(self.config.seed)
        module = self._build_module(ctx)
        model = module.model

        self._ensure_teacher(module, ctx)
        self._ensure_penalty_set(module, ctx)
        ctx.raise_if_cancelled()

        ctx.progress(0.05, "fitting teacher channel statistics")
        stats_stream = _BatchStream(train, ctx.preprocessing, self.config.stats_batch_size, torch)
        model.mean_std.update(module.teacher_channel_mean_std(stats_stream))
        ctx.raise_if_cancelled()

        self._emit_architecture(ctx, model, torch)
        self._emit_teacher_view(ctx, model, train, torch)

        total = self.config.max_steps
        optimizer = self._build_optimizer(model, torch)
        scheduler = self._build_scheduler(optimizer, total, torch)

        self._module = module
        self._generator = np.random.default_rng(self.config.seed)
        self._completed = 0

        ctx.log(f"training for {total} steps at batch size 1")
        self._run_steps(module, optimizer, scheduler, train, ctx, steps=total, total=total)

        self._fit_quantiles(module, ctx, train, torch)
        model.eval()
        self._capture_resume_state(optimizer, scheduler, torch)

    # ------------------------------------------------------------------ resume

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
        """Continue a loaded model for `additional_steps` further steps.

        Refuses a format-1 checkpoint **by name** rather than restarting the optimizer.
        Adam's moments are most of what a long run has learned about its own gradients;
        throwing them away and calling the result a continuation would be a fabricated
        number on a screen, which is the thing this codebase refuses everywhere else.
        """
        import torch

        if self._module is None:
            msg = "efficientad_anomalib cannot continue: no model is loaded"
            raise RuntimeError(msg)
        if self._optimizer_state is None or self._scheduler_state is None:
            msg = (
                "this checkpoint was written before optimizer state was saved, so training "
                "cannot be continued exactly from it. Train from scratch once, and that "
                "run can then be continued."
            )
            raise RuntimeError(msg)

        # Rebuilt through `_build_module`, and the loaded weights moved into it.
        #
        # `load` has no `TrainContext` and so cannot know where the penalty set lives; the
        # module it makes is fine for `predict` and cannot train, because
        # `prepare_imagenette_data` resolves against an `imagenet_dir` it was never given.
        # Found by running a continuation end to end, which failed on a missing class
        # folder several layers below the cause.
        module = self._build_module(ctx)
        module.model.load_state_dict(self._module.model.state_dict())
        self._module = module
        model = module.model

        self._ensure_teacher(module, ctx)
        self._ensure_penalty_set(module, ctx)
        ctx.raise_if_cancelled()

        done = self._completed
        total = done + additional_steps

        optimizer = self._build_optimizer(model, torch)
        optimizer.load_state_dict(self._optimizer_state)
        # The schedule is recomputed against the **new** total, which is what makes a
        # 4000 + 4000 continue close to what a single 8000-step run would have done — and
        # it means the learning rate returns to its base value at the resume point, since
        # the drop moves from 3800 to 7600. Surprising enough that the caller prints it.
        # It is also only true because `_build_scheduler` sets the rate rather than letting
        # the restored optimizer's decayed one stand; see its docstring.
        scheduler = self._build_scheduler(optimizer, total, torch, last_epoch=done - 1)

        self._restore_rng(torch, ctx)
        ctx.log(
            f"continuing from step {done} for {additional_steps} more, to {total}; "
            f"the learning-rate drop moves to step {int(0.95 * total)}"
        )
        ctx.log(
            "the optimizer, the schedule, the step counter and the training image order "
            "resume exactly; the ImageNette penalty batch order restarts",
            level="warning",
        )

        self._run_steps(
            module, optimizer, scheduler, train, ctx, steps=additional_steps, total=total
        )

        self._fit_quantiles(module, ctx, train, torch)
        model.eval()
        self._capture_resume_state(optimizer, scheduler, torch)

    def _capture_resume_state(self, optimizer: Any, scheduler: Any, torch_module: Any) -> None:
        """Leave the instance in exactly the state a reloaded one would be in.

        Without the two RNG snapshots, a model continued in the same process would take a
        different path from the identical model continued after a reload — the in-memory
        one having no state to restore. The handler always reloads, so it is unreachable
        today; recording it anyway is what keeps "the checkpoint loses nothing" a
        statement about the checkpoint rather than about the call order.
        """
        self._optimizer_state = optimizer.state_dict()
        self._scheduler_state = scheduler.state_dict()
        self._torch_rng_state = torch_module.get_rng_state()
        self._generator_state = (
            None if self._generator is None else self._generator.bit_generator.state
        )

    def _build_optimizer(self, model: Any, torch_module: Any) -> Any:
        return torch_module.optim.Adam(
            list(model.student.parameters()) + list(model.ae.parameters()),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

    def _build_scheduler(
        self, optimizer: Any, total: int, torch_module: Any, *, last_epoch: int = -1
    ) -> Any:
        """anomalib's own schedule: one tenfold drop near the end of training.

        Resuming makes this harder than it looks, and getting it wrong is invisible.
        `StepLR.get_lr` is *multiplicative on the group's current learning rate*, and
        `Adam.load_state_dict` restores the rate the previous leg ended on — which, since
        every leg anneals over its own last 5%, is the decayed one. Left to itself each
        continuation would start where the last drop left it and then drop again: measured
        at 1e-5 instead of 1e-4 on the first resume, reaching 1e-9 by the fifth. So the rate
        is *computed* from the schedule's closed form at the resume point rather than
        inherited from the restored optimizer.

        `last_epoch` other than -1 also makes PyTorch read `initial_lr` off every param
        group, which `Adam` does not set — so it is seeded here. Without this, resuming
        raises a `KeyError` a long way from its cause.
        """
        step_size = max(1, int(0.95 * total))
        if last_epoch >= 0:
            for group in optimizer.param_groups:
                group["initial_lr"] = self.config.learning_rate
                group["lr"] = self.config.learning_rate * _LR_GAMMA ** (last_epoch // step_size)
        return torch_module.optim.lr_scheduler.StepLR(
            optimizer, step_size=step_size, gamma=_LR_GAMMA, last_epoch=last_epoch
        )

    def _restore_rng(self, torch_module: Any, ctx: TrainContext) -> None:
        """Put both random streams back where the previous run left them.

        The numpy generator picks training images and the torch generator drives anything
        stochastic in the modules. Restoring them is what makes "resume" mean the same
        sequence of work rather than merely the same weights.
        """
        if self._torch_rng_state is not None:
            torch_module.set_rng_state(self._torch_rng_state)
        if self._generator_state is not None:
            generator = np.random.default_rng()
            generator.bit_generator.state = self._generator_state
            self._generator = generator
        else:  # pragma: no cover - only for a checkpoint written without it
            ctx.log(
                "this checkpoint carries no image-order state, so the training image "
                "sequence restarts from the seed",
                level="warning",
            )
            self._generator = np.random.default_rng(self.config.seed)

    def _run_steps(
        self,
        module: Any,
        optimizer: Any,
        scheduler: Any,
        train: Sequence[ImageRecord],
        ctx: TrainContext,
        *,
        steps: int,
        total: int,
    ) -> None:
        """The training loop, shared by `fit` and `fit_more`.

        **Steps reported to `ctx.metric` are absolute across the experiment's training**,
        so a continued run's curve is a continuation of the first rather than a second
        curve starting at zero — no stitching in the chart, and no extra log reads, which
        is the cliff ADR-0020 named.
        """
        import torch

        model = module.model
        model.train()
        generator = self._generator
        started = time.perf_counter()

        for index in range(steps):
            # Checked every step rather than every epoch, which is the practical benefit
            # of owning the loop: cancelling a training run stops it in ~100 ms.
            ctx.raise_if_cancelled()
            step = self._completed

            record = train[int(generator.integers(len(train)))]
            image = torch.from_numpy(
                to_chw(load_array(record.path, ctx.preprocessing))[np.newaxis]
            ).to(ctx.device.value)
            penalty = _next_penalty_batch(module).to(ctx.device.value)

            optimizer.zero_grad()
            loss_st, loss_ae, loss_stae = model(batch=image, batch_imagenet=penalty)
            loss = loss_st + loss_ae + loss_stae
            loss.backward()
            optimizer.step()
            scheduler.step()
            self._completed = step + 1

            if index % _LOSS_LOG_EVERY == 0 or index == steps - 1:
                # Scalar series reuse the `metric` event that already exists in the job
                # protocol (ADR-0009). No new channel, no protocol change — which is the
                # test the diagnostics contract was designed to pass.
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

    def _fit_quantiles(
        self,
        module: Any,
        ctx: TrainContext,
        train: Sequence[ImageRecord],
        torch_module: Any,
    ) -> None:
        """Fit the map-normalization quantiles, saying which images they came from.

        EfficientAD calibrates on **held-out** normals. VisA's official one-class protocol
        has no `val` subset at all, so there are none, and the only alternative is the
        training normals themselves. That is a real weakening — the quantiles are then
        fitted on data the student has already memorized — and it is logged as a warning
        rather than silently substituted.

        It is worth being precise about *what* it weakens, because the obvious reading is
        wrong. The normalization looks like a display convenience that a threshold-free
        metric would be immune to, and it is not. Writing the score out:

            s = c + max_p [ w_st * map_st[p] + w_ae * map_stae[p] ]

        with ``w_st = 0.05 / (qb_st - qa_st)`` and ``w_ae = 0.05 / (qb_ae - qa_ae)``. A
        shared scale and the offset ``c`` are monotone and cannot move a ranking — but the
        two weights come from *different* quantile pairs, so what the fit really decides is
        their **ratio**, the relative weight of the two branches before the max. Change
        that and images reorder, which is precisely what ROC-AUC measures. Quantiles fitted
        on memorized data understate the student-teacher spread, inflating ``w_st``, and
        the combined map tilts toward the branch with the least honest calibration.
        """
        source = ctx.val
        if source:
            origin = f"{len(source)} held-out normals"
        else:
            source = train
            origin = "the training normals themselves"
            ctx.log(
                "this split has no val subset, so the score-normalization quantiles are "
                "fitted on the training normals themselves. The normalization is "
                "optimistic as a result; the ROC-AUC, being threshold-free, is not "
                "affected by it.",
                level="warning",
            )

        # anomalib's quantile routine holds every map it computes in a list before taking
        # a quantile over the lot, so the memory is linear in the number of images. 900
        # maps is gigabytes; a sampled subset estimates the same quantile for a fraction
        # of it. Capped here rather than there, and said out loud rather than assumed.
        chosen = evenly_spaced(len(source), self.config.quantile_images)
        calibration = [source[index] for index in chosen]
        if len(calibration) < len(source):
            ctx.log(
                f"fitting quantiles on {len(calibration)} of {len(source)} images from "
                f"{origin}, sampled evenly (quantile_images={self.config.quantile_images})"
            )
        else:
            ctx.log(f"fitting score-normalization quantiles on {origin}")

        ctx.progress(0.92, "fitting score-normalization quantiles")
        stream = _BatchStream(
            calibration, ctx.preprocessing, self.config.stats_batch_size, torch_module
        )
        quantiles = module.map_norm_quantiles(stream)
        module.model.quantiles.update(quantiles)
        self._emit_normalization(ctx, quantiles, origin=origin, images=len(calibration))

    def _emit_normalization(
        self,
        ctx: TrainContext,
        quantiles: dict[str, Any],
        *,
        origin: str,
        images: int,
    ) -> None:
        """Record what the fit above decided, as a table rather than a chart.

        These are four numbers, not a series, so the training charts are the wrong shape
        for them — but they are the most consequential four numbers the run produces after
        the weights. The ratio of the two branch weights they imply is what reorders
        images, and the `origin` row is the difference the M3 measurement put at +0.025
        sample ROC-AUC. Both belong somewhere a reader will look.
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

    # ---------------------------------------------------------------- diagnostics

    def _emit_architecture(self, ctx: TrainContext, model: Any, torch_module: Any) -> None:
        """Capture the real architecture from a dry forward pass, not from a diagram.

        Shapes and parameter counts read off the running model cannot go stale the way a
        hand-drawn figure does, which is the whole basis for M4's architecture view.

        Every module is recorded, not only the three branches, through the shared helper in
        `models/introspect.py` — so this method contributes only *which roots to walk* and
        *how they are wired*, and any other torch method inherits the same view by doing the
        same (ADR-0024). The branch nodes remain, as the depth-0 rows, so a reader who knows
        the old three-card picture is not lost.
        """
        if not ctx.diagnostics.enabled:
            return

        probe = torch_module.zeros(
            1, 3, ctx.preprocessing.height, ctx.preprocessing.width, device=ctx.device.value
        )
        records: list[ModuleRecord] = []
        for name in ("teacher", "student", "ae"):
            records.extend(collect(getattr(model, name), probe, prefix=f"{name}."))

        payload = build_tree(records)
        # The wiring between the branches is the one thing hooks cannot see: it lives in
        # the training loop's losses, not in any module's forward. Stated here, by the code
        # that knows it, rather than inferred by the helper from names it should not read.
        payload["edges"] = [
            {"source": "teacher", "target": "student", "label": "distillation"},
            {"source": "student", "target": "ae", "label": "reconstruction"},
        ]
        payload["total_parameters"] = int(sum(p.numel() for p in model.parameters()))

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
        self,
        ctx: TrainContext,
        model: Any,
        train: Sequence[ImageRecord],
        torch_module: Any,
    ) -> None:
        """What the teacher sees on one training image — the basis of M4's inspector."""
        if not ctx.diagnostics.enabled or not train:
            return

        image = torch_module.from_numpy(
            to_chw(load_array(train[0].path, ctx.preprocessing))[np.newaxis]
        ).to(ctx.device.value)
        with torch_module.no_grad():
            features = model.teacher(image)[0].detach().cpu().numpy()

        ctx.emit_diagnostic(
            "teacher_features_pca",
            "Teacher features (PCA composite)",
            DiagnosticKind.IMAGE,
            pca_to_rgb(features),
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

        if self._module is None:
            msg = "efficientad_anomalib was asked to predict before it was fitted or loaded"
            raise RuntimeError(msg)

        # `load` restores onto CPU so a checkpoint can be read on a machine without the
        # device it was trained on; the move to the inference device happens here.
        self._module = self._module.to(ctx.device.value)
        model = self._module.model
        model.eval()
        predictions: list[Prediction] = []

        for index, record in enumerate(images):
            ctx.raise_if_cancelled()
            started = time.perf_counter()

            image = torch.from_numpy(
                to_chw(load_array(record.path, ctx.preprocessing))[np.newaxis]
            ).to(ctx.device.value)

            with torch.no_grad():
                output = model(image)
                # The two branches apart, which is the diagnostic that makes EfficientAD
                # legible: a student-teacher hit and an autoencoder hit mean different
                # things about what went wrong.
                map_st, map_stae = model.get_maps(image, normalize=True)

            # `(B, 1, H, W)` from anomalib; indexing the batch away still leaves the
            # channel axis, so squeeze rather than assume.
            anomaly_map = output.anomaly_map[0].squeeze().detach().cpu().numpy().astype(np.float32)
            score = float(output.pred_score[0].detach().cpu())
            map_path = ctx.write_map(record.image_id, anomaly_map)
            elapsed_ms = (time.perf_counter() - started) * 1000.0

            ctx.emit_diagnostic(
                "map_student_teacher",
                "Student-teacher error",
                DiagnosticKind.MAP,
                map_st[0, 0].detach().cpu().numpy().astype(np.float32),
                image_id=record.image_id,
                description="Where the student failed to reproduce the teacher's features.",
            )
            ctx.emit_diagnostic(
                "map_autoencoder",
                "Autoencoder-student error",
                DiagnosticKind.MAP,
                map_stae[0, 0].detach().cpu().numpy().astype(np.float32),
                image_id=record.image_id,
                description="Where the autoencoder and the student disagree — the global branch.",
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

    # ---------------------------------------------------------------- persistence

    def save(self, artifact_dir: Path) -> None:
        """Write everything a *continuation* needs, not only everything inference needs.

        Format 2 (ADR-0025) carries the optimizer moments, the LR-schedule state, the step
        counter and both random streams. That is roughly three times the size of the
        weights alone — 32 MB becomes about 96 MB — and there is deliberately no option to
        skip it: an option would make "can I continue this run?" depend on a flag someone
        chose before they knew the answer.
        """
        import torch

        if self._module is None:
            msg = "efficientad_anomalib has nothing to save; it was never fitted"
            raise RuntimeError(msg)

        # `mean_std` and `quantiles` are ParameterDicts, so the state dict carries the
        # fitted normalization as well as the weights — nothing extra to serialize.
        payload: dict[str, Any] = {
            "format": CHECKPOINT_FORMAT,
            "state_dict": self._module.model.state_dict(),
            "model_size": self.config.model_size,
            "completed_steps": self._completed,
            "optimizer": self._optimizer_state,
            "scheduler": self._scheduler_state,
            "torch_rng_state": torch.get_rng_state(),
            "generator_state": (
                None if self._generator is None else self._generator.bit_generator.state
            ),
        }
        # Guarded: `torch.mps` exists only on an Apple-silicon build, and a checkpoint
        # written on one machine should still load on another.
        if hasattr(torch, "mps") and hasattr(torch.mps, "get_rng_state"):
            with contextlib.suppress(Exception):
                payload["mps_rng_state"] = torch.mps.get_rng_state()

        torch.save(payload, artifact_dir / STATE_FILENAME)

    def load(self, artifact_dir: Path) -> None:
        """Restore a fitted model, and whatever of the training state was written.

        A checkpoint with no `format` is format 1 — weights only. It loads, and `predict`
        is unaffected; only `fit_more` refuses it, and by name.
        """
        import torch
        from anomalib.models import EfficientAd

        stored = torch.load(artifact_dir / STATE_FILENAME, map_location="cpu", weights_only=False)
        module = EfficientAd(model_size=stored.get("model_size", self.config.model_size))
        module.model.load_state_dict(stored["state_dict"])
        self._module = module

        self._completed = int(stored.get("completed_steps", 0))
        self._optimizer_state = stored.get("optimizer")
        self._scheduler_state = stored.get("scheduler")
        self._torch_rng_state = stored.get("torch_rng_state")
        self._generator_state = stored.get("generator_state")


def _next_penalty_batch(module: Any) -> Any:
    """One batch from the penalty set, cycling the loader when it runs out."""
    try:
        return next(module.imagenet_iterator)[0]
    except StopIteration:
        module.imagenet_iterator = iter(module.imagenet_loader)
        return next(module.imagenet_iterator)[0]


def _penalty_set_is_extracted(penalty_dir: Path) -> bool:
    """Whether the penalty set is usable, rather than merely present.

    `ImageFolder` needs class subdirectories holding images. A lone `.tgz` means the
    download was interrupted before extraction.
    """
    return any(child.is_dir() for child in penalty_dir.iterdir())


def _refuse_download_if_missing() -> None:
    """Fail rather than fetch, when downloads are turned off and the weights are absent."""
    with contextlib.suppress(Exception):
        from anomalib.utils.path import get_pretrained_weights_dir

        if (get_pretrained_weights_dir() / "efficientad_pretrained_weights").is_dir():
            return
    msg = (
        "EfficientAD needs its pretrained teacher weights and allow_downloads is off. "
        "Turn allow_downloads back on for one run, or place the weights where anomalib "
        "expects them."
    )
    raise RuntimeError(msg)
