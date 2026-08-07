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
import inspect
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
from anomaly_lab.models.preprocessing import load_array, to_chw
from anomaly_lab.schemas import API_MODEL_CONFIG

if TYPE_CHECKING:  # pragma: no cover - import cost is the whole point of deferring it
    import torch

STATE_FILENAME = "efficientad.pt"
PENALTY_SUBDIR = "imagenette"

# The paper trains for 70000 steps. That is roughly two and a half hours on this Mac at
# the measured 123 ms/step, which is not a default anyone will wait through the first
# time. The default here trades a known amount of accuracy for a runnable first result,
# and says so in the field description rather than in a comment nobody reads.
DEFAULT_MAX_STEPS = 4000

_LOSS_LOG_EVERY = 20


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


def _pca_to_rgb(features: np.ndarray) -> np.ndarray:
    """Reduce a `(C, H, W)` feature map to `(H, W, 3)` in [0, 1] by PCA.

    384 teacher channels cannot be looked at directly. The three leading components
    carry what varies spatially, which is what a reader wants to see; the result is a
    false-colour image and is labelled as one in the UI.
    """
    channels, height, width = features.shape
    flat = features.reshape(channels, height * width).T.astype(np.float64)
    centred = flat - flat.mean(axis=0, keepdims=True)
    # SVD rather than an explicit covariance eigendecomposition: 384x384 is small, but
    # the covariance route squares the condition number for no benefit here.
    _, _, components = np.linalg.svd(centred, full_matrices=False)
    projected = centred @ components[:3].T
    projected = projected.reshape(height, width, 3)
    low = projected.min(axis=(0, 1), keepdims=True)
    high = projected.max(axis=(0, 1), keepdims=True)
    return ((projected - low) / np.maximum(high - low, 1e-9)).astype(np.float32)


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

    @classmethod
    def config_model(cls) -> type[BaseModel]:
        return EfficientAdConfig

    @classmethod
    def capabilities(cls) -> Capabilities:
        return Capabilities(
            requires_training=True,
            produces_anomaly_map=True,
            produces_diagnostics=True,
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

        optimizer = torch.optim.Adam(
            list(model.student.parameters()) + list(model.ae.parameters()),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        # anomalib's own schedule: one tenfold drop near the end of training.
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=int(0.95 * self.config.max_steps), gamma=0.1
        )

        model.train()
        generator = np.random.default_rng(self.config.seed)
        ctx.log(f"training for {self.config.max_steps} steps at batch size 1")
        started = time.perf_counter()

        for step in range(self.config.max_steps):
            # Checked every step rather than every epoch, which is the practical benefit
            # of owning the loop: cancelling a training run stops it in ~100 ms.
            ctx.raise_if_cancelled()

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

            if step % _LOSS_LOG_EVERY == 0 or step == self.config.max_steps - 1:
                # Scalar series reuse the `metric` event that already exists in the job
                # protocol (ADR-0009). No new channel, no protocol change — which is the
                # test the diagnostics contract was designed to pass.
                ctx.metric("loss_st", float(loss_st.item()), step=step)
                ctx.metric("loss_ae", float(loss_ae.item()), step=step)
                ctx.metric("loss_stae", float(loss_stae.item()), step=step)
                ctx.metric("loss_total", float(loss.item()), step=step)
                ctx.metric("learning_rate", float(scheduler.get_last_lr()[0]), step=step)
                ctx.progress(
                    0.1 + 0.8 * (step + 1) / self.config.max_steps,
                    f"step {step + 1}/{self.config.max_steps}, loss {loss.item():.4f}",
                )

        elapsed = time.perf_counter() - started
        ctx.log(
            f"trained {self.config.max_steps} steps in {elapsed:.0f}s "
            f"({elapsed / self.config.max_steps * 1000:.0f} ms/step on {ctx.device.value})"
        )

        self._fit_quantiles(module, ctx, train, torch)
        model.eval()
        self._module = module

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
        fitted on data the student has already memorized, so the normalization is
        optimistic — and it is logged as a warning rather than silently substituted.
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
        module.model.quantiles.update(module.map_norm_quantiles(stream))

    # ---------------------------------------------------------------- diagnostics

    def _emit_architecture(self, ctx: TrainContext, model: Any, torch_module: Any) -> None:
        """Capture the real architecture from a dry forward pass, not from a diagram.

        Shapes and parameter counts read off the running model cannot go stale the way a
        hand-drawn figure does, which is the whole basis for M4's architecture view.
        """
        if not ctx.diagnostics.enabled:
            return

        probe = torch_module.zeros(
            1, 3, ctx.preprocessing.height, ctx.preprocessing.width, device=ctx.device.value
        )
        nodes: list[dict[str, Any]] = []
        for name in ("teacher", "student", "ae"):
            branch = getattr(model, name)
            with torch_module.no_grad():
                # The branches do not share a signature: the autoencoder needs the target
                # size, because its decoder upsamples to it rather than inferring it. The
                # signature is inspected rather than special-cased by branch name, so a
                # future branch with its own arguments does not silently break this.
                output = branch(probe, **_probe_arguments(branch, probe))
            nodes.append(
                {
                    "id": name,
                    "label": name,
                    "type": type(branch).__name__,
                    "parameters": int(sum(p.numel() for p in branch.parameters())),
                    "input_shape": list(probe.shape),
                    "output_shape": list(output.shape),
                }
            )

        ctx.emit_diagnostic(
            "architecture",
            "Model architecture",
            DiagnosticKind.GRAPH,
            {
                "nodes": nodes,
                "edges": [
                    {"source": "teacher", "target": "student", "label": "distillation"},
                    {"source": "student", "target": "ae", "label": "reconstruction"},
                ],
                "total_parameters": int(sum(p.numel() for p in model.parameters())),
            },
            description="Branch shapes and parameter counts, read from a dry forward pass.",
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
        import torch

        if self._module is None:
            msg = "efficientad_anomalib has nothing to save; it was never fitted"
            raise RuntimeError(msg)
        # `mean_std` and `quantiles` are ParameterDicts, so the state dict carries the
        # fitted normalization as well as the weights — nothing extra to serialize.
        torch.save(
            {
                "state_dict": self._module.model.state_dict(),
                "model_size": self.config.model_size,
            },
            artifact_dir / STATE_FILENAME,
        )

    def load(self, artifact_dir: Path) -> None:
        import torch
        from anomalib.models import EfficientAd

        stored = torch.load(artifact_dir / STATE_FILENAME, map_location="cpu", weights_only=True)
        module = EfficientAd(model_size=stored.get("model_size", self.config.model_size))
        module.model.load_state_dict(stored["state_dict"])
        self._module = module


def _next_penalty_batch(module: Any) -> Any:
    """One batch from the penalty set, cycling the loader when it runs out."""
    try:
        return next(module.imagenet_iterator)[0]
    except StopIteration:
        module.imagenet_iterator = iter(module.imagenet_loader)
        return next(module.imagenet_iterator)[0]


def _probe_arguments(branch: Any, probe: Any) -> dict[str, Any]:
    """Extra keyword arguments a branch needs for a dry forward pass.

    Only `image_size` so far, which the autoencoder requires because its decoder
    upsamples to an explicit target rather than inferring one.
    """
    try:
        parameters = inspect.signature(branch.forward).parameters
    except (TypeError, ValueError):  # pragma: no cover - a C-implemented forward
        return {}
    return {"image_size": probe.shape[-2:]} if "image_size" in parameters else {}


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
