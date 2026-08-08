"""The `distill` job handler — a frozen source model compressed into the PDN teacher.

One entry in `jobs/handlers.py` and this function, like every other kind. It belongs to no
experiment, so its `experiment_id` is null and its log lands under `data/jobs/logs/`, exactly
as an import job's does. What it produces is an **asset**: a teacher in the model cache that
experiments then name in their configuration.

The mechanics — the feature source, the aggregation, the corpus — are in `teacher_distill`.
This file is the loop, the checkpoint and the manifest, which is the part that has to behave
like the rest of the application: progress on the job stream, `loss` as a metric event so the
existing chart draws it, cancellation honoured at a step boundary, and a resume that picks up
where a stopped run left off rather than starting again.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from anomaly_lab.jobs.context import JobCancelledError, JobContext
from anomaly_lab.models.teacher_distill import (
    GRAYSCALE_PROBABILITY,
    IMAGENET_MEAN,
    IMAGENET_STD,
    MANIFEST_FILENAME,
    PDN_GRID,
    WEIGHTS_FILENAME,
    DistillConfig,
    build_source,
    corpus_images,
    teacher_dir,
)
from anomaly_lab.schemas import API_MODEL_CONFIG

CHECKPOINT_FORMAT = 1
_LOSS_LOG_EVERY = 20


class DistillParams(BaseModel):
    """What a distill job is given: the configuration, whole, and nothing else.

    Unlike `train`, there is no row in the database to carry the configuration, because a
    distilled teacher is not an experiment. The manifest beside the weights is the record.
    """

    model_config = API_MODEL_CONFIG

    config: DistillConfig
    resume: bool = Field(
        default=True,
        description=(
            "Continue an interrupted run of the same name. Off, an existing directory of "
            "that name is refused rather than overwritten."
        ),
    )


def _load_batch(paths: list[Path], sizes: tuple[int, int], grayscale: list[bool]) -> Any:
    """Two views of the same images: one for the source, one for the PDN.

    The augmentation is applied to the *pair*, before either resize, because the two views
    have to remain two views of one image. Doing it per-view would teach the PDN to predict
    the colour of a picture it was not shown.

    Deliberately **not** `models.preprocessing.load_array`: that function exists so every
    *method* sees identical dataset pixels, and these are not dataset pixels. The transform
    here is the reference's, and routing it through the shared bridge would silently replace
    it with a plain resize.
    """
    from PIL import Image

    source_size, pdn_size = sizes
    source_batch = np.empty((len(paths), 3, source_size, source_size), dtype=np.float32)
    pdn_batch = np.empty((len(paths), 3, pdn_size, pdn_size), dtype=np.float32)
    for index, path in enumerate(paths):
        opened = Image.open(path)
        image = opened.convert("RGB") if opened.mode != "RGB" else opened
        if grayscale[index]:
            image = image.convert("L").convert("RGB")
        for target, size in ((source_batch, source_size), (pdn_batch, pdn_size)):
            resized = image.resize((size, size), Image.Resampling.BILINEAR)
            array = np.asarray(resized, dtype=np.float32) / 255.0
            target[index] = array.transpose(2, 0, 1)
    return source_batch, pdn_batch


def _normalize(batch: Any, torch_module: Any, device: str) -> Any:
    """ImageNet standardization, on the device, matching what the source was trained under."""
    tensor = torch_module.from_numpy(batch).to(device)
    mean = torch_module.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch_module.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
    return (tensor - mean) / std


class _Corpus:
    """An endless, seeded supply of image paths, whose position fits in a checkpoint."""

    def __init__(self, files: list[Path], seed: int) -> None:
        self._files = files
        self._generator = np.random.default_rng(seed)
        self._epoch_seed = int(self._generator.integers(2**31))
        self._cursor = 0

    def take(self, count: int) -> tuple[list[Path], list[bool]]:
        paths: list[Path] = []
        while len(paths) < count:
            order = np.random.default_rng(self._epoch_seed).permutation(len(self._files))
            if self._cursor >= len(order):
                self._epoch_seed = int(self._generator.integers(2**31))
                self._cursor = 0
                continue
            paths.append(self._files[int(order[self._cursor])])
            self._cursor += 1
        flips = [bool(self._generator.random() < GRAYSCALE_PROBABILITY) for _ in paths]
        return paths, flips

    def state(self) -> dict[str, Any]:
        return {
            "epoch_seed": self._epoch_seed,
            "cursor": self._cursor,
            "files": len(self._files),
            "generator": self._generator.bit_generator.state,
        }

    def restore(self, state: dict[str, Any], reporter: Any) -> None:
        if int(state.get("files", 0)) != len(self._files):
            reporter.log(
                f"the corpus now holds {len(self._files)} images where the checkpoint was "
                f"written against {state.get('files')}, so its order restarts",
                "warning",
            )
            return
        self._epoch_seed = int(state["epoch_seed"])
        self._cursor = int(state["cursor"])
        self._generator.bit_generator.state = state["generator"]


def _measure_source_statistics(
    source: Any,
    corpus: _Corpus,
    config: DistillConfig,
    ctx: JobContext,
    torch_module: Any,
    device: str,
) -> tuple[Any, Any]:
    """Channel mean and standard deviation of the source's features over the corpus.

    Two passes, as the reference does, rather than one accumulating both: the variance is
    measured against the *final* mean, and a running estimate of both at once would be
    measuring each batch's deviation from a mean that was still moving.
    """
    batches = max(1, config.normalization_images // config.batch_size)
    means = []
    for index in range(batches):
        ctx.raise_if_cancelled()
        paths, flips = corpus.take(config.batch_size)
        source_view, _ = _load_batch(paths, (source.input_size, 256), flips)
        features = source.features(_normalize(source_view, torch_module, device))
        means.append(features.mean(dim=(0, 2, 3)))
        if index % 16 == 0:
            ctx.progress(0.02 * index / batches, f"measuring source statistics {index}/{batches}")
    mean = torch_module.stack(means).mean(dim=0).view(1, -1, 1, 1)

    variances = []
    for index in range(batches):
        ctx.raise_if_cancelled()
        paths, flips = corpus.take(config.batch_size)
        source_view, _ = _load_batch(paths, (source.input_size, 256), flips)
        features = source.features(_normalize(source_view, torch_module, device))
        variances.append(((features - mean) ** 2).mean(dim=(0, 2, 3)))
        if index % 16 == 0:
            ctx.progress(
                0.02 + 0.02 * index / batches, f"measuring source spread {index}/{batches}"
            )
    std = torch_module.sqrt(torch_module.stack(variances).mean(dim=0)).view(1, -1, 1, 1)
    ctx.log(
        f"source statistics over {batches * config.batch_size} images: "
        f"mean in [{float(mean.min()):.3f}, {float(mean.max()):.3f}], "
        f"std in [{float(std.min()):.3f}, {float(std.max()):.3f}]"
    )
    return mean, std


def run_distill_job(ctx: JobContext) -> dict[str, Any]:
    """Distil a frozen source model into the compact PDN teacher."""
    import torch

    from anomaly_lab.models.base import Device
    from anomaly_lab.models.device import resolve_device
    from anomaly_lab.models.efficientad_nets import PatchDescriptionNetwork

    params = DistillParams.model_validate(dict(ctx.params))
    config = params.config
    cache_dir = ctx.settings.model_cache_dir
    directory = teacher_dir(cache_dir, config.name)
    checkpoint_path = directory / "checkpoint.pt"

    if directory.exists() and not params.resume:
        msg = (
            f"a distilled teacher named {config.name!r} already exists at {directory}. "
            "Choose another name, or allow resume."
        )
        raise FileExistsError(msg)
    directory.mkdir(parents=True, exist_ok=True)

    resolved = resolve_device(Device.MPS)
    device = resolved.device.value
    ctx.log(f"distilling into a {config.model_size} PDN on device {device}")
    ctx.log(f"device: {resolved.reason}")

    torch.manual_seed(config.seed)
    files = list(corpus_images(config, cache_dir, ctx))
    ctx.log(f"corpus '{config.corpus}' holds {len(files)} images")
    corpus = _Corpus(files, config.seed)

    source = build_source(config, device, ctx)
    try:
        # `padding=True`: the reference distils onto the source's 64x64 grid and detects
        # with padding off, where the same weights give a smaller map that is padded
        # afterwards. Regressing onto a grid of the wrong size would train a different
        # network and still finish without complaint.
        pdn = PatchDescriptionNetwork(
            out_channels=source.out_channels, size=config.model_size, padding=True
        ).to(device)
        optimizer = torch.optim.Adam(
            pdn.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
        )

        done = 0
        mean = std = None
        if checkpoint_path.is_file():
            stored = torch.load(checkpoint_path, map_location=device, weights_only=False)
            if int(stored.get("format", 0)) != CHECKPOINT_FORMAT:
                msg = (
                    f"the checkpoint at {checkpoint_path} declares format "
                    f"{stored.get('format')} and this build reads {CHECKPOINT_FORMAT}"
                )
                raise RuntimeError(msg)
            pdn.load_state_dict(stored["state_dict"])
            optimizer.load_state_dict(stored["optimizer"])
            done = int(stored["completed_steps"])
            mean = stored["source_mean"].to(device)
            std = stored["source_std"].to(device)
            corpus.restore(stored["corpus"], ctx)
            ctx.log(f"resuming at step {done} of {config.steps}")

        if mean is None or std is None:
            mean, std = _measure_source_statistics(source, corpus, config, ctx, torch, device)

        if done >= config.steps:
            ctx.log(f"already at {done} steps; nothing to do")
        else:
            _run_steps(
                pdn,
                optimizer,
                source,
                corpus,
                config,
                ctx,
                torch,
                device,
                mean=mean,
                std=std,
                done=done,
                checkpoint_path=checkpoint_path,
            )
            done = config.steps

        weights_path = directory / WEIGHTS_FILENAME
        torch.save(pdn.state_dict(), weights_path)
        manifest = {
            "format": CHECKPOINT_FORMAT,
            "name": config.name,
            "model_size": config.model_size,
            "out_channels": source.out_channels,
            "source": source.name,
            "source_input": source.input_size,
            "pdn_input": 256,
            "pdn_grid": PDN_GRID,
            "distilled_with_padding": True,
            "preprocessing": {
                "resize": "bilinear to a square, aspect not preserved",
                "normalization": "imagenet",
                "mean": list(IMAGENET_MEAN),
                "std": list(IMAGENET_STD),
                "random_grayscale": 0.1,
            },
            "corpus": config.corpus,
            "corpus_images": len(files),
            "steps": done,
            "config": config.model_dump(),
            "source_feature_mean": mean.flatten().tolist(),
            "source_feature_std": std.flatten().tolist(),
        }
        (directory / MANIFEST_FILENAME).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        ctx.log(f"teacher written to {weights_path}")
        return {
            "name": config.name,
            "directory": str(directory),
            "steps": done,
            "corpus_images": len(files),
            "source": source.name,
        }
    finally:
        source.close()


def _run_steps(
    pdn: Any,
    optimizer: Any,
    source: Any,
    corpus: _Corpus,
    config: DistillConfig,
    ctx: JobContext,
    torch_module: Any,
    device: str,
    *,
    mean: Any,
    std: Any,
    done: int,
    checkpoint_path: Path,
) -> None:
    """The distillation loop. Cancellation leaves a checkpoint, not a wasted hour."""

    def checkpoint(step: int) -> None:
        torch_module.save(
            {
                "format": CHECKPOINT_FORMAT,
                "state_dict": pdn.state_dict(),
                "optimizer": optimizer.state_dict(),
                "completed_steps": step,
                "source_mean": mean.cpu(),
                "source_std": std.cpu(),
                "corpus": corpus.state(),
            },
            checkpoint_path,
        )

    pdn.train()
    started = time.perf_counter()
    step = done
    try:
        for step in range(done, config.steps):
            ctx.raise_if_cancelled()
            paths, flips = corpus.take(config.batch_size)
            source_view, pdn_view = _load_batch(paths, (source.input_size, 256), flips)

            target = (source.features(_normalize(source_view, torch_module, device)) - mean) / std
            prediction = pdn(torch_module.from_numpy(pdn_view).to(device))
            if prediction.shape[-2:] != target.shape[-2:]:
                msg = (
                    f"the PDN emits {tuple(prediction.shape[-2:])} where the source's grid "
                    f"is {tuple(target.shape[-2:])}. Distilling onto a mismatched grid "
                    "would train a different network; check that padding is on."
                )
                raise RuntimeError(msg)
            loss = torch_module.mean((target - prediction) ** 2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % _LOSS_LOG_EVERY == 0:
                ctx.metric("distillation_loss", float(loss.item()), step=step)
            fraction = 0.04 + 0.96 * (step + 1) / config.steps
            elapsed = time.perf_counter() - started
            rate = (step + 1 - done) / max(elapsed, 1e-9)
            ctx.progress(
                fraction,
                f"step {step + 1}/{config.steps}, loss {float(loss.item()):.4f}, "
                f"{1000.0 / max(rate, 1e-9):.0f} ms/step",
            )
            if (step + 1) % config.checkpoint_every == 0:
                checkpoint(step + 1)
    except JobCancelledError:
        checkpoint(step)
        ctx.log(f"cancelled at step {step}; the checkpoint is written and this run resumes")
        raise
    checkpoint(config.steps)
