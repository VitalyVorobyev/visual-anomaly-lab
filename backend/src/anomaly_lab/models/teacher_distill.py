"""Distilling a frozen source model into the compact PDN teacher.

EfficientAD's teacher is a 2.7M-parameter PDN that has been taught to reproduce the local
features of a much larger network. The paper distils a WideResNet-101 on ImageNet; every
published teacher is somebody's run of that procedure. This module is ours.

**Why it is here at all.** Measured on `candle` at a fixed budget, swapping which *published*
teacher is loaded moved AU-PRO from 0.560 to 0.916 — the largest single effect in this
project. A weight file nobody in this repository produced was therefore the most important
input the workbench did not control. Producing it is what turns the teacher from something we
are handed into something we can measure.

**The source model is training-only.** What ships is the same PDN, at the same inference
cost. Nothing about deployment changes because a WideResNet was involved in making the
weights, and that asymmetry is the whole point of distillation.

**The recipe is the reference one**, because a teacher distilled by a different procedure
would make every comparison against a published teacher partly a measurement of the
procedure: a frozen `wide_resnet101_2` at 512x512, `layer2` and `layer3` patch-aggregated to
384 channels on a 64x64 grid, channel-normalized by statistics measured over the corpus, and
matched by MSE against the PDN's output on the same image at 256x256. Adam, 1e-4, weight
decay 1e-5.

**The source is a protocol, from the first commit.** `FeatureSource` is what the distillation
loop consumes, and `WideResNet101Source` is one implementation. A frozen DINOv2-S is a second
one — a class, not a rewrite of the loop — and paying that cost now was nearly free.

**Torch is imported inside functions**, as everywhere else in this package, so importing the
module to read its configuration costs nothing.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal, Protocol

from pydantic import BaseModel, Field

from anomaly_lab.schemas import API_MODEL_CONFIG

DISTILLED_SUBDIR = "efficientad-teacher-distilled"
"""Where a produced teacher lands in the model cache, one directory per run name."""

MANIFEST_FILENAME = "distillation.json"
WEIGHTS_FILENAME = "teacher.pth"

PDN_GRID = 64
"""The PDN's output grid at 256 px input *with padding on*, and the target grid it learns.

The reference distils with `padding=True` so the PDN's output is 64x64 and lines up with the
extractor's patch grid, then runs anomaly detection with padding *off*, where the same
weights produce a smaller map that is padded afterwards. Both are true at once because
padding changes a convolution's extent and not its weights — but a distillation run that
forgot it would be regressing onto a grid of the wrong size.
"""

SOURCE_INPUT = 512
"""What the source model sees. Twice the PDN's input, which is what makes its stride-8
`layer2` land on the same 64x64 grid the PDN emits from 256 px."""

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

GRAYSCALE_PROBABILITY = 0.1
"""The reference's augmentation, applied to *both* views of an image so the pair stays a pair."""


class FeatureSource(Protocol):
    """A frozen model that produces the target the PDN is taught to reproduce.

    Deliberately narrow. Everything a distillation run needs to know about a source is its
    name, the resolution it wants, and how to turn a batch into a `(N, C, 64, 64)` target.
    A source that needs the loop to change is a source this protocol has described wrongly.
    """

    @property
    def name(self) -> str:
        """Recorded in the manifest, so a teacher can say what it was distilled from."""

    @property
    def input_size(self) -> int:
        """The square resolution this source wants its images at."""

    @property
    def out_channels(self) -> int:
        """Target channel count — 384, the PDN's output width."""

    def features(self, batch: Any) -> Any:
        """`(N, 3, input_size, input_size)` normalized pixels to `(N, out_channels, 64, 64)`."""

    def close(self) -> None:
        """Release whatever the source holds. Part of the contract, not a detail: the one
        implementation here hooks a frozen backbone, and a hook left registered outlives
        the run that made it."""


class DistillConfig(BaseModel):
    """Everything that decides what a distilled teacher is.

    Stored beside the weights, because a teacher whose provenance is not recorded is exactly
    the problem this module exists to fix.
    """

    model_config = API_MODEL_CONFIG

    name: str = Field(
        min_length=1,
        max_length=64,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$",
        description=(
            "Names the produced teacher, and its directory in the model cache. An "
            "experiment refers to it by this name."
        ),
    )
    model_size: Literal["small", "medium"] = Field(
        default="small",
        description="Which PDN width to distil into. Must match the student's model_size.",
    )
    source: Literal["wide_resnet101_2"] = Field(
        default="wide_resnet101_2",
        description=(
            "The frozen model being compressed. Only the paper's WideResNet is implemented; "
            "the loop takes any FeatureSource, which is what a DINOv2 source would be."
        ),
    )
    corpus: Literal["imagenette", "directory"] = Field(
        default="imagenette",
        description=(
            "Images to distil over. 'imagenette' is the 13k-image smoke corpus already "
            "downloaded for the penalty set. 'directory' reads corpus_path, which is how "
            "ImageNet-1K is used — never by default, because it is a multi-day run here."
        ),
    )
    corpus_path: str = Field(
        default="",
        description="Root of the image tree when corpus is 'directory'. Walked recursively.",
    )
    steps: int = Field(
        default=10_000,
        ge=10,
        le=1_000_000,
        description=(
            "Distillation steps. The reference uses 60000 at batch 16 over ImageNet; the "
            "default here is a fraction of that so a first run finishes in an evening."
        ),
    )
    batch_size: int = Field(
        default=4,
        ge=1,
        le=64,
        description=(
            "Images per step. The reference uses 16. The binding cost is the patch tensor "
            "of the source's layer3, which is ~150 MB per image at float32, so this is the "
            "knob that fits the run into unified memory."
        ),
    )
    learning_rate: float = Field(default=1e-4, gt=0.0, le=1.0, description="Adam learning rate.")
    weight_decay: float = Field(default=1e-5, ge=0.0, le=1.0, description="Adam weight decay.")
    normalization_images: int = Field(
        default=1024,
        ge=16,
        le=100_000,
        description=(
            "Images used to measure the source's channel mean and standard deviation "
            "before distillation starts. The reference uses 10000; the cost is linear and "
            "the estimate converges long before that."
        ),
    )
    seed: int = Field(default=0, description="Seeds the corpus order and the augmentation.")
    allow_downloads: bool = Field(
        default=True,
        description="Permit fetching the source model's ImageNet weights and the smoke corpus.",
    )
    checkpoint_every: int = Field(
        default=1000,
        ge=1,
        le=100_000,
        description="Steps between checkpoints. A run this long must survive being stopped.",
    )


def teacher_dir(cache_dir: Path, name: str) -> Path:
    """Where a distilled teacher by this name lives."""
    return cache_dir / DISTILLED_SUBDIR / name


def load_manifest(cache_dir: Path, name: str) -> dict[str, Any]:
    """A distilled teacher's recorded provenance, or a refusal naming what is missing."""
    path = teacher_dir(cache_dir, name) / MANIFEST_FILENAME
    if not path.is_file():
        known = sorted(p.name for p in (cache_dir / DISTILLED_SUBDIR).glob("*") if p.is_dir())
        listing = ", ".join(known) if known else "none"
        msg = (
            f"no distilled teacher named {name!r} at {path.parent}. "
            f"Distilled teachers present: {listing}."
        )
        raise FileNotFoundError(msg)
    return dict(json.loads(path.read_text(encoding="utf-8")))


POOL_CHUNK = 4096
"""Patch positions pooled at once, so peak memory is flat in batch size rather than linear.

The flattened `layer3` patches are `1024 * 9 = 9216` floats per position and there are
`batch * 4096` positions — 2.4 GB for one intermediate at batch 16. Chunking makes batch size
a throughput decision instead of a memory cliff, which is what lets this run on a laptop.
"""


def _patchify(feature: Any, torch_module: Any) -> tuple[Any, tuple[int, int]]:
    """3x3 neighbourhoods at stride 1 — `(batch, positions, channels * 9)`."""
    _batch, channels, height, width = feature.shape
    flat = torch_module.nn.functional.unfold(feature, kernel_size=3, padding=1, stride=1)
    # `unfold` gives (batch, channels * 9, positions), channel-major then kernel position,
    # which is exactly the (C, 3, 3) flattening order the reference pools over.
    return flat.transpose(1, 2).reshape(flat.shape[0], height * width, channels * 9), (
        height,
        width,
    )


def _pool_last(flat: Any, target: int, torch_module: Any) -> Any:
    """Adaptive average pool over the last dimension, chunked across positions."""
    batch, positions, _ = flat.shape
    out = torch_module.empty(batch, positions, target, device=flat.device, dtype=flat.dtype)
    for start in range(0, positions, POOL_CHUNK):
        stop = min(start + POOL_CHUNK, positions)
        piece = flat[:, start:stop]
        shape = piece.shape
        pooled = torch_module.nn.functional.adaptive_avg_pool1d(
            piece.reshape(-1, 1, shape[-1]), target
        )
        out[:, start:stop] = pooled.reshape(shape[0], shape[1], target)
    return out


def aggregate_patch_features(fine: Any, coarse: Any, out_channels: int, torch_module: Any) -> Any:
    """Two backbone feature maps to one `(N, out_channels, H, W)` target on the finer grid.

    PatchCore's aggregation, which the reference reuses verbatim and which is therefore
    reproduced rather than improved: 3x3 patches at stride 1 from each map, the coarse map's
    **patch grid** resampled up to the fine one's, each patch flattened and adaptively pooled
    to 1024, the two stacked and adaptively pooled again to `out_channels`.

    Split out of the source class so it can be pinned by a test against a transcription of
    the reference without downloading a 243 MB backbone to do it. The subtle part is that the
    resampling happens on the *patch* grid — interpolating the feature map and patchifying
    afterwards is a different function, and it is the one you write by accident.
    """
    patched_fine, (grid_h, grid_w) = _patchify(fine, torch_module)
    patched_coarse, (small_h, small_w) = _patchify(coarse, torch_module)

    size = patched_coarse.shape[-1]
    grid = patched_coarse.reshape(patched_coarse.shape[0], small_h, small_w, size)
    grid = grid.permute(0, 3, 1, 2)
    grid = torch_module.nn.functional.interpolate(
        grid, size=(grid_h, grid_w), mode="bilinear", align_corners=False
    )
    patched_coarse = grid.permute(0, 2, 3, 1).reshape(
        patched_coarse.shape[0], grid_h * grid_w, size
    )

    mapped = torch_module.stack(
        [
            _pool_last(patched_fine, 1024, torch_module),
            _pool_last(patched_coarse, 1024, torch_module),
        ],
        dim=2,
    )
    mapped = mapped.reshape(mapped.shape[0], mapped.shape[1], -1)
    aggregated = _pool_last(mapped, out_channels, torch_module)
    return aggregated.reshape(aggregated.shape[0], grid_h, grid_w, out_channels).permute(0, 3, 1, 2)


class WideResNet101Source:
    """The paper's source: a frozen `wide_resnet101_2`, `layer2` and `layer3`.

    The model is held in eval mode with gradients off, and the two layers are read by
    forward hooks that are removed in `close()`. The aggregation itself is
    `aggregate_patch_features`, which is where the arithmetic lives and where it is tested.
    """

    def __init__(self, device: str, *, allow_downloads: bool, reporter: Any) -> None:
        import torch
        import torchvision
        from torchvision.models import Wide_ResNet101_2_Weights

        self._torch = torch
        weights = Wide_ResNet101_2_Weights.IMAGENET1K_V1
        if not allow_downloads:
            reporter.log(
                "allow_downloads is off, so the WideResNet weights must already be in "
                "torch's hub cache; this will fail by name if they are not",
                level="warning",
            )
        backbone = torchvision.models.wide_resnet101_2(weights=weights)
        self._backbone = backbone.to(device).eval()
        for parameter in self._backbone.parameters():
            parameter.requires_grad_(False)
        self._device = device
        self._captured: dict[str, Any] = {}
        self._handles = [
            self._backbone.layer2.register_forward_hook(self._capture("layer2")),
            self._backbone.layer3.register_forward_hook(self._capture("layer3")),
        ]

    def _capture(self, key: str) -> Any:
        def hook(_module: Any, _inputs: Any, output: Any) -> None:
            self._captured[key] = output

        return hook

    def close(self) -> None:
        """Remove the hooks. A hook left on a frozen model is a leak, not a bug you see."""
        for handle in self._handles:
            handle.remove()
        self._handles = []

    @property
    def name(self) -> str:
        return "wide_resnet101_2/layer2+layer3"

    @property
    def input_size(self) -> int:
        return SOURCE_INPUT

    @property
    def out_channels(self) -> int:
        return 384

    def features(self, batch: Any) -> Any:
        torch = self._torch
        with torch.no_grad():
            self._captured.clear()
            self._backbone(batch)
            return aggregate_patch_features(
                self._captured["layer2"], self._captured["layer3"], self.out_channels, torch
            )


def build_source(config: DistillConfig, device: str, reporter: Any) -> FeatureSource:
    """The configured source. One `if`, because there is one implementation — and the
    protocol is what makes adding the second one a class rather than an edit here."""
    if config.source == "wide_resnet101_2":
        return WideResNet101Source(
            device, allow_downloads=config.allow_downloads, reporter=reporter
        )
    msg = f"no feature source is implemented for {config.source!r}"
    raise ValueError(msg)


def corpus_images(config: DistillConfig, cache_dir: Path, reporter: Any) -> Sequence[Path]:
    """Every image the distillation will draw from, in a deterministic order."""
    from anomaly_lab.models.efficientad_assets import penalty_images

    if config.corpus == "imagenette":
        # The same tree the penalty set uses. It is already on disk, and "natural images
        # that are not the dataset" is exactly what both jobs want — but they are read for
        # different reasons, and this one owns no transform of the other's.
        files = penalty_images(cache_dir, allow_downloads=config.allow_downloads, reporter=reporter)
        return list(files)

    root = Path(config.corpus_path).expanduser()
    if not root.is_dir():
        msg = (
            f"corpus is 'directory' and corpus_path {root} is not a directory. "
            "ImageNet-1K is used by pointing this at its train tree."
        )
        raise FileNotFoundError(msg)
    suffixes = {".jpeg", ".jpg", ".png", ".bmp", ".webp"}
    files = sorted(p for p in root.rglob("*") if p.suffix.lower() in suffixes and p.is_file())
    if not files:
        msg = f"the corpus at {root} holds no images"
        raise FileNotFoundError(msg)
    return files
