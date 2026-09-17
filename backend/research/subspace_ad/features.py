"""The encoder half: pixels in, patch features out, with every convention named.

This is the only module in the campaign that imports torch, and the only one whose cost
scales with anything. Everything downstream of it -- tau, rho, the shot count, the seed --
is arithmetic on what it returns (`subspace`), which is what makes a sweep of thousands of
arms affordable.

**Three conventions decide the numbers, and none of them is stated in the paper.** Each is
an explicit field here rather than a line of code, because a silent difference in any of
them is indistinguishable from a difference in the method:

  * **Which blocks.** The paper reads layers 22-28 of DINOv2-G's 40. That is a statement
    about *relative depth* (0.55 to 0.70) or about *a count of seven ending at 70%* -- on a
    40-block encoder the two are the same window, and on a 12-block ViT-S they are two
    layers and seven layers respectively. `LayerBand` expresses both, and which one
    transfers is a question the campaign answers rather than assumes.
  * **Final norm or not.** timm's `forward_intermediates(norm=True)` applies the encoder's
    *last* LayerNorm to every intermediate block it returns. DINOv2's own
    `get_intermediate_layers` defaults to the same thing, so it is the convention most
    published feature extraction follows -- but it is a convention, and the un-normed
    residual stream has a different scale per block.
  * **Whether layers are L2-normalized before pooling.** `dino_backbone.extract_patch_features`
    does, deliberately, because a nearest-neighbour bank compares by Euclidean distance and
    block norms are not comparable across depth. SubspaceAD does *not*: it takes a plain
    mean (Eq. 2). Pooling normalized layers is a different feature, and one this campaign
    can measure rather than argue about.

**Rotations invent pixels, and the campaign decides what to do about that.** The fit set is
augmented by random rotations, which for a square frame means the corners are filled with
something. Under `zeros` they are filled with black, and those black patches enter the
covariance as a genuine direction of "normal" variation that no test image will ever show.
Under `masked` they are excluded: a rotated copy of an all-ones frame says exactly which
patches are entirely real, and only those are folded in.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from anomaly_lab.media import decode
from anomaly_lab.models.dino_backbone import BACKBONES, DinoBackbone, load_backbone, patch_grid
from anomaly_lab.regions.transform import SpatialTransform

ROTATION_MAX_DEGREES = 345.0
"""The paper's upper bound: rotations are drawn from 0 to 345 degrees."""


class RotationFill(StrEnum):
    """What a rotated frame's corners contribute to the fitted subspace."""

    ZEROS = "zeros"
    MASKED = "masked"


class Aggregation(StrEnum):
    MEAN = "mean"
    CONCAT = "concat"


@dataclass(frozen=True)
class LayerBand:
    """A window of transformer blocks, expressed so it survives a change of encoder.

    Depths differ across the menu -- 12 blocks for ViT-S and ViT-B, 24 for ViT-L, 40 for
    the DINOv2-G the paper used -- so an absolute block number means a different thing on
    each. Both readings of "layers 22-28" are representable:

      * `fixed_count=None` keeps the **relative band**: every block between the two depth
        fractions, so the window covers the same part of the encoder and its size follows
        the depth.
      * `fixed_count=n` keeps **n blocks ending at `end_fraction`**, so the window is the
        same size everywhere and sits at the same depth at its deep end.
    """

    name: str
    end_fraction: float
    start_fraction: float | None = None
    fixed_count: int | None = None

    def blocks(self, depth: int) -> tuple[int, ...]:
        """One-based block numbers for an encoder of this depth, ascending."""
        if depth < 1:
            msg = f"an encoder needs at least one block; got depth {depth}"
            raise ValueError(msg)
        last = min(depth, max(1, round(self.end_fraction * depth)))
        if self.fixed_count is not None:
            first = max(1, last - self.fixed_count + 1)
        elif self.start_fraction is not None:
            first = min(last, max(1, math.ceil(self.start_fraction * depth)))
        else:
            first = last
        return tuple(range(first, last + 1))

    def indices(self, depth: int) -> tuple[int, ...]:
        """The same window as zero-based indices, which is what timm takes."""
        return tuple(block - 1 for block in self.blocks(depth))


# The paper's own windows first, each under both readings, then the two this repository's
# `FeatureLayers` already offers so a SubspaceAD arm can be read against `dino_memory`.
LAYER_BANDS: dict[str, LayerBand] = {
    "mid_band": LayerBand("mid_band", end_fraction=0.70, start_fraction=0.55),
    "mid7": LayerBand("mid7", end_fraction=0.70, fixed_count=7),
    "mid4": LayerBand("mid4", end_fraction=0.70, fixed_count=4),
    "final_band": LayerBand("final_band", end_fraction=1.0, start_fraction=0.85),
    "final7": LayerBand("final7", end_fraction=1.0, fixed_count=7),
    "last": LayerBand("last", end_fraction=1.0, fixed_count=1),
    "last_two": LayerBand("last_two", end_fraction=1.0, fixed_count=2),
    "last_four": LayerBand("last_four", end_fraction=1.0, fixed_count=4),
    "upper_half": LayerBand("upper_half", end_fraction=1.0, start_fraction=0.5),
}


@dataclass(frozen=True)
class FeatureView:
    """One way of turning a stack of block outputs into the vector PCA sees."""

    band: LayerBand
    aggregation: Aggregation = Aggregation.MEAN
    l2_normalize: bool = False

    @property
    def name(self) -> str:
        parts = [self.band.name, self.aggregation.value]
        if self.l2_normalize:
            parts.append("l2")
        return "-".join(parts)

    def dimension(self, embedding_dim: int, depth: int) -> int:
        if self.aggregation is Aggregation.CONCAT:
            return embedding_dim * len(self.band.blocks(depth))
        return embedding_dim

    def apply(self, stack: Any, positions: dict[int, int], depth: int) -> Any:
        """Reduce a `(B, L, D, P)` tensor of every requested block to `(B, D_view, P)`.

        `positions` maps a block index to its row in the stack, so one forward pass over
        the union of every view's blocks serves all of them.
        """
        import torch

        chosen = [positions[index] for index in self.band.indices(depth)]
        selected = stack[:, chosen]
        if self.l2_normalize:
            selected = torch.nn.functional.normalize(selected, p=2.0, dim=2)
        if self.aggregation is Aggregation.CONCAT:
            return selected.flatten(1, 2)
        return selected.mean(dim=1)


@dataclass(frozen=True)
class PreparedImage:
    """One frame as the encoder will see it, with the transform that produced it.

    The transform is kept because the ground-truth mask has to travel the same path: a
    score map in the prepared frame can only be compared against a mask that was cropped,
    resized and padded identically.
    """

    array: np.ndarray
    transform: SpatialTransform


def prepare(path: Path, size: int) -> PreparedImage:
    """Decode one source image into a `size` x `size` RGB frame in `[0, 1]`.

    The geometry is `SpatialTransform` with no region -- the application's identity region
    profile, reached directly. Sharing it is the point: a number measured here and a number
    measured by a run inside the workbench then differ by the method, not by a resize.
    """
    image = decode.load(path)
    if image.mode != "RGB":
        image = image.convert("RGB")
    transform = SpatialTransform.resolve(source_size=image.size, prepared_size=(size, size))
    prepared = transform.prepare_image(image, resample=Image.Resampling.BILINEAR)
    return PreparedImage(
        array=np.asarray(prepared, dtype=np.float32) / 255.0,
        transform=transform,
    )


def rotations(
    prepared: np.ndarray,
    *,
    count: int,
    rng: np.random.Generator,
    fill: RotationFill,
) -> list[tuple[np.ndarray, np.ndarray | None]]:
    """The original frame plus `count` rotated copies, each with its validity mask.

    The mask is `None` for the unrotated original and for every copy under `zeros`, where
    the invented corners are deliberately kept. Under `masked` it is a pixel-level map of
    where the frame is real, produced by rotating an all-ones image through exactly the
    same call -- which is more reliable than deriving the polygon, because it inherits
    whatever rounding Pillow's own resampling does.
    """
    frames: list[tuple[np.ndarray, np.ndarray | None]] = [(prepared, None)]
    if count <= 0:
        return frames
    source = Image.fromarray((prepared * 255.0).round().astype(np.uint8), mode="RGB")
    ones = Image.fromarray(np.full(prepared.shape[:2], 255, dtype=np.uint8), mode="L")
    for angle in rng.uniform(0.0, ROTATION_MAX_DEGREES, size=count):
        turned = source.rotate(float(angle), resample=Image.Resampling.BILINEAR, fillcolor=0)
        array = np.asarray(turned, dtype=np.float32) / 255.0
        valid = None
        if fill is RotationFill.MASKED:
            spun = ones.rotate(float(angle), resample=Image.Resampling.NEAREST, fillcolor=0)
            valid = np.asarray(spun) > 0
        frames.append((array, valid))
    return frames


def valid_patches(valid: np.ndarray | None, grid: tuple[int, int], patch: int) -> np.ndarray | None:
    """Which tokens of a `grid` layout are backed entirely by real pixels.

    All-or-nothing per patch. A token whose receptive field is nine-tenths real is still a
    token the encoder computed partly from invented black, and admitting it would put a
    dimmed edge into the subspace of normal appearance.
    """
    if valid is None:
        return None
    rows, columns = grid
    blocks = valid[: rows * patch, : columns * patch].reshape(rows, patch, columns, patch)
    return blocks.all(axis=(1, 3)).reshape(-1)


class PatchEncoder:
    """A frozen encoder, pinned to one input size, that returns per-view patch features.

    Holds the union of every view's blocks so a forward pass serves all of them at once.
    That is the second half of the cost argument: the layer-aggregation axis costs one
    extra slice of an activation that was computed anyway, not another pass.
    """

    def __init__(
        self,
        backbone: DinoBackbone,
        *,
        size: int,
        views: Sequence[FeatureView],
        cache_dir: Path,
        device: str | None = None,
        seed: int = 0,
        final_norm: bool = True,
        allow_downloads: bool = True,
        pretrained: bool = True,
    ) -> None:
        import torch

        from anomaly_lab.models.base import Device
        from anomaly_lab.models.device import resolve_device

        self.backbone = backbone
        self.spec = BACKBONES[backbone]
        self.size = size
        self.views = tuple(views)
        self.final_norm = final_norm
        self.grid = patch_grid(backbone, size, size)
        self.depth = self.spec.depth
        resolved = resolve_device(Device.MPS)
        self.device_reason = resolved.reason
        self.device = torch.device(device or resolved.device.value)

        blocks = sorted({index for view in views for index in view.band.indices(self.depth)})
        self.indices = tuple(blocks)
        self.positions = {index: position for position, index in enumerate(blocks)}

        self.model = load_backbone(
            backbone,
            pretrained=pretrained,
            allow_downloads=allow_downloads,
            cache_dir=cache_dir,
            seed=seed,
            method="subspace_ad_research",
        ).to(self.device)

        # The checkpoint's own statistics rather than a hard-coded ImageNet triple.
        # `preprocessing.IMAGENET_MEAN` exists and would be right for every entry in
        # `BACKBONES` today, but standardizing pixels for a pretrained network is a
        # property of *that network*, and a future entry trained under different
        # statistics would be silently fed the wrong ones.
        config = self.model.pretrained_cfg
        self.mean = torch.tensor(config["mean"], dtype=torch.float32).view(1, 3, 1, 1)
        self.mean = self.mean.to(self.device)
        self.std = torch.tensor(config["std"], dtype=torch.float32).view(1, 3, 1, 1)
        self.std = self.std.to(self.device)

    @property
    def patch_count(self) -> int:
        return self.grid[0] * self.grid[1]

    def dimensions(self) -> dict[str, int]:
        return {
            view.name: view.dimension(self.spec.embedding_dim, self.depth) for view in self.views
        }

    def encode(self, frames: Iterable[np.ndarray]) -> dict[str, np.ndarray]:
        """Patch features for one batch, as `{view name: (B, P, D_view)}` float32.

        `no_grad` rather than `inference_mode` for one boring reason: an inference-mode
        tensor cannot later be used by anything that records autograd metadata, and this
        model is shared with nothing that does -- but the difference has cost a debugging
        session elsewhere in this repository, so the weaker guarantee is the one taken.
        """
        import torch

        batch = np.stack([np.ascontiguousarray(frame.transpose(2, 0, 1)) for frame in frames])
        tensor = torch.from_numpy(batch).to(self.device)
        tensor = (tensor - self.mean) / self.std
        with torch.no_grad():
            maps = self.model.forward_intermediates(
                tensor,
                indices=list(self.indices),
                norm=self.final_norm,
                output_fmt="NCHW",
                intermediates_only=True,
            )
            stack = torch.stack(maps, dim=1).flatten(3)
            out: dict[str, np.ndarray] = {}
            for view in self.views:
                features = view.apply(stack, self.positions, self.depth)
                out[view.name] = features.transpose(1, 2).float().cpu().numpy()
        return out
