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
    layers and seven layers respectively. `dino_backbone.LayerBand` expresses both, and
    which one transfers is a question the campaign answers rather than assumes.
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

**Rotations invent pixels, and `subspace.RotationFill` decides what to do about that.** The
augmentation itself is torch-free and therefore lives beside the covariance it feeds; what
stays here is the half that needs an encoder.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from anomaly_lab.media import decode
from anomaly_lab.models.dino_backbone import (
    BACKBONES,
    LAYER_BANDS,
    DinoBackbone,
    LayerBand,
    load_backbone,
    patch_grid,
)
from anomaly_lab.models.subspace import RotationFill, rotations, valid_patches
from anomaly_lab.regions.transform import SpatialTransform

__all__ = [
    "LAYER_BANDS",
    "Aggregation",
    "FeatureView",
    "LayerBand",
    "PatchEncoder",
    "PreparedImage",
    "RotationFill",
    "prepare",
    "rotations",
    "valid_patches",
]
"""Re-exported so a campaign module imports its whole vocabulary from one place. The
definitions themselves live in `anomaly_lab` because the `subspace_ad` plugin needs the same
ones, and the campaign that chose its defaults must not be running different code (ADR-0038).
"""


class Aggregation(StrEnum):
    MEAN = "mean"
    CONCAT = "concat"


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
