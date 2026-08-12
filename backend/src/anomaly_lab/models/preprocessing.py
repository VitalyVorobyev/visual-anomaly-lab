"""The model-input bridge: what every method is made to see.

A comparison between two methods only means something if they were shown the same
pixels. Left to themselves the libraries disagree — anomalib resizes to its own default,
a numpy baseline would read the file as it lies on disk — and the resulting difference in
AUROC would partly measure the resize.

Spatial preparation belongs to the experiment's pinned region build. This bridge only
applies the remaining colour policy and verifies that the prepared artifact has the
dimensions frozen into the experiment. A model that decodes an image any other way is a
bug, not a variation.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path

import numpy as np
from PIL import Image
from pydantic import BaseModel, Field

from anomaly_lab.media import decode
from anomaly_lab.schemas import API_MODEL_CONFIG


class ColorMode(StrEnum):
    RGB = "rgb"
    GRAYSCALE = "grayscale"


_UINT8_MAX = 255.0

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
"""The statistics every ImageNet-pretrained backbone was trained under.

Here, and **not** in this module's job. `load_array` decides decode, resize and colour —
what every method is made to see — and stops there. Standardizing those pixels for a
particular pretrained network is a property of *that network*, not of the experiment: a
method whose backbone wants ImageNet statistics is not seeing different pixels from one
that does not, it is applying its own first layer to the same ones.

The constants live here anyway because this is the module about what a method is fed, and
because three plugins now need them. Getting the seam wrong is silent in a specific way:
an unnormalized ImageNet backbone runs, produces maps, and quietly scores features from
outside its training distribution. `efficientad_nets.imagenet_normalize` applies them
inside `forward` for the same reason, which is why that wrapper can hand a `[0, 1]` array
straight to anomalib's EfficientAD — and why PatchCore, whose module does *not* normalize
internally, has to do it itself.
"""


class PreprocessingOptions(BaseModel):
    """The user-selectable model-input policy after spatial preparation."""

    model_config = API_MODEL_CONFIG

    color: ColorMode = Field(
        default=ColorMode.RGB,
        description=(
            "Colour to feed the model. Grayscale datasets are expanded to three "
            "identical channels under 'rgb', which is what pretrained backbones expect."
        ),
    )


class PreprocessingConfig(PreprocessingOptions):
    """Resolved prepared-image contract frozen into an experiment at creation."""

    model_config = API_MODEL_CONFIG

    width: int = Field(
        default=256,
        ge=8,
        le=2048,
        description="Model input width in pixels. EfficientAD is designed around 256.",
    )
    height: int = Field(
        default=256,
        ge=8,
        le=2048,
        description="Model input height in pixels.",
    )

    @property
    def size(self) -> tuple[int, int]:
        """`(width, height)`, in Pillow's order rather than numpy's."""
        return (self.width, self.height)

    @property
    def channels(self) -> int:
        return 3 if self.color is ColorMode.RGB else 1


def load_array(path: Path, config: PreprocessingConfig) -> np.ndarray:
    """Decode one prepared artifact into `(height, width, channels)` float32 in `[0, 1]`.

    No resize is permitted here. Crop, contain-resize and padding were already resolved
    by the pinned region build; doing spatial work again would make the stored transform
    cease to describe what the method actually saw.
    """
    image = decode.load(path)
    if image.size != config.size:
        raise ValueError(
            f"prepared image {path} is {image.size}, expected frozen input {config.size}"
        )
    target_mode = "RGB" if config.color is ColorMode.RGB else "L"
    if image.mode != target_mode:
        image = image.convert(target_mode)

    array = np.asarray(image, dtype=np.float32) / _UINT8_MAX
    if array.ndim == 2:
        array = array[:, :, np.newaxis]
    return array


def to_chw(array: np.ndarray) -> np.ndarray:
    """`(H, W, C)` to `(C, H, W)` — the layout torch wants, contiguous."""
    return np.ascontiguousarray(array.transpose(2, 0, 1))


def load_mask(path: Path, size: tuple[int, int] | None = None) -> np.ndarray:
    """Read a ground-truth mask as a boolean array, optionally resized.

    Any non-zero pixel is defect. Masks arrive as 0/255 PNGs, as 0/1 PNGs, and
    occasionally as palette images; thresholding above zero reads all three correctly
    where a fixed 127 threshold would silently drop the 0/1 encoding entirely.

    The resize uses NEAREST, because interpolating a label map invents labels.
    """
    image = decode.load(path)
    if image.mode != "L":
        image = image.convert("L")
    if size is not None and image.size != size:
        image = image.resize(size, Image.Resampling.NEAREST)
    return np.asarray(image) > 0
