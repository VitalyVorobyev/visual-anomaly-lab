"""Turning a coarse foreground probability into one at the prepared image's resolution.

A frozen ViT answers once per patch — every 14 or 16 pixels — so a few-shot method's raw
output is a grid, and its edges are wherever the grid happens to fall. Refinement brings
it to pixel resolution:

- `bilinear` interpolates, and nothing else: the edges stay where the patches put them.
- `guided` interpolates, then runs He et al.'s guided filter with the image as the guide,
  so the probability's edges move to the image's own edges within the filter's radius.

numpy only, so every method can use it and it is tested without torch. A dense CRF is the
common third choice; it waits in the backlog until a maintained package is chosen and a
measurement asks for it.
"""

from __future__ import annotations

from enum import StrEnum

import numpy as np
from PIL import Image

GUIDED_RADIUS = 4
"""The guided filter's window half-width, in prepared pixels."""
GUIDED_EPS = 1e-3
"""The guided filter's regulariser: how strong an image edge must be to move a transition."""


class Refinement(StrEnum):
    BILINEAR = "bilinear"
    GUIDED = "guided"


def upsample(grid: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """`(rows, cols)` to `(height, width)` by bilinear interpolation. `size` is `(w, h)`."""
    image = Image.fromarray(np.asarray(grid, dtype=np.float32), mode="F")
    return np.asarray(image.resize(size, Image.Resampling.BILINEAR), dtype=np.float32)


def _box(values: np.ndarray, radius: int) -> np.ndarray:
    """Mean over a `(2r+1)` square, clipped at the edges, through integral images."""
    padded = np.pad(values, ((1, 0), (1, 0)))
    integral = padded.cumsum(axis=0).cumsum(axis=1)
    height, width = values.shape
    rows = np.arange(height)
    cols = np.arange(width)
    top = np.clip(rows - radius, 0, height)
    bottom = np.clip(rows + radius + 1, 0, height)
    left = np.clip(cols - radius, 0, width)
    right = np.clip(cols + radius + 1, 0, width)
    total = (
        integral[bottom][:, right]
        - integral[top][:, right]
        - integral[bottom][:, left]
        + integral[top][:, left]
    )
    count = (bottom - top)[:, None] * (right - left)[None, :]
    return np.asarray(total / count, dtype=np.float64)


def guided_filter(source: np.ndarray, guide: np.ndarray, radius: int, eps: float) -> np.ndarray:
    """He, Sun and Tang's guided filter with a grey guide (ECCV 2010), in float64."""
    guide = guide.astype(np.float64)
    source = source.astype(np.float64)
    mean_guide = _box(guide, radius)
    mean_source = _box(source, radius)
    covariance = _box(guide * source, radius) - mean_guide * mean_source
    variance = _box(guide * guide, radius) - mean_guide**2
    a = covariance / (variance + eps)
    b = mean_source - a * mean_guide
    return np.asarray(_box(a, radius) * guide + _box(b, radius), dtype=np.float64)


def refine(
    grid: np.ndarray,
    image: np.ndarray,
    method: Refinement,
    *,
    radius: int = GUIDED_RADIUS,
    eps: float = GUIDED_EPS,
) -> np.ndarray:
    """A patch-grid probability at the resolution of `image` (`(H, W, C)` in `[0, 1]`).

    The result stays in `[0, 1]`: the guided filter is a local linear fit and can overshoot.
    """
    height, width = image.shape[:2]
    upsampled = upsample(grid, (width, height))
    if method is Refinement.BILINEAR:
        return np.asarray(np.clip(upsampled, 0.0, 1.0), dtype=np.float32)
    grey = image.mean(axis=2) if image.ndim == 3 else image
    refined = guided_filter(upsampled, grey, radius, eps)
    return np.asarray(np.clip(refined, 0.0, 1.0), dtype=np.float32)
