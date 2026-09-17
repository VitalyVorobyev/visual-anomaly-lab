"""Did the anomaly map fire on the defect, or somewhere else on the frame?

Three primitives, numpy only and deliberately dependency-free, so the rule can be read and
tested without a database, a model or a file on disk. The policy — which images qualify,
which mask counts as truth, what a sample's verdict is — lives in `eval/runner.py`; what is
here is only the geometry.

**It is the map's peak, not the score's location.** A stored map has been blurred, upsampled
from the patch grid and projected back into source coordinates, so its argmax sits near the
grid cell that produced the score rather than exactly on it — and a method that scores at a
percentile below 100 is not reading a single cell at all. The verdict is therefore a
statement about *the picture the reviewer is looking at*, which is the only thing that can be
checked against a region drawn by hand.
"""

from __future__ import annotations

import math

import numpy as np


def peak_of(array: np.ndarray) -> tuple[int, int] | None:
    """`(x, y)` of the map's largest finite value, or `None` if it has none.

    NaN means "not observed" everywhere in this app — a region-prepared map is NaN outside
    the selected crop — so the maximum is taken over the covered pixels alone. A map that is
    entirely NaN has no peak, and saying so is the honest answer; `(0, 0)` would be a
    coordinate nobody measured.

    Returned as `(x, y)` rather than numpy's `(row, column)` because every consumer of this
    number — the stored columns, the viewer, the window test below — speaks in image
    coordinates, and one flipped pair at the boundary is a bug that looks like a
    near-miss.
    """
    values = np.asarray(array, dtype=np.float32)
    # `write_map` rejects anything that is not 2-D, so a stray shape here is not a map at
    # all; reporting no peak keeps a corrupt file from being handed on as a coordinate.
    if values.ndim != 2 or values.size == 0:
        return None
    if bool(np.isnan(values).all()):
        return None
    row, column = np.unravel_index(int(np.nanargmax(values)), values.shape)
    return int(column), int(row)


def tolerance_px(width: int, height: int, fraction: float) -> int:
    """A pixel radius from a fraction of the image diagonal, never below 1.

    A fraction rather than a fixed pixel count, because the map's true resolution is not the
    image's: a DINO-family method scores on a patch grid whose stride is 14 or 16 source
    pixels before any upsampling, so "within 8 px" means *inside the same patch* on a small
    frame and *four patches away* on a large one. Scaling with the diagonal keeps one
    configured number comparable across datasets of different sizes.

    The floor of 1 is why a tolerance of exactly zero is still a 3x3 window: a radius of 0
    would make the verdict turn on a single pixel of a bilinearly resampled map, which is
    below the resolution the map actually carries.
    """
    return max(1, round(fraction * math.hypot(width, height)))


def hits(mask: np.ndarray, peak: tuple[int, int], radius: int) -> bool:
    """Is any true pixel of `mask` within `radius` (Chebyshev) of `peak`?

    A square window rather than a disc: the difference is the corners of a box whose half-
    width is already a tolerance, and a disc would buy a rounder number at the cost of a
    rule that cannot be read off the slice. The window is clamped at the frame's edges, so a
    peak in a corner tests the part of the window that exists rather than wrapping — a
    negative start index would silently select from the far side of the image.

    A peak outside the mask's frame is a miss, not an error: the two can disagree in size
    when an imported mask was published at a different resolution, and the caller scales
    before it gets here.
    """
    if mask.ndim != 2 or mask.size == 0:
        return False
    x, y = peak
    height, width = mask.shape
    if not (0 <= x < width and 0 <= y < height):
        return False
    window = mask[max(0, y - radius) : y + radius + 1, max(0, x - radius) : x + radius + 1]
    # `np.bool_` is not `bool`, and it reaches SQLite and pydantic as neither.
    return bool(window.any())
