"""Turning a patch-grid score map into a pixel map, in numpy and without scipy.

Two operations, both of which the paper specifies and neither of which is free to get
subtly wrong: bilinear upsampling from the token grid to the prepared frame, and a
Gaussian blur at sigma = 4.

They are reimplemented here rather than imported for the reason `eval/metrics.py` gives
for reimplementing ROC-AUC. scipy is not a dependency of this project at all, and torch's
`interpolate` is behind the optional `dl` extra; taking either would mean a map depends on
which extras the machine happened to have. Forty lines of array code keep it identical
everywhere, and keep it testable in the torch-free job.

Like `subspace`, this is shared between the `subspace_ad` plugin and the sweep that set
its defaults, so the localization the campaign measured is the localization the workbench
draws.

**`align_corners=False` is the convention, and the choice is visible.** It is torch's
default and OpenCV's behaviour, and it is the one that treats a pixel as a square with an
area rather than as a point: the source coordinate of output pixel j is
`(j + 0.5) * h/H - 0.5`. The alternative stretches the grid so the outermost token centres
land exactly on the outermost pixels, which shifts every score inward by half a token and
would show up as a systematic localization bias against the ground-truth masks.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

GAUSSIAN_TRUNCATE = 4.0
"""Kernel radius in standard deviations. scipy's default, and past four sigma the omitted
tail is under 1e-4 of the mass -- far below the quantization the pixel metrics apply
anyway."""


def _axis_weights(source: int, target: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Lower index, upper index and upper weight for one resampled axis."""
    positions = (np.arange(target, dtype=np.float64) + 0.5) * (source / target) - 0.5
    positions = np.clip(positions, 0.0, source - 1.0)
    lower = np.floor(positions).astype(np.int64)
    upper = np.minimum(lower + 1, source - 1)
    return lower, upper, (positions - lower).astype(np.float32)


def upsample_bilinear(grid: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Resample a `(h, w)` score map to `(height, width)`.

    A 48x48 grid becoming a 672x672 frame is a 14x magnification, so this is where most of
    an anomaly map's apparent detail actually comes from. Keeping it exact and shared means
    two arms differing only in their encoder are not also differing in their interpolation.
    """
    if grid.ndim != 2:
        msg = f"expected a 2-D score map; got shape {grid.shape}"
        raise ValueError(msg)
    height, width = size
    if height <= 0 or width <= 0:
        msg = f"target size must be positive; got {size}"
        raise ValueError(msg)
    values = np.asarray(grid, dtype=np.float32)
    row_lo, row_hi, row_weight = _axis_weights(values.shape[0], height)
    col_lo, col_hi, col_weight = _axis_weights(values.shape[1], width)

    top = values[row_lo]
    bottom = values[row_hi]
    rows = top + (bottom - top) * row_weight[:, None]
    left = rows[:, col_lo]
    right = rows[:, col_hi]
    resampled: np.ndarray = left + (right - left) * col_weight[None, :]
    return resampled


def gaussian_kernel(sigma: float) -> np.ndarray:
    """A normalized 1-D Gaussian, truncated at `GAUSSIAN_TRUNCATE` standard deviations."""
    if sigma <= 0.0:
        msg = f"sigma must be positive; got {sigma}"
        raise ValueError(msg)
    radius = int(GAUSSIAN_TRUNCATE * sigma + 0.5)
    offsets = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-(offsets**2) / (2.0 * sigma**2))
    normalized: np.ndarray = (kernel / kernel.sum()).astype(np.float32)
    return normalized


def _convolve_rows(values: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """One separable pass along the last axis, reflecting at the edges.

    A loop over the kernel's taps rather than over the image's rows. The two obvious
    alternatives are both worse here: `np.convolve` per row spends more time in Python call
    overhead than in arithmetic, and a `sliding_window_view` matrix product materializes a
    33-deep copy of the whole frame -- sixty megabytes of traffic per axis for a blur that
    touches every one of a sweep's maps.
    """
    radius = kernel.size // 2
    padded = np.pad(values, ((0, 0), (radius, radius)), mode="reflect")
    width = values.shape[-1]
    out = np.zeros_like(values, dtype=np.float32)
    for offset, weight in enumerate(kernel):
        out += weight * padded[:, offset : offset + width]
    return out


def gaussian_blur(values: np.ndarray, sigma: float) -> np.ndarray:
    """Smooth a 2-D map with a separable Gaussian, reflecting at the edges.

    Reflection rather than zero padding, and the difference is not cosmetic: a defect that
    touches the frame edge would be blurred against a border of invented zeros and lose
    part of the score that a defect in the middle keeps. Reflection is scipy's default for
    the same reason.
    """
    if values.ndim != 2:
        msg = f"expected a 2-D map; got shape {values.shape}"
        raise ValueError(msg)
    kernel = gaussian_kernel(sigma)
    horizontal = _convolve_rows(np.asarray(values, dtype=np.float32), kernel)
    return _convolve_rows(np.ascontiguousarray(horizontal.T), kernel).T


@lru_cache(maxsize=16)
def separable_operator(source: int, target: int, sigma: float) -> np.ndarray:
    """The `(target, source)` matrix that upsamples one axis and then blurs it.

    Built by pushing the identity through the very primitives above, so it cannot drift
    away from them: column *i* is what a blurred upsample does to a map that is one at
    token *i* and zero everywhere else.

    Cached because a sweep asks for the same `(48, 672, 4.0)` operator for every image of
    every arm of a category, and building it costs more than applying it. The result is
    shared, so callers must treat it as read-only.
    """
    lower, upper, weight = _axis_weights(source, target)
    resample = np.zeros((target, source), dtype=np.float32)
    rows = np.arange(target)
    np.add.at(resample, (rows, lower), 1.0 - weight)
    np.add.at(resample, (rows, upper), weight)
    smoothed = _convolve_rows(np.ascontiguousarray(resample.T), gaussian_kernel(sigma)).T
    operator: np.ndarray = np.ascontiguousarray(smoothed)
    return operator


def pixel_map(grid: np.ndarray, size: tuple[int, int], *, sigma: float) -> np.ndarray:
    """The paper's localization map: upsample to the frame, then smooth.

    In that order. Smoothing the token grid first and upsampling afterwards would apply a
    sigma of 4 *tokens* -- at 672 px with a 14-pixel patch, a blur fifty-six pixels wide
    instead of four.

    Written as two matrix products rather than as that composition, because both halves
    are separable and linear and therefore collapse: upsampling is `U g U'` and blurring
    is `K v K'`, so the whole map is `(K U) g (K U)'` with one `(672, 48)` operator per
    axis. The arithmetic is the same one -- a test pins it against the composition -- but
    the cost is not. The composition walks a 672x672 frame once per kernel tap, 66
    memory-bound passes over 1.8 MB, and measures 6.0 ms per image; the factored form is
    23 MFLOP of cached BLAS at 0.04 ms. A sweep calls this once per image per arm, which
    on one VisA category is ninety arms over two hundred images, so the category's metric
    phase falls from about two minutes to thirty seconds. Worth having, and no more than
    that: the forward pass it accompanies costs eighty.
    """
    if grid.ndim != 2:
        msg = f"expected a 2-D score map; got shape {grid.shape}"
        raise ValueError(msg)
    height, width = size
    if height <= 0 or width <= 0:
        msg = f"target size must be positive; got {size}"
        raise ValueError(msg)
    values = np.ascontiguousarray(grid, dtype=np.float32)
    rows = separable_operator(values.shape[0], height, sigma)
    columns = separable_operator(values.shape[1], width, sigma)
    mapped: np.ndarray = rows @ values @ columns.T
    return mapped
