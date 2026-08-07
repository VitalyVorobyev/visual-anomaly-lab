"""Rendering anomaly maps and ground-truth masks for display.

Stored maps are raw float32 (ADR-0007). Everything visual — the colormap, the range it is
stretched over, the opacity — is applied here, at view time, so none of it is baked into
data that took minutes to compute.

The colormap is a small table rather than a matplotlib import. matplotlib would be a new
top-level dependency, pulled in for one lookup table, in a backend whose baseline is numpy
and Pillow so that `pixel_reference` runs without the optional deep-learning group.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
from PIL import Image

_LUT_SIZE = 256
_UINT8_MAX = 255

# Perceptually-ordered anchors sampled from `inferno`: monotonic in lightness, so a
# brighter pixel always means a higher score even in greyscale or to a colourblind reader.
_ANCHORS: list[tuple[float, tuple[int, int, int]]] = [
    (0.00, (0, 0, 4)),
    (0.10, (16, 8, 47)),
    (0.20, (51, 7, 89)),
    (0.30, (88, 17, 109)),
    (0.40, (124, 29, 111)),
    (0.50, (159, 42, 99)),
    (0.60, (192, 58, 79)),
    (0.70, (220, 82, 52)),
    (0.80, (240, 116, 22)),
    (0.90, (250, 159, 10)),
    (0.95, (251, 197, 49)),
    (1.00, (252, 255, 164)),
]

ALPHA_GAMMA = 2.0
"""How sharply the overlay's opacity rises with the score. Above 1 the low end fades out
faster than linear, which keeps a map's large quiet regions from veiling the whole part."""

CONTOUR_RGB = (34, 211, 138)
"""Ground-truth contours are drawn in a green that no colormap entry comes close to."""


def _build_lut() -> np.ndarray:
    positions = np.array([position for position, _ in _ANCHORS], dtype=np.float64)
    colours = np.array([colour for _, colour in _ANCHORS], dtype=np.float64)
    grid = np.linspace(0.0, 1.0, _LUT_SIZE)
    return np.stack(
        [np.interp(grid, positions, colours[:, channel]) for channel in range(3)],
        axis=1,
    ).astype(np.uint8)


_LUT = _build_lut()


def read_display_range(maps_dir: Path) -> tuple[float, float] | None:
    """The run-wide display range written by the inference job, if it wrote one."""
    path = maps_dir / "range.json"
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return float(payload["low"]), float(payload["high"])
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None


def render_anomaly_map(
    array: np.ndarray,
    *,
    value_range: tuple[float, float] | None = None,
    size: tuple[int, int] | None = None,
) -> bytes:
    """Colormap one map to an RGBA PNG whose **alpha follows the score**.

    `value_range` should be the *run's* range, not this image's, so two images from one
    experiment are comparable by eye. Falling back to this image's own extremes makes a
    clean part look exactly as alarming as a defective one, so it is only the last resort.

    The alpha channel is what makes this readable, and it was arrived at by looking at the
    result. An opaque colormap laid over a photograph — at any opacity, under any blend
    mode — tints the whole frame, because a map's *low* end is still a colour. The regions
    where the model found nothing then look as processed as the region where it found
    something, which is the opposite of what an overlay is for. Scaling alpha with the
    score instead leaves the source image untouched wherever the score is low, so the
    overlay marks a place rather than covering the picture.

    The curve is deliberately steeper than linear: a map is mostly low values, and a
    linear ramp veils the whole part in a thin wash before the peak is visible at all.
    """
    values = np.asarray(array, dtype=np.float32)
    if value_range is None:
        low, high = float(values.min()), float(values.max())
    else:
        low, high = value_range
    span = high - low if high > low else 1.0

    normalized = np.clip((values - low) / span, 0.0, 1.0)
    indices = (normalized * (_LUT_SIZE - 1)).astype(np.uint8)

    rgba = np.empty((*values.shape, 4), dtype=np.uint8)
    rgba[..., :3] = _LUT[indices]
    rgba[..., 3] = (np.power(normalized, ALPHA_GAMMA) * _UINT8_MAX).astype(np.uint8)

    image = Image.fromarray(rgba, mode="RGBA")
    if size is not None and image.size != size:
        image = image.resize(size, Image.Resampling.BILINEAR)

    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()


def render_mask_contour(
    mask: np.ndarray,
    *,
    size: tuple[int, int] | None = None,
    width: int = 2,
) -> bytes:
    """Draw a mask's outline as a transparent RGBA PNG.

    An outline rather than a filled overlay: a filled ground-truth mask hides the very
    pixels the reader is trying to judge the model's map against.

    The boundary is `mask AND NOT erode(mask)`, with erosion done by shifting and
    AND-ing — a handful of numpy operations, and no morphology dependency for one shape.
    """
    binary = np.asarray(mask, dtype=bool)
    eroded = binary.copy()
    for _ in range(max(1, width)):
        shifted = eroded.copy()
        shifted[1:, :] &= eroded[:-1, :]
        shifted[:-1, :] &= eroded[1:, :]
        shifted[:, 1:] &= eroded[:, :-1]
        shifted[:, :-1] &= eroded[:, 1:]
        # Edge pixels have no neighbour outside the frame, so treat the frame as
        # background: a defect touching the border still gets an outline drawn.
        shifted[0, :] = False
        shifted[-1, :] = False
        shifted[:, 0] = False
        shifted[:, -1] = False
        eroded = shifted

    boundary = binary & ~eroded
    rgba = np.zeros((*binary.shape, 4), dtype=np.uint8)
    rgba[boundary, 0] = CONTOUR_RGB[0]
    rgba[boundary, 1] = CONTOUR_RGB[1]
    rgba[boundary, 2] = CONTOUR_RGB[2]
    rgba[boundary, 3] = _UINT8_MAX

    image = Image.fromarray(rgba, mode="RGBA")
    if size is not None and image.size != size:
        # NEAREST keeps the outline one solid colour instead of fading it to nothing.
        image = image.resize(size, Image.Resampling.NEAREST)

    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()
