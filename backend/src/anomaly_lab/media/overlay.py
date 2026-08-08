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

PREDICTION_RGB = (129, 140, 248)
"""The model's own claim, in a periwinkle that reads clearly against the ground-truth green.

The pair has to be told apart at a glance, because the whole point of drawing both is to
see where they disagree. It also has to survive being drawn over the colormap: `inferno`
runs black -> deep purple -> red -> orange -> pale yellow, and its purple end is far darker
and more saturated than this, so a filled region never reads as part of the heatmap."""

PREDICTION_FILL_ALPHA = 90
"""Translucent enough to judge the pixels underneath, opaque enough to find. A filled
region is right here for the reason an outline is right for ground truth: the reader is
comparing the model's *extent* against a known one, and an outline of a ragged
threshold crossing is a tangle of contours rather than a shape."""


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
    alpha_follows_score: bool = True,
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

    `alpha_follows_score=False` returns the same colormap fully opaque. That is for a map
    shown **on its own** — a diagnostics panel, not an overlay — where score-driven alpha
    would dissolve the quiet regions into the page background and leave the reader unable
    to tell "the model found nothing here" from "nothing was rendered here".
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
    if alpha_follows_score:
        rgba[..., 3] = (np.power(normalized, ALPHA_GAMMA) * _UINT8_MAX).astype(np.uint8)
    else:
        rgba[..., 3] = _UINT8_MAX

    image = Image.fromarray(rgba, mode="RGBA")
    if size is not None and image.size != size:
        image = image.resize(size, Image.Resampling.BILINEAR)

    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()


def render_rgb_image(array: np.ndarray, *, size: tuple[int, int] | None = None) -> bytes:
    """Render an `(H, W, 3)` float32 array in `[0, 1]` as an opaque PNG.

    No colormap and no range stretch: the `image` diagnostic kind is already a picture —
    a PCA-to-RGB composite of a teacher's features, say — and the three channels mean
    something to whoever produced them. Rescaling would be this layer inventing an
    interpretation. Values outside `[0, 1]` are clipped rather than renormalized, for the
    same reason.
    """
    values = np.clip(np.asarray(array, dtype=np.float32), 0.0, 1.0)
    rgb = (values * _UINT8_MAX).astype(np.uint8)

    image = Image.fromarray(rgb, mode="RGB")
    if size is not None and image.size != size:
        image = image.resize(size, Image.Resampling.BILINEAR)

    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()


def boundary_of(binary: np.ndarray, width: int = 2) -> np.ndarray:
    """The outline of a binary region, as `region AND NOT erode(region)`.

    Erosion is done by shifting and AND-ing — a handful of numpy operations, and no
    morphology dependency for one shape.
    """
    region = np.asarray(binary, dtype=bool)
    eroded = region.copy()
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

    return np.asarray(region & ~eroded, dtype=bool)


def _paint(
    stencil: np.ndarray,
    rgb: tuple[int, int, int],
    alpha: int,
    size: tuple[int, int] | None,
) -> bytes:
    """One flat colour wherever `stencil` is set, transparent everywhere else."""
    rgba = np.zeros((*stencil.shape, 4), dtype=np.uint8)
    rgba[stencil, 0] = rgb[0]
    rgba[stencil, 1] = rgb[1]
    rgba[stencil, 2] = rgb[2]
    rgba[stencil, 3] = alpha

    image = Image.fromarray(rgba, mode="RGBA")
    if size is not None and image.size != size:
        # NEAREST keeps the shape one solid colour instead of fading its edge to nothing.
        image = image.resize(size, Image.Resampling.NEAREST)

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
    """
    binary = np.asarray(mask, dtype=bool)
    return _paint(boundary_of(binary, width), CONTOUR_RGB, _UINT8_MAX, size)


def render_prediction_region(
    array: np.ndarray,
    threshold: float,
    *,
    size: tuple[int, int] | None = None,
    outline: bool = False,
    width: int = 2,
) -> bytes:
    """Draw where the map crosses `threshold`, as the model's own segmentation.

    This is the overlay that answers "how does what the model found differ from what was
    annotated?" — a question the heatmap cannot answer, because a heatmap has no edge. Laid
    under the ground-truth outline, agreement reads as a green line hugging a filled shape
    and disagreement reads as the two coming apart.

    `threshold` is in **map units** and is a *display* decision, not an evaluation one.
    Nothing here feeds a metric: the pixel metrics integrate over every threshold (ADR-0017)
    and the sample-score threshold on the results screen classifies whole samples, not
    pixels. Keeping this parameter explicit and in the map's own units is what stops the
    three being confused for each other.

    Filled by default, outlined on request. Two filled regions would hide each other, so a
    comparison of two experiments' predictions on one sample outlines instead.
    """
    values = np.asarray(array, dtype=np.float32)
    region = values >= threshold
    if outline:
        return _paint(boundary_of(region, width), PREDICTION_RGB, _UINT8_MAX, size)
    return _paint(region, PREDICTION_RGB, PREDICTION_FILL_ALPHA, size)
