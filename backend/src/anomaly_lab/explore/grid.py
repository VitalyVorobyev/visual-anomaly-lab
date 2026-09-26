"""The arithmetic of Explore, on a grid of patch features: numpy only.

A frozen encoder turns one image into a `(rows, cols, D)` grid of unit vectors. Everything a
person can ask of that grid by clicking is one of three small computations here, and none of
them needs torch — which is why they are tested in the CI job that installs without the `dl`
extra, and why the resident child only has to produce the grid.

**The frame is fixed per patch size, never per image.** Every image is contain-resized into
a square whose side is `GRID_CELLS` patches — 448 pixels for a /14 encoder, 512 for a /16 —
through the same `SpatialTransform` a region profile pins, so a map projects back into the
source frame by the transform's own inverse. A frame chosen per image would make the patch
size in source pixels a property of the image's aspect, and two images of one dataset would
be seen at two resolutions.

**Only cells over source pixels take part.** Contain-resize pads the short side with edge
pixels, and a padded patch is a smear of the image's border rather than anything in it.
`covered_cells` is the rectangle of cells whose centres fall on the resized image; k-means
and the false-colour PCA are fitted on it alone, so a letterbox cannot claim a cluster or a
principal direction.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from anomaly_lab.models.dino_backbone import BACKBONES, DinoBackbone, FeatureLayers
from anomaly_lab.models.feature_view import pca_to_rgb
from anomaly_lab.regions.transform import SpatialTransform

GRID_CELLS = 32
"""Patches per side of the explore frame: 1024 cells, a forward pass of well under a second
on MPS for ViT-B, and fine enough that a click lands on a part rather than on a quadrant."""

LAYERS = FeatureLayers.LAST_TWO
"""The blocks `dino_memory` reads by default, so what Explore shows is what that method sees."""

MIN_CLUSTERS = 2
MAX_CLUSTERS = 12
DEFAULT_CLUSTERS = 6
KMEANS_ITERATIONS = 40
MAX_POINTS = 32
"""A prompt of more points than this is somebody clicking, not somebody asking."""
MIN_DISPLAY_SPAN = 0.1
"""The narrowest cosine range a similarity map is stretched over (`display_range`)."""
DISPLAY_PERCENTILES = (50.0, 99.0)
"""The percentiles of the covered cells a similarity map is stretched between."""


class ExploreMode(StrEnum):
    SIMILAR = "similar"
    """Cosine similarity to clicked patches, minus similarity to patches marked unlike them."""
    CLUSTERS = "clusters"
    """k-means over the image's own patches: which parts the encoder groups together."""
    PCA = "pca"
    """The three leading principal components as false colour."""


@dataclass(frozen=True)
class CellWindow:
    """The rectangle of grid cells whose centres lie on source pixels, half-open."""

    top: int
    bottom: int
    left: int
    right: int


def frame_side(backbone: DinoBackbone) -> int:
    return GRID_CELLS * BACKBONES[backbone].patch_size


def frame_transform(source_size: tuple[int, int], backbone: DinoBackbone) -> SpatialTransform:
    """The contain-resize from a source image into the explore frame, full image, no crop."""
    side = frame_side(backbone)
    return SpatialTransform.resolve(source_size=source_size, prepared_size=(side, side))


def covered_cells(transform: SpatialTransform, patch: int) -> CellWindow:
    def span(pad: int, size: int, cells: int) -> tuple[int, int]:
        start = max(0, math.ceil(pad / patch - 0.5))
        stop = min(cells, math.ceil((pad + size) / patch - 0.5))
        # A sliver of an image thinner than half a patch still owns the cell it lies in.
        return (start, stop) if stop > start else (min(start, cells - 1), min(start, cells - 1) + 1)

    rows = transform.prepared_height // patch
    cols = transform.prepared_width // patch
    top, bottom = span(transform.pad_top, transform.resized_height, rows)
    left, right = span(transform.pad_left, transform.resized_width, cols)
    return CellWindow(top=top, bottom=bottom, left=left, right=right)


def source_cell(
    transform: SpatialTransform, patch: int, window: CellWindow, point: tuple[float, float]
) -> tuple[int, int]:
    """The `(row, col)` of the cell a source pixel falls in, clamped to the covered window."""
    x, y = transform.source_to_prepared(point)
    row = min(max(math.floor(y / patch), window.top), window.bottom - 1)
    col = min(max(math.floor(x / patch), window.left), window.right - 1)
    return row, col


def similarity(
    features: np.ndarray,
    positives: list[tuple[int, int]],
    negatives: list[tuple[int, int]],
) -> np.ndarray:
    """`max cos(positives) - max cos(negatives)`, clipped to `[0, 1]`, per cell.

    The features are unit vectors (`image_patch_features` normalises them), so a dot product
    *is* the cosine. Max rather than mean over the clicked patches: two clicks on two unlike
    parts ask "anything like either", and a mean would answer "like neither".
    """
    if not positives:
        msg = "similarity needs at least one positive point"
        raise ValueError(msg)
    rows, cols, width = features.shape
    flat = features.reshape(rows * cols, width).astype(np.float32, copy=False)

    def best(cells: list[tuple[int, int]]) -> np.ndarray:
        chosen = flat[[row * cols + col for row, col in cells]]
        return np.asarray((flat @ chosen.T).max(axis=1), dtype=np.float32)

    score = best(positives)
    if negatives:
        score = score - best(negatives)
    return np.asarray(np.clip(score, 0.0, 1.0).reshape(rows, cols), dtype=np.float32)


def kmeans(
    vectors: np.ndarray, k: int, *, seed: int, iterations: int = KMEANS_ITERATIONS
) -> np.ndarray:
    """Seeded k-means++ then Lloyd, pure numpy; labels `0..k-1` ordered by cluster size."""
    return kmeans_fit(vectors, k, seed=seed, iterations=iterations)[0]


def kmeans_fit(
    vectors: np.ndarray, k: int, *, seed: int, iterations: int = KMEANS_ITERATIONS
) -> tuple[np.ndarray, np.ndarray]:
    """`kmeans`, with the centres: `(labels, centres)`, centre `i` being label `i`'s.

    Every random draw comes from one `default_rng(seed)`, so the same seed gives the same
    labels and a different one is free to differ. Labels are renumbered largest cluster first,
    so a colour means "the biggest group" rather than "whichever centre was drawn first".
    """
    count = int(vectors.shape[0])
    if count == 0:
        return np.zeros(0, dtype=np.int64), np.zeros((0, vectors.shape[-1]), dtype=np.float64)
    k = max(1, min(int(k), count))
    data = np.asarray(vectors, dtype=np.float64)
    rng = np.random.default_rng(seed)
    norms = np.einsum("ij,ij->i", data, data)

    centres = np.empty((k, data.shape[1]), dtype=np.float64)
    centres[0] = data[int(rng.integers(count))]
    nearest = np.maximum(norms - 2.0 * data @ centres[0] + centres[0] @ centres[0], 0.0)
    for index in range(1, k):
        total = float(nearest.sum())
        pick = (
            int(rng.integers(count)) if total <= 0.0 else int(rng.choice(count, p=nearest / total))
        )
        centres[index] = data[pick]
        distance = np.maximum(
            norms - 2.0 * data @ centres[index] + centres[index] @ centres[index], 0.0
        )
        nearest = np.minimum(nearest, distance)

    labels = np.full(count, -1, dtype=np.int64)
    for _ in range(iterations):
        distances = (
            norms[:, None] - 2.0 * data @ centres.T + np.einsum("ij,ij->i", centres, centres)
        )
        updated = np.argmin(distances, axis=1)
        if np.array_equal(updated, labels):
            break
        labels = updated
        for index in range(k):
            members = labels == index
            # An empty cluster keeps its centre rather than collapsing to NaN.
            if members.any():
                centres[index] = data[members].mean(axis=0)

    sizes = np.bincount(labels, minlength=k)
    order = np.argsort(-sizes, kind="stable")
    rank = np.empty(k, dtype=np.int64)
    rank[order] = np.arange(k)
    return rank[labels], centres[order]


def cluster_affinity(
    features: np.ndarray, window: CellWindow, k: int, *, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Per-cell soft assignment to the image's k-means clusters, and the hard labels.

    Returns `(affinity, labels)`: a `(rows, cols, k)` float32 softmax of the negative squared
    distance to each centre, and the `(rows, cols)` uint8 `cluster_grid` of its argmax.

    **The soft assignment is what gets drawn.** A cluster index is a name, so projecting the
    label grid can only be nearest-neighbour, and a 32x32 grid drawn that way has patch
    squares for edges. Each cluster's affinity is a quantity: it is bilinear between patch
    centres like any map, and the argmax of the interpolated planes (`store.to_source`) puts a
    boundary where two clusters are equally likely, which runs between patch centres at any
    angle instead of along the grid.

    **The temperature is the typical squared distance between two centres**, so the softmax is
    graded rather than hard: a patch halfway between two centres is about even between them,
    and the interpolated planes cross where the features do. A hard assignment would put the
    crossing at the midpoint between two patch centres — on the patch edge again — and a
    temperature in one encoder's units would be hard on one encoder and flat on another.
    Cells off the image take the affinity of the nearest covered cell: a letterbox is a smear
    of the border and should not pull a boundary toward itself.
    """
    rows, cols, width = features.shape
    inside = features[window.top : window.bottom, window.left : window.right]
    points = inside.reshape(-1, width)
    # The hard labels are the argmax below, which is what the picture draws.
    _, centres = kmeans_fit(points, k, seed=seed)
    data = np.asarray(points, dtype=np.float64)
    distances = np.maximum(
        np.einsum("ij,ij->i", data, data)[:, None]
        - 2.0 * data @ centres.T
        + np.einsum("ij,ij->i", centres, centres),
        0.0,
    )
    nearest = distances.min(axis=1, keepdims=True)
    logits = -(distances - nearest) / _temperature(centres)
    weights = np.exp(logits)
    soft = (weights / weights.sum(axis=1, keepdims=True)).astype(np.float32)

    shape = inside.shape[:2]
    affinity = np.pad(
        soft.reshape(*shape, -1),
        ((window.top, rows - window.bottom), (window.left, cols - window.right), (0, 0)),
        mode="edge",
    )
    grid = np.zeros((rows, cols), dtype=np.uint8)
    grid[window.top : window.bottom, window.left : window.right] = (
        np.argmax(soft, axis=1).reshape(shape) + 1
    ).astype(np.uint8)
    return np.ascontiguousarray(affinity, dtype=np.float32), grid


def _temperature(centres: np.ndarray) -> float:
    """The median squared distance between two distinct centres; 1 when there is no pair."""
    count = int(centres.shape[0])
    if count < 2:
        return 1.0
    norms = np.einsum("ij,ij->i", centres, centres)
    pairs = norms[:, None] - 2.0 * centres @ centres.T + norms[None, :]
    upper = pairs[np.triu_indices(count, k=1)]
    typical = float(np.median(np.maximum(upper, 0.0)))
    return typical if typical > 0.0 else 1.0


def cluster_grid(features: np.ndarray, window: CellWindow, k: int, *, seed: int) -> np.ndarray:
    """A `(rows, cols)` uint8 label grid: 0 off the image, `1..k` over it."""
    return cluster_affinity(features, window, k, seed=seed)[1]


def display_range(values: np.ndarray, window: CellWindow) -> tuple[float, float]:
    """The range a similarity map is coloured over: from this image's median to its top 1 %.

    Coloured on the fixed [0, 1] a similarity map is a faint wash: the clicked patch is 1 and
    most of the image sits in a band well below it, which the heatmap's score-driven alpha
    all but hides. Its own min and max change nothing — the click itself is always the max.
    Stretched from the median of the covered cells (what a typical patch scores against the
    click) to their 99th percentile, everything more like the click than most of the image
    shows, and the few patches most like it saturate. The range is never narrower than
    `MIN_DISPLAY_SPAN`, so a flat map stays flat instead of amplifying noise; the threshold
    that cuts a mask is in absolute cosine and does not move with it.
    """
    inside = values[window.top : window.bottom, window.left : window.right]
    low = float(np.percentile(inside, DISPLAY_PERCENTILES[0]))
    high = float(np.percentile(inside, DISPLAY_PERCENTILES[1]))
    if high - low < MIN_DISPLAY_SPAN:
        high = min(1.0, low + MIN_DISPLAY_SPAN)
        low = high - MIN_DISPLAY_SPAN
    return low, high


def pca_grid(features: np.ndarray, window: CellWindow) -> np.ndarray:
    """`(rows, cols, 3)` false colour in `[0, 1]`, fitted on the covered cells alone."""
    rows, cols, _ = features.shape
    inside = features[window.top : window.bottom, window.left : window.right]
    grid = np.zeros((rows, cols, 3), dtype=np.float32)
    if inside.shape[0] * inside.shape[1] >= 3:
        grid[window.top : window.bottom, window.left : window.right] = pca_to_rgb(
            np.ascontiguousarray(inside.transpose(2, 0, 1))
        )
    return grid
