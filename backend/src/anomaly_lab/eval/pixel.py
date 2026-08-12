"""Pixel-level evaluation, at constant memory.

ADR-0011 said image-level metrics only, on the honest ground that the dataset in hand had
no masks. The public reference datasets do, so this exists — and the constraint it was
built under is the interesting part.

**A hundred test images at 1.5 MPix in float32 is ~600 MB if the maps are accumulated to
compute a curve**, and a comparison view over several experiments would multiply that.
So nothing is accumulated. Each image is folded into fixed-bin histograms and discarded,
and every curve is read back out of those histograms:

  * pixel ROC-AUC comes from a positive and a negative score histogram;
  * AU-PRO comes from one further accumulator holding, per bin, the sum over ground-truth
    regions of that region's *fraction* of pixels in the bin. Dividing by the region size
    before accumulating is what collapses "one histogram per region" into one array, and
    it is exact rather than an approximation of the per-region computation.

Memory is therefore constant in the number of test images and in their resolution. The
only approximation is the score quantization, at 65536 bins across the observed range.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

DEFAULT_BINS = 1 << 16
PRO_MAX_FPR = 0.3
"""AU-PRO integrates to 30% false-positive rate, the convention every paper reports."""


def _bin_indices(values: np.ndarray, vmin: float, vmax: float, bins: int) -> np.ndarray:
    span = vmax - vmin
    if span <= 0:
        return np.zeros(values.shape, dtype=np.int64)
    scaled = (values.astype(np.float64) - vmin) / span * (bins - 1)
    return np.clip(scaled, 0, bins - 1).astype(np.int64)


def connected_regions(mask: np.ndarray) -> list[np.ndarray]:
    """8-connected components of a boolean mask, as flat index arrays.

    Union-find over the positive pixels only. A defect mask is sparse — thousands of
    pixels in a megapixel image — so working over `nonzero` rather than the full grid is
    what keeps this from dominating the evaluation.
    """
    flat_positions = np.flatnonzero(mask.ravel())
    if flat_positions.size == 0:
        return []

    height, width = mask.shape
    lookup = np.full(mask.size, -1, dtype=np.int64)
    lookup[flat_positions] = np.arange(flat_positions.size)

    rows, columns = np.divmod(flat_positions, width)
    parent = np.arange(flat_positions.size, dtype=np.int64)

    def find(node: int) -> int:
        root = node
        while parent[root] != root:
            root = int(parent[root])
        while parent[node] != root:  # path compression, iterative
            parent[node], node = root, int(parent[node])
        return root

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    # Four offsets suffice for 8-connectivity when every pixel is visited: the other four
    # are the same edges seen from the opposite end.
    for row_step, column_step in ((0, 1), (1, 0), (1, 1), (1, -1)):
        neighbour_rows = rows + row_step
        neighbour_columns = columns + column_step
        inside = (neighbour_rows < height) & (neighbour_columns >= 0) & (neighbour_columns < width)
        if not inside.any():
            continue
        neighbour_flat = neighbour_rows[inside] * width + neighbour_columns[inside]
        neighbour_index = lookup[neighbour_flat]
        linked = neighbour_index >= 0
        for source, target in zip(
            np.flatnonzero(inside)[linked], neighbour_index[linked], strict=True
        ):
            union(int(source), int(target))

    grouped: dict[int, list[int]] = {}
    for index in range(flat_positions.size):
        grouped.setdefault(find(index), []).append(index)
    return [flat_positions[np.array(members)] for members in grouped.values()]


@dataclass
class PixelAccumulator:
    """Streams anomaly maps and masks into the histograms every pixel curve needs."""

    vmin: float
    vmax: float
    bins: int = DEFAULT_BINS

    positive: np.ndarray = field(init=False)
    negative: np.ndarray = field(init=False)
    region_fraction: np.ndarray = field(init=False)
    region_count: int = field(default=0, init=False)
    image_count: int = field(default=0, init=False)
    total_pixels: int = field(default=0, init=False)
    uncovered_defect_pixels: int = field(default=0, init=False)
    uncovered_normal_pixels: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self.positive = np.zeros(self.bins, dtype=np.int64)
        self.negative = np.zeros(self.bins, dtype=np.int64)
        self.region_fraction = np.zeros(self.bins, dtype=np.float64)

    def add(self, anomaly_map: np.ndarray, mask: np.ndarray) -> None:
        """Fold one image in. `anomaly_map` and `mask` must already be the same shape."""
        if anomaly_map.shape != mask.shape:
            msg = f"map {anomaly_map.shape} and mask {mask.shape} must have the same shape"
            raise ValueError(msg)

        valid = np.isfinite(anomaly_map)
        uncovered = ~valid
        self.total_pixels += int(valid.size)
        self.uncovered_defect_pixels += int((uncovered & mask).sum())
        self.uncovered_normal_pixels += int((uncovered & ~mask).sum())

        # A crop is part of the experiment, not a reason to shrink the denominator.
        # Pixels the method could not see receive the run floor: missed defect pixels
        # therefore become false negatives, while uncovered background remains negative.
        covered_values = np.where(valid, anomaly_map, self.vmin)
        indices = _bin_indices(covered_values, self.vmin, self.vmax, self.bins)
        flat_indices = indices.ravel()
        flat_mask = mask.ravel()

        self.positive += np.bincount(flat_indices[flat_mask], minlength=self.bins)
        self.negative += np.bincount(flat_indices[~flat_mask], minlength=self.bins)

        for region in connected_regions(mask):
            counts = np.bincount(flat_indices[region], minlength=self.bins)
            self.region_fraction += counts / region.size
            self.region_count += 1

        self.image_count += 1

    @property
    def has_both_classes(self) -> bool:
        return bool(self.positive.sum() > 0 and self.negative.sum() > 0)

    def _rates(self) -> tuple[np.ndarray, np.ndarray]:
        """`(fpr, tpr)` at every bin edge, ordered by increasing false-positive rate.

        Reversed cumulative sums: thresholding at bin `k` means "score in bin k or above",
        so sweeping k downward sweeps the threshold downward and both rates upward.
        """
        total_positive = self.positive.sum()
        total_negative = self.negative.sum()
        tpr = np.cumsum(self.positive[::-1]) / total_positive
        fpr = np.cumsum(self.negative[::-1]) / total_negative
        return np.r_[0.0, fpr, 1.0], np.r_[0.0, tpr, 1.0]

    def roc_auc(self) -> float | None:
        """Pixel-level ROC-AUC, or `None` when a class is absent from every mask."""
        if not self.has_both_classes:
            return None
        fpr, tpr = self._rates()
        return float(np.trapezoid(tpr, fpr))

    def au_pro(self, max_fpr: float = PRO_MAX_FPR) -> float | None:
        """Area under the per-region-overlap curve, normalized over `[0, max_fpr]`.

        PRO weights every ground-truth region equally regardless of its size, which is
        the point: a metric dominated by the one large defect says nothing about whether
        the small ones were found at all.
        """
        if self.region_count == 0 or self.negative.sum() == 0:
            return None

        pro = np.cumsum(self.region_fraction[::-1]) / self.region_count
        fpr = np.cumsum(self.negative[::-1]) / self.negative.sum()
        pro = np.r_[0.0, pro]
        fpr = np.r_[0.0, fpr]

        within = fpr <= max_fpr
        if not within.any():
            return None
        clipped_fpr = fpr[within]
        clipped_pro = pro[within]

        # Interpolate the final point so the integration limit is exactly `max_fpr`,
        # rather than the last bin edge that happened to fall below it.
        if clipped_fpr[-1] < max_fpr and within.sum() < fpr.size:
            next_index = int(within.sum())
            span = fpr[next_index] - clipped_fpr[-1]
            weight = (max_fpr - clipped_fpr[-1]) / span if span > 0 else 0.0
            interpolated = clipped_pro[-1] + weight * (pro[next_index] - clipped_pro[-1])
            clipped_fpr = np.r_[clipped_fpr, max_fpr]
            clipped_pro = np.r_[clipped_pro, interpolated]

        return float(np.trapezoid(clipped_pro, clipped_fpr) / max_fpr)

    def summary(self) -> dict[str, float | int | None]:
        uncovered = self.uncovered_defect_pixels + self.uncovered_normal_pixels
        return {
            "pixel_roc_auc": self.roc_auc(),
            "au_pro": self.au_pro(),
            "mask_images": self.image_count,
            "mask_regions": self.region_count,
            "defect_pixels": int(self.positive.sum()),
            "normal_pixels": int(self.negative.sum()),
            "covered_pixel_fraction": (
                1.0 - uncovered / self.total_pixels if self.total_pixels else None
            ),
            "uncovered_defect_pixels": self.uncovered_defect_pixels,
            "uncovered_normal_pixels": self.uncovered_normal_pixels,
        }
