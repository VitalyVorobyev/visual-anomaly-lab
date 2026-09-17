"""The peak-in-mask geometry, with no database, no model and no file on disk.

These three primitives decide whether a defect was *found* or merely scored, so each rule
they encode is pinned here rather than inferred from a metric several layers up.
"""

from __future__ import annotations

import numpy as np

from anomaly_lab.eval.localization import hits, peak_of, tolerance_px


def _mask(height: int, width: int, *points: tuple[int, int]) -> np.ndarray:
    """A boolean mask with `(x, y)` points set."""
    mask = np.zeros((height, width), dtype=bool)
    for x, y in points:
        mask[y, x] = True
    return mask


# ------------------------------------------------------------------- peak_of


def test_a_map_with_no_observed_pixel_has_no_peak() -> None:
    """A fully uncovered map is `None`, never `(0, 0)` — that is a coordinate nobody
    measured, and it would put a marker on the corner of every such image."""
    assert peak_of(np.full((4, 4), np.nan, dtype=np.float32)) is None


def test_an_empty_map_has_no_peak() -> None:
    assert peak_of(np.zeros((0, 0), dtype=np.float32)) is None


def test_the_peak_ignores_the_uncovered_margin() -> None:
    """A region-prepared map is NaN outside its crop; the maximum is over what was seen."""
    array = np.full((6, 8), np.nan, dtype=np.float32)
    array[2:4, 3:6] = 1.0
    array[3, 5] = 9.0

    assert peak_of(array) == (5, 3)


def test_the_peak_is_x_then_y_not_numpy_s_row_then_column() -> None:
    """One flipped pair at this boundary is a bug that looks exactly like a near-miss."""
    array = np.zeros((10, 4), dtype=np.float32)
    array[7, 1] = 5.0

    assert peak_of(array) == (1, 7)


def test_a_map_that_is_not_two_dimensional_reports_no_peak() -> None:
    """`write_map` rejects these, so one reaching here is a corrupt file, not a map."""
    assert peak_of(np.zeros((2, 3, 4), dtype=np.float32)) is None


# -------------------------------------------------------------- tolerance_px


def test_the_tolerance_scales_with_the_diagonal() -> None:
    """A fraction, not a pixel count: the same fraction has to mean the same thing on a
    thumbnail and on a 12-megapixel frame, because the map's real resolution — the patch
    stride — scales with the frame too."""
    assert tolerance_px(300, 400, 0.02) == 10  # hypot = 500
    assert tolerance_px(600, 800, 0.02) == 20
    assert tolerance_px(300, 400, 0.04) == 20


def test_a_tolerance_never_falls_below_one_pixel() -> None:
    """Including a tolerance of exactly zero: a radius of 0 would decide the verdict on a
    single pixel of a bilinearly resampled map, below the resolution the map carries."""
    assert tolerance_px(16, 16, 0.0) == 1
    assert tolerance_px(16, 16, 0.001) == 1


# --------------------------------------------------------------------- hits


def test_a_peak_on_the_region_is_a_hit_at_any_tolerance() -> None:
    assert hits(_mask(20, 20, (10, 10)), (10, 10), 1) is True


def test_the_window_is_inclusive_at_exactly_the_radius_and_excludes_one_more() -> None:
    """Pinned on both axes, because a window built from a half-open slice is off by one in
    exactly one direction and the metric would still look plausible."""
    mask = _mask(30, 30, (15, 15))

    assert hits(mask, (15 + 3, 15), 3) is True
    assert hits(mask, (15 - 3, 15), 3) is True
    assert hits(mask, (15, 15 + 3), 3) is True
    assert hits(mask, (15, 15 - 3), 3) is True

    assert hits(mask, (15 + 4, 15), 3) is False
    assert hits(mask, (15 - 4, 15), 3) is False
    assert hits(mask, (15, 15 + 4), 3) is False
    assert hits(mask, (15, 15 - 4), 3) is False


def test_the_window_is_clamped_at_every_edge_rather_than_wrapping() -> None:
    """A negative slice start selects from the far side of the image, which would score a
    peak in the top-left corner against a defect in the bottom-right."""
    height, width = 12, 12
    far = _mask(height, width, (width - 1, height - 1))
    near = _mask(height, width, (0, 0))

    # Top-left peak with a radius wider than the frame must not reach the far corner.
    assert hits(far, (0, 0), 5) is False
    assert hits(near, (0, 0), 5) is True
    # …and the same from the other three corners.
    assert hits(near, (width - 1, 0), 5) is False
    assert hits(near, (0, height - 1), 5) is False
    assert hits(near, (width - 1, height - 1), 5) is False
    assert hits(far, (width - 1, height - 1), 5) is True


def test_a_peak_outside_the_frame_is_a_miss_rather_than_an_error() -> None:
    """Sizes can disagree when a benchmark published its masks at another resolution."""
    assert hits(_mask(8, 8, (4, 4)), (99, 4), 4) is False
    assert hits(_mask(8, 8, (4, 4)), (4, -1), 4) is False


def test_the_answer_is_a_plain_bool() -> None:
    """`np.bool_` is neither `bool` nor an integer SQLite will store, and pydantic will not
    serialize it. The cast is the contract, so it is asserted rather than assumed."""
    answer = hits(_mask(8, 8, (4, 4)), (4, 4), 1)
    assert type(answer) is bool


def test_an_empty_mask_is_a_miss() -> None:
    assert hits(np.zeros((0, 0), dtype=bool), (0, 0), 3) is False
    assert hits(np.zeros((8, 8), dtype=bool), (4, 4), 3) is False
