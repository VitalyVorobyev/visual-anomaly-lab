"""Pixel-level evaluation: connected components, the histogram accumulator, AU-PRO.

The point of the histogram design is that it gives the *same* answer as accumulating
every map would, at constant memory. So the tests here compare it against values computed
directly on tiny arrays, not against itself at a different size.
"""

from __future__ import annotations

import numpy as np
import pytest

from anomaly_lab.eval.pixel import PixelAccumulator, connected_regions


def test_a_single_blob_is_one_region() -> None:
    mask = np.zeros((6, 6), dtype=bool)
    mask[1:4, 1:4] = True
    regions = connected_regions(mask)
    assert len(regions) == 1
    assert regions[0].size == 9


def test_two_separated_blobs_are_two_regions() -> None:
    mask = np.zeros((8, 8), dtype=bool)
    mask[0:2, 0:2] = True
    mask[5:7, 5:7] = True
    assert len(connected_regions(mask)) == 2


def test_diagonal_contact_counts_as_connected() -> None:
    """8-connectivity: a one-pixel-wide diagonal scratch is one defect, not many."""
    mask = np.zeros((5, 5), dtype=bool)
    for index in range(5):
        mask[index, index] = True
    assert len(connected_regions(mask)) == 1


def test_an_empty_mask_has_no_regions() -> None:
    assert connected_regions(np.zeros((4, 4), dtype=bool)) == []


def test_a_full_mask_is_one_region() -> None:
    regions = connected_regions(np.ones((3, 3), dtype=bool))
    assert len(regions) == 1
    assert regions[0].size == 9


def test_perfectly_localized_map_scores_one() -> None:
    """A map that is high exactly on the defect and low everywhere else."""
    mask = np.zeros((10, 10), dtype=bool)
    mask[2:5, 2:5] = True
    anomaly_map = np.where(mask, 1.0, 0.0).astype(np.float32)

    accumulator = PixelAccumulator(vmin=0.0, vmax=1.0, bins=256)
    accumulator.add(anomaly_map, mask)

    assert accumulator.roc_auc() == pytest.approx(1.0, abs=1e-6)
    assert accumulator.au_pro() == pytest.approx(1.0, abs=1e-6)


def test_an_inverted_map_scores_zero() -> None:
    mask = np.zeros((10, 10), dtype=bool)
    mask[2:5, 2:5] = True
    anomaly_map = np.where(mask, 0.0, 1.0).astype(np.float32)

    accumulator = PixelAccumulator(vmin=0.0, vmax=1.0, bins=256)
    accumulator.add(anomaly_map, mask)

    assert accumulator.roc_auc() == pytest.approx(0.0, abs=1e-6)


def test_a_constant_map_scores_one_half() -> None:
    mask = np.zeros((8, 8), dtype=bool)
    mask[0:4, 0:4] = True
    anomaly_map = np.full((8, 8), 0.5, dtype=np.float32)

    accumulator = PixelAccumulator(vmin=0.0, vmax=1.0, bins=256)
    accumulator.add(anomaly_map, mask)

    assert accumulator.roc_auc() == pytest.approx(0.5, abs=1e-6)


def test_the_accumulator_matches_a_direct_computation() -> None:
    """The claim the whole design rests on: histograms lose nothing but precision.

    Compared against a directly computed rank-based AUC over every pixel, which is what
    the streaming version exists to avoid having to materialize.
    """
    generator = np.random.default_rng(7)
    mask = np.zeros((32, 32), dtype=bool)
    mask[8:16, 8:20] = True
    anomaly_map = generator.normal(0.0, 1.0, size=(32, 32)).astype(np.float32)
    anomaly_map[mask] += 1.5

    accumulator = PixelAccumulator(
        vmin=float(anomaly_map.min()), vmax=float(anomaly_map.max()), bins=1 << 16
    )
    accumulator.add(anomaly_map, mask)

    from anomaly_lab.eval.metrics import roc_auc

    direct = roc_auc(mask.ravel(), anomaly_map.ravel().astype(np.float64))
    assert direct is not None
    assert accumulator.roc_auc() == pytest.approx(direct, abs=1e-4)


def test_memory_does_not_grow_with_the_number_of_images() -> None:
    """The design constraint, asserted rather than assumed.

    Ten images folded in leave the accumulator exactly as large as one did — the arrays
    are sized by `bins`, never by the data.
    """
    accumulator = PixelAccumulator(vmin=0.0, vmax=1.0, bins=1024)
    mask = np.zeros((16, 16), dtype=bool)
    mask[4:8, 4:8] = True

    for _ in range(10):
        accumulator.add(np.where(mask, 0.9, 0.1).astype(np.float32), mask)

    footprint = accumulator.positive.nbytes + accumulator.negative.nbytes
    footprint += accumulator.region_fraction.nbytes
    assert footprint == 1024 * (8 + 8 + 8)
    assert accumulator.image_count == 10
    assert accumulator.region_count == 10


def test_pro_weights_every_region_equally() -> None:
    """A large region found and a small one missed is not a good result.

    One 100-pixel blob detected perfectly and one 4-pixel blob missed entirely gives a
    pixel-level view dominated by the large one. PRO averages over regions, so it lands
    near 0.5 rather than near 1.
    """
    mask = np.zeros((40, 40), dtype=bool)
    mask[2:12, 2:12] = True  # 100 pixels, will be found
    mask[30:32, 30:32] = True  # 4 pixels, will be missed

    anomaly_map = np.zeros((40, 40), dtype=np.float32)
    anomaly_map[2:12, 2:12] = 1.0

    accumulator = PixelAccumulator(vmin=0.0, vmax=1.0, bins=256)
    accumulator.add(anomaly_map, mask)

    assert accumulator.region_count == 2
    au_pro = accumulator.au_pro()
    assert au_pro is not None
    assert 0.4 < au_pro < 0.65


def test_normal_images_contribute_negative_pixels() -> None:
    """An all-normal image folded in with an empty mask adds no regions and no positives."""
    accumulator = PixelAccumulator(vmin=0.0, vmax=1.0, bins=256)
    mask = np.zeros((8, 8), dtype=bool)
    mask[0:2, 0:2] = True
    accumulator.add(np.where(mask, 0.9, 0.1).astype(np.float32), mask)

    before = int(accumulator.negative.sum())
    accumulator.add(np.full((8, 8), 0.1, dtype=np.float32), np.zeros((8, 8), dtype=bool))

    assert int(accumulator.negative.sum()) == before + 64
    assert accumulator.region_count == 1


def test_no_defect_pixels_at_all_means_no_pixel_metrics() -> None:
    accumulator = PixelAccumulator(vmin=0.0, vmax=1.0, bins=256)
    accumulator.add(np.zeros((8, 8), dtype=np.float32), np.zeros((8, 8), dtype=bool))
    assert accumulator.roc_auc() is None
    assert accumulator.au_pro() is None


def test_mismatched_shapes_are_refused() -> None:
    accumulator = PixelAccumulator(vmin=0.0, vmax=1.0)
    with pytest.raises(ValueError, match="same shape"):
        accumulator.add(np.zeros((8, 8), dtype=np.float32), np.zeros((4, 4), dtype=bool))


def test_uncovered_source_pixels_stay_in_the_denominator_at_the_score_floor() -> None:
    anomaly_map = np.full((8, 8), np.nan, dtype=np.float32)
    anomaly_map[2:6, 2:6] = 0.25
    mask = np.zeros((8, 8), dtype=bool)
    mask[0:2, 0:2] = True
    accumulator = PixelAccumulator(vmin=0.0, vmax=1.0, bins=256)

    accumulator.add(anomaly_map, mask)

    summary = accumulator.summary()
    assert summary["defect_pixels"] == 4
    assert summary["normal_pixels"] == 60
    assert summary["covered_pixel_fraction"] == 0.25
    assert summary["uncovered_defect_pixels"] == 4
    assert summary["uncovered_normal_pixels"] == 44
    assert accumulator.region_count == 1
