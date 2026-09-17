"""The SubspaceAD campaign's arithmetic, checked without torch.

The method is a covariance, an eigendecomposition and a subtraction, and the campaign's
whole cost argument rests on two claims about that arithmetic: that sweeping tau is exactly
free, and that sweeping rho is exactly free. "Exactly" is a testable word, so it is tested
here against the definitions the paper writes down rather than against a previous run of the
same code.

No torch, deliberately. These tests run in CI's torch-free backend job, which is where a
reproduction gate belongs: a verdict that can only be re-checked on a machine with the
optional deep-learning extra installed is a verdict most checkouts have to take on trust.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from research.subspace_ad.benchmarks import (
    BenchmarkMissingError,
    gkn_splits,
    mvtec_splits,
)
from research.subspace_ad.features import LAYER_BANDS, Aggregation, FeatureView, LayerBand
from research.subspace_ad.maps import (
    gaussian_blur,
    gaussian_kernel,
    pixel_map,
    upsample_bilinear,
)
from research.subspace_ad.subspace import (
    CovarianceAccumulator,
    fit_subspace,
    residual_basis,
    tail_value_at_risk,
)


def _sample(rows: int = 400, dimension: int = 24, seed: int = 7) -> np.ndarray:
    """Patch-like features: a genuinely low-rank signal plus small isotropic noise."""
    rng = np.random.default_rng(seed)
    latent = rng.normal(size=(rows, 5))
    basis = rng.normal(size=(5, dimension))
    return (latent @ basis + 0.05 * rng.normal(size=(rows, dimension)) + 3.0).astype(np.float32)


def _fitted(data: np.ndarray) -> tuple[CovarianceAccumulator, np.ndarray]:
    accumulator = CovarianceAccumulator(data.shape[1])
    accumulator.add(data)
    return accumulator, data


def test_streaming_covariance_matches_a_single_shot_one() -> None:
    """Folding batches in one at a time is the same matrix as computing it all at once."""
    data = _sample()
    streamed = CovarianceAccumulator(data.shape[1])
    for start in range(0, data.shape[0], 37):
        streamed.add(data[start : start + 37])
    mean, covariance, count = streamed.snapshot()

    assert count == data.shape[0]
    np.testing.assert_allclose(mean, data.mean(axis=0), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(covariance, np.cov(data, rowvar=False), rtol=1e-5, atol=1e-6)


def test_snapshot_does_not_consume_the_accumulator() -> None:
    """Nested k-shot arms depend on this: one pass, a snapshot as it crosses each k."""
    data = _sample()
    accumulator = CovarianceAccumulator(data.shape[1])
    accumulator.add(data[:100])
    _, first, first_count = accumulator.snapshot()
    accumulator.add(data[100:])
    _, second, second_count = accumulator.snapshot()

    assert (first_count, second_count) == (100, data.shape[0])
    np.testing.assert_allclose(first, np.cov(data[:100], rowvar=False), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(second, np.cov(data, rowvar=False), rtol=1e-5, atol=1e-6)


def test_residual_at_rank_equals_the_papers_projection_formula() -> None:
    """`total - cumsum(a^2)` is the same number as `||(x-mu) - C C^T (x-mu)||^2`.

    This identity is what makes the tau axis free, so it is checked against Eq. 5 and 6
    written out longhand rather than against another arrangement of the same shortcut.
    """
    data = _sample()
    accumulator, _ = _fitted(data)
    fit = fit_subspace(accumulator)
    probes = _sample(rows=50, seed=11)
    basis = residual_basis(fit, probes, rank=fit.available_rank)

    centred = probes.astype(np.float64) - fit.mean
    for rank in (1, 3, 9, fit.available_rank):
        components = fit.components[:rank].astype(np.float64)
        projected = centred @ components.T @ components
        longhand = np.einsum("nd,nd->n", centred - projected, centred - projected)
        np.testing.assert_allclose(basis.at_rank(rank), longhand, rtol=1e-4, atol=1e-4)


def test_every_retained_component_is_orthonormal() -> None:
    """The identity above holds only for an orthonormal basis; `eigh` is why it does."""
    accumulator, _ = _fitted(_sample())
    fit = fit_subspace(accumulator)
    gram = fit.components.astype(np.float64) @ fit.components.astype(np.float64).T
    np.testing.assert_allclose(gram, np.eye(gram.shape[0]), rtol=0, atol=1e-5)


def test_eigenvalues_descend_and_rank_follows_the_variance_threshold() -> None:
    accumulator, data = _fitted(_sample())
    fit = fit_subspace(accumulator)

    assert np.all(np.diff(fit.eigenvalues) <= 1e-9)
    assert fit.rank_for(0.5) <= fit.rank_for(0.9) <= fit.rank_for(0.99)
    # Five latent directions carry essentially all of the variance by construction.
    assert fit.rank_for(0.95) <= 5
    assert fit.dimension == data.shape[1]


def test_full_rank_destroys_the_score_which_is_what_tau_of_one_measures() -> None:
    """The paper's tau=1.00 row collapses to chance, and this is the mechanism."""
    accumulator, _ = _fitted(_sample())
    fit = fit_subspace(accumulator)
    assert fit.rank_for(1.0) == fit.dimension

    probes = _sample(rows=50, seed=11)
    basis = residual_basis(fit, probes, rank=fit.available_rank)
    scores = basis.at_rank(fit.dimension)
    spread = float(np.ptp(basis.at_rank(5)))
    assert float(np.abs(scores).max()) < spread * 1e-3


def test_a_threshold_never_asks_for_more_directions_than_the_space_has() -> None:
    """The cumulative fraction reaches 1.0 only up to rounding, and tau=1.0 sits on it.

    On a real covariance the tail eigenvalues are small negatives, clipped to zero, and the
    normalized cumulative sum lands a few ulps below one. Without the clamp, tau=1.0 selects
    rank D+1 and every arm at that threshold fails instead of collapsing -- which is the
    behaviour Table 4 is about.
    """
    data = _sample(rows=120, dimension=40, seed=19)
    accumulator, _ = _fitted(data)
    fit = fit_subspace(accumulator)

    assert fit.rank_for(1.0) == fit.dimension
    basis = residual_basis(fit, data[:5], rank=fit.rank_for(1.0))
    assert basis.at_rank(fit.rank_for(1.0)).shape == (5,)


def test_rank_zero_is_the_distance_from_the_mean() -> None:
    accumulator, _ = _fitted(_sample())
    fit = fit_subspace(accumulator)
    probes = _sample(rows=20, seed=3)
    basis = residual_basis(fit, probes, rank=4)
    centred = probes.astype(np.float64) - fit.mean
    np.testing.assert_allclose(
        basis.at_rank(0), np.einsum("nd,nd->n", centred, centred), rtol=1e-5, atol=1e-5
    )


def test_scoring_beyond_the_kept_rank_is_refused_rather_than_silently_lowered() -> None:
    accumulator, _ = _fitted(_sample())
    fit = fit_subspace(accumulator, max_rank=6)
    with pytest.raises(ValueError, match="refit with a larger max_rank"):
        residual_basis(fit, _sample(rows=5, seed=2), rank=7)


def test_trimming_the_basis_leaves_the_threshold_comparing_against_the_whole_spectrum() -> None:
    """A truncated fit must not also truncate the denominator tau is a fraction of."""
    accumulator, _ = _fitted(_sample())
    full = fit_subspace(accumulator)
    trimmed = fit_subspace(accumulator, max_rank=6)

    assert trimmed.available_rank == 6
    assert trimmed.eigenvalues.size == full.eigenvalues.size
    assert trimmed.rank_for(0.95) == full.rank_for(0.95)


def test_tail_value_at_risk_is_the_mean_of_the_top_fraction() -> None:
    rng = np.random.default_rng(1)
    scores = rng.normal(size=(32, 32))
    results = tail_value_at_risk(scores, [0.01, 0.1, 1.0])

    flat = np.sort(scores.ravel())[::-1]
    assert results[0.01] == pytest.approx(flat[: max(1, round(0.01 * flat.size))].mean())
    assert results[0.1] == pytest.approx(flat[: round(0.1 * flat.size)].mean())
    assert results[1.0] == pytest.approx(flat.mean())
    # A wider tail can only dilute the top of the distribution.
    assert results[0.01] >= results[0.1] >= results[1.0]


def test_tail_value_at_risk_always_takes_at_least_one_patch() -> None:
    """One percent of a small grid rounds to zero, and an empty mean is not a score."""
    scores = np.arange(9, dtype=np.float64).reshape(3, 3)
    assert tail_value_at_risk(scores, [0.01])[0.01] == pytest.approx(8.0)


def test_layer_band_reproduces_the_papers_window_under_both_readings() -> None:
    """Layers 22-28 of DINOv2-G's 40 blocks, as a relative band and as a fixed count."""
    assert LAYER_BANDS["mid_band"].blocks(40) == tuple(range(22, 29))
    assert LAYER_BANDS["mid7"].blocks(40) == tuple(range(22, 29))
    assert LAYER_BANDS["final_band"].blocks(40) == tuple(range(34, 41))
    assert LAYER_BANDS["final7"].blocks(40) == tuple(range(34, 41))
    assert LAYER_BANDS["last"].blocks(40) == (40,)


def test_the_two_readings_diverge_on_a_shallower_encoder_which_is_the_whole_question() -> None:
    """On 12 blocks the same words mean two layers or seven, and the campaign measures which."""
    assert LAYER_BANDS["mid_band"].blocks(12) == (7, 8)
    assert LAYER_BANDS["mid7"].blocks(12) == (2, 3, 4, 5, 6, 7, 8)
    assert LAYER_BANDS["mid_band"].blocks(24) == (14, 15, 16, 17)
    assert LAYER_BANDS["mid7"].blocks(24) == (11, 12, 13, 14, 15, 16, 17)


def test_layer_band_indices_are_zero_based_and_never_leave_the_encoder() -> None:
    band = LayerBand("wide", end_fraction=1.0, fixed_count=99)
    assert band.blocks(12) == tuple(range(1, 13))
    assert band.indices(12) == tuple(range(0, 12))


def test_a_view_names_itself_and_knows_its_own_width() -> None:
    mean_view = FeatureView(LAYER_BANDS["mid7"])
    concat_view = FeatureView(LAYER_BANDS["mid7"], aggregation=Aggregation.CONCAT)
    normalized = FeatureView(LAYER_BANDS["last"], l2_normalize=True)

    assert (mean_view.name, concat_view.name) == ("mid7-mean", "mid7-concat")
    assert normalized.name == "last-mean-l2"
    assert mean_view.dimension(1024, 40) == 1024
    assert concat_view.dimension(1024, 40) == 7168


def test_bilinear_upsampling_preserves_a_constant_and_stays_inside_the_range() -> None:
    grid = np.full((6, 6), 2.5, dtype=np.float32)
    np.testing.assert_allclose(upsample_bilinear(grid, (48, 48)), 2.5, rtol=0, atol=1e-6)

    rng = np.random.default_rng(4)
    noisy = rng.random((8, 8)).astype(np.float32)
    resampled = upsample_bilinear(noisy, (112, 112))
    assert resampled.shape == (112, 112)
    assert float(resampled.min()) >= float(noisy.min()) - 1e-6
    assert float(resampled.max()) <= float(noisy.max()) + 1e-6


def test_bilinear_upsampling_uses_the_half_pixel_convention() -> None:
    """A 1-D ramp doubled: the four samples must be the half-pixel centres, not the corners.

    `align_corners=True` would put 0 and 1 at the ends; this convention puts them a quarter
    of a step inside, which is what keeps a score map aligned with the mask it is scored
    against.
    """
    ramp = np.array([[0.0, 1.0]], dtype=np.float32)
    np.testing.assert_allclose(
        upsample_bilinear(ramp, (1, 4))[0], [0.0, 0.25, 0.75, 1.0], rtol=0, atol=1e-6
    )


def test_gaussian_kernel_is_normalized_symmetric_and_truncated_at_four_sigma() -> None:
    kernel = gaussian_kernel(4.0)
    assert kernel.size == 2 * 16 + 1
    assert float(kernel.sum()) == pytest.approx(1.0, abs=1e-6)
    np.testing.assert_allclose(kernel, kernel[::-1], rtol=0, atol=1e-7)


def test_gaussian_blur_preserves_a_constant_field_including_at_the_border() -> None:
    """Reflection rather than zero padding is the reason the border survives."""
    field = np.full((40, 40), 7.0, dtype=np.float32)
    np.testing.assert_allclose(gaussian_blur(field, 4.0), 7.0, rtol=0, atol=1e-4)


def test_gaussian_blur_is_separable_and_therefore_isotropic() -> None:
    impulse = np.zeros((41, 41), dtype=np.float32)
    impulse[20, 20] = 1.0
    blurred = gaussian_blur(impulse, 3.0)
    np.testing.assert_allclose(blurred, blurred.T, rtol=0, atol=1e-7)
    assert float(blurred.sum()) == pytest.approx(1.0, abs=1e-4)
    assert blurred.argmax() == 20 * 41 + 20


def test_pixel_map_upsamples_before_it_smooths() -> None:
    """Smoothing the token grid first would apply sigma in tokens, not in pixels.

    A single hot token should stay a compact blob: at 14x magnification a sigma of 4 pixels
    is well under one token, so the blob's width is set by the upsample, not by the blur.
    """
    grid = np.zeros((8, 8), dtype=np.float32)
    grid[4, 4] = 1.0
    frame = pixel_map(grid, (112, 112), sigma=4.0)
    above_half = int((frame > 0.5 * frame.max()).sum())
    assert 100 < above_half < 600


def test_the_factored_map_is_the_composition_it_replaced() -> None:
    """`pixel_map` collapses blur-after-upsample into two matmuls; this is the equality.

    It is the load-bearing test of that rewrite. Both halves are separable linear
    operators, so `K(U g U') K'` regroups exactly as `(KU) g (KU)'` -- but "exactly" is a
    statement about real arithmetic, and the point of writing it down is that a wrong
    operator (a transposed axis, a resample built with `=` where the clipped right edge
    needs `+=`) still produces a plausible blurred map. Only the naive composition can
    say it is the *same* map.

    float32 matrix products reassociate, so the agreement is to a few ulps of the peak
    rather than bit-exact.
    """
    rng = np.random.default_rng(11)
    grid = rng.normal(size=(9, 13)).astype(np.float32)
    naive = gaussian_blur(upsample_bilinear(grid, (126, 168)), 4.0)
    factored = pixel_map(grid, (126, 168), sigma=4.0)
    assert factored.shape == naive.shape
    assert np.abs(factored - naive).max() < 1e-5 * np.abs(naive).max()


def test_gkn_reserves_its_fit_pool_so_no_arm_is_scored_on_what_it_was_fitted_to(
    tmp_path: Path,
) -> None:
    root = tmp_path / "GKN"
    for folder, count in (("Good", 60), ("Nick", 4), ("Scratch", 6)):
        directory = root / "Data_GKN" / folder
        directory.mkdir(parents=True)
        for index in range(count):
            Image.new("RGB", (8, 8)).save(directory / f"{index:03d}.png")
    (root / "Data_GKN" / ".DS_Store").write_bytes(b"not an image")

    split = gkn_splits(root)[0]
    assert len(split.fit_pool) == 50
    assert split.anomaly_count == 10
    assert len(split.test_items) == 20
    assert not set(split.fit_pool) & {item.path for item in split.test_items}
    assert split.mask_count == 0
    assert split.rotation_safe


def test_a_missing_benchmark_is_a_named_prerequisite_rather_than_a_path_error(
    tmp_path: Path,
) -> None:
    with pytest.raises(BenchmarkMissingError, match="not unpacked"):
        mvtec_splits(tmp_path / "MVTec-AD")


def test_mvtec_transistor_is_the_one_category_rotation_is_withheld_from(
    tmp_path: Path,
) -> None:
    """Rotating an orientation-sensitive category teaches the subspace that the defect
    is normal."""
    root = tmp_path / "MVTec-AD"
    for category in ("transistor", "bottle"):
        train = root / "images" / "train" / category / "good"
        train.mkdir(parents=True)
        Image.new("RGB", (8, 8)).save(train / "000.png")
        for defect in ("good", "broken"):
            folder = root / "images" / "test" / category / defect
            folder.mkdir(parents=True)
            Image.new("RGB", (8, 8)).save(folder / "000.png")
            if defect != "good":
                masks = root / "masks" / "test" / category / defect
                masks.mkdir(parents=True)
                Image.new("L", (8, 8)).save(masks / "000_mask.png")

    splits = {split.category: split for split in mvtec_splits(root)}
    assert splits["bottle"].rotation_safe
    assert not splits["transistor"].rotation_safe
    assert splits["bottle"].anomaly_count == 1
    assert splits["bottle"].mask_count == 1


def test_an_anomalous_image_without_a_mask_is_a_fault_not_a_quiet_omission(
    tmp_path: Path,
) -> None:
    root = tmp_path / "MVTec-AD"
    train = root / "images" / "train" / "bottle" / "good"
    train.mkdir(parents=True)
    Image.new("RGB", (8, 8)).save(train / "000.png")
    broken = root / "images" / "test" / "bottle" / "broken"
    broken.mkdir(parents=True)
    Image.new("RGB", (8, 8)).save(broken / "000.png")

    with pytest.raises(BenchmarkMissingError, match="has no mask"):
        mvtec_splits(root)
