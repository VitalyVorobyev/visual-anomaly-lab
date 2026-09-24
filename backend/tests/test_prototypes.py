"""FSSDINO's arithmetic, without torch: mask coverage on the patch grid, seeded cosine
k-means, Gram energy, and the mean-times-max decision."""

from __future__ import annotations

import numpy as np
import pytest

from anomaly_lab.models.prototypes import (
    class_maps,
    combined_score,
    foreground_probability,
    gram_energy,
    gram_matrix,
    patch_coverage,
    spherical_kmeans,
)


def _unit(rows: np.ndarray) -> np.ndarray:
    return np.asarray(rows / np.linalg.norm(rows, axis=-1, keepdims=True), dtype=np.float32)


def test_a_patch_is_covered_by_the_share_of_its_pixels_under_the_mask() -> None:
    mask = np.zeros((8, 8), dtype=bool)
    mask[:, :4] = True  # the left half: two of four 4x4 patches entirely
    coverage = patch_coverage(mask, (2, 2))
    assert coverage.shape == (4,)
    assert coverage[0] > 0.5 > coverage[1]


def test_k_means_is_seeded_in_both_directions_and_clips_k() -> None:
    features = _unit(np.random.default_rng(0).standard_normal((200, 16)))
    first = spherical_kmeans(features, 5, iterations=10, rng=np.random.default_rng(1))
    again = spherical_kmeans(features, 5, iterations=10, rng=np.random.default_rng(1))
    other = spherical_kmeans(features, 5, iterations=10, rng=np.random.default_rng(2))
    assert first.shape == (5, 16)
    assert np.array_equal(first, again)
    assert not np.allclose(first, other)
    assert np.linalg.norm(first, axis=1) == pytest.approx(np.ones(5), abs=1e-5)
    assert spherical_kmeans(features[:3], 5, iterations=10, rng=np.random.default_rng(1)).shape == (
        3,
        16,
    )


def test_k_means_finds_two_separated_directions() -> None:
    rng = np.random.default_rng(3)
    north = _unit(np.array([1.0, 0.0, 0.0]) + 0.05 * rng.standard_normal((50, 3)))
    east = _unit(np.array([0.0, 1.0, 0.0]) + 0.05 * rng.standard_normal((50, 3)))
    centroids = spherical_kmeans(np.concatenate([north, east]), 2, iterations=20, rng=rng)
    assert sorted(np.argmax(np.abs(centroids), axis=1).tolist()) == [0, 1]


def test_gram_energy_ranks_patches_that_share_the_class_s_channels() -> None:
    gram = gram_matrix(_unit(np.array([[1.0, 0.0], [0.9, 0.1]])))
    energy = gram_energy(_unit(np.array([[1.0, 0.0], [0.0, 1.0], [0.7, 0.7]])), gram)
    assert energy.min() == 0.0 and energy.max() == 1.0
    assert energy[0] == 1.0 and energy[1] == 0.0


def test_the_class_whose_prototypes_match_wins_the_pixel() -> None:
    # A 2x2 grid: the left column matches the foreground prototype, the right the background.
    fg, bg = np.array([[1.0, 0.0]], dtype=np.float32), np.array([[0.0, 1.0]], dtype=np.float32)
    query = np.array([[1, 0], [0, 1], [1, 0], [0, 1]], dtype=np.float32)
    grid, size = (2, 2), (8, 8)
    fg_score = combined_score(class_maps(query, fg, gram_matrix(fg)), grid, size)
    bg_score = combined_score(class_maps(query, bg, gram_matrix(bg)), grid, size)
    decided = fg_score > bg_score
    assert decided[:, 0].all() and not decided[:, -1].any()

    probability = foreground_probability(fg_score, bg_score)
    assert np.array_equal(probability > 0.5, decided)
    assert foreground_probability(np.array([-1.0]), np.array([-2.0]))[0] == 0.5
