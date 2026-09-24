"""The shared few-shot helpers that need no torch: patch-grid refinement, and INSID3's
positional debiasing arithmetic."""

from __future__ import annotations

import numpy as np
import pytest

from anomaly_lab.models.positional import debias, positional_basis, resolved_rank
from anomaly_lab.models.refine import Refinement, refine, upsample


def test_upsampling_keeps_the_corners_and_fills_between() -> None:
    grid = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=np.float32)
    up = upsample(grid, (8, 4))
    assert up.shape == (4, 8)
    assert up[:, 0].max() < 0.2 and up[:, -1].min() > 0.8
    assert np.all(np.diff(up[0]) >= -1e-6)


def test_the_guided_filter_concentrates_the_transition_on_the_image_s_edge() -> None:
    # The patch grid only knows "left: no, right: yes", which bilinear spreads as a ramp.
    # The guided filter is edge-preserving smoothing, not a threshold: in flat regions it
    # keeps the local mean, and across the image's own edge it makes the step.
    image = np.zeros((32, 32, 3), dtype=np.float32)
    image[:, 16:] = 1.0
    grid = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=np.float32)

    soft = refine(grid, image, Refinement.BILINEAR)
    sharp = refine(grid, image, Refinement.GUIDED, radius=4, eps=1e-4)
    assert sharp.min() >= 0.0 and sharp.max() <= 1.0
    soft_step = float((soft[:, 16] - soft[:, 15]).mean())
    sharp_step = float((sharp[:, 16] - sharp[:, 15]).mean())
    assert sharp_step > 3 * soft_step


def _synthetic(seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Features that are content plus a position code living in 4 known directions."""
    rng = np.random.default_rng(seed)
    width, patches = 64, 100
    directions = np.linalg.qr(rng.standard_normal((width, 4)))[0]  # (D, 4)
    position = rng.standard_normal((patches, 4)) @ directions.T * 3.0  # (P, D)
    noise_image = position + 0.05 * rng.standard_normal((patches, width))
    return noise_image, position, directions


def test_the_basis_recovers_the_directions_position_lives_in() -> None:
    noise_image, _, directions = _synthetic(0)
    basis = positional_basis(noise_image, rank=4)
    assert basis.shape == (64, 4)
    captured = np.linalg.norm(basis.T @ directions, axis=0)  # each true direction's reach
    assert captured.min() > 0.99


def test_debiasing_makes_the_same_content_at_two_places_the_same_feature() -> None:
    noise_image, position, _ = _synthetic(1)
    basis = positional_basis(noise_image, rank=4)
    content = np.random.default_rng(2).standard_normal(64)
    here, there = content + position[3], content + position[70]

    def cosine(a: np.ndarray, b: np.ndarray) -> float:
        return float(a @ b / np.linalg.norm(a) / np.linalg.norm(b))

    cleaned = debias(np.stack([here, there]).astype(np.float32), basis)
    assert cosine(cleaned[0], cleaned[1]) > 0.99 > cosine(here, there)
    assert np.linalg.norm(cleaned, axis=1) == pytest.approx([1.0, 1.0], abs=1e-5)


def test_the_rank_is_clipped_to_what_the_grid_can_hold() -> None:
    # INSID3's 500 against a 16x16 grid of a 384-wide encoder: at most half of 256.
    assert resolved_rank(500, patches=256, width=384) == 128
    assert resolved_rank(500, patches=1369, width=1024) == 500
    assert positional_basis(np.ones((2, 8), dtype=np.float32), rank=5).shape == (8, 1)
    empty = positional_basis(np.ones((1, 8), dtype=np.float32), rank=5)
    features = np.ones((3, 8), dtype=np.float32)
    assert empty.shape == (8, 0) and debias(features, empty) is features
