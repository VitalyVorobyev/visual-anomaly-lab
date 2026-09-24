"""The shared foreground-probability calibration: Platt's fit on synthetic pairs, the
leave-one-out bookkeeping, and the identity it falls back to when it cannot hold."""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pytest

from anomaly_lab.models.calibration import (
    IDENTITY,
    PlattScale,
    fit_platt,
    leave_one_out,
)


def _logistic_pairs(slope: float, bias: float, count: int, seed: int = 0) -> tuple[np.ndarray, ...]:
    """Probabilities whose truth is drawn from `sigmoid(slope * logit(p) + bias)`."""
    rng = np.random.default_rng(seed)
    logit = rng.normal(0.0, 3.0, count)
    probability = 1.0 / (1.0 + np.exp(-logit))
    truth = rng.random(count) < 1.0 / (1.0 + np.exp(-(slope * logit + bias)))
    return probability, truth


def test_platt_recovers_the_scale_that_drew_the_truth() -> None:
    probability, truth = _logistic_pairs(0.6, -2.5, 200_000)
    scale = fit_platt(probability, truth)
    assert scale is not None
    assert scale.slope == pytest.approx(0.6, abs=0.03)
    assert scale.bias == pytest.approx(-2.5, abs=0.06)


def test_a_calibrated_map_is_already_calibrated() -> None:
    probability, truth = _logistic_pairs(1.0, 0.0, 200_000, seed=1)
    scale = fit_platt(probability, truth)
    assert scale is not None
    assert scale.slope == pytest.approx(1.0, abs=0.03)
    assert scale.bias == pytest.approx(0.0, abs=0.03)


def test_perfectly_separated_pairs_still_give_a_finite_increasing_scale() -> None:
    probability = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    truth = probability > 0.5
    scale = fit_platt(probability, truth)
    assert scale is not None
    assert np.isfinite(scale.slope) and scale.slope > 0.0
    assert np.isfinite(scale.bias)
    assert scale.apply_score(0.1) < 0.5 < scale.apply_score(0.9)


def test_pairs_of_one_class_or_a_reversed_ranking_fit_nothing() -> None:
    probability = np.linspace(0.01, 0.99, 50)
    assert fit_platt(probability, np.zeros(50, dtype=bool)) is None
    assert fit_platt(probability, np.ones(50, dtype=bool)) is None
    assert fit_platt(probability, probability < 0.5) is None


def test_the_scale_is_monotone_and_keeps_uncovered_pixels_uncovered() -> None:
    scale = PlattScale(slope=0.5, bias=-3.0)
    values = np.array([0.0, 0.1, 0.5, 0.9, 1.0, np.nan], dtype=np.float32)
    scaled = scale.apply(values)
    assert scaled.dtype == np.float32
    assert np.isnan(scaled[-1])
    assert np.all(np.diff(scaled[:-1]) >= 0)
    assert np.all(np.diff(scaled[1:4]) > 0)
    assert np.all((scaled[:-1] >= 0.0) & (scaled[:-1] <= 1.0))
    # A negative bias pulls 0.5 down: a pixel has to be surer than it was to clear the cut.
    assert scale.apply_score(0.5) < 0.5


def test_the_identity_returns_the_map_untouched_and_survives_storage() -> None:
    values = np.array([0.0, 0.25, 1.0], dtype=np.float32)
    assert IDENTITY.is_identity
    np.testing.assert_array_equal(IDENTITY.apply(values), values)
    assert PlattScale.from_array(None) == IDENTITY
    stored = PlattScale(slope=0.7, bias=-1.25)
    assert PlattScale.from_array(stored.to_array()) == stored


def test_one_reference_cannot_be_left_out() -> None:
    logged: list[str] = []
    calls: list[int] = []

    def held_out(fold: int) -> tuple[np.ndarray, np.ndarray]:
        calls.append(fold)
        return np.zeros(4), np.zeros(4, dtype=bool)

    assert leave_one_out(1, held_out, logged.append) == IDENTITY
    assert calls == []
    assert any("cannot be left out" in line for line in logged), logged


def test_every_reference_is_held_out_once_and_the_pairs_are_bounded() -> None:
    probability, truth = _logistic_pairs(0.8, -1.0, 3 * 10_000, seed=2)
    folds = [(probability[i::3], truth[i::3]) for i in range(3)]
    logged: list[str] = []
    calls: list[int] = []

    def held_out(fold: int) -> tuple[np.ndarray, np.ndarray]:
        calls.append(fold)
        return folds[fold]

    scale = leave_one_out(3, held_out, logged.append, max_pixels=3_000)
    assert calls == [0, 1, 2]
    assert any("3000 of 30000 held-out pixels" in line for line in logged), logged
    assert any("slope" in line for line in logged), logged
    assert not scale.is_identity
    assert scale.slope == pytest.approx(0.8, abs=0.2)


def test_a_fold_that_shows_nothing_to_fit_leaves_the_scale_alone() -> None:
    logged: list[str] = []
    scale = leave_one_out(2, lambda fold: (np.full(8, 0.3), np.zeros(8, dtype=bool)), logged.append)
    assert scale == IDENTITY
    assert any("stays unscaled" in line for line in logged), logged


def test_a_map_and_a_truth_of_different_sizes_are_refused() -> None:
    with pytest.raises(ValueError, match="held-out reference 0"):
        leave_one_out(2, lambda fold: (np.zeros(4), np.zeros(5, dtype=bool)), lambda _: None)


def test_a_fold_the_others_cannot_model_is_skipped_and_counted() -> None:
    probability, truth = _logistic_pairs(0.8, -1.0, 2 * 5_000, seed=3)
    logged: list[str] = []

    def held_out(fold: int) -> tuple[np.ndarray, np.ndarray] | None:
        return None if fold == 1 else (probability[fold::2], truth[fold::2])

    scale = leave_one_out(3, held_out, logged.append)
    assert not scale.is_identity
    assert any("1 of 3 references could not be left out" in line for line in logged), logged
    assert any("leave-one-out over 2 references" in line for line in logged), logged
    logged.clear()
    assert leave_one_out(2, lambda fold: None, logged.append) == IDENTITY
    assert any("no reference could be scored" in line for line in logged), logged


def test_nearly_saturated_scores_keep_their_order() -> None:
    scale = PlattScale(slope=0.23, bias=-5.3)
    scores = [1e-200, 1e-9, 1e-7, 0.5, 1.0 - 1e-7, 1.0 - 1e-9, 1.0 - 1e-13]
    scaled = [scale.apply_score(value) for value in scores]
    assert all(low < high for low, high in pairwise(scaled))
