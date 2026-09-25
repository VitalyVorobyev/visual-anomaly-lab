"""The generic parity rule the export job and the export-parity gate share (ADR-0034)."""

from __future__ import annotations

import numpy as np
import pytest

from anomaly_lab.deployment.parity import (
    ParityReading,
    compare_outputs,
    portable_score,
    summarize,
)
from anomaly_lab.deployment.schema import (
    MaxReducer,
    PercentileReducer,
    ScalarTensorSpec,
    TensorDtype,
    TensorScore,
    TopKMeanReducer,
)

MAP = np.linspace(0.0, 1.0, 64, dtype=np.float32).reshape(8, 8)


def test_identical_outputs_pass_with_zero_error() -> None:
    reading = compare_outputs(MAP, 0.5, MAP.copy(), 0.5, absolute_tolerance=0, relative_tolerance=0)
    assert reading == ParityReading(0.0, 0.0, map_within=True, score_within=True)
    assert reading.passed


def test_a_map_error_past_both_bounds_fails_and_is_measured() -> None:
    shifted = MAP.copy()
    shifted[3, 4] += 1e-3
    reading = compare_outputs(
        MAP, 0.5, shifted, 0.5, absolute_tolerance=1e-4, relative_tolerance=1e-4
    )
    assert not reading.map_within
    assert reading.score_within
    assert not reading.passed
    assert reading.map_max_absolute_error == pytest.approx(1e-3, rel=1e-3)


def test_the_relative_bound_admits_a_proportional_error_on_a_large_value() -> None:
    large = MAP * 1000.0
    reading = compare_outputs(
        large, 0.0, large * (1 + 5e-5), 0.0, absolute_tolerance=1e-4, relative_tolerance=1e-4
    )
    assert reading.map_within


def test_the_score_is_held_to_the_absolute_bound_alone() -> None:
    """A relative bound on one number would loosen exactly as the score grows."""
    reading = compare_outputs(
        MAP, 1000.0, MAP, 1000.05, absolute_tolerance=1e-4, relative_tolerance=1e-4
    )
    assert reading.map_within
    assert not reading.score_within


def test_a_non_finite_portable_map_fails() -> None:
    broken = MAP.copy()
    broken[0, 0] = np.nan
    reading = compare_outputs(MAP, 0.0, broken, 0.0, absolute_tolerance=1, relative_tolerance=1)
    assert not reading.passed


def test_a_map_of_another_shape_is_refused_rather_than_broadcast() -> None:
    with pytest.raises(ValueError, match="shape"):
        compare_outputs(MAP, 0.0, MAP[:1], 0.0, absolute_tolerance=1, relative_tolerance=1)


def test_host_reducers_read_the_emitted_map() -> None:
    assert portable_score({}, MAP, MaxReducer()) == pytest.approx(1.0)
    assert portable_score({}, MAP, PercentileReducer(percentile=50.0)) == pytest.approx(0.5)
    top = portable_score({}, MAP, TopKMeanReducer(top_k=2))
    assert top == pytest.approx(float(np.sort(MAP.ravel())[-2:].mean()))


def test_a_tensor_score_is_read_from_the_named_graph_output() -> None:
    contract = TensorScore(tensor=ScalarTensorSpec(name="score", dtype=TensorDtype.FLOAT32))
    assert portable_score({"score": np.asarray([0.25])}, MAP, contract) == pytest.approx(0.25)
    with pytest.raises(ValueError, match="no declared score output"):
        portable_score({}, MAP, contract)
    with pytest.raises(ValueError, match="invalid shape"):
        portable_score({"score": np.zeros(2)}, MAP, contract)


def test_a_summary_fails_on_any_failure_and_on_nothing_compared() -> None:
    good = ParityReading(1e-6, 2e-6, map_within=True, score_within=True)
    bad = ParityReading(1e-2, 0.0, map_within=False, score_within=True)

    passing = summarize([good, good])
    assert passing.passed
    assert passing.worst_score_absolute_error == pytest.approx(2e-6)

    failing = summarize([good, bad])
    assert not failing.passed
    assert failing.failures == 1
    assert failing.worst_map_absolute_error == pytest.approx(1e-2)

    assert not summarize([]).passed
