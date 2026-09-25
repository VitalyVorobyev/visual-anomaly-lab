"""The Python-versus-portable parity rule, for every method and every caller (ADR-0034).

One definition of "the exported graph computes what the method computes", used by the export
job on its dataset-free fixture and by `scripts/export-parity-gate.py` on every test image of
a real fit. Both sides of a comparison are handed the same prepared `[0, 1]` NCHW tensor; the
Python side is the method's `portable_reference`, the portable side is the graph's outputs
read through the manifest's score contract. Numpy only: a bundle is checked without torch.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from anomaly_lab.deployment.schema import (
    MaxReducer,
    PercentileReducer,
    ScoreContract,
    TensorScore,
    TopKMeanReducer,
)


@dataclass(frozen=True)
class ParityReading:
    """One input's disagreement between the Python path and the portable runtime."""

    map_max_absolute_error: float
    score_absolute_error: float
    map_within: bool
    score_within: bool

    @property
    def passed(self) -> bool:
        return self.map_within and self.score_within


def compare_outputs(
    expected_map: np.ndarray,
    expected_score: float,
    actual_map: np.ndarray,
    actual_score: float,
    *,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> ParityReading:
    """Map by `allclose(atol, rtol)`, score by `atol` alone.

    The score is held to the absolute bound only because it is one number whose scale the
    method owns, and a relative bound on it would loosen exactly as the score grows. A map
    whose shape differs, or which is not finite where the reference is, fails outright
    rather than broadcasting into a pass.
    """
    expected = np.asarray(expected_map, dtype=np.float32)
    actual = np.asarray(actual_map, dtype=np.float32)
    if expected.shape != actual.shape:
        raise ValueError(
            f"portable map has shape {actual.shape} and the Python map {expected.shape}"
        )
    difference = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
    map_error = float(np.max(difference)) if difference.size else 0.0
    score_error = abs(float(actual_score) - float(expected_score))
    map_within = bool(
        np.allclose(actual, expected, atol=absolute_tolerance, rtol=relative_tolerance)
    )
    return ParityReading(
        map_max_absolute_error=map_error if np.isfinite(map_error) else float("inf"),
        score_absolute_error=score_error if np.isfinite(score_error) else float("inf"),
        map_within=map_within,
        score_within=bool(score_error <= absolute_tolerance),
    )


def portable_score(
    outputs: Mapping[str, np.ndarray],
    anomaly_map: np.ndarray,
    score: ScoreContract,
) -> float:
    """The image score a host resolves from a graph's outputs under the manifest's contract."""
    if isinstance(score, PercentileReducer):
        return float(np.percentile(anomaly_map, score.percentile))
    if isinstance(score, MaxReducer):
        return float(np.max(anomaly_map))
    if isinstance(score, TopKMeanReducer):
        flat = np.asarray(anomaly_map).ravel()
        count = min(score.top_k, flat.size)
        return float(np.partition(flat, flat.size - count)[-count:].mean())
    if isinstance(score, TensorScore):
        if score.tensor.name not in outputs:
            raise ValueError(f"exported graph has no declared score output {score.tensor.name!r}")
        value = np.asarray(outputs[score.tensor.name], dtype=np.float32).squeeze()
        if value.ndim != 0:
            raise ValueError(f"exported score tensor has invalid shape {value.shape}")
        return float(value)
    raise AssertionError(f"unhandled score contract {type(score).__name__}")


@dataclass(frozen=True)
class ParitySummary:
    """Many readings as the handful of numbers a verdict is stated in."""

    inputs: int
    failures: int
    worst_map_absolute_error: float
    worst_score_absolute_error: float

    @property
    def passed(self) -> bool:
        return self.inputs > 0 and self.failures == 0


def summarize(readings: Sequence[ParityReading]) -> ParitySummary:
    """No readings is not a pass: a gate that compared nothing has proven nothing."""
    return ParitySummary(
        inputs=len(readings),
        failures=sum(1 for reading in readings if not reading.passed),
        worst_map_absolute_error=max(
            (reading.map_max_absolute_error for reading in readings), default=0.0
        ),
        worst_score_absolute_error=max(
            (reading.score_absolute_error for reading in readings), default=0.0
        ),
    )
