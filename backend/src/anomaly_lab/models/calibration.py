"""Calibrating a few-shot method's foreground probability on its own references.

A few-shot method's map is a probability by construction — a softmax share, a posterior,
a ratio of scores — but not by measurement: nothing ties its 0.5 to the class being there
as often as not, and on a small class most of an image can clear it. The evaluator cuts
every map at one fixed rule (`eval/segmentation.py`), so an uncalibrated map makes that
cut mean something different on every class.

`leave_one_out` measures the scale on the references themselves. Each reference in turn is
held out, the method is fitted on the others and scores it, and the held-out pixels'
(probability, truth) pairs are pooled. A Platt scale — a logistic regression on the
probability's logit, `sigmoid(slope * logit(p) + bias)` — is fitted to them, and the method
applies it to every map it writes and to its presence score.

- **Monotone, so rankings survive.** The scale is refused unless its slope is positive: it
  never reorders two pixels of a run, nor two presence scores, so pixel average precision
  and presence ROC-AUC read the same ranking (up to the evaluator's histogram bins).
- **One reference cannot be left out.** With a single reference there is no model of the
  others to score it with; the scale is the identity, and the fit says so.
- **Bounded.** Pairs are capped at `MAX_CALIBRATION_PIXELS`, evenly spaced within each
  held-out reference and shared equally between them, and the cap is logged.
- **The references' prior, not the queries'.** Every reference shows the class, so the
  scale is fitted to images where the class is present. On images where it is absent the
  scale is, if anything, generous.

numpy only, so every method can use it and it is tested without torch.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from anomaly_lab.models.base import evenly_spaced

MAX_CALIBRATION_PIXELS = 262_144
"""At most this many held-out (probability, truth) pairs fit the scale, over all folds."""
LOGIT_LOW = float(np.finfo(np.float64).tiny)
LOGIT_HIGH = 1.0 - float(np.finfo(np.float64).epsneg)
"""A probability is clipped to `[LOGIT_LOW, LOGIT_HIGH]` before its logit, so 0 and 1 stay
finite. The bounds are float64's own: a wider clip would merge nearly saturated values into
ties, and a presence ranking would move."""
NEWTON_ITERATIONS = 100


class Calibration(StrEnum):
    NONE = "none"
    """The method's own probability, unscaled."""
    LEAVE_ONE_OUT = "leave_one_out"
    """A Platt scale fitted on the references, each scored by a model of the others."""


@dataclass(frozen=True)
class PlattScale:
    """`sigmoid(slope * logit(p) + bias)`; the identity is `slope = 1`, `bias = 0`."""

    slope: float = 1.0
    bias: float = 0.0

    @property
    def is_identity(self) -> bool:
        return self.slope == 1.0 and self.bias == 0.0

    def apply(self, probability: np.ndarray) -> np.ndarray:
        """The scaled map, same shape, `float32`; `NaN` (uncovered pixels) stays `NaN`."""
        values = np.asarray(probability, dtype=np.float32)
        if self.is_identity:
            return values
        z = self.slope * _logit(values.astype(np.float64)) + self.bias
        return np.asarray(_sigmoid(z), dtype=np.float32)

    def apply_score(self, value: float) -> float:
        """One value, in float64; the identity returns it unchanged."""
        if self.is_identity:
            return float(value)
        z = self.slope * _logit(np.array([value], dtype=np.float64)) + self.bias
        return float(_sigmoid(z)[0])

    def to_array(self) -> np.ndarray:
        return np.array([self.slope, self.bias], dtype=np.float64)

    @classmethod
    def from_array(cls, stored: np.ndarray | None) -> PlattScale:
        """What was saved, or the identity for a checkpoint saved before calibration existed."""
        if stored is None:
            return cls()
        slope, bias = (float(value) for value in np.asarray(stored).ravel()[:2])
        return cls(slope=slope, bias=bias)


IDENTITY = PlattScale()


def _logit(probability: np.ndarray) -> np.ndarray:
    clipped = np.clip(probability, LOGIT_LOW, LOGIT_HIGH)
    return np.asarray(np.log(clipped) - np.log1p(-clipped), dtype=np.float64)


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return np.asarray(0.5 * (1.0 + np.tanh(0.5 * z)), dtype=np.float64)


def fit_platt(probability: np.ndarray, truth: np.ndarray) -> PlattScale | None:
    """Platt's logistic fit of `truth` on `logit(probability)`, or `None` if it cannot hold.

    Platt's smoothed targets (`(N+ + 1) / (N+ + 2)` and `1 / (N- + 2)`) keep the fit finite
    when the pairs separate perfectly, which a handful of references easily do. `None` when
    the pairs hold only one class, or the fitted slope is not positive — a scale that would
    reorder the map is not a calibration of it.
    """
    x = _logit(np.asarray(probability, dtype=np.float64).ravel())
    y = np.asarray(truth, dtype=bool).ravel()
    finite = np.isfinite(x)
    x, y = x[finite], y[finite]
    positives = int(np.count_nonzero(y))
    negatives = len(y) - positives
    if positives == 0 or negatives == 0:
        return None
    target = np.where(y, (positives + 1.0) / (positives + 2.0), 1.0 / (negatives + 2.0))

    def loss(slope: float, bias: float) -> float:
        z = slope * x + bias
        # -[t log s(z) + (1 - t) log(1 - s(z))] = log(1 + e^z) - t z, computed stably.
        return float(np.sum(np.logaddexp(0.0, z) - target * z))

    slope, bias = 0.0, float(np.log((positives + 1.0) / (negatives + 1.0)))
    current = loss(slope, bias)
    for _ in range(NEWTON_ITERATIONS):
        p = _sigmoid(slope * x + bias)
        residual = p - target
        weight = np.maximum(p * (1.0 - p), 1e-12)
        gradient = np.array([np.dot(residual, x), residual.sum()])
        hessian = np.array(
            [[np.dot(weight, x * x), np.dot(weight, x)], [np.dot(weight, x), weight.sum()]]
        )
        hessian += 1e-9 * np.eye(2)
        step = np.linalg.solve(hessian, gradient)
        scale = 1.0
        while scale > 1e-10:
            trial = loss(slope - scale * step[0], bias - scale * step[1])
            if trial <= current:
                break
            scale *= 0.5
        else:
            break
        slope, bias = slope - scale * float(step[0]), bias - scale * float(step[1])
        improvement = current - trial
        current = trial
        if improvement <= 1e-10 * max(1.0, abs(current)):
            break
    if not np.isfinite(slope) or not np.isfinite(bias) or slope <= 0.0:
        return None
    return PlattScale(slope=slope, bias=bias)


HeldOut = Callable[[int], tuple[np.ndarray, np.ndarray] | None]
"""Fold `i`: reference `i`'s probability map from a model of the others, and its truth — or
`None` when the others cannot make a model (they hold no pixel of one side)."""


def leave_one_out(
    references: int,
    held_out: HeldOut,
    log: Callable[[str], None],
    *,
    max_pixels: int = MAX_CALIBRATION_PIXELS,
    cancelled: Callable[[], None] | None = None,
) -> PlattScale:
    """The Platt scale fitted on every reference scored by a model of the others.

    The identity, logged, when there is one reference, when no fold could be scored, when the
    pairs show only one class, or when the fit would reorder the map. A reference whose
    fold cannot be built — the others show no pixel of the class, which an absent reference
    beside a single present one does — is skipped and counted.
    """
    if references < 2:
        log(
            f"calibration: {references} reference cannot be left out; the foreground "
            "probability stays unscaled"
        )
        return IDENTITY
    per_fold = max(1, max_pixels // references)
    scores: list[np.ndarray] = []
    truths: list[np.ndarray] = []
    total = 0
    skipped = 0
    for fold in range(references):
        if cancelled is not None:
            cancelled()
        answered = held_out(fold)
        if answered is None:
            skipped += 1
            continue
        probability, truth = answered
        flat_probability = np.asarray(probability, dtype=np.float64).ravel()
        flat_truth = np.asarray(truth, dtype=bool).ravel()
        if flat_probability.shape != flat_truth.shape:
            msg = (
                f"held-out reference {fold}: a map of {flat_probability.size} pixels against "
                f"a truth of {flat_truth.size}"
            )
            raise ValueError(msg)
        total += flat_probability.size
        chosen = np.asarray(evenly_spaced(flat_probability.size, per_fold), dtype=np.int64)
        scores.append(flat_probability[chosen])
        truths.append(flat_truth[chosen])
    if skipped:
        log(
            f"calibration: {skipped} of {references} references could not be left out — the "
            "others show no pixel of one side"
        )
    if not scores:
        log("calibration: no reference could be scored; the foreground probability stays unscaled")
        return IDENTITY
    pooled_scores, pooled_truth = np.concatenate(scores), np.concatenate(truths)
    folds = references - skipped
    if len(pooled_scores) < total:
        log(
            f"calibration: {len(pooled_scores)} of {total} held-out pixels, sampled evenly "
            f"within each of {folds} references (max {max_pixels})"
        )
    scale = fit_platt(pooled_scores, pooled_truth)
    positives = int(np.count_nonzero(pooled_truth))
    if scale is None:
        log(
            f"calibration: no increasing scale fits {len(pooled_scores)} held-out pixels "
            f"({positives} of the class); the foreground probability stays unscaled"
        )
        return IDENTITY
    log(
        f"calibration: leave-one-out over {folds} references, {len(pooled_scores)} "
        f"pixels ({positives} of the class): slope {scale.slope:.4g}, bias {scale.bias:.4g}; "
        f"0.5 now falls at the unscaled probability {_uncalibrated_half(scale):.6g} "
        f"(logit {-scale.bias / scale.slope:.3g})"
    )
    return scale


def _uncalibrated_half(scale: PlattScale) -> float:
    """The unscaled probability the scale maps to 0.5 — where the evaluator's cut now falls."""
    return float(_sigmoid(np.array([-scale.bias / scale.slope]))[0])
