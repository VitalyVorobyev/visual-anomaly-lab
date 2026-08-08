"""Threshold-free metrics, implemented rather than imported.

scikit-learn is already in the tree as a transitive dependency of anomalib, but the
evaluation layer must work when the optional deep-learning group is *not* installed —
`pixel_reference` is meant to run on a checkout with numpy and Pillow and nothing else.
These are a few dozen lines each, and having them here means the headline number does not
change depending on which extras someone installed.

Every function returns `None` rather than a number when the input cannot support one: a
subset with no defects has no ROC-AUC, and inventing 0.5 for it would put a fabricated
value on the results screen.
"""

from __future__ import annotations

import numpy as np


def _average_ranks(sorted_values: np.ndarray) -> np.ndarray:
    """Ranks 1..n with ties sharing their mean rank.

    Ties matter here: a model that returns the same score for many images — a saturated
    map, a degenerate fit — would otherwise score an AUC that depends on the sort's
    arbitrary tie order rather than on anything the model did.
    """
    count = sorted_values.size
    ranks = np.arange(1, count + 1, dtype=np.float64)
    start = 0
    while start < count:
        stop = start + 1
        while stop < count and sorted_values[stop] == sorted_values[start]:
            stop += 1
        if stop - start > 1:
            ranks[start:stop] = ranks[start:stop].mean()
        start = stop
    return ranks


def roc_auc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    """Area under the ROC curve, with 1 meaning defect.

    Computed as the Mann-Whitney U statistic, which is exact and tie-aware, rather than
    by integrating a sampled curve.
    """
    labels = np.asarray(labels).astype(bool)
    scores = np.asarray(scores, dtype=np.float64)
    positives = int(labels.sum())
    negatives = int(labels.size - positives)
    if positives == 0 or negatives == 0:
        return None

    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(scores.size, dtype=np.float64)
    ranks[order] = _average_ranks(scores[order])
    rank_sum = float(ranks[labels].sum())
    return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def _cut_points(
    labels: np.ndarray, scores: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """`(true_positives, predicted, threshold)` at one cut per distinct score, descending.

    Cutting per distinct score rather than per row is what keeps every curve here
    independent of the sort's arbitrary tie order, the same concern as in `roc_auc`.

    The third array is the score *at* each cut, and it is what makes a curve readable
    against the threshold slider rather than only against its own other axis. The cut is
    taken at the **last** index carrying a score, so every row at or above that score is
    on the predicted-defect side — the same rule `eval.threshold.classify` applies with
    `score >= threshold`. That correspondence is the whole value of returning it, and it
    is pinned by a test.
    """
    order = np.argsort(-scores, kind="mergesort")
    sorted_labels = labels[order]
    sorted_scores = scores[order]

    # One cut point per distinct score, at the last index carrying that score.
    distinct = np.nonzero(np.diff(sorted_scores))[0]
    cuts = np.r_[distinct, sorted_labels.size - 1]
    return np.cumsum(sorted_labels)[cuts], cuts + 1, sorted_scores[cuts]


def pr_curve(
    labels: np.ndarray, scores: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """`(recall, precision, threshold)` per distinct score, or `None` for one class.

    The arrays `average_precision` has always computed, returned rather than discarded so
    the chart and the scalar come from one implementation and cannot disagree.
    """
    labels = np.asarray(labels).astype(bool)
    scores = np.asarray(scores, dtype=np.float64)
    positives = int(labels.sum())
    if positives == 0 or labels.size == positives:
        return None

    true_positives, predicted, threshold = _cut_points(labels, scores)
    return true_positives / positives, true_positives / predicted, threshold


def average_precision(labels: np.ndarray, scores: np.ndarray) -> float | None:
    """Average precision, summed over distinct score thresholds."""
    curve = pr_curve(labels, scores)
    if curve is None:
        return None
    recall, precision, _ = curve
    previous_recall = np.r_[0.0, recall[:-1]]
    return float(np.sum((recall - previous_recall) * precision))


def roc_curve(labels: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """`(fpr, tpr)` for plotting, one point per distinct score plus the two endpoints.

    `None` for a one-class input, like every other function here. The chance diagonal a
    degenerate input would otherwise produce is a picture of a model that guesses, which
    is not what a subset with no defects in it means.
    """
    labels = np.asarray(labels).astype(bool)
    scores = np.asarray(scores, dtype=np.float64)
    positives = int(labels.sum())
    negatives = int(labels.size - positives)
    if positives == 0 or negatives == 0:
        return None

    true_positives, predicted, _ = _cut_points(labels, scores)
    false_positives = predicted - true_positives
    return (
        np.r_[0.0, false_positives / negatives, 1.0],
        np.r_[0.0, true_positives / positives, 1.0],
    )


def timing_summary(milliseconds: np.ndarray) -> dict[str, float]:
    """Mean, median, p95 and total, in milliseconds.

    Reported because accuracy at any latency is not a comparison — methods here differ by
    orders of magnitude in cost, and that is a first-class part of the answer (§8).
    """
    if milliseconds.size == 0:
        return {}
    values = np.asarray(milliseconds, dtype=np.float64)
    return {
        "mean_ms": float(values.mean()),
        "median_ms": float(np.median(values)),
        "p95_ms": float(np.percentile(values, 95)),
        "total_ms": float(values.sum()),
    }
