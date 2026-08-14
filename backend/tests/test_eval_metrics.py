"""The metric implementations, against hand-computable cases.

These are reimplemented rather than imported from scikit-learn (see `eval/metrics.py`),
so they have to be checked against values worked out by hand — a test that compares them
to another implementation would only prove the two agree, which is not the question.
"""

from __future__ import annotations

import numpy as np
import pytest

from anomaly_lab.db.repositories.results import ScoredImage, ScoredSample
from anomaly_lab.domain.entities import Aggregation, ChannelNormalization, Label, Subset
from anomaly_lab.eval.aggregate import (
    aggregate_scores,
    build_sample_results,
    normalize_by_channel,
)
from anomaly_lab.eval.metrics import (
    average_precision,
    pr_curve,
    roc_auc,
    roc_curve,
    timing_summary,
)
from anomaly_lab.eval.threshold import report


def test_perfect_separation_is_one() -> None:
    labels = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    assert roc_auc(labels, scores) == pytest.approx(1.0)


def test_reversed_separation_is_zero() -> None:
    labels = np.array([0, 0, 1, 1])
    scores = np.array([0.9, 0.8, 0.2, 0.1])
    assert roc_auc(labels, scores) == pytest.approx(0.0)


def test_all_scores_tied_is_one_half() -> None:
    """The case that separates a tie-aware AUC from one that trusts the sort order.

    Every score identical means the model has said nothing, and 0.5 is the only honest
    answer. An implementation that ranked by position would return 0 or 1 here depending
    on how the input happened to be ordered.
    """
    labels = np.array([0, 1, 0, 1])
    scores = np.array([0.5, 0.5, 0.5, 0.5])
    assert roc_auc(labels, scores) == pytest.approx(0.5)


def test_partial_ties_are_averaged() -> None:
    """Two normals at 0.3, one defect at 0.3, one defect at 0.9.

    Pairs: (0.9 defect) beats both normals -> 2 wins. (0.3 defect) ties both -> 1.0.
    Total 2 + 1 = 3 out of 4 pairs, so 0.75.
    """
    labels = np.array([0, 0, 1, 1])
    scores = np.array([0.3, 0.3, 0.3, 0.9])
    assert roc_auc(labels, scores) == pytest.approx(0.75)


def test_a_subset_with_one_class_has_no_auc() -> None:
    """`None`, not 0.5. A fabricated number on the results screen is worse than a gap."""
    assert roc_auc(np.array([1, 1, 1]), np.array([0.1, 0.5, 0.9])) is None
    assert roc_auc(np.array([0, 0]), np.array([0.1, 0.5])) is None


def test_average_precision_on_a_hand_worked_case() -> None:
    """Ranked: D N D N. Precision at the two defects is 1/1 and 2/3, recall steps 0.5.

    AP = 0.5 * 1.0 + 0.5 * (2/3) = 0.8333...
    """
    labels = np.array([1, 0, 1, 0])
    scores = np.array([0.9, 0.8, 0.7, 0.6])
    assert average_precision(labels, scores) == pytest.approx(5 / 6)


def test_average_precision_needs_both_classes() -> None:
    assert average_precision(np.array([1, 1]), np.array([0.2, 0.4])) is None


def test_roc_curve_starts_at_the_origin_and_ends_at_one() -> None:
    curve = roc_curve(np.array([0, 1, 0, 1]), np.array([0.1, 0.4, 0.35, 0.8]))
    assert curve is not None
    fpr, tpr = curve
    assert (fpr[0], tpr[0]) == (0.0, 0.0)
    assert (fpr[-1], tpr[-1]) == (1.0, 1.0)
    assert np.all(np.diff(fpr) >= 0)


def test_a_one_class_subset_has_no_curve_rather_than_a_chance_diagonal() -> None:
    """The same rule as every scalar here: absent, never fabricated."""
    assert roc_curve(np.array([0, 0, 0]), np.array([0.1, 0.2, 0.3])) is None
    assert pr_curve(np.array([1, 1, 1]), np.array([0.1, 0.2, 0.3])) is None


def test_the_roc_curve_integrates_to_the_roc_auc_it_is_drawn_beside() -> None:
    """One implementation for the chart and the number, so they cannot disagree."""
    generator = np.random.default_rng(7)
    labels = generator.integers(0, 2, size=200)
    scores = generator.normal(labels * 0.8, 1.0)

    curve = roc_curve(labels, scores)
    assert curve is not None
    fpr, tpr = curve
    assert float(np.trapezoid(tpr, fpr)) == pytest.approx(roc_auc(labels, scores), abs=1e-9)


def test_the_pr_curve_carries_the_points_average_precision_sums() -> None:
    labels = np.array([1, 0, 1, 0])
    scores = np.array([0.9, 0.8, 0.7, 0.6])

    curve = pr_curve(labels, scores)
    assert curve is not None
    recall, precision, threshold = curve
    assert recall == pytest.approx([0.5, 0.5, 1.0, 1.0])
    assert precision == pytest.approx([1.0, 0.5, 2 / 3, 0.5])
    assert threshold == pytest.approx([0.9, 0.8, 0.7, 0.6])

    steps = np.r_[0.0, recall[:-1]]
    assert float(np.sum((recall - steps) * precision)) == pytest.approx(
        average_precision(labels, scores)
    )


def test_the_pr_thresholds_are_the_ones_the_threshold_report_would_use() -> None:
    """The curve is drawn under a slider that reports the same numbers, so they must agree.

    `classify` calls a sample defective at `score >= threshold`; `_cut_points` cuts at the
    *last* index carrying each distinct score. Those two rules coincide, and this is the
    test that says so — if either moves, the chart under the slider stops describing the
    confusion matrix beside it, and nothing else would notice.
    """
    generator = np.random.default_rng(11)
    labels = generator.integers(0, 2, size=120).astype(bool)
    scores = generator.normal(labels * 0.8, 1.0)

    curve = pr_curve(labels, scores)
    assert curve is not None
    recall, precision, threshold = curve

    samples = [
        ScoredSample(
            sample_id=index,
            group_key="g",
            external_id=str(index),
            label=Label.DEFECT if bool(label) else Label.NORMAL,
            notes=None,
            agg_score=float(score),
            aggregation=Aggregation.MAX,
            subset=Subset.TEST,
        )
        for index, (label, score) in enumerate(zip(labels, scores, strict=True))
    ]

    for index, cut in enumerate(threshold):
        at_cut = report(samples, float(cut))
        assert at_cut.precision == pytest.approx(float(precision[index]))
        assert at_cut.recall == pytest.approx(float(recall[index]))


def test_curves_ignore_tie_order_the_way_the_scalars_do() -> None:
    """A saturated map returns one score for many images; the curve must not care."""
    labels = np.array([1, 0, 1, 0])
    tied = np.array([0.5, 0.5, 0.5, 0.5])

    curve = roc_curve(labels, tied)
    assert curve is not None
    fpr, tpr = curve
    # One distinct score, so one interior cut plus the two endpoints.
    assert fpr.size == 3
    assert float(np.trapezoid(tpr, fpr)) == pytest.approx(0.5)


def test_timing_summary_reports_the_shape_of_the_distribution() -> None:
    summary = timing_summary(np.array([10.0, 20.0, 30.0, 40.0]))
    assert summary["mean_ms"] == pytest.approx(25.0)
    assert summary["median_ms"] == pytest.approx(25.0)
    assert summary["total_ms"] == pytest.approx(100.0)


def test_timing_summary_of_nothing_is_empty_not_zero() -> None:
    assert timing_summary(np.array([])) == {}


def _image(sample_id: int, score: float) -> ScoredImage:
    return ScoredImage(
        image_id=sample_id * 100 + int(score * 10),
        sample_id=sample_id,
        channel=None,
        path="/x.png",
        width=8,
        height=8,
        score=score,
        map_path=None,
        inference_ms=1.0,
        label=Label.NORMAL,
        subset=None,
    )


def test_max_aggregation_keeps_single_channel_evidence() -> None:
    """The reason `max` is the default: two quiet views must not bury one loud one."""
    images = [_image(1, 0.1), _image(1, 0.2), _image(1, 0.9)]
    assert aggregate_scores(images, Aggregation.MAX) == {1: pytest.approx(0.9)}


def test_mean_aggregation_dilutes_it() -> None:
    images = [_image(1, 0.1), _image(1, 0.2), _image(1, 0.9)]
    assert aggregate_scores(images, Aggregation.MEAN) == {1: pytest.approx(0.4)}


def test_aggregation_groups_by_sample_not_by_image() -> None:
    images = [_image(1, 0.9), _image(2, 0.1), _image(2, 0.2)]
    aggregated = aggregate_scores(images, Aggregation.MAX)
    assert aggregated == {1: pytest.approx(0.9), 2: pytest.approx(0.2)}


# ------------------------------------------- per-channel normalization (ADR-0011)


def _channelled(image_id: int, sample_id: int, channel: str, score: float) -> ScoredImage:
    return ScoredImage(
        image_id=image_id,
        sample_id=sample_id,
        channel=channel,
        path="/x.png",
        width=8,
        height=8,
        score=score,
        map_path=None,
        inference_ms=1.0,
        label=Label.NORMAL,
        subset=None,
    )


def _two_channel_run() -> list[ScoredImage]:
    """`dark` scores a hundred times higher than `bright` on every part, and carries no
    information: its ordering is identical for the defective and the normal sample. `bright`
    is the channel that actually separates them."""
    return [
        _channelled(1, 1, "bright", 0.10),
        _channelled(2, 1, "dark", 50.0),
        _channelled(3, 2, "bright", 0.90),
        _channelled(4, 2, "dark", 50.5),
    ]


def test_without_normalization_the_loudest_channel_wins_every_sample() -> None:
    """The caveat ADR-0011 recorded, made visible: `max` is reading the scale, not the part."""
    scores = aggregate_scores(_two_channel_run(), Aggregation.MAX)

    assert scores[1] == pytest.approx(50.0)
    assert scores[2] == pytest.approx(50.5)
    # The bright-field evidence — 0.10 against 0.90 — has been erased entirely.
    assert scores[2] - scores[1] == pytest.approx(0.5)


def test_robust_z_lets_the_informative_channel_be_heard() -> None:
    scores = aggregate_scores(_two_channel_run(), Aggregation.MAX, ChannelNormalization.ROBUST_Z)

    # Both channels are now centred on their own median, so the gap between the two
    # samples reflects the separation each channel actually provides.
    assert scores[2] > scores[1]
    assert scores[2] - scores[1] > 1.0


def test_normalization_is_monotone_within_one_channel() -> None:
    """A single-view dataset must be unaffected: every transform here is monotone over one
    population, so nothing about a one-channel run may change."""
    images = [_channelled(i, i, "only", score) for i, score in enumerate([0.3, 0.1, 0.9, 0.5])]

    for method in (ChannelNormalization.ROBUST_Z, ChannelNormalization.RANK):
        normalized = aggregate_scores(images, Aggregation.MAX, method)
        raw = aggregate_scores(images, Aggregation.MAX)
        assert sorted(normalized, key=lambda s: normalized[s]) == sorted(raw, key=lambda s: raw[s])


def test_rank_normalization_throws_away_how_far_above_the_rest_a_score_sits() -> None:
    """The documented cost of `rank`, pinned rather than discovered in a metric.

    Rank keeps only the ordering inside a channel. A part that is dramatically the most
    anomalous thing bright-field ever saw and a part that is marginally the highest of a
    flat dark-field distribution both become exactly 1.0, so under `max` they tie — and
    "how anomalous" was the question. `robust_z` keeps the distance and separates them.
    """
    images = [
        _channelled(1, 1, "bright", 0.10),
        _channelled(2, 2, "bright", 0.11),
        _channelled(3, 3, "bright", 5.00),  # a dramatic outlier
        _channelled(4, 1, "dark", 50.0),  # marginally the highest of a flat channel
        _channelled(5, 2, "dark", 49.9),
        _channelled(6, 3, "dark", 49.8),
    ]

    ranked = aggregate_scores(images, Aggregation.MAX, ChannelNormalization.RANK)
    assert ranked[1] == pytest.approx(ranked[3])

    robust = aggregate_scores(images, Aggregation.MAX, ChannelNormalization.ROBUST_Z)
    assert robust[3] > robust[1]


def test_unassigned_images_normalize_as_their_own_group() -> None:
    """An image belonging to no channel is not a member of every channel."""
    images = [
        _channelled(1, 1, "bright", 1.0),
        ScoredImage(
            image_id=2,
            sample_id=1,
            channel=None,
            path="/x.png",
            width=8,
            height=8,
            score=1000.0,
            map_path=None,
            inference_ms=1.0,
            label=Label.NORMAL,
            subset=None,
        ),
    ]

    normalized = normalize_by_channel(images, ChannelNormalization.ROBUST_Z)

    # Each is alone in its group, so each is exactly at its own median.
    assert normalized[1] == pytest.approx(0.0)
    assert normalized[2] == pytest.approx(0.0)


def test_the_normalization_is_recorded_on_every_stored_row() -> None:
    """A stored result stays self-describing after the default changes."""
    rows = build_sample_results(
        7, _two_channel_run(), Aggregation.MAX, ChannelNormalization.ROBUST_Z
    )

    assert {row.normalization for row in rows} == {ChannelNormalization.ROBUST_Z}
    assert {row.aggregation for row in rows} == {Aggregation.MAX}
