"""Threshold selection, and what it says when it cannot do the honest thing.

VisA's official one-class split has no `val` subset, so threshold selection routinely has
nothing to fit against. The requirement is not that it produce a good number in that case
— it cannot — but that it say plainly what it did instead.
"""

from __future__ import annotations

import pytest

from anomaly_lab.db.repositories.results import ScoredSample
from anomaly_lab.domain.entities import Aggregation, Label
from anomaly_lab.eval.threshold import classify, report, suggest_threshold


def _sample(
    sample_id: int,
    score: float,
    label: Label,
    *,
    localized: bool | None = None,
) -> ScoredSample:
    return ScoredSample(
        sample_id=sample_id,
        group_key="g",
        external_id=str(sample_id),
        label=label,
        notes=None,
        agg_score=score,
        aggregation=Aggregation.MAX,
        subset=None,
        localized=localized,
    )


def test_a_separable_subset_gets_a_threshold_that_separates_it() -> None:
    samples = [
        _sample(1, 0.1, Label.NORMAL),
        _sample(2, 0.2, Label.NORMAL),
        _sample(3, 0.8, Label.DEFECT),
        _sample(4, 0.9, Label.DEFECT),
    ]
    threshold, rationale = suggest_threshold(samples)
    found = report(samples, threshold)

    assert "F1" in rationale
    assert found.confusion.true_positive == 2
    assert found.confusion.false_positive == 0
    assert found.f1 == pytest.approx(1.0)


def test_with_no_defects_the_fallback_is_named() -> None:
    """The VisA case. A number is still returned; what it means is stated."""
    samples = [_sample(index, 0.1 * index, Label.NORMAL) for index in range(1, 5)]
    threshold, rationale = suggest_threshold(samples)

    assert threshold == pytest.approx(0.4)
    assert "no defects" in rationale


def test_with_no_labels_at_all_the_fallback_is_also_named() -> None:
    samples = [_sample(index, float(index), Label.UNLABELED) for index in range(1, 4)]
    threshold, rationale = suggest_threshold(samples)

    assert threshold == pytest.approx(2.0)
    assert "no labels" in rationale


def test_an_empty_subset_does_not_crash_the_slider() -> None:
    threshold, rationale = suggest_threshold([])
    assert threshold == 0.0
    assert "no scored samples" in rationale


def test_unlabeled_samples_are_ranked_but_never_counted() -> None:
    """Ranking them is the triage value; counting them would be contamination."""
    samples = [
        _sample(1, 0.9, Label.UNLABELED),
        _sample(2, 0.5, Label.DEFECT),
        _sample(3, 0.1, Label.NORMAL),
    ]
    found = report(samples, 0.4)

    assert found.confusion.labelled == 2
    assert found.unlabeled == 1
    assert [verdict.sample_id for verdict in classify(samples, 0.4)] == [1, 2, 3]
    assert classify(samples, 0.4)[0].outcome == "unlabeled"


def test_rates_are_absent_rather_than_zero_when_undefined() -> None:
    """Precision with nothing predicted positive is not 0; it does not exist."""
    samples = [_sample(1, 0.1, Label.NORMAL), _sample(2, 0.2, Label.NORMAL)]
    found = report(samples, 10.0)

    assert found.precision is None
    assert found.recall is None
    assert found.f1 is None
    assert found.accuracy == pytest.approx(1.0)


def test_the_threshold_is_inclusive_at_its_own_value() -> None:
    """`>=`, stated once here so the slider and the metrics cannot disagree about it."""
    samples = [_sample(1, 0.5, Label.DEFECT)]
    assert report(samples, 0.5).confusion.true_positive == 1
    assert report(samples, 0.500001).confusion.false_negative == 1


def test_the_localization_verdict_does_not_move_with_the_slider() -> None:
    """It compares a map against ground truth, so no threshold can change it. The same
    defect is a `tp` at one cut and an `fn` at another while still having peaked on — or
    beside — its annotated region."""
    samples = [
        _sample(1, 0.9, Label.DEFECT, localized=True),
        _sample(2, 0.8, Label.DEFECT, localized=False),
        _sample(3, 0.7, Label.DEFECT),
        _sample(4, 0.1, Label.NORMAL),
    ]

    for threshold in (0.0, 0.75, 0.85, 10.0):
        by_id = {v.sample_id: v for v in classify(samples, threshold)}
        assert by_id[1].localized is True
        assert by_id[2].localized is False
        assert by_id[3].localized is None
        assert by_id[4].localized is None

    # …while the outcome of the same rows does move.
    assert {v.sample_id: v.outcome for v in classify(samples, 0.85)}[2] == "fn"
    assert {v.sample_id: v.outcome for v in classify(samples, 0.75)}[2] == "tp"


def test_the_outcome_vocabulary_is_still_exactly_four_names_plus_unlabeled() -> None:
    """`localized` is an orthogonal qualifier, not a fifth bucket. The comparison layer and
    the frontend key on these five strings, and a run whose maps all miss must still report
    the true positives it found."""
    samples = [
        _sample(1, 0.9, Label.DEFECT, localized=False),
        _sample(2, 0.1, Label.DEFECT, localized=False),
        _sample(3, 0.9, Label.NORMAL),
        _sample(4, 0.1, Label.NORMAL),
        _sample(5, 0.5, Label.UNLABELED, localized=True),
    ]

    outcomes = {verdict.outcome for verdict in classify(samples, 0.5)}
    assert outcomes == {"tp", "fn", "fp", "tn", "unlabeled"}

    found = report(samples, 0.5)
    assert found.confusion.true_positive == 1
    assert found.unlabeled == 1
