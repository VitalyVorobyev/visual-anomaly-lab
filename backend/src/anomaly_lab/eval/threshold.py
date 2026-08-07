"""Threshold-dependent outputs, computed on demand and never stored (ADR-0011).

A confusion matrix is a function of data already in the database. Persisting one per
threshold would multiply rows to store a derived value, and would make the UI's threshold
slider feel like a database write instead of an instant filter.

Unlabeled samples appear in the ranked lists and in no count. Ranking them is one of the
most useful things the workbench does — it turns a trained model into a labelling aid —
but letting one into a reported number would be contamination.
"""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel, Field

from anomaly_lab.db.repositories.results import ScoredSample
from anomaly_lab.domain.entities import Label
from anomaly_lab.schemas import API_MODEL_CONFIG


class ConfusionCounts(BaseModel):
    model_config = API_MODEL_CONFIG

    true_positive: int = 0
    false_positive: int = 0
    true_negative: int = 0
    false_negative: int = 0

    @property
    def labelled(self) -> int:
        return self.true_positive + self.false_positive + self.true_negative + self.false_negative


class SampleVerdict(BaseModel):
    """One sample as the results grid shows it at the current threshold."""

    model_config = API_MODEL_CONFIG

    sample_id: int
    group_key: str
    external_id: str
    label: Label
    notes: str | None = None
    score: float
    predicted_defect: bool
    outcome: str = Field(description="tp, fp, tn, fn — or 'unlabeled' for a ranked-only row.")


class ThresholdReport(BaseModel):
    model_config = API_MODEL_CONFIG

    threshold: float
    confusion: ConfusionCounts
    precision: float | None = None
    recall: float | None = None
    f1: float | None = None
    accuracy: float | None = None
    unlabeled: int = 0


def _outcome(label: Label, predicted_defect: bool) -> str:
    if label is Label.UNLABELED:
        return "unlabeled"
    if label is Label.DEFECT:
        return "tp" if predicted_defect else "fn"
    return "fp" if predicted_defect else "tn"


def classify(samples: Sequence[ScoredSample], threshold: float) -> list[SampleVerdict]:
    """Label every scored sample against one threshold, ranked most anomalous first."""
    verdicts = [
        SampleVerdict(
            sample_id=sample.sample_id,
            group_key=sample.group_key,
            external_id=sample.external_id,
            label=sample.label,
            notes=sample.notes,
            score=sample.agg_score,
            predicted_defect=sample.agg_score >= threshold,
            outcome=_outcome(sample.label, sample.agg_score >= threshold),
        )
        for sample in samples
    ]
    verdicts.sort(key=lambda verdict: verdict.score, reverse=True)
    return verdicts


def report(samples: Sequence[ScoredSample], threshold: float) -> ThresholdReport:
    """Confusion matrix and the rates derived from it, for one threshold."""
    tallies = dict.fromkeys(("tp", "fp", "tn", "fn", "unlabeled"), 0)
    for sample in samples:
        tallies[_outcome(sample.label, sample.agg_score >= threshold)] += 1

    counts = ConfusionCounts(
        true_positive=tallies["tp"],
        false_positive=tallies["fp"],
        true_negative=tallies["tn"],
        false_negative=tallies["fn"],
    )
    unlabeled = tallies["unlabeled"]

    predicted_positive = counts.true_positive + counts.false_positive
    actual_positive = counts.true_positive + counts.false_negative

    precision = counts.true_positive / predicted_positive if predicted_positive else None
    recall = counts.true_positive / actual_positive if actual_positive else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and (precision + recall) > 0
        else None
    )
    accuracy = (
        (counts.true_positive + counts.true_negative) / counts.labelled if counts.labelled else None
    )

    return ThresholdReport(
        threshold=threshold,
        confusion=counts,
        precision=precision,
        recall=recall,
        f1=f1,
        accuracy=accuracy,
        unlabeled=unlabeled,
    )


def suggest_threshold(samples: Sequence[ScoredSample]) -> tuple[float, str]:
    """A starting position for the slider, and a plain statement of how it was chosen.

    The rationale is returned with the number because the three cases are not equally
    trustworthy, and a slider that silently opens at a fabricated position invites the
    operator to read it as a recommendation.
    """
    labelled = [sample for sample in samples if sample.label is not Label.UNLABELED]
    defects = [sample.agg_score for sample in labelled if sample.label is Label.DEFECT]
    normals = [sample.agg_score for sample in labelled if sample.label is Label.NORMAL]

    if defects and normals:
        best_score, best_threshold = -1.0, 0.0
        for candidate in sorted({*defects, *normals}):
            found = report(labelled, candidate)
            if found.f1 is not None and found.f1 > best_score:
                best_score, best_threshold = found.f1, candidate
        return best_threshold, "maximizes F1 over the labelled samples in this subset"

    if normals:
        # No defects to separate from, so the only defensible rule is "unusual for a
        # normal" — the highest normal score seen, which is the empirical operating point
        # a one-class method would be held to.
        return max(normals), "highest normal score in this subset — no defects to fit against"

    scores = [sample.agg_score for sample in samples]
    if scores:
        midpoint = (min(scores) + max(scores)) / 2
        return midpoint, "midpoint of the score range — this subset has no labels at all"
    return 0.0, "no scored samples"
