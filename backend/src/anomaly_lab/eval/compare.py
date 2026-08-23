"""Reading several runs against each other (ADR-0028).

The premise of this module is one fact: **a score has no meaning outside its own run.**
`pixel_reference` reports a robust z and peaks around 8; `efficientad_custom` reports a
quantile-normalized sum on its own, unrelated scale. Nothing relates the two, because there
is no shared quantity underneath — so nothing here is ever compared in score units.

What is shared across runs is a *rule*. Each run's threshold is derived by the same stated
rule from its own distribution, and the caller is expected to print both the rule and the
value it produced; a confusion matrix with no threshold beside it is a claim about an
operating point the reader cannot name.

Like the rest of `eval/`, this imports no model and re-runs no inference (ADR-0011). Its
inputs are the scored samples already in the database.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

from anomaly_lab.db.repositories.results import ScoredSample
from anomaly_lab.domain.entities import Label
from anomaly_lab.eval.threshold import outcome_of, suggest_threshold


class OperatingPoint(StrEnum):
    """How each run's own threshold is chosen from its own scores."""

    F1 = "f1"
    RECALL = "recall"


@dataclass(frozen=True)
class OperatingThreshold:
    """One run's cut, in that run's units, and the sentence that explains it.

    `value` is `None` when the rule cannot be applied to this run's data — nothing scored
    at all, or a target recall no threshold reaches. A run with no operating point shows no
    confusion matrix, which is the honest answer; the nearest achievable point would be a
    different operating point wearing this one's label.
    """

    value: float | None
    rationale: str


def resolve_threshold(
    samples: Sequence[ScoredSample],
    point: OperatingPoint,
    *,
    recall_target: float = 0.95,
) -> OperatingThreshold:
    """This run's threshold under one shared rule.

    `F1` delegates to `suggest_threshold` rather than reimplementing the search. That is
    deliberate and it costs something: the search is quadratic in the number of samples.
    A second F1-optimal implementation here would be free to drift from the one the
    results screen opens its slider at, and a comparison that disagrees with the screen it
    was reached from is worse than a slow one.

    A run with nothing scored in this subset has no operating point at all.
    `suggest_threshold` answers `0.0` there, which is right for a *slider* that has to open
    somewhere and wrong for a confusion matrix: four zeros under a threshold of 0.000 reads
    as a run that predicted nothing, rather than one that was never asked.
    """
    if not samples:
        return OperatingThreshold(value=None, rationale="nothing scored in this subset")
    if point is OperatingPoint.F1:
        value, rationale = suggest_threshold(samples)
        return OperatingThreshold(value=value, rationale=rationale)
    return _at_recall(samples, recall_target)


def _at_recall(samples: Sequence[ScoredSample], target: float) -> OperatingThreshold:
    """The highest threshold that still reaches `target` recall on this subset.

    Highest, not merely any: recall only falls as the threshold rises, so every lower cut
    reaches the target too, and picking a lower one would hand the run false alarms it did
    not need to accept. Concretely, with `k = ceil(target * defects)`, the answer is the
    k-th highest defect score — at that cut exactly the top k defects are caught, and any
    higher cut catches one fewer.
    """
    defects = sorted(
        (sample.agg_score for sample in samples if sample.label is Label.DEFECT),
        reverse=True,
    )
    if not defects:
        return OperatingThreshold(
            value=None,
            rationale=f"no defects in this subset, so {target:.0%} recall is not defined",
        )
    if not 0.0 < target <= 1.0:  # pragma: no cover - the route validates the range
        return OperatingThreshold(value=None, rationale="the recall target must be in (0, 1]")

    needed = max(1, math.ceil(target * len(defects)))
    return OperatingThreshold(
        value=defects[needed - 1],
        rationale=(
            f"the highest cut still catching {needed} of {len(defects)} defects "
            f"({needed / len(defects):.0%} recall)"
        ),
    )


@dataclass(frozen=True)
class SampleAgreement:
    """One sample as every compared run judged it, index-aligned with the run list.

    `agree` is derived from the **predictions**, not from the outcomes. An unlabeled sample
    is tagged `unlabeled` by every run whatever it predicted, so agreement read off the
    outcome would call every unlabeled row unanimous — and unlabeled rows are exactly where
    a disagreement between two methods is worth a human's attention.
    """

    sample_id: int
    group_key: str
    external_id: str
    label: Label
    scores: list[float | None]
    predicted: list[bool | None]
    outcomes: list[str | None]
    agree: bool


def agreement(
    runs: Sequence[Sequence[ScoredSample]],
    thresholds: Sequence[float | None],
) -> list[SampleAgreement]:
    """One row per sample, with each run's verdict at its own threshold.

    Computed here rather than in the client for the reason `ThresholdReport` carries its
    classified rows: the rule "a score at or above the threshold is a defect" already
    exists once, in Python, and re-deriving it in TypeScript for N runs at N thresholds is
    that bug multiplied by N.

    A sample one run scored and another did not gets `None` in that run's column — the two
    runs share a split, but a run interrupted mid-inference has fewer rows, and inventing a
    verdict for it would be the fabrication this codebase renders as a dash everywhere else.
    """
    if len(runs) != len(thresholds):  # pragma: no cover - callers build both from one list
        raise ValueError("every run needs a threshold slot, even an empty one")

    # Identity comes from the first run that has the sample; every run reads the same
    # `sample` rows, so the label and the keys cannot disagree.
    order: list[int] = []
    identity: dict[int, ScoredSample] = {}
    by_run: list[dict[int, ScoredSample]] = []
    for samples in runs:
        indexed = {sample.sample_id: sample for sample in samples}
        by_run.append(indexed)
        for sample in samples:
            if sample.sample_id not in identity:
                identity[sample.sample_id] = sample
                order.append(sample.sample_id)

    rows: list[SampleAgreement] = []
    for sample_id in order:
        first = identity[sample_id]
        scores: list[float | None] = []
        predicted: list[bool | None] = []
        outcomes: list[str | None] = []
        for indexed, threshold in zip(by_run, thresholds, strict=True):
            found = indexed.get(sample_id)
            if found is None or threshold is None:
                scores.append(None if found is None else found.agg_score)
                predicted.append(None)
                outcomes.append(None)
                continue
            is_defect = found.agg_score >= threshold
            scores.append(found.agg_score)
            predicted.append(is_defect)
            outcomes.append(outcome_of(found.label, is_defect))

        decided = [value for value in predicted if value is not None]
        rows.append(
            SampleAgreement(
                sample_id=sample_id,
                group_key=first.group_key,
                external_id=first.external_id,
                label=first.label,
                scores=scores,
                predicted=predicted,
                outcomes=outcomes,
                # A row nobody could judge is not a disagreement.
                agree=len(set(decided)) <= 1,
            )
        )
    return rows
