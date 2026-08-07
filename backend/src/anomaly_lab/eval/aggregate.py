"""Channel to sample aggregation — the one substantive decision in this layer.

Models emit per-image scores; labels and splits belong to the sample (ADR-0005). Three
views of one part therefore produce three numbers and one verdict is needed. That
reduction is a detection decision, not a formatting step, so it lives here, applies
identically to every method, and is recorded on every row it produces.

`max` is the default: a defect visible under any single illumination makes the part
defective, and averaging dilutes exactly that evidence. The caveat ADR-0011 states
honestly still stands — `max` assumes per-channel scores are comparable in scale, which
is not automatic for a deep model. Per-channel quantile normalization remains a backlog
item rather than a silent default.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from anomaly_lab.db.repositories.results import ScoredImage
from anomaly_lab.domain.entities import Aggregation, SampleResult


def aggregate_scores(
    images: Sequence[ScoredImage],
    aggregation: Aggregation,
) -> dict[int, float]:
    """Per-sample score from per-image scores, keyed by sample id."""
    grouped: dict[int, list[float]] = {}
    for image in images:
        grouped.setdefault(image.sample_id, []).append(image.score)

    def reduce(scores: list[float]) -> float:
        values = np.asarray(scores, dtype=np.float64)
        return float(values.max() if aggregation is Aggregation.MAX else values.mean())

    return {sample_id: reduce(scores) for sample_id, scores in grouped.items()}


def build_sample_results(
    experiment_id: int,
    images: Sequence[ScoredImage],
    aggregation: Aggregation,
) -> list[SampleResult]:
    """Sample rows ready to persist, with the aggregation recorded on each one.

    Recorded per row rather than only in `eval_config` so that a stored result stays
    self-describing after the default changes (ADR-0011).
    """
    return [
        SampleResult(
            experiment_id=experiment_id,
            sample_id=sample_id,
            agg_score=score,
            aggregation=aggregation,
        )
        for sample_id, score in sorted(aggregate_scores(images, aggregation).items())
    ]
