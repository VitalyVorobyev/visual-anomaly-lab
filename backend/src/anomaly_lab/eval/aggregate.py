"""Channel to sample aggregation — the one substantive decision in this layer.

Models emit per-image scores; labels and splits belong to the sample (ADR-0005). Three
views of one part therefore produce three numbers and one verdict is needed. That
reduction is a detection decision, not a formatting step, so it lives here, applies
identically to every method, and is recorded on every row it produces.

`max` is the default: a defect visible under any single illumination makes the part
defective, and averaging dilutes exactly that evidence.

**The comparability caveat ADR-0011 recorded is now a configured step rather than a
footnote.** `max` assumes a part's per-channel scores are on one scale, which is not
automatic for a deep model: if the dark-field distribution simply sits higher than the
bright-field one, every maximum comes from dark-field and the sample score measures which
illumination the model finds noisiest, not which part is defective. `channel_normalization`
puts them on one scale first. It stays `none` by default, so no stored experiment silently
changes meaning.

**The transform is fitted over every image the experiment has scored, with labels
ignored.** Two properties force this rather than making it a preference. `sample_result`
is keyed `(experiment_id, sample_id)` with no subset column and `build_sample_results` runs
over the whole run at once, so a per-subset fit is not representable in the storage shape.
And using labels to build the transform would make the metric partly a function of the
answer. It is transductive — a test image's score depends on the other test images — which
is the honest cost, and the same cost `eval/pixel.py` already accepts when it adapts its
histogram bins to a run's observed range.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from anomaly_lab.db.repositories.results import ScoredImage
from anomaly_lab.domain.entities import Aggregation, ChannelNormalization, SampleResult

MAD_TO_SIGMA = 1.4826
"""Scales a median absolute deviation to a standard deviation under a normal distribution,
so `robust_z` reads on roughly the same scale as an ordinary z-score."""

_SCALE_FLOOR = 1e-9
"""A channel whose scores are all equal has no spread to divide by. Flooring maps it to
zero deviation, which is the truthful answer: nothing in that channel distinguishes
anything."""


def normalize_by_channel(
    images: Sequence[ScoredImage],
    method: ChannelNormalization,
) -> dict[int, float]:
    """Per-image scores on one cross-channel scale, keyed by image id.

    Images with no channel form their own group, exactly as a named channel does — an
    unassigned image is not a member of every channel, and a single-view dataset therefore
    lands in one group where any of these transforms is monotone and changes no ordering.
    """
    if method is ChannelNormalization.NONE:
        return {image.image_id: image.score for image in images}

    by_channel: dict[str | None, list[ScoredImage]] = {}
    for image in images:
        by_channel.setdefault(image.channel, []).append(image)

    normalized: dict[int, float] = {}
    for group in by_channel.values():
        scores = np.asarray([image.score for image in group], dtype=np.float64)
        if method is ChannelNormalization.ROBUST_Z:
            median = float(np.median(scores))
            scale = max(float(np.median(np.abs(scores - median))) * MAD_TO_SIGMA, _SCALE_FLOOR)
            values = (scores - median) / scale
        else:
            # Average ranks, so tied scores stay tied rather than being ordered by the
            # accident of row order. Divided by n-1 so the group spans [0, 1] exactly.
            order = scores.argsort()
            ranks = np.empty(len(scores), dtype=np.float64)
            ranks[order] = np.arange(len(scores), dtype=np.float64)
            for value in np.unique(scores):
                tied = scores == value
                ranks[tied] = ranks[tied].mean()
            values = ranks / max(len(scores) - 1, 1)
        for image, value in zip(group, values, strict=True):
            normalized[image.image_id] = float(value)
    return normalized


def aggregate_scores(
    images: Sequence[ScoredImage],
    aggregation: Aggregation,
    normalization: ChannelNormalization = ChannelNormalization.NONE,
) -> dict[int, float]:
    """Per-sample score from per-image scores, keyed by sample id."""
    scaled = normalize_by_channel(images, normalization)

    grouped: dict[int, list[float]] = {}
    for image in images:
        grouped.setdefault(image.sample_id, []).append(scaled[image.image_id])

    def reduce(scores: list[float]) -> float:
        values = np.asarray(scores, dtype=np.float64)
        return float(values.max() if aggregation is Aggregation.MAX else values.mean())

    return {sample_id: reduce(scores) for sample_id, scores in grouped.items()}


def build_sample_results(
    experiment_id: int,
    images: Sequence[ScoredImage],
    aggregation: Aggregation,
    normalization: ChannelNormalization = ChannelNormalization.NONE,
) -> list[SampleResult]:
    """Sample rows ready to persist, with both decisions recorded on each one.

    Recorded per row rather than only in `eval_config` so that a stored result stays
    self-describing after the default changes (ADR-0011).
    """
    return [
        SampleResult(
            experiment_id=experiment_id,
            sample_id=sample_id,
            agg_score=score,
            aggregation=aggregation,
            normalization=normalization,
        )
        for sample_id, score in sorted(aggregate_scores(images, aggregation, normalization).items())
    ]
