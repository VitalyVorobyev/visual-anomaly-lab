"""Channel to sample aggregation — the one substantive decision in this layer.

Models emit per-image scores; labels and splits belong to the sample (ADR-0005). Three
views of one part therefore produce three numbers and one verdict is needed. That
reduction is a detection decision, not a formatting step, so it lives here, applies
identically to every method, and is recorded on every row it produces.

`max` is the default: a defect visible under any single illumination makes the part
defective, and averaging dilutes exactly that evidence.

**The comparability caveat of `max` is a configured step rather than a
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
from dataclasses import dataclass

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


@dataclass(frozen=True)
class SampleScore:
    """One part's reduced score, and which image it came from.

    `source_image_id` is the image that **produced** the number. Under `max` that is the
    argmax after normalization; under `mean` no single image produced it, so the field is
    `None` rather than an arbitrary member — a winner nominated by row order would be a
    fact that reads as measured and is not.
    """

    score: float
    source_image_id: int | None


def aggregate_with_source(
    images: Sequence[ScoredImage],
    aggregation: Aggregation,
    normalization: ChannelNormalization = ChannelNormalization.NONE,
) -> dict[int, SampleScore]:
    """Per-sample score *and* the image behind it, keyed by sample id.

    The winner was always computed here and always discarded. It is kept now because the
    localization verdict has to follow the evidence the score actually came from: asking
    "did **any** channel's map land on the defect" would report a hit for a part whose
    score came from a channel that fired on the background, which is exactly the failure
    the verdict exists to expose.

    Under `feature_concat`-style methods one map is computed per sample and written once per
    image, so every channel carries the same peak and the argmax picks between identical
    verdicts — arbitrary, and harmless for that reason.
    """
    scaled = normalize_by_channel(images, normalization)

    grouped: dict[int, list[ScoredImage]] = {}
    for image in images:
        grouped.setdefault(image.sample_id, []).append(image)

    reduced: dict[int, SampleScore] = {}
    for sample_id, group in grouped.items():
        values = np.asarray([scaled[image.image_id] for image in group], dtype=np.float64)
        if aggregation is Aggregation.MAX:
            winner = int(values.argmax())
            reduced[sample_id] = SampleScore(
                score=float(values[winner]), source_image_id=group[winner].image_id
            )
        else:
            reduced[sample_id] = SampleScore(score=float(values.mean()), source_image_id=None)
    return reduced


def aggregate_scores(
    images: Sequence[ScoredImage],
    aggregation: Aggregation,
    normalization: ChannelNormalization = ChannelNormalization.NONE,
) -> dict[int, float]:
    """Per-sample score from per-image scores, keyed by sample id."""
    return {
        sample_id: found.score
        for sample_id, found in aggregate_with_source(images, aggregation, normalization).items()
    }


def sample_localization(
    group: Sequence[ScoredImage],
    source_image_id: int | None,
) -> bool | None:
    """One part's localization verdict from its images'.

    Two rules, because the two aggregations answer different questions. With a winning image
    the verdict is *that image's*, whatever it is — including `None`, which truthfully says
    the channel that decided this part carries no ground truth to check against. With no
    winner (`mean`) every image contributed, so a hit anywhere is a hit; a miss is reported
    only when something was actually checked and nothing landed.
    """
    if source_image_id is not None:
        for image in group:
            if image.image_id == source_image_id:
                return image.localized
        return None
    verdicts = [image.localized for image in group if image.localized is not None]
    if not verdicts:
        return None
    return any(verdicts)


def build_sample_results(
    experiment_id: int,
    images: Sequence[ScoredImage],
    aggregation: Aggregation,
    normalization: ChannelNormalization = ChannelNormalization.NONE,
) -> list[SampleResult]:
    """Sample rows ready to persist, with both decisions recorded on each one.

    Recorded per row rather than only in `eval_config` so that a stored result stays
    self-describing after the default changes (handbook evaluation.md).

    The localization verdict is carried up here too, from `ScoredImage.localized` — so a
    caller that re-derives sample scores without re-reading a single map still writes a
    sample verdict that agrees with its images.
    """
    grouped: dict[int, list[ScoredImage]] = {}
    for image in images:
        grouped.setdefault(image.sample_id, []).append(image)

    return [
        SampleResult(
            experiment_id=experiment_id,
            sample_id=sample_id,
            agg_score=found.score,
            aggregation=aggregation,
            normalization=normalization,
            localized=sample_localization(grouped[sample_id], found.source_image_id),
        )
        for sample_id, found in sorted(
            aggregate_with_source(images, aggregation, normalization).items()
        )
    ]
