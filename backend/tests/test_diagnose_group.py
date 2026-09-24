"""Which records `diagnose` hands a model for one image (handbook diagnostics.md)."""

from __future__ import annotations

import pytest

from anomaly_lab.db.repositories.images import SplitImage
from anomaly_lab.domain.entities import Label, Subset
from anomaly_lab.experiments.diagnose import DiagnoseError, records_to_score


def _image(image_id: int, sample_id: int, channel: str | None) -> SplitImage:
    return SplitImage(
        image_id=image_id,
        sample_id=sample_id,
        channel=channel,
        path=f"/fixture/{image_id}.png",
        sha256=f"hash-{image_id}",
        label=Label.NORMAL,
        subset=Subset.TEST,
    )


SELECTED = [
    _image(1, 10, "bright"),
    _image(2, 10, "dark"),
    _image(3, 10, "side"),
    _image(4, 11, "bright"),
]


def test_a_channel_aware_model_gets_the_whole_sample() -> None:
    chosen = records_to_score(SELECTED, 2, channel_aware=True)

    assert [image.image_id for image in chosen] == [1, 2, 3]


def test_a_per_image_model_gets_the_image_alone() -> None:
    chosen = records_to_score(SELECTED, 2, channel_aware=False)

    assert [image.image_id for image in chosen] == [2]


def test_an_image_outside_the_split_is_refused_by_name() -> None:
    with pytest.raises(DiagnoseError, match="split"):
        records_to_score(SELECTED, 99, channel_aware=True)
