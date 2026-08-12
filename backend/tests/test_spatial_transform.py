"""Deterministic synthetic geometry fixtures for the M10 input bridge (ADR-0033)."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image
from pydantic import ValidationError

from anomaly_lab.regions import PixelBounds, SpatialTransform


def test_resolve_expands_clips_contains_and_persists_actual_scales() -> None:
    transform = SpatialTransform.resolve(
        source_size=(100, 80),
        prepared_size=(64, 64),
        region=PixelBounds(left=10, top=20, right=50, bottom=60),
        padding_fraction=0.05,
    )

    assert (
        transform.crop_left,
        transform.crop_top,
        transform.crop_right,
        transform.crop_bottom,
    ) == (8, 18, 52, 62)
    assert (transform.resized_width, transform.resized_height) == (64, 64)
    assert transform.model_validate_json(transform.model_dump_json()) == transform


@pytest.mark.parametrize(
    ("source_size", "prepared_size", "region"),
    [
        ((101, 73), (256, 256), None),
        ((101, 73), (257, 191), PixelBounds(left=7.2, top=5.4, right=81.8, bottom=63.1)),
        ((37, 211), (128, 96), PixelBounds(left=-4, top=19, right=20, bottom=199)),
    ],
)
def test_point_round_trip_is_exact_to_floating_precision(
    source_size: tuple[int, int],
    prepared_size: tuple[int, int],
    region: PixelBounds | None,
) -> None:
    transform = SpatialTransform.resolve(
        source_size=source_size,
        prepared_size=prepared_size,
        region=region,
    )
    points = [
        (float(transform.crop_left), float(transform.crop_top)),
        ((transform.crop_left + transform.crop_right - 1) / 2, 27.125),
        (float(transform.crop_right - 1), float(transform.crop_bottom - 1)),
    ]

    for point in points:
        restored = transform.prepared_to_source(transform.source_to_prepared(point))
        assert restored == pytest.approx(point, abs=1e-12)


def test_image_uses_edge_padding_and_never_stretches_aspect_ratio() -> None:
    source = np.zeros((4, 8), dtype=np.uint8)
    source[:, :] = np.arange(8, dtype=np.uint8)
    transform = SpatialTransform.resolve(source_size=(8, 4), prepared_size=(8, 8))

    prepared = np.asarray(
        transform.prepare_image(
            Image.fromarray(source, mode="L"), resample=Image.Resampling.NEAREST
        )
    )

    assert prepared.shape == (8, 8)
    assert np.array_equal(prepared[2:6], source)
    assert np.array_equal(prepared[:2], np.repeat(source[:1], 2, axis=0))
    assert np.array_equal(prepared[6:], np.repeat(source[-1:], 2, axis=0))


def test_mask_round_trip_is_exact_for_an_aligned_synthetic_region() -> None:
    source = np.zeros((12, 20), dtype=bool)
    source[4:8, 6:14] = True
    transform = SpatialTransform.resolve(
        source_size=(20, 12),
        prepared_size=(40, 24),
        region=PixelBounds(left=2, top=2, right=18, bottom=10),
        padding_fraction=0,
    )

    restored = transform.project_mask(transform.prepare_mask(source))

    assert np.array_equal(restored, source)


def test_map_projection_removes_letterbox_and_marks_uncovered_source_pixels() -> None:
    transform = SpatialTransform.resolve(
        source_size=(20, 12),
        prepared_size=(16, 16),
        region=PixelBounds(left=4, top=3, right=16, bottom=9),
        padding_fraction=0,
    )
    prepared = np.full((16, 16), 0.75, dtype=np.float32)

    projected = transform.project_map(prepared)

    assert projected.shape == (12, 20)
    assert np.all(projected[3:9, 4:16] == pytest.approx(0.75))
    assert np.isnan(projected[:3]).all()
    assert np.isnan(projected[:, :4]).all()


def test_transform_rejects_inconsistent_or_mismatched_frames() -> None:
    transform = SpatialTransform.resolve(source_size=(10, 8), prepared_size=(16, 16))

    with pytest.raises(ValueError, match="source mask shape"):
        transform.prepare_mask(np.zeros((8, 9), dtype=bool))
    with pytest.raises(ValueError, match="prepared shape"):
        transform.project_map(np.zeros((15, 16), dtype=np.float32))
    with pytest.raises(ValidationError, match="do not fill"):
        SpatialTransform(**(transform.model_dump() | {"pad_right": transform.pad_right + 1}))
