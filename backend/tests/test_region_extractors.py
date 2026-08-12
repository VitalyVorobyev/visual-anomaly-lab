"""Pure registry and classical extractor tests; no optional DL runtime is required."""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from anomaly_lab.regions.base import RegionExtractionError
from anomaly_lab.regions.mobile_sam import MobileSamRegionConfig, MobileSamRegionExtractor
from anomaly_lab.regions.registry import (
    UnknownRegionExtractorError,
    build,
    describe_all,
    get_extractor_class,
    registered_keys,
)


def test_registry_is_lazy_and_every_extractor_has_a_schema() -> None:
    assert registered_keys() == ("identity", "foreground_threshold", "mobile_sam")
    descriptions = describe_all()

    assert [item.key for item in descriptions] == list(registered_keys())
    assert all(item.title and item.summary for item in descriptions)
    assert all(item.config_schema["type"] == "object" for item in descriptions)
    assert descriptions[-1].required_assets == ["mobile-sam-vit-t"]


def test_unknown_extractor_fails_with_the_known_keys() -> None:
    with pytest.raises(UnknownRegionExtractorError, match="foreground_threshold"):
        get_extractor_class("magic")


def test_identity_returns_the_source_frame() -> None:
    result = build("identity", {}).extract(np.zeros((31, 47, 3), dtype=np.uint8))

    assert result.bounds.model_dump() == {"left": 0.0, "top": 0.0, "right": 47.0, "bottom": 31.0}
    assert result.metadata["coverage_fraction"] == 1.0


def test_threshold_finds_the_largest_contrasting_component() -> None:
    image = np.full((120, 160, 3), 230, dtype=np.uint8)
    image[30:90, 45:125] = 25
    image[5:10, 5:10] = 0
    extractor = build(
        "foreground_threshold",
        {"min_contrast": 8, "min_area_fraction": 0.01, "max_analysis_side": 512},
    )

    result = extractor.extract(image)

    assert result.bounds.model_dump() == {
        "left": 45.0,
        "top": 30.0,
        "right": 125.0,
        "bottom": 90.0,
    }
    assert result.metadata["area_pixels"] == 60 * 80


def test_threshold_maps_a_bounded_analysis_grid_back_to_source_edges() -> None:
    image = np.full((800, 1200, 3), 255, dtype=np.uint8)
    image[200:600, 300:900] = 0
    extractor = build(
        "foreground_threshold",
        {"min_contrast": 8, "min_area_fraction": 0.01, "max_analysis_side": 120},
    )

    result = extractor.extract(image)

    assert result.metadata["analysis_width"] == 120
    assert result.metadata["analysis_height"] == 80
    assert result.bounds.left == pytest.approx(300, abs=10)
    assert result.bounds.right == pytest.approx(900, abs=10)
    assert result.bounds.top == pytest.approx(200, abs=10)
    assert result.bounds.bottom == pytest.approx(600, abs=10)


def test_threshold_fails_instead_of_returning_identity_for_no_foreground() -> None:
    image = np.full((64, 64, 3), 128, dtype=np.uint8)

    with pytest.raises(RegionExtractionError, match="largest foreground component"):
        build("foreground_threshold", {}).extract(image)


def test_schema_rejects_an_unbounded_mobile_sam_grid() -> None:
    with pytest.raises(ValidationError):
        build("mobile_sam", {"points_per_side": 128})


def test_mobile_sam_chooses_the_largest_credible_mask_without_loading_in_the_registry() -> None:
    class FakeGenerator:
        def generate(self, _image: np.ndarray) -> list[dict[str, object]]:
            return [
                {
                    "area": 9800,
                    "bbox": [0, 0, 100, 100],
                    "predicted_iou": 0.99,
                    "stability_score": 0.99,
                },
                {
                    "area": 3600,
                    "bbox": [20, 15, 60, 60],
                    "predicted_iou": 0.88,
                    "stability_score": 0.93,
                },
                {
                    "area": 1200,
                    "bbox": [40, 35, 30, 40],
                    "predicted_iou": 0.96,
                    "stability_score": 0.97,
                },
            ]

    extractor = MobileSamRegionExtractor(MobileSamRegionConfig(max_area_fraction=0.9), assets={})
    extractor._generator = FakeGenerator()

    result = extractor.extract(np.zeros((100, 100, 3), dtype=np.uint8))

    assert result.bounds.model_dump() == {
        "left": 20.0,
        "top": 15.0,
        "right": 80.0,
        "bottom": 75.0,
    }
    assert result.confidence == 0.88
    assert result.metadata["coverage_fraction"] == 0.36


def test_mobile_sam_asset_absence_is_an_explicit_extraction_failure() -> None:
    extractor = MobileSamRegionExtractor(MobileSamRegionConfig(), assets={})

    with pytest.raises(RegionExtractionError, match="required model asset"):
        extractor.extract(np.zeros((32, 32, 3), dtype=np.uint8))


def test_mobile_sam_retries_the_known_float64_prompt_failure_on_cpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class MpsGenerator:
        def generate(self, _image: np.ndarray) -> list[dict[str, object]]:
            raise TypeError(
                "Cannot convert a MPS Tensor to float64 dtype as the MPS framework "
                "doesn't support float64"
            )

    class CpuGenerator:
        def generate(self, _image: np.ndarray) -> list[dict[str, object]]:
            return [
                {
                    "area": 1600,
                    "bbox": [10, 10, 40, 40],
                    "predicted_iou": 0.9,
                    "stability_score": 0.95,
                }
            ]

    extractor = MobileSamRegionExtractor(MobileSamRegionConfig(), assets={})
    extractor._device = "mps"
    calls: list[bool] = []

    def load_generator(*, force_cpu: bool = False) -> object:
        calls.append(force_cpu)
        if force_cpu:
            extractor._device = "cpu"
            return CpuGenerator()
        return MpsGenerator()

    monkeypatch.setattr(extractor, "_load_generator", load_generator)

    result = extractor.extract(np.zeros((100, 100, 3), dtype=np.uint8))

    assert calls == [False, True]
    assert result.metadata["device"] == "cpu"
    assert result.bounds.model_dump() == {
        "left": 10.0,
        "top": 10.0,
        "right": 50.0,
        "bottom": 50.0,
    }


def test_mobile_sam_does_not_hide_an_unrelated_type_error() -> None:
    class BrokenGenerator:
        def generate(self, _image: np.ndarray) -> list[dict[str, object]]:
            raise TypeError("unexpected prompt representation")

    extractor = MobileSamRegionExtractor(MobileSamRegionConfig(), assets={})
    extractor._generator = BrokenGenerator()
    extractor._device = "mps"

    with pytest.raises(TypeError, match="unexpected prompt representation"):
        extractor.extract(np.zeros((32, 32, 3), dtype=np.uint8))
