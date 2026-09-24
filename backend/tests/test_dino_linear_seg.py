"""`dino_linear_seg`'s torch-free half: the pixel plan, the loss weights, the interpolation the
head is trained through, and the configuration the form is generated from. What needs an encoder
is in `test_dl_dino_linear_seg.py`."""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from anomaly_lab.domain.entities import Task
from anomaly_lab.models.base import IGNORE_INDEX
from anomaly_lab.models.dino_backbone import DinoBackbone, FeatureLayers
from anomaly_lab.models.dino_linear_seg import (
    MAX_FEATURE_BYTES,
    ClassBalancing,
    DinoLinearSegConfig,
    DinoLinearSegModel,
    PixelSampling,
    allocate_pixels,
    class_weights,
    feature_dim,
    pixel_features,
    plan_pixels,
    sample_pixels,
)
from anomaly_lab.models.preprocessing import PreprocessingConfig
from anomaly_lab.models.refine import upsample
from anomaly_lab.models.registry import describe


def test_it_declares_semantic_segmentation_alone_and_no_export() -> None:
    description = describe("dino_linear_seg")
    assert description.capabilities.tasks == [Task.SEMANTIC_SEGMENTATION]
    assert description.capabilities.portable_formats == []
    schema = description.config_schema
    for name, field in schema["properties"].items():
        assert field.get("description"), name


def test_the_plan_keeps_every_image_when_the_total_allows() -> None:
    plan = plan_pixels(10, pixels_per_image=1024, max_training_pixels=131_072, feature_dim=768)
    assert (plan.images_used, plan.pixels_per_image, plan.max_pixels) == (10, 1024, 10_240)
    assert plan.feature_bytes == 10_240 * 768 * 4
    assert "all 10 training images" in plan.describe()


def test_the_plan_divides_the_total_among_images_before_it_drops_any() -> None:
    plan = plan_pixels(500, pixels_per_image=1024, max_training_pixels=10_000, feature_dim=384)
    assert (plan.images_used, plan.pixels_per_image) == (500, 20)
    assert plan.max_pixels <= 10_000

    crowded = plan_pixels(5_000, pixels_per_image=1024, max_training_pixels=1_000, feature_dim=384)
    assert (crowded.images_used, crowded.pixels_per_image) == (1_000, 1)
    assert "1000 of 5000 training images, sampled evenly" in crowded.describe()


def test_the_plan_refuses_a_sample_that_would_not_fit_in_memory() -> None:
    width = feature_dim(DinoBackbone.DINOV2_VIT_L14, FeatureLayers.LAST_FOUR)
    assert width == 4096
    assert 2_000_000 * width * 4 > MAX_FEATURE_BYTES
    with pytest.raises(ValueError, match="lower max_training_pixels"):
        plan_pixels(4_000, pixels_per_image=1024, max_training_pixels=2_000_000, feature_dim=width)
    with pytest.raises(ValueError, match="at least one training image"):
        plan_pixels(0, pixels_per_image=16, max_training_pixels=1_000, feature_dim=8)


def test_each_present_class_gets_an_equal_share_and_gives_back_what_it_cannot_use() -> None:
    assert allocate_pixels(np.array([10_000, 5_000]), 100).tolist() == [50, 50]
    # A class smaller than its share takes all it has; the others split the rest.
    assert allocate_pixels(np.array([10_000, 7, 0, 5_000]), 100).tolist() == [46, 7, 0, 47]
    # The budget is spent whenever the image holds that many labelled pixels, and never exceeded.
    assert allocate_pixels(np.array([3, 4]), 100).tolist() == [3, 4]
    assert allocate_pixels(np.array([0, 0]), 100).tolist() == [0, 0]
    rng = np.random.default_rng(0)
    for _ in range(200):
        counts = rng.integers(0, 50, size=4) * rng.integers(0, 2, size=4)
        budget = int(rng.integers(1, 120))
        taken = allocate_pixels(counts, budget)
        assert np.all(taken <= counts)
        assert taken.sum() == min(budget, counts.sum())


def _frame_with_a_small_defect() -> np.ndarray:
    truth = np.zeros((64, 64), dtype=np.uint8)
    truth[40:44, 10:14] = 1  # 16 defect pixels of 4 096
    truth[:, :4] = IGNORE_INDEX
    return truth.reshape(-1)


def test_per_class_sampling_reaches_a_class_that_raster_sampling_misses() -> None:
    truth = _frame_with_a_small_defect()
    raster = sample_pixels(truth, 2, 64, PixelSampling.RASTER)
    per_class = sample_pixels(truth, 2, 64, PixelSampling.PER_CLASS)
    assert len(raster) == len(per_class) == 64
    assert int((truth[raster] == 1).sum()) <= 1
    assert np.bincount(truth[per_class], minlength=2).tolist() == [48, 16]
    for picked in (raster, per_class):
        assert not np.any(truth[picked] == IGNORE_INDEX)
        assert np.array_equal(picked, np.sort(np.unique(picked)))
    # Deterministic: nothing is drawn at random.
    assert np.array_equal(per_class, sample_pixels(truth, 2, 64, PixelSampling.PER_CLASS))


def test_per_class_sampling_spreads_each_class_over_its_own_pixels() -> None:
    truth = _frame_with_a_small_defect()
    picked = sample_pixels(truth, 2, 16, PixelSampling.PER_CLASS)
    defect = picked[truth[picked] == 1]
    background = picked[truth[picked] == 0]
    assert len(defect) == len(background) == 8
    # Spread from the first background pixel to the last, not the first eight of them.
    members = np.flatnonzero(truth == 0)
    assert (background[0], background[-1]) == (members[0], members[-1])


def test_a_class_outside_the_run_is_never_sampled() -> None:
    truth = np.array([0, 0, 3, 3, 1, IGNORE_INDEX], dtype=np.uint8)
    for sampling in PixelSampling:
        assert set(truth[sample_pixels(truth, 2, 10, sampling)].tolist()) == {0, 1}
    assert (
        len(sample_pixels(np.full(4, IGNORE_INDEX, np.uint8), 2, 10, PixelSampling.PER_CLASS)) == 0
    )


def test_inverse_frequency_gives_every_sampled_class_the_same_total_weight() -> None:
    counts = np.array([900, 90, 0, 10])
    weights = class_weights(counts, ClassBalancing.INVERSE_FREQUENCY)
    assert weights[2] == 0.0
    totals = weights * counts
    assert totals[0] == pytest.approx(totals[1]) == pytest.approx(totals[3])
    assert np.array_equal(class_weights(counts, ClassBalancing.NONE), [1, 1, 0, 1])


@pytest.mark.parametrize(
    ("grid", "size"), [((4, 6), (84, 56)), ((3, 3), (48, 48)), ((5, 2), (7, 9))]
)
def test_pixel_features_are_the_upsampled_grid(
    grid: tuple[int, int], size: tuple[int, int]
) -> None:
    """The head is trained on these rows and predicts through `upsample`; one function only
    if the two agree at every pixel."""
    rows, cols = grid
    width, height = size
    values = np.random.default_rng(3).normal(size=(rows * cols, 2)).astype(np.float32)
    every = np.arange(width * height)
    sampled = pixel_features(values, every, grid, size)
    for channel in range(2):
        expected = upsample(values[:, channel].reshape(rows, cols), size).reshape(-1)
        np.testing.assert_allclose(sampled[:, channel], expected, rtol=1e-5, atol=1e-5)


def test_the_configuration_is_bounded() -> None:
    with pytest.raises(ValidationError):
        DinoLinearSegConfig(pixels_per_image=0)
    with pytest.raises(ValidationError):
        DinoLinearSegConfig(learning_rate=0.0)
    with pytest.raises(ValidationError):
        DinoLinearSegConfig.model_validate({"class_balancing": "median"})
    assert DinoLinearSegConfig().class_balancing is ClassBalancing.INVERSE_FREQUENCY
    assert DinoLinearSegConfig().pixel_sampling is PixelSampling.PER_CLASS
    with pytest.raises(ValidationError):
        DinoLinearSegConfig.model_validate({"pixel_sampling": "random"})


def test_a_frame_the_patch_does_not_divide_is_refused_at_creation() -> None:
    config = DinoLinearSegConfig()
    DinoLinearSegModel.check_input(config, PreprocessingConfig(width=224, height=224))
    with pytest.raises(ValueError, match="divisible by 14"):
        DinoLinearSegModel.check_input(config, PreprocessingConfig(width=256, height=224))
