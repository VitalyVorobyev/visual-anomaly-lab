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
    LogitBias,
    PixelSampling,
    allocate_pixels,
    class_weights,
    feature_dim,
    fit_class_bias,
    pixel_features,
    pixel_weights,
    plan_pixels,
    prior_shift,
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


def test_under_inverse_frequency_the_shift_is_the_log_prior_of_the_images() -> None:
    available = np.array([99_000, 900, 0, 100])
    counts = np.array([500, 400, 0, 100])
    weights = class_weights(counts, ClassBalancing.INVERSE_FREQUENCY)
    shift = prior_shift(available, counts, weights)
    # The fit saw every learned class as equally common, so only the images' prior is left.
    assert shift[0] == 0.0 and shift[2] == 0.0
    assert shift[1] == pytest.approx(np.log(900 / 99_000), rel=1e-5)
    assert shift[3] == pytest.approx(np.log(100 / 99_000), rel=1e-5)


def test_an_unweighted_fit_on_a_representative_sample_needs_no_shift() -> None:
    available = np.array([8_000, 1_600, 400])
    counts = available // 100
    shift = prior_shift(available, counts, class_weights(counts, ClassBalancing.NONE))
    np.testing.assert_allclose(shift, 0.0, atol=1e-6)


def test_an_unweighted_fit_on_a_per_class_sample_is_shifted_by_what_sampling_did() -> None:
    available = np.array([9_900, 100])
    counts = np.array([50, 50])
    shift = prior_shift(available, counts, class_weights(counts, ClassBalancing.NONE))
    assert shift[1] == pytest.approx(np.log(0.01 / 0.5) - np.log(0.99 / 0.5), rel=1e-5)


def test_restoring_the_prior_hands_an_even_pixel_back_to_the_common_class() -> None:
    """A pixel the balanced head calls a toss-up is, among the images' pixels, background."""
    available = np.array([9_990, 10])
    counts = np.array([100, 100])
    shift = prior_shift(available, counts, class_weights(counts, ClassBalancing.INVERSE_FREQUENCY))
    balanced = np.array([0.0, 0.5])
    assert int(np.argmax(balanced)) == 1
    assert int(np.argmax(balanced + shift)) == 0
    # A class the head is sure enough of survives it.
    assert int(np.argmax(np.array([0.0, 8.0]) + shift)) == 1


def test_a_class_the_fit_never_learned_is_not_shifted() -> None:
    available = np.array([100, 0, 5])
    counts = np.array([10, 0, 0])
    shift = prior_shift(available, counts, class_weights(counts, ClassBalancing.NONE))
    np.testing.assert_array_equal(shift, [0.0, 0.0, 0.0])


def test_a_sampled_pixel_stands_for_its_share_of_its_class_in_its_image() -> None:
    weights = pixel_weights(np.array([0, 0, 1, 0]), np.array([900, 5, 0]))
    np.testing.assert_allclose(weights, [300, 300, 5, 300])


def _binary(class_logits: list[float]) -> np.ndarray:
    return np.stack([np.zeros(len(class_logits)), np.array(class_logits)], axis=1)


def test_the_held_out_bias_is_the_cut_with_the_highest_iou() -> None:
    logits = _binary([5, 4, 3, 2, 1])
    labels = np.array([1, 1, 0, 1, 0])
    # Taking the top 1..5 pixels gives IoU 1/3, 2/3, 1/2, 3/4, 3/5: the best keeps four, so
    # the class needs a logit above 1.5, midway between the fourth pixel's and the fifth's.
    bias, reached = fit_class_bias(logits, labels, np.ones(5), np.array([True, True]))
    assert bias[0] == 0.0
    assert bias[1] == pytest.approx(-1.5)
    assert reached[1] == pytest.approx(0.75)
    assert np.isnan(reached[0])


def test_the_held_out_bias_reads_the_pixels_as_the_frames_they_stand_for() -> None:
    """A background pixel standing for ten outweighs the class pixel ranked after it."""
    logits = _binary([5, 4, 3, 2, 1])
    labels = np.array([1, 1, 0, 1, 0])
    weights = np.array([1.0, 1.0, 10.0, 1.0, 1.0])
    bias, reached = fit_class_bias(logits, labels, weights, np.array([True, True]))
    assert bias[1] == pytest.approx(-3.5)
    assert reached[1] == pytest.approx(2 / 3)


def test_a_class_no_cut_overlaps_or_the_head_never_learned_keeps_its_answer() -> None:
    logits = np.stack([np.zeros(4), np.arange(4.0), np.arange(4.0)], axis=1)
    labels = np.array([0, 0, 0, 2])
    bias, reached = fit_class_bias(logits, labels, np.ones(4), np.array([True, True, False]))
    np.testing.assert_array_equal(bias, [0.0, 0.0, 0.0])
    assert np.isnan(reached).all()


def test_a_cut_that_takes_every_pixel_still_lands_above_the_last_one() -> None:
    logits = _binary([3, 2, 1])
    bias, reached = fit_class_bias(logits, np.array([1, 1, 1]), np.ones(3), np.ones(2, bool))
    assert reached[1] == pytest.approx(1.0)
    assert np.all(logits[:, 1] + bias[1] > 0)


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
    assert DinoLinearSegConfig().logit_bias is LogitBias.HELD_OUT_IOU
    with pytest.raises(ValidationError):
        DinoLinearSegConfig.model_validate({"logit_bias": "test_prior"})
    with pytest.raises(ValidationError):
        DinoLinearSegConfig.model_validate({"pixel_sampling": "random"})


def test_a_frame_the_patch_does_not_divide_is_refused_at_creation() -> None:
    config = DinoLinearSegConfig()
    DinoLinearSegModel.check_input(config, PreprocessingConfig(width=224, height=224))
    with pytest.raises(ValueError, match="divisible by 14"):
        DinoLinearSegModel.check_input(config, PreprocessingConfig(width=256, height=224))
