"""The plugin layer: preprocessing, diagnostics, the registry, and `pixel_reference`.

`pixel_reference` is exercised as a whole plugin — fit, predict, save, load — because it
is the reference implementation of the interface. If the contract in `models/base.py` is
wrong, it is wrong here first.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from anomaly_lab.models.base import (
    Capabilities,
    Device,
    ImageRecord,
    InferContext,
    NullReporter,
    TrainContext,
    evenly_spaced,
)
from anomaly_lab.models.device import resolve_device
from anomaly_lab.models.diagnostics import (
    DiagnosticError,
    DiagnosticKind,
    DiagnosticScope,
    DiagnosticWriter,
    load_index,
)
from anomaly_lab.models.dinomaly_anomalib import (
    BASE_LR,
    FINAL_LR,
    DinomalyConfig,
    learning_rate,
    plan_training,
    validate_prepared_size,
)
from anomaly_lab.models.feature_view import pca_to_rgb
from anomaly_lab.models.glass_anomalib import GlassConfig
from anomaly_lab.models.glass_anomalib import plan_training as plan_glass_training
from anomaly_lab.models.patchcore_anomalib import plan_bank
from anomaly_lab.models.pixel_reference import PixelReferenceConfig, PixelReferenceModel
from anomaly_lab.models.preprocessing import (
    ColorMode,
    PreprocessingConfig,
    load_array,
    load_mask,
    to_chw,
)
from anomaly_lab.models.registry import UnknownModelError, describe_all, get_model_class
from tests.conftest import write_image

# ----------------------------------------------------------------- preprocessing


def test_model_input_refuses_pixels_outside_the_frozen_prepared_shape(tmp_path: Path) -> None:
    """Spatial preparation happens once; this bridge may not quietly resize it again."""
    small = write_image(tmp_path / "small.png", size=(11, 7))
    config = PreprocessingConfig(width=32, height=24)

    with pytest.raises(ValueError, match="expected frozen input"):
        load_array(small, config)


def test_pixels_arrive_as_float32_in_zero_to_one(tmp_path: Path) -> None:
    array = load_array(write_image(tmp_path / "a.png"), PreprocessingConfig(width=8, height=8))
    assert array.dtype == np.float32
    assert float(array.min()) >= 0.0 and float(array.max()) <= 1.0


def test_grayscale_mode_produces_one_channel(tmp_path: Path) -> None:
    config = PreprocessingConfig(width=16, height=16, color=ColorMode.GRAYSCALE)
    assert load_array(write_image(tmp_path / "g.png", mode="L", size=(16, 16)), config).shape == (
        16,
        16,
        1,
    )


def test_a_grayscale_file_still_expands_to_three_under_rgb(tmp_path: Path) -> None:
    """Pretrained backbones expect three channels; a mono dataset must not be a special case."""
    config = PreprocessingConfig(width=16, height=16, color=ColorMode.RGB)
    assert load_array(write_image(tmp_path / "g.png", mode="L", size=(16, 16)), config).shape == (
        16,
        16,
        3,
    )


def test_to_chw_gives_torch_the_layout_it_wants(tmp_path: Path) -> None:
    array = load_array(
        write_image(tmp_path / "a.png", size=(16, 8)),
        PreprocessingConfig(width=16, height=8),
    )
    assert to_chw(array).shape == (3, 8, 16)


def test_a_mask_is_any_nonzero_pixel(tmp_path: Path) -> None:
    """0/1 and 0/255 encodings both occur; a fixed 127 threshold would drop the first."""
    path = tmp_path / "m.png"
    write_image(path, mode="L", size=(4, 4), colour=0)
    from PIL import Image

    image = Image.new("L", (4, 4), 0)
    image.putpixel((1, 1), 1)
    image.putpixel((2, 2), 255)
    image.save(path)

    mask = load_mask(path)
    assert mask.sum() == 2


def test_a_mask_resize_never_interpolates(tmp_path: Path) -> None:
    """Interpolating a label map invents labels that are neither 0 nor 1."""
    from PIL import Image

    path = tmp_path / "m.png"
    array = np.zeros((8, 8), dtype=np.uint8)
    array[2:4, 2:4] = 255
    Image.fromarray(array, mode="L").save(path)

    resized = load_mask(path, size=(4, 4))
    assert resized.dtype == np.bool_
    assert set(np.unique(resized).tolist()) <= {True, False}


# ----------------------------------------------------------------- diagnostics


def test_a_disabled_writer_accepts_everything_and_writes_nothing(tmp_path: Path) -> None:
    """A plugin must never have to ask whether diagnostics are wanted."""
    writer = DiagnosticWriter(tmp_path / "diag", enabled=False)
    writer.emit("m", "M", DiagnosticKind.MAP, np.zeros((4, 4), dtype=np.float32))

    assert writer.flush().entries == []
    assert not (tmp_path / "diag").exists()


def test_an_array_diagnostic_lands_as_float32_npy_with_an_index(tmp_path: Path) -> None:
    root = tmp_path / "diag"
    writer = DiagnosticWriter(root)
    writer.emit("z_map", "Z map", DiagnosticKind.MAP, np.ones((4, 5), dtype=np.float64))
    index = writer.flush()

    entry = index.entries[0]
    assert entry.kind is DiagnosticKind.MAP
    assert entry.scope is DiagnosticScope.MODEL
    assert entry.shape == [4, 5]
    assert entry.path is not None
    assert np.load(root / entry.path).dtype == np.float32
    assert load_index(root).entries == index.entries


def test_an_image_scoped_diagnostic_is_filed_under_its_image(tmp_path: Path) -> None:
    writer = DiagnosticWriter(tmp_path / "diag")
    writer.emit("e", "E", DiagnosticKind.MAP, np.zeros((2, 2), dtype=np.float32), image_id=17)
    entry = writer.flush().entries[0]

    assert entry.scope is DiagnosticScope.IMAGE
    assert entry.image_id == 17
    assert entry.path == "image-17/e.npy"


def test_json_kinds_are_stored_inline_rather_than_as_files(tmp_path: Path) -> None:
    writer = DiagnosticWriter(tmp_path / "diag")
    writer.emit("architecture", "Arch", DiagnosticKind.GRAPH, {"nodes": [], "edges": []})
    entry = writer.flush().entries[0]

    assert entry.path is None
    assert entry.payload == {"nodes": [], "edges": []}


def test_the_image_budget_is_enforced_and_reported(tmp_path: Path) -> None:
    """A silent truncation would read as 'this is all there was' (`no silent caps`)."""
    writer = DiagnosticWriter(tmp_path / "diag", image_budget=2)
    for image_id in (1, 2, 3, 4):
        writer.emit("m", "M", DiagnosticKind.MAP, np.zeros((2, 2), np.float32), image_id=image_id)

    index = writer.flush()
    assert len(index.entries) == 2
    assert index.image_budget == 2
    assert index.truncated_images == 2


def test_the_kept_images_are_spread_across_the_run_not_taken_from_its_front(
    tmp_path: Path,
) -> None:
    """Twelve neighbours are not a sample of five hundred images.

    The budget used to be filled by arrival order, so on the reference run it kept images
    151-162 out of a scored range of 151-1249 — twelve consecutive files from one corner
    of the dataset, presented as what the run recorded. `evenly_spaced` exists for exactly
    this and is used everywhere else a cost is capped.
    """
    scored = list(range(100, 200))
    chosen = [scored[index] for index in evenly_spaced(len(scored), 5)]
    writer = DiagnosticWriter(tmp_path / "diag", image_budget=5, keep_images=chosen)
    for image_id in scored:
        writer.emit("m", "M", DiagnosticKind.MAP, np.zeros((2, 2), np.float32), image_id=image_id)

    index = writer.flush()
    kept = sorted(entry.image_id for entry in index.entries if entry.image_id is not None)
    assert kept == [100, 125, 150, 174, 199], "both ends and the middle, not the first five"
    assert index.truncated_images == 95


def test_without_a_chosen_set_the_budget_still_bounds_the_run(tmp_path: Path) -> None:
    """A caller that cannot know its image list up front still gets a bounded run."""
    writer = DiagnosticWriter(tmp_path / "diag", image_budget=2)
    for image_id in (7, 8, 9):
        writer.emit("m", "M", DiagnosticKind.MAP, np.zeros((2, 2), np.float32), image_id=image_id)

    index = writer.flush()
    assert len(index.entries) == 2
    assert index.truncated_images == 1


def test_an_image_already_within_budget_keeps_all_of_its_diagnostics(tmp_path: Path) -> None:
    """The budget counts images, not emissions — half a diagnostic set is worse than none."""
    writer = DiagnosticWriter(tmp_path / "diag", image_budget=1)
    writer.emit("a", "A", DiagnosticKind.MAP, np.zeros((2, 2), np.float32), image_id=5)
    writer.emit("b", "B", DiagnosticKind.MAP, np.zeros((2, 2), np.float32), image_id=5)

    assert len(writer.flush().entries) == 2


def test_a_wrongly_shaped_diagnostic_is_a_plugin_bug_not_a_silent_write(tmp_path: Path) -> None:
    writer = DiagnosticWriter(tmp_path / "diag")
    with pytest.raises(DiagnosticError, match="must be 2-D"):
        writer.emit("m", "M", DiagnosticKind.MAP, np.zeros((2, 2, 2), dtype=np.float32))
    with pytest.raises(DiagnosticError, match="3 channels"):
        writer.emit("i", "I", DiagnosticKind.IMAGE, np.zeros((2, 2, 4), dtype=np.float32))


def test_a_key_that_would_escape_the_directory_is_refused(tmp_path: Path) -> None:
    writer = DiagnosticWriter(tmp_path / "diag")
    with pytest.raises(DiagnosticError, match="identifier"):
        writer.emit("../escape", "X", DiagnosticKind.MAP, np.zeros((2, 2), dtype=np.float32))


def test_reading_an_index_that_was_never_written_is_empty_not_an_error(tmp_path: Path) -> None:
    assert load_index(tmp_path / "nothing").entries == []


def test_one_key_gets_one_range_across_every_image_that_emitted_it(tmp_path: Path) -> None:
    """Per-array normalization makes a clean part look as alarming as a defective one."""
    writer = DiagnosticWriter(tmp_path / "diag")
    writer.emit("m", "M", DiagnosticKind.MAP, np.full((8, 8), 0.2, np.float32), image_id=1)
    writer.emit("m", "M", DiagnosticKind.MAP, np.full((8, 8), 5.0, np.float32), image_id=2)

    span = writer.flush().ranges["m"]
    assert span.low == pytest.approx(0.2)
    assert span.high == pytest.approx(5.0)


def test_the_range_high_end_is_a_percentile_so_one_hot_pixel_cannot_flatten_a_run(
    tmp_path: Path,
) -> None:
    """The same choice `write_map` makes, for the same reason."""
    values = np.zeros((32, 32), dtype=np.float32)
    values[0, 0] = 1000.0

    writer = DiagnosticWriter(tmp_path / "diag")
    writer.emit("m", "M", DiagnosticKind.MAP, values)

    assert writer.flush().ranges["m"].high < 1000.0


def test_a_kind_rendered_without_a_colormap_records_no_range(tmp_path: Path) -> None:
    """An `image` payload is already a picture; a range for it would never be read."""
    writer = DiagnosticWriter(tmp_path / "diag")
    writer.emit("pca", "PCA", DiagnosticKind.IMAGE, np.zeros((4, 4, 3), np.float32))
    writer.emit("arch", "Arch", DiagnosticKind.GRAPH, {"nodes": []})

    assert writer.flush().ranges == {}


def test_re_running_inference_replaces_its_whole_per_image_sample(tmp_path: Path) -> None:
    """A second run's index is that run's sample, not the union of two.

    Invisible while the budget kept the first N: every run chose the same images, so every
    entry was superseded by id. Once the budget spread across the run, two runs sampled
    different images and the index became 74 images under a stated budget of 64 — a cap
    that reads as broken, over a selection neither run made.
    """
    root = tmp_path / "diag"

    training = DiagnosticWriter(root)
    training.emit("teacher", "T", DiagnosticKind.MAP, np.zeros((4, 4), np.float32))
    training.flush()

    first = DiagnosticWriter(root, image_budget=2, keep_images=[1, 2])
    for image_id in (1, 2):
        first.emit("err", "E", DiagnosticKind.MAP, np.zeros((4, 4), np.float32), image_id=image_id)
    first.flush()

    second = DiagnosticWriter(root, image_budget=2, keep_images=[3, 4])
    for image_id in (3, 4):
        second.emit("err", "E", DiagnosticKind.MAP, np.zeros((4, 4), np.float32), image_id=image_id)
    merged = second.flush()

    per_image = sorted(entry.image_id for entry in merged.entries if entry.image_id is not None)
    assert per_image == [3, 4], "the earlier run's images are gone, not merged in"
    assert len(per_image) <= (merged.image_budget or 0), "the stated budget is the real one"
    # The training run's own diagnostic is untouched — that is the M3 bug this merge exists
    # to prevent, and narrowing the rule must not reintroduce it.
    assert any(entry.key == "teacher" for entry in merged.entries)


def test_a_later_run_keeps_the_earlier_run_s_ranges_and_replaces_its_own(
    tmp_path: Path,
) -> None:
    """Ranges merge by the same rule as entries — the M3 bug, one level coarser."""
    root = tmp_path / "diag"

    training = DiagnosticWriter(root)
    training.emit("teacher", "T", DiagnosticKind.MAP, np.full((4, 4), 3.0, np.float32))
    training.flush()

    inference = DiagnosticWriter(root)
    inference.emit("err", "E", DiagnosticKind.MAP, np.full((4, 4), 9.0, np.float32), image_id=1)
    merged = inference.flush()

    assert set(merged.ranges) == {"teacher", "err"}
    assert merged.ranges["teacher"].high == pytest.approx(3.0)
    assert merged.ranges["err"].high == pytest.approx(9.0)
    assert set(load_index(root).ranges) == {"teacher", "err"}


def test_an_index_written_before_ranges_existed_still_loads(tmp_path: Path) -> None:
    """The field is additive, so `INDEX_VERSION` did not move; prove the reader agrees."""
    root = tmp_path / "diag"
    root.mkdir()
    (root / "diagnostics.json").write_text(
        '{"version": 1, "entries": [], "truncated_images": 0}', encoding="utf-8"
    )

    assert load_index(root).ranges == {}


# ----------------------------------------------------------------- registry


def test_dinomaly_plan_is_bounded_and_torch_free() -> None:
    plan = plan_training(DinomalyConfig(max_steps=123), 17, 392, 196)

    assert plan.steps == 123
    assert plan.images_available == 17
    assert "123 batch-1 steps" in plan.describe()
    assert "246 MB" in plan.describe()


def test_dinomaly_refuses_a_partial_patch_before_training() -> None:
    with pytest.raises(ValueError, match=r"divisible by 14.*390x392"):
        validate_prepared_size(390, 392)


def test_dinomaly_schedule_has_a_fixed_resume_safe_horizon() -> None:
    assert learning_rate(0) == 0.0
    assert learning_rate(99) == pytest.approx(BASE_LR)
    assert learning_rate(100) == pytest.approx(BASE_LR)
    assert learning_rate(5_000) == pytest.approx(FINAL_LR)
    assert learning_rate(50_000) == pytest.approx(FINAL_LR)


def test_glass_plan_bounds_and_announces_every_center_pass() -> None:
    plan = plan_glass_training(
        GlassConfig(
            max_steps=1_000,
            center_images=128,
            center_refresh_steps=392,
            mining_steps=20,
        ),
        500,
        288,
        288,
    )

    assert plan.center_images == 128
    assert plan.center_images_dropped == 372
    assert plan.center_passes == 3
    assert "3 centre passes x 128/500" in plan.describe()
    assert "20-step mining" in plan.describe()


def test_glass_refuses_an_unusable_prepared_frame_before_importing_torch() -> None:
    with pytest.raises(ValueError, match=r"at least 64 pixels.*63x288"):
        plan_glass_training(GlassConfig(max_steps=1), 5, 63, 288)


def test_every_registered_method_describes_itself_without_importing_torch() -> None:
    """`describe_all` is called whenever the picker opens; it must stay cheap."""
    described = describe_all()
    keys = {entry.key for entry in described}
    assert {
        "pixel_reference",
        "efficientad_anomalib",
        "dinomaly_anomalib",
        "glass_anomalib",
    } <= keys

    for entry in described:
        assert entry.title
        assert entry.summary
        assert entry.config_schema["type"] == "object"
        assert isinstance(entry.capabilities, Capabilities)


def test_an_unknown_key_names_the_ones_that_exist() -> None:
    with pytest.raises(UnknownModelError, match="pixel_reference"):
        get_model_class("no_such_method")


def test_a_method_with_missing_dependencies_says_how_to_install_them() -> None:
    """Availability is a UI affordance, not an exception minutes into a job."""
    from anomaly_lab.models.base import module_available

    unavailable = module_available("definitely_not_installed", "dl", "Something")
    assert unavailable.available is False
    assert "uv sync" in (unavailable.reason or "")


def test_cpu_resolution_never_touches_torch() -> None:
    resolved = resolve_device(Device.CPU)
    assert resolved.device is Device.CPU
    assert "preferred" in resolved.reason


def test_device_resolution_always_produces_a_device_and_a_reason() -> None:
    """It must never raise: an unavailable accelerator is a slower run, not a failure."""
    for preferred in (Device.CPU, Device.MPS, Device.CUDA):
        resolved = resolve_device(preferred)
        assert resolved.device in set(Device)
        assert resolved.reason


# ----------------------------------------------------------------- pixel_reference


def _contexts(tmp_path: Path, config: PreprocessingConfig) -> tuple[TrainContext, InferContext]:
    (tmp_path / "artifacts").mkdir(parents=True, exist_ok=True)
    return (
        TrainContext(
            artifact_dir=tmp_path / "artifacts",
            cache_dir=tmp_path / "cache",
            preprocessing=config,
            device=Device.CPU,
            reporter=NullReporter(),
            diagnostics=DiagnosticWriter(tmp_path / "d1"),
        ),
        InferContext(
            artifact_dir=tmp_path / "artifacts",
            cache_dir=tmp_path / "cache",
            preprocessing=config,
            device=Device.CPU,
            reporter=NullReporter(),
            diagnostics=DiagnosticWriter(tmp_path / "d2"),
        ),
    )


def _normal(path: Path, seed: int) -> Path:
    """A textured but consistent image — the same scene with a little noise."""
    from PIL import Image

    generator = np.random.default_rng(seed)
    base = np.linspace(40, 200, 16 * 16).reshape(16, 16)
    noisy = np.clip(base + generator.normal(0, 3, size=(16, 16)), 0, 255).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(noisy, mode="L").convert("RGB").save(path)
    return path


def _defective(path: Path, seed: int) -> Path:
    """The same scene with a bright square stamped into it."""
    from PIL import Image

    generator = np.random.default_rng(seed)
    base = np.linspace(40, 200, 16 * 16).reshape(16, 16)
    noisy = np.clip(base + generator.normal(0, 3, size=(16, 16)), 0, 255)
    noisy[5:10, 5:10] = 255
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(noisy.astype(np.uint8), mode="L").convert("RGB").save(path)
    return path


def test_pixel_reference_separates_a_stamped_defect_from_normals(tmp_path: Path) -> None:
    """The floor baseline has to actually work, or it is not a floor."""
    config = PreprocessingConfig(width=16, height=16)
    train_ctx, infer_ctx = _contexts(tmp_path, config)

    train = [
        ImageRecord(
            image_id=index, sample_id=index, path=_normal(tmp_path / f"n{index}.png", index)
        )
        for index in range(8)
    ]
    model = PixelReferenceModel(PixelReferenceConfig(smoothing_sigma=1.0))
    model.fit(train, train_ctx)

    probe = [
        ImageRecord(image_id=100, sample_id=100, path=_normal(tmp_path / "test-n.png", 99)),
        ImageRecord(image_id=101, sample_id=101, path=_defective(tmp_path / "test-d.png", 98)),
    ]
    predictions = model.predict(probe, infer_ctx)

    assert len(predictions) == 2
    assert predictions[1].score > predictions[0].score


def test_predictions_come_back_one_per_input_in_order(tmp_path: Path) -> None:
    """The contract the infer handler checks; assert it at the plugin too."""
    config = PreprocessingConfig(width=16, height=16)
    train_ctx, infer_ctx = _contexts(tmp_path, config)
    train = [
        ImageRecord(
            image_id=index, sample_id=index, path=_normal(tmp_path / f"n{index}.png", index)
        )
        for index in range(4)
    ]
    model = PixelReferenceModel(PixelReferenceConfig())
    model.fit(train, train_ctx)

    predictions = model.predict(train, infer_ctx)
    assert [p.image_id for p in predictions] == [record.image_id for record in train]


def test_maps_are_written_as_float32_npy(tmp_path: Path) -> None:
    config = PreprocessingConfig(width=16, height=16)
    train_ctx, infer_ctx = _contexts(tmp_path, config)
    train = [
        ImageRecord(
            image_id=index, sample_id=index, path=_normal(tmp_path / f"n{index}.png", index)
        )
        for index in range(4)
    ]
    model = PixelReferenceModel(PixelReferenceConfig())
    model.fit(train, train_ctx)
    prediction = model.predict(train[:1], infer_ctx)[0]

    assert prediction.anomaly_map is not None
    stored = np.load(prediction.anomaly_map)
    assert stored.dtype == np.float32
    assert stored.shape == (16, 16)


def test_a_saved_model_reloads_to_the_same_scores(tmp_path: Path) -> None:
    """Reopening a past experiment must reproduce its numbers, not approximate them."""
    config = PreprocessingConfig(width=16, height=16)
    train_ctx, infer_ctx = _contexts(tmp_path, config)
    train = [
        ImageRecord(
            image_id=index, sample_id=index, path=_normal(tmp_path / f"n{index}.png", index)
        )
        for index in range(6)
    ]

    model = PixelReferenceModel(PixelReferenceConfig())
    model.fit(train, train_ctx)
    before = [p.score for p in model.predict(train, infer_ctx)]

    saved = tmp_path / "model"
    saved.mkdir()
    model.save(saved)

    restored = PixelReferenceModel(PixelReferenceConfig())
    restored.load(saved)
    after = [p.score for p in restored.predict(train, infer_ctx)]

    assert before == pytest.approx(after)


def test_the_reference_cap_samples_across_the_set_not_the_first_n(tmp_path: Path) -> None:
    """Datasets arrive in acquisition order; the first N is often one batch."""
    from anomaly_lab.models.base import evenly_spaced

    assert evenly_spaced(4, 10) == [0, 1, 2, 3]
    chosen = evenly_spaced(100, 5)
    assert chosen[0] == 0
    assert chosen[-1] == 99
    assert len(chosen) == 5


def test_predicting_before_fitting_is_an_error_not_a_zero(tmp_path: Path) -> None:
    _, infer_ctx = _contexts(tmp_path, PreprocessingConfig())
    model = PixelReferenceModel(PixelReferenceConfig())
    with pytest.raises(RuntimeError, match="before it was fitted"):
        model.predict([ImageRecord(image_id=1, sample_id=1, path=tmp_path / "x.png")], infer_ctx)


def test_pixel_reference_records_diagnostics_through_the_contract(tmp_path: Path) -> None:
    """It produces them without torch, which is what makes the contract testable at all."""
    config = PreprocessingConfig(width=16, height=16)
    train_ctx, infer_ctx = _contexts(tmp_path, config)
    train = [
        ImageRecord(
            image_id=index, sample_id=index, path=_normal(tmp_path / f"n{index}.png", index)
        )
        for index in range(4)
    ]
    model = PixelReferenceModel(PixelReferenceConfig())
    model.fit(train, train_ctx)
    model.predict(train[:1], infer_ctx)

    train_keys = {entry.key for entry in train_ctx.diagnostics.flush().entries}
    infer_index = infer_ctx.diagnostics.flush()
    assert {"reference_median", "reference_scale"} <= train_keys
    assert any(entry.key == "z_map_raw" for entry in infer_index.entries)


def test_the_display_range_is_run_wide_not_per_image(tmp_path: Path) -> None:
    """Two maps on different scales would make a clean part look like a defective one."""
    config = PreprocessingConfig(width=16, height=16)
    train_ctx, infer_ctx = _contexts(tmp_path, config)
    train = [
        ImageRecord(
            image_id=index, sample_id=index, path=_normal(tmp_path / f"n{index}.png", index)
        )
        for index in range(4)
    ]
    model = PixelReferenceModel(PixelReferenceConfig())
    model.fit(train, train_ctx)
    model.predict(train, infer_ctx)

    span = infer_ctx.display_range()
    assert span is not None
    assert span[1] > span[0]


def test_a_later_run_does_not_erase_an_earlier_run_s_diagnostics(tmp_path: Path) -> None:
    """An experiment's diagnostics come from two runs, and both write this file.

    `fit` records the architecture; `predict` records the per-image error maps. Replacing
    the index rather than merging it made inference silently erase training's entries —
    the arrays stayed on disk, unreferenced, and the architecture view had nothing to draw.
    """
    root = tmp_path / "diag"

    training = DiagnosticWriter(root)
    training.emit("architecture", "Arch", DiagnosticKind.GRAPH, {"nodes": []})
    training.emit("teacher_magnitude", "T", DiagnosticKind.MAP, np.zeros((2, 2), np.float32))
    training.flush()

    inference = DiagnosticWriter(root)
    inference.emit("map_st", "ST", DiagnosticKind.MAP, np.zeros((2, 2), np.float32), image_id=1)
    merged = inference.flush()

    expected = {"architecture", "teacher_magnitude", "map_st"}
    assert {entry.key for entry in merged.entries} == expected
    assert {entry.key for entry in load_index(root).entries} == expected


def test_re_running_replaces_its_own_entries_rather_than_duplicating_them(tmp_path: Path) -> None:
    root = tmp_path / "diag"
    index = None
    for _ in range(3):
        writer = DiagnosticWriter(root)
        writer.emit("m", "M", DiagnosticKind.MAP, np.zeros((2, 2), np.float32), image_id=7)
        index = writer.flush()

    assert index is not None
    assert len(index.entries) == 1


def test_a_map_with_a_stray_channel_axis_is_squeezed_not_stored(tmp_path: Path) -> None:
    """Torch hands back `(1, H, W)`; a stored 3-D map fails much later, in evaluation."""
    _, infer_ctx = _contexts(tmp_path, PreprocessingConfig(width=16, height=16))
    path = infer_ctx.write_map(1, np.zeros((1, 16, 16), dtype=np.float32))
    assert np.load(path).shape == (16, 16)


def test_a_genuinely_wrong_map_shape_is_reported_at_the_plugin(tmp_path: Path) -> None:
    _, infer_ctx = _contexts(tmp_path, PreprocessingConfig(width=16, height=16))
    with pytest.raises(ValueError, match="must be 2-D"):
        infer_ctx.write_map(1, np.zeros((3, 16, 16), dtype=np.float32))


def test_a_map_projector_runs_before_persistence_and_ignores_uncovered_extremes(
    tmp_path: Path,
) -> None:
    _, infer_ctx = _contexts(tmp_path, PreprocessingConfig(width=8, height=8))

    def project(_: int, values: np.ndarray) -> np.ndarray:
        source = np.full((12, 12), np.nan, dtype=np.float32)
        source[2:10, 2:10] = values
        return source

    infer_ctx.map_projector = project
    path = infer_ctx.write_map(7, np.arange(64, dtype=np.float32).reshape(8, 8))

    stored = np.load(path)
    assert stored.shape == (12, 12)
    assert np.isnan(stored[0, 0])
    assert infer_ctx.display_range() == pytest.approx((0.0, 62.937))


# ----------------------------------------------------------- patchcore, torch-free parts

# `plan_bank` and `pca_to_rgb` are deliberately pure — plain numbers and numpy in, a plan or
# an image out — the same split `introspect.build_tree` has from `introspect.collect`. They
# live here rather than in `test_dl_patchcore.py` so the arithmetic that decides whether a
# fit exhausts the machine is checked by the CI job that installs *without* the `dl` extra.


def test_a_bank_that_fits_under_both_caps_keeps_everything() -> None:
    plan = plan_bank(
        images_available=100,
        patches_per_image=1024,
        embedding_dim=1536,
        max_bank_images=512,
        max_candidate_vectors=1_000_000,
        coreset_ratio=0.1,
    )
    assert plan.images_used == 100
    assert plan.patches_kept_per_image == 1024
    assert plan.candidates_kept == plan.candidates_generated == 102_400
    assert plan.coreset_size == 10_240
    assert plan.images_dropped == 0 and plan.patches_dropped_per_image == 0


def test_the_image_cap_binds_before_the_vector_cap() -> None:
    """Images are dropped first and patches thinned second, and the order is not arbitrary.

    Patches inside one image overlap through the 3x3 pooling and are largely redundant;
    two images differ by whatever the process actually varies. So a bank sees as many
    images as it is allowed and thins within them, never the reverse.
    """
    plan = plan_bank(
        images_available=900,
        patches_per_image=1024,
        embedding_dim=1536,
        max_bank_images=512,
        max_candidate_vectors=100_000,
        coreset_ratio=0.1,
    )
    assert plan.images_used == 512
    assert plan.patches_kept_per_image == 195
    assert plan.candidates_kept == 512 * 195
    assert plan.candidates_kept <= 100_000
    assert plan.images_dropped == 388


def test_the_vector_cap_is_never_overshot() -> None:
    """The cap is a ceiling, and a plan that exceeded it would be a cap that does not work."""
    for images in (1, 7, 64, 513, 5000):
        plan = plan_bank(
            images_available=images,
            patches_per_image=1024,
            embedding_dim=384,
            max_bank_images=512,
            max_candidate_vectors=10_000,
            coreset_ratio=0.1,
        )
        assert plan.candidates_kept <= 10_000
        assert plan.coreset_size <= plan.candidates_kept


def test_a_cap_below_the_image_count_takes_images_off_too() -> None:
    """One patch per image is the floor, so a tiny vector cap has to reduce images as well.

    Degenerate, and reachable from the form — which is exactly why it must not silently
    produce a pool larger than the number it was given.
    """
    plan = plan_bank(
        images_available=500,
        patches_per_image=1024,
        embedding_dim=384,
        max_bank_images=500,
        max_candidate_vectors=40,
        coreset_ratio=0.5,
    )
    assert plan.images_used == 40
    assert plan.patches_kept_per_image == 1
    assert plan.candidates_kept == 40


def test_a_ratio_that_would_select_nothing_selects_one() -> None:
    """An empty bank is not a smaller bank, it is a model that cannot score at all."""
    plan = plan_bank(
        images_available=1,
        patches_per_image=4,
        embedding_dim=16,
        max_bank_images=1,
        max_candidate_vectors=1000,
        coreset_ratio=0.001,
    )
    assert plan.coreset_size == 1


def test_the_plan_reports_the_footprint_the_caps_avoided() -> None:
    plan = plan_bank(
        images_available=900,
        patches_per_image=1024,
        embedding_dim=1536,
        max_bank_images=512,
        max_candidate_vectors=100_000,
        coreset_ratio=0.1,
    )
    # The 5.66 GB the module docstring quotes, and the reason the caps exist.
    assert plan.unbounded_candidates == 921_600
    assert plan.unbounded_candidates * plan.embedding_dim * 4 == pytest.approx(5.66e9, rel=0.01)
    assert "uncapped" in plan.describe()
    assert "921,600" in plan.describe()


def test_a_plan_refuses_a_grid_it_could_not_have_measured() -> None:
    with pytest.raises(ValueError, match="patch grid"):
        plan_bank(
            images_available=10,
            patches_per_image=0,
            embedding_dim=384,
            max_bank_images=8,
            max_candidate_vectors=100,
            coreset_ratio=0.1,
        )


def test_pca_to_rgb_produces_a_renderable_image_payload() -> None:
    """The `image` kind's contract: rank 3, three trailing channels, already in [0, 1]."""
    rng = np.random.default_rng(0)
    features = rng.normal(size=(64, 8, 12)).astype(np.float32)

    composite = pca_to_rgb(features)

    assert composite.shape == (8, 12, 3)
    assert composite.dtype == np.float32
    assert float(composite.min()) >= 0.0 and float(composite.max()) <= 1.0


def test_pca_to_rgb_survives_a_feature_map_that_does_not_vary() -> None:
    """A constant component divides by zero, and NaNs reach the renderer as an unexplained
    black tile rather than as an error."""
    composite = pca_to_rgb(np.ones((8, 4, 4), dtype=np.float32))
    assert bool(np.isfinite(composite).all())


def test_pca_to_rgb_refuses_a_shape_it_cannot_reduce() -> None:
    with pytest.raises(ValueError, match="at least 3 channels"):
        pca_to_rgb(np.zeros((2, 4, 4), dtype=np.float32))
    with pytest.raises(ValueError, match="\\(C, H, W\\)"):
        pca_to_rgb(np.zeros((4, 4), dtype=np.float32))
