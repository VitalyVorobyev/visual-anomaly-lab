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
from anomaly_lab.models.coreset import johnson_lindenstrauss_dim
from anomaly_lab.models.device import resolve_device
from anomaly_lab.models.diagnostics import (
    DiagnosticError,
    DiagnosticKind,
    DiagnosticScope,
    DiagnosticWriter,
    load_index,
)
from anomaly_lab.models.dino_backbone import (
    ACCESS_REQUEST_URLS,
    BACKBONES,
    DinoBackbone,
    FeatureLayers,
    patch_grid,
)
from anomaly_lab.models.dino_backbone import validate_prepared_size as validate_dino_size
from anomaly_lab.models.dino_memory import (
    ChannelFusion,
    DinoMemoryConfig,
    MemoryPlan,
    Scoring,
    Shrinkage,
    channel_order,
    condition_covariance,
    plan_memory,
    precision_from_covariance,
)
from anomaly_lab.models.dinomaly_custom import (
    TARGET_LAYERS,
    DinomalyCustomConfig,
    DinomalyCustomModel,
    fuse_groups,
    trainable_parameter_count,
)
from anomaly_lab.models.dinomaly_custom import plan_training as plan_custom_training
from anomaly_lab.models.feature_view import pca_to_rgb
from anomaly_lab.models.glass_anomalib import GlassConfig, planned_center_refreshes
from anomaly_lab.models.glass_anomalib import plan_training as plan_glass_training
from anomaly_lab.models.patchcore_anomalib import plan_bank
from anomaly_lab.models.pixel_reference import (
    PixelReferenceConfig,
    PixelReferenceModel,
    ReferenceScope,
)
from anomaly_lab.models.preprocessing import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    ColorMode,
    PreprocessingConfig,
    expand_planes,
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


def test_grayscale_stays_one_plane_through_the_bridge(tmp_path: Path) -> None:
    """The seam that makes the colour option mean something.

    If `load_array` expanded to three planes itself, `grayscale` and `rgb` would produce
    identical arrays for a mono file — the experiment would record a choice that changed
    nothing. Plane expansion is the *method's* business, so this must keep failing to
    happen here.
    """
    config = PreprocessingConfig(width=16, height=16, color=ColorMode.GRAYSCALE)
    array = load_array(write_image(tmp_path / "g.png", mode="L", size=(16, 16)), config)
    assert to_chw(array).shape == (1, 16, 16)
    assert config.channels == 1


def test_expand_planes_repeats_a_single_plane() -> None:
    one = np.arange(6, dtype=np.float32).reshape(1, 2, 3)
    widened = expand_planes(one, 3)
    assert widened.shape == (3, 2, 3)
    for plane in range(3):
        assert np.array_equal(widened[plane], one[0])


def test_expand_planes_passes_through_an_already_wide_array() -> None:
    """Callers apply it unconditionally, so the no-op case has to be free and exact."""
    three = np.random.default_rng(0).random((3, 4, 5)).astype(np.float32)
    assert expand_planes(three, 3) is three


def test_expand_planes_refuses_a_count_it_cannot_mean() -> None:
    """Two planes are not a mono image and not an RGB one; guessing would be the bug."""
    with pytest.raises(ValueError, match="cannot expand 2 planes to 3"):
        expand_planes(np.zeros((2, 4, 4), dtype=np.float32), 3)


def test_expanding_then_standardizing_matches_the_broadcast_it_replaces() -> None:
    """PatchCore's grayscale path was right by accident; making it explicit must not move it.

    `(B, 1, H, W) - (1, 3, 1, 1)` broadcasts to the same numbers as expanding first. That
    is why no grayscale PatchCore run was ever wrong — and why the change is safe.
    """
    mono = np.random.default_rng(1).random((1, 1, 4, 4)).astype(np.float32)
    mean = np.asarray(IMAGENET_MEAN, dtype=np.float32).reshape(1, 3, 1, 1)
    std = np.asarray(IMAGENET_STD, dtype=np.float32).reshape(1, 3, 1, 1)

    broadcast = (mono - mean) / std
    explicit = (np.stack([expand_planes(mono[0], 3)]) - mean) / std

    assert np.array_equal(broadcast, explicit)


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


# ------------------------------------------------------- dinomaly_custom, torch-free
#
# Everything below decides whether a `dinomaly_custom` run is legal, reproducible and
# affordable, and all of it is arithmetic. It lives here rather than in
# `test_dl_dinomaly_custom.py` for the reason `plan_bank` does: the CI job that installs
# *without* the `dl` extra is what measures the torch-free boundary, and importing the
# plugin at the top of this file is what asserts the module stays on the right side of it.


def test_dinomaly_custom_plan_is_bounded_and_torch_free() -> None:
    plan = plan_custom_training(
        DinomalyCustomConfig(max_steps=250, batch_size=4, decoder_depth=6), 40, 392, 196
    )

    assert plan.steps == 250
    assert plan.images_available == 40
    assert plan.images_seen == 1_000
    assert (plan.grid_rows, plan.grid_cols) == (14, 28)
    assert plan.positions == 392
    assert plan.embedding_dim == 384
    assert plan.num_heads == 6
    assert plan.decoder_groups == ((0, 1, 2), (3, 4, 5))
    assert plan.encoder_groups == ((0, 1, 2, 3), (4, 5, 6, 7))
    assert plan.trainable_parameters == trainable_parameter_count(384, 6)
    # Weights plus StableAdamW's three amsgrad moment buffers, in float32.
    assert plan.estimated_checkpoint_bytes == plan.trainable_parameters * 16

    described = plan.describe()
    assert "250 steps at batch 4" in described
    assert "dinov2_vit_s14_reg4" in described
    assert "14x28 grid" in described
    assert "6 blocks x 6 heads" in described
    assert [row[0] for row in plan.table()["rows"]].count("decoder") == 1


def test_dinomaly_custom_refuses_a_partial_patch_before_touching_torch() -> None:
    with pytest.raises(ValueError, match=r"divisible by 14.*390x392"):
        plan_custom_training(DinomalyCustomConfig(max_steps=1), 5, 390, 392)


def test_dinomaly_custom_refuses_an_empty_training_set() -> None:
    with pytest.raises(ValueError, match="at least one normal training image"):
        plan_custom_training(DinomalyCustomConfig(max_steps=1), 0, 112, 112)


def test_the_fusion_split_follows_the_decoder_depth() -> None:
    """The claim the retired anomalib wrapper could not make, as arithmetic.

    anomalib hard-codes `[[0, 1, 2, 3], [4, 5, 6, 7]]` for both sides, so its `decoder_depth`
    argument only works at eight. Deriving the split is what makes depth a real field.
    """
    assert fuse_groups(8) == ((0, 1, 2, 3), (4, 5, 6, 7))
    assert fuse_groups(4) == ((0, 1), (2, 3))
    assert fuse_groups(2) == ((0,), (1,))
    # An odd count puts the spare output in the second group, and never leaves one empty.
    assert fuse_groups(5) == ((0, 1), (2, 3, 4))
    for count in range(2, 13):
        first, second = fuse_groups(count)
        assert first and second
        assert first + second == tuple(range(count))

    with pytest.raises(ValueError, match="at least two layer outputs"):
        fuse_groups(1)


def test_the_dinomaly_custom_parameter_count_is_closed_form() -> None:
    """8d² for the bottleneck and 12d² + 8d per decoder block.

    `test_dl_dinomaly_custom.py` checks the same formula against a real network's `numel()`.
    """
    assert trainable_parameter_count(384, 8) == 8 * 384**2 + 8 * (12 * 384**2 + 8 * 384)
    assert trainable_parameter_count(768, 8) > trainable_parameter_count(384, 8)
    with pytest.raises(ValueError, match="positive width and depth"):
        trainable_parameter_count(0, 8)


def test_the_dinomaly_custom_form_offers_the_encoder_menu_and_the_depth() -> None:
    """No frontend work: the picker and the bounds come from this schema alone (ADR-0007)."""
    schema = DinomalyCustomModel.config_model().model_json_schema()
    encoder = schema["properties"]["encoder"]
    # A `$ref` into `$defs`, which is exactly the shape `SchemaForm` resolves into a picker.
    resolved = schema["$defs"][encoder["$ref"].rsplit("/", 1)[-1]]
    assert set(resolved["enum"]) == {backbone.value for backbone in DinoBackbone}
    assert encoder["default"] == DinoBackbone.DINOV2_VIT_S14_REG4.value

    depth = schema["properties"]["decoder_depth"]
    assert (depth["default"], depth["minimum"], depth["maximum"]) == (8, 2, 12)
    for name, field in schema["properties"].items():
        assert field.get("description"), name


def test_dinomaly_custom_availability_names_the_optional_extra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Greyed out with the command that fixes it, never offered and failing inside a worker."""
    import importlib.util

    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    availability = DinomalyCustomModel.availability()

    assert availability.available is False
    assert "--extra dl" in (availability.reason or "")


def test_the_dinomaly_custom_target_layers_are_the_middle_of_the_stack() -> None:
    assert TARGET_LAYERS == (2, 3, 4, 5, 6, 7, 8, 9)
    assert len(TARGET_LAYERS) == 8
    assert max(TARGET_LAYERS) < min(spec.depth for spec in BACKBONES.values())


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


def test_glass_center_refresh_schedule_is_absolute_across_continuations() -> None:
    assert planned_center_refreshes(0, 1_000, 392) == 3
    assert planned_center_refreshes(650, 500, 392) == 1
    assert planned_center_refreshes(784, 1, 392) == 1
    assert planned_center_refreshes(785, 100, 392) == 0


@pytest.mark.parametrize("values", [(-1, 1, 1), (0, 0, 1), (0, 1, 0)])
def test_glass_center_refresh_schedule_rejects_invalid_bounds(
    values: tuple[int, int, int],
) -> None:
    with pytest.raises(ValueError, match="start step"):
        planned_center_refreshes(*values)


def test_every_registered_method_describes_itself_without_importing_torch() -> None:
    """`describe_all` is called whenever the picker opens; it must stay cheap."""
    described = describe_all()
    keys = {entry.key for entry in described}
    assert {
        "pixel_reference",
        "dinomaly_custom",
        "glass_anomalib",
        "dino_memory",
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


# ------------------------------------------------- the DINO backbone, torch-free parts

# The table, the size guard, the grid arithmetic and the projection bound are all pure. They
# live here rather than in `test_dl_dino_backbone.py` for the reason `plan_bank` does: what
# decides whether a run is legal, reproducible and affordable is checked by the CI job that
# installs *without* the `dl` extra, and the modules that hold it must therefore import with
# no torch present at all — which importing them at the top of this file is what asserts.


def test_every_backbone_has_a_spec_and_the_spec_is_self_consistent() -> None:
    assert set(BACKBONES) == set(DinoBackbone)
    for backbone, spec in BACKBONES.items():
        assert spec.timm_name
        assert spec.patch_size in {14, 16}
        assert spec.embedding_dim in {384, 768}
        assert spec.depth == 12
        # A width splits evenly into its heads, or `dinomaly_custom`'s decoder cannot be
        # built from the table at all.
        assert spec.embedding_dim % spec.num_heads == 0
        assert spec.embedding_dim // spec.num_heads == 64
        assert spec.license_note
        assert (backbone in ACCESS_REQUEST_URLS) is spec.gated


def test_the_dinov2_entries_are_the_ungated_ones() -> None:
    """The ungated-default promise, made checkable before anything depends on it.

    A method that picks a gated encoder by default is a method that fails on a fresh machine
    with a 401 rather than a result, so which entries need an approved account is a property
    of the table rather than a note in a docstring.
    """
    ungated = {key for key, spec in BACKBONES.items() if not spec.gated}
    assert ungated == {
        DinoBackbone.DINOV2_VIT_S14,
        DinoBackbone.DINOV2_VIT_S14_REG4,
        DinoBackbone.DINOV2_VIT_B14,
    }
    for key in ungated:
        assert "Apache-2.0" in BACKBONES[key].license_note


def test_feature_layers_index_from_the_end_so_one_name_fits_every_depth() -> None:
    """Negative indices are resolved against the model's own depth by timm."""
    for layers in FeatureLayers:
        indices = layers.indices
        assert indices
        assert all(-12 <= index <= -1 for index in indices)
        assert list(indices) == sorted(indices)
    assert FeatureLayers.LAST.indices == (-1,)
    assert FeatureLayers.MID_LATE.indices == (-7, -4)
    assert len(FeatureLayers.LAST_FOUR.indices) == 4


def test_a_frame_that_does_not_divide_is_refused_by_name() -> None:
    with pytest.raises(ValueError, match=r"dinov3_vit_s16.*divisible by 16.*450x448"):
        validate_dino_size(DinoBackbone.DINOV3_VIT_S16, 450, 448)
    with pytest.raises(ValueError, match=r"dinov2_vit_s14.*divisible by 14.*448x450"):
        validate_dino_size(DinoBackbone.DINOV2_VIT_S14, 448, 450)


def test_the_refusal_names_the_offending_dimension_and_the_sizes_that_work() -> None:
    with pytest.raises(ValueError) as failure:
        validate_dino_size(DinoBackbone.DINOV3_VIT_S16, 450, 448)
    message = str(failure.value)
    assert "width does not divide" in message
    assert "448 or 464" in message


def test_448_is_the_size_both_families_share() -> None:
    """One prepared size that divides by 14 and by 16, so the two families see one resize."""
    for backbone in DinoBackbone:
        validate_dino_size(backbone, 448, 448)
    assert patch_grid(DinoBackbone.DINOV2_VIT_S14_REG4, 448, 448) == (32, 32)
    assert patch_grid(DinoBackbone.DINOV3_VIT_S16, 448, 448) == (28, 28)


def test_a_grid_is_never_produced_for_a_size_that_would_not_run() -> None:
    with pytest.raises(ValueError, match="divisible by 14"):
        patch_grid(DinoBackbone.DINOV2_VIT_B14, 448, 450)


def test_the_projection_bound_is_the_classical_one_and_grows_with_the_pool() -> None:
    """284 at eps 0.9 and N=100 000 is the dimension anomalib's own projection picks."""
    assert johnson_lindenstrauss_dim(100_000, eps=0.9) == 284

    previous = 0
    for count in (10, 1_000, 25_000, 100_000, 250_000):
        current = johnson_lindenstrauss_dim(count, eps=0.9)
        assert current >= previous
        previous = current


def test_the_projection_bound_never_collapses_to_zero_dimensions() -> None:
    """A single point implies `ln(1) = 0`, and a zero-width projection is not a projection."""
    assert johnson_lindenstrauss_dim(1, eps=0.9) == 1
    with pytest.raises(ValueError, match="at least one point"):
        johnson_lindenstrauss_dim(0, eps=0.9)
    with pytest.raises(ValueError, match=r"eps must lie in \(0, 1\)"):
        johnson_lindenstrauss_dim(100, eps=1.5)


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


# ------------------------------------------------- dino_memory, torch-free parts

# `plan_memory`, `channel_order` and the covariance conditioning are deliberately pure —
# integers and numpy in, a plan or a matrix out. They live here rather than in
# `test_dl_dino_memory.py` for `plan_bank`'s reason: what decides whether a fit exhausts the
# machine, and whether a fitted Gaussian can be inverted at all, is checked by the CI job
# that installs *without* the `dl` extra.


def _memory_config(**overrides: object) -> DinoMemoryConfig:
    return DinoMemoryConfig(**overrides)  # type: ignore[arg-type]


def _visa_class_plan(**overrides: object) -> MemoryPlan:
    """One VisA class through the encoder the plugin defaults to: 900 images, 32x32, 768."""
    return plan_memory(
        _memory_config(**overrides),
        samples_available=900,
        images_available=900,
        grid=(32, 32),
        embedding_dim=768,
        channels=("",),
    )


def test_a_global_memory_is_bounded_by_both_caps_and_says_what_it_dropped() -> None:
    """Exit criterion one is not "it fits", it is "it fits and says what it cost".

    Both caps bind at once here: 900 images against a limit of 512, and 512x1024 patches
    against a pool of 100 000. Images are dropped first and patches thinned second, because
    two images differ by whatever the process varies while neighbouring transformer tokens
    attend to overlapping context.
    """
    plan = _visa_class_plan(max_bank_images=512, max_candidate_vectors=100_000)

    assert plan.units_used == 512 and plan.units_available == 900
    assert plan.patches_kept_per_image == 195 and plan.patches_per_image == 1024
    assert plan.candidates_kept == 512 * 195 <= 100_000
    assert plan.coreset_size == 9_984
    assert plan.images_dropped == 388 and plan.patches_dropped_per_image == 829
    # 284 is the dimension anomalib's own projection picks at this pool size and eps 0.9.
    assert plan.projection_dim == 284

    described = plan.describe()
    assert "uncapped this would have been 921,600 vectors" in described
    assert "2.83 GB" in described
    assert "388 of 900 images dropped" in described


def test_a_memory_that_fits_under_every_cap_drops_nothing() -> None:
    """The caps must be bounds, not a tax every run pays."""
    plan = plan_memory(
        _memory_config(max_bank_images=512, max_candidate_vectors=1_000_000),
        samples_available=100,
        images_available=100,
        grid=(16, 16),
        embedding_dim=384,
        channels=("",),
    )

    assert plan.units_used == 100
    assert plan.patches_kept_per_image == plan.patches_per_image == 256
    assert plan.candidates_kept == 25_600
    assert plan.images_dropped == 0 and plan.patches_dropped_per_image == 0
    assert "nothing dropped" in plan.describe()


def test_a_per_position_bank_is_bounded_by_its_own_cap() -> None:
    """The local memory's cost is positions x K x width, and K is what bounds it."""
    plan = _visa_class_plan(
        scoring=Scoring.LOCAL_KNN,
        max_bank_images=256,
        per_position_images=64,
        window_radius=1,
    )

    assert plan.per_position_images == 64, "capped by per_position_images, not by the units"
    assert plan.samples_per_position == 64
    assert plan.candidates_kept == 1024 * 64
    assert plan.bank_bytes == 1024 * 64 * 768 * 4
    # (2r+1)^2 x K: how many bank vectors one query position is compared against.
    assert plan.window_candidates == 9 * 64
    assert plan.coreset_size == 0 and plan.projection_dim == 0
    assert "window radius 1 gives 576 candidates per position" in plan.describe()


def test_an_unreduced_gaussian_names_the_footprint_mahalanobis_dims_avoids() -> None:
    """The quadratic cost is why `mahalanobis_dims` exists, so the plan has to state it.

    A 32x32 grid of 768x768 inverse covariances is 2.42 GB stored as float32 — and twice
    that while the fit holds them in float64. A run that discovered this after the forward
    pass would be a machine that had already started swapping.
    """
    plan = _visa_class_plan(scoring=Scoring.LOCAL_GAUSSIAN, mahalanobis_dims=0)

    assert plan.reduced_dim == plan.fused_dim == 768
    assert plan.precision_bytes == 1024 * 768 * 768 * 4
    assert plan.precision_bytes / 1e9 == pytest.approx(2.42, abs=0.01)
    assert "2.42 GB" in plan.describe()


def test_the_gaussian_sample_deficit_is_a_field_not_a_sentence() -> None:
    """A rank-deficient covariance inverted silently produces confident nonsense.

    64 samples of 128 dimensions is 65 short of full rank, and that number is a field so a
    test — and the diagnostics table — can assert it rather than parse it out of a log line.
    """
    plan = _visa_class_plan(
        scoring=Scoring.LOCAL_GAUSSIAN, mahalanobis_dims=128, per_position_images=64
    )

    assert plan.reduced_dim == 128 and plan.samples_per_position == 64
    assert plan.sample_deficit == 65
    assert "sample deficit 65" in plan.describe()

    ample = _visa_class_plan(
        scoring=Scoring.LOCAL_GAUSSIAN, mahalanobis_dims=16, per_position_images=64
    )
    assert ample.sample_deficit == 0


def test_a_fused_plan_scales_with_the_channel_count_and_names_no_number() -> None:
    """Channel count is data, never schema: nothing here is a constant.

    Three channels make a fused vector three times as wide and every footprint three times
    as large, and the *unit* of banking becomes the sample — so one three-channel sample
    reads three images while contributing one vector per position.
    """

    def fused(count: int) -> MemoryPlan:
        return plan_memory(
            _memory_config(channel_fusion=ChannelFusion.FEATURE_CONCAT, max_bank_images=50),
            samples_available=100,
            images_available=100 * count,
            grid=(8, 8),
            embedding_dim=384,
            channels=tuple(f"channel-{index}" for index in range(count)),
        )

    one, three = fused(1), fused(3)

    assert one.channels_fused == 1 and three.channels_fused == 3
    assert one.fused_dim == 384 and three.fused_dim == 3 * 384
    assert three.bank_bytes == 3 * one.bank_bytes
    # The cap bounds samples; `images_used` reports the files the forward pass opens.
    assert one.units_used == three.units_used == 50
    assert one.images_used == 50 and three.images_used == 150
    assert three.images_dropped == 150 and one.images_dropped == 50


def test_per_image_fusion_keeps_a_vector_one_channel_wide() -> None:
    """`per_image` scores each channel image on its own, so nothing is concatenated."""
    plan = plan_memory(
        _memory_config(channel_fusion=ChannelFusion.PER_IMAGE),
        samples_available=100,
        images_available=300,
        grid=(8, 8),
        embedding_dim=384,
        channels=("a", "b", "c"),
    )

    assert plan.channels_fused == 1
    assert plan.fused_dim == 384
    assert plan.units_available == 300, "a unit is an image here, not a sample"


def test_a_memory_plan_refuses_a_geometry_it_could_not_have_measured() -> None:
    for grid, width, channels in (((0, 8), 384, ("",)), ((8, 8), 0, ("",)), ((8, 8), 384, ())):
        with pytest.raises(ValueError):
            plan_memory(
                _memory_config(),
                samples_available=4,
                images_available=4,
                grid=grid,
                embedding_dim=width,
                channels=channels,
            )


def test_the_channel_order_does_not_depend_on_the_order_records_arrive_in() -> None:
    """A fused vector's meaning is positional, so its channel order has to be a property of
    the dataset rather than of a query's row order — otherwise two runs of one dataset build
    silently incomparable banks."""
    records = [
        ImageRecord(image_id=index, sample_id=index // 3, channel=name, path=Path("x.png"))
        for index, name in enumerate(["dark", "bright", "dome", "bright", "dome", "dark"])
    ]
    shuffled = list(reversed(records))

    assert channel_order(records) == ("bright", "dark", "dome")
    assert channel_order(shuffled) == channel_order(records)
    # No channel at all is a one-entry order rather than a special case.
    assert channel_order([ImageRecord(image_id=1, sample_id=1, path=Path("x.png"))]) == ("",)


def test_ledoit_wolf_conditions_a_covariance_that_has_no_inverse() -> None:
    """The property the whole local_gaussian mode rests on.

    A position sees fewer samples than dimensions — that is the ordinary case, not a
    degenerate one — so its raw covariance is singular by construction and inverting it would
    produce a confident, meaningless Mahalanobis distance. Shrinkage toward a scaled identity
    is what makes the inverse exist, and Ledoit-Wolf derives how much from the samples.
    """
    rng = np.random.default_rng(0)
    samples = rng.normal(size=(8, 32))
    samples = samples - samples.mean(axis=0)

    raw = samples.T @ samples / samples.shape[0]
    assert np.linalg.matrix_rank(raw) < 32, "n < d, so the raw covariance is rank-deficient"
    with pytest.raises(np.linalg.LinAlgError):
        np.linalg.cholesky(raw)

    conditioned, lam = condition_covariance(
        samples, shrinkage=Shrinkage.LEDOIT_WOLF, ridge_epsilon=0.01
    )

    assert 0.0 < lam <= 1.0
    precision = precision_from_covariance(conditioned)
    assert np.allclose(precision @ conditioned, np.eye(32), atol=1e-8)


def test_the_ridge_is_a_fraction_of_the_features_own_scale() -> None:
    """An absolute ridge would swamp a unit-norm feature's covariance and say nothing.

    These features are unit vectors, so a per-position covariance has entries orders of
    magnitude below 0.01. `S + 0.01 I` would be the identity with a rounding error attached,
    and the Mahalanobis distance would quietly become a Euclidean one that still runs.
    Expressed relative to `tr(S)/d`, scaling the features scales the conditioned matrix by
    exactly the same factor.
    """
    rng = np.random.default_rng(1)
    samples = rng.normal(size=(12, 16)) * 1e-2
    samples = samples - samples.mean(axis=0)

    small, lam = condition_covariance(samples, shrinkage=Shrinkage.RIDGE, ridge_epsilon=0.01)
    larger, _ = condition_covariance(samples * 5.0, shrinkage=Shrinkage.RIDGE, ridge_epsilon=0.01)

    assert lam == pytest.approx(0.01)
    assert np.allclose(larger, 25.0 * small)

    # What an absolute ridge would have produced instead: a matrix whose condition number is
    # 1.03, which is the identity with a rounding error attached. The scale-aware one is
    # still shaped by the data it was fitted from, which is the whole point of fitting it.
    absolute = samples.T @ samples / samples.shape[0] + 0.01 * np.eye(16)
    assert np.linalg.cond(absolute) < 1.1
    assert np.linalg.cond(small) > 100.0


def test_the_dino_memory_schema_renders_its_choices_as_pickers() -> None:
    """No frontend work: `SchemaForm` reads `enum` through `$defs` and bounds off the field."""
    schema = DinoMemoryConfig.model_json_schema()
    defs = schema["$defs"]

    def choices(field: str) -> list[str]:
        reference = str(schema["properties"][field]["$ref"]).rsplit("/", 1)[-1]
        return [str(value) for value in defs[reference]["enum"]]

    assert choices("scoring") == ["global_knn", "local_knn", "local_gaussian"]
    assert "dinov2_vit_s14_reg4" in choices("backbone")
    assert choices("shrinkage") == ["ledoit_wolf", "ridge"]
    assert choices("channel_fusion") == ["per_image", "feature_concat"]
    assert schema["properties"]["window_radius"]["maximum"] == 8
    assert schema["properties"]["scoring"]["default"] == "global_knn"
    for name, field in schema["properties"].items():
        assert field.get("description"), f"{name} has no description, so the form has no help"


# ------------------------------------------------ channel-aware reference (ADR-0007)


def _bright(path: Path, seed: int) -> Path:
    """A bright-field-looking scene: a light gradient."""
    from PIL import Image

    generator = np.random.default_rng(seed)
    base = np.linspace(150, 220, 16 * 16).reshape(16, 16)
    noisy = np.clip(base + generator.normal(0, 2, size=(16, 16)), 0, 255).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(noisy, mode="L").convert("RGB").save(path)
    return path


def _dark(path: Path, seed: int) -> Path:
    """A dark-field-looking scene: mostly black with a bright rim. Nothing like `_bright`."""
    from PIL import Image

    generator = np.random.default_rng(seed + 500)
    base = np.full((16, 16), 12.0)
    base[:2, :] = 200
    base[-2:, :] = 200
    noisy = np.clip(base + generator.normal(0, 2, size=(16, 16)), 0, 255).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(noisy, mode="L").convert("RGB").save(path)
    return path


def _two_channel_train(tmp_path: Path) -> list[ImageRecord]:
    records = []
    for index in range(6):
        records.append(
            ImageRecord(
                image_id=index,
                sample_id=index,
                channel="bright",
                path=_bright(tmp_path / f"b{index}.png", index),
            )
        )
        records.append(
            ImageRecord(
                image_id=100 + index,
                sample_id=index,
                channel="dark",
                path=_dark(tmp_path / f"d{index}.png", index),
            )
        )
    return records


def test_a_reference_is_fitted_per_channel(tmp_path: Path) -> None:
    config = PreprocessingConfig(width=16, height=16)
    train_ctx, _ = _contexts(tmp_path, config)

    model = PixelReferenceModel(PixelReferenceConfig(smoothing_sigma=1.0))
    model.fit(_two_channel_train(tmp_path), train_ctx)

    assert set(model._references) == {"bright", "dark"}
    # Two illuminations that look nothing alike must not have produced the same reference.
    bright, dark = model._references["bright"], model._references["dark"]
    assert not np.allclose(bright.median, dark.median)


def _bright_with_defect(path: Path, seed: int) -> Path:
    from PIL import Image

    generator = np.random.default_rng(seed)
    base = np.linspace(150, 220, 16 * 16).reshape(16, 16)
    noisy = np.clip(base + generator.normal(0, 2, size=(16, 16)), 0, 255)
    noisy[5:10, 5:10] = 20
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(noisy.astype(np.uint8), mode="L").convert("RGB").save(path)
    return path


def test_pooling_two_illuminations_desensitizes_the_detector(tmp_path: Path) -> None:
    """Why this is a correctness fix and not a refinement.

    One reference over two illuminations has a per-pixel MAD dominated by the difference
    *between* the channels rather than by variation among normals. Every deviation is then
    divided by that inflated scale, so a real defect is flattened towards the same z-value
    as ordinary noise — the baseline still runs, still produces maps, and quietly stops
    being able to tell them apart.
    """
    config = PreprocessingConfig(width=16, height=16)
    train_ctx, infer_ctx = _contexts(tmp_path, config)
    train = _two_channel_train(tmp_path)

    per_channel = PixelReferenceModel(
        PixelReferenceConfig(smoothing_sigma=1.0, reference_scope=ReferenceScope.CHANNEL)
    )
    per_channel.fit(train, train_ctx)
    pooled = PixelReferenceModel(
        PixelReferenceConfig(smoothing_sigma=1.0, reference_scope=ReferenceScope.DATASET)
    )
    pooled.fit(train, train_ctx)

    normal = ImageRecord(
        image_id=900, sample_id=900, channel="bright", path=_bright(tmp_path / "hb.png", 41)
    )
    defective = ImageRecord(
        image_id=901,
        sample_id=901,
        channel="bright",
        path=_bright_with_defect(tmp_path / "hbd.png", 42),
    )

    def separation(model: PixelReferenceModel) -> float:
        scores = model.predict([normal, defective], infer_ctx)
        return scores[1].score - scores[0].score

    # The pooled scale is inflated by the gap between the illuminations...
    assert np.median(pooled._references[""].scale) > np.median(
        per_channel._references["bright"].scale
    )
    # ...and that is what costs it the ability to separate a defect from a normal.
    assert separation(per_channel) > separation(pooled)


def test_scoring_a_channel_that_was_never_fitted_is_refused_by_name(tmp_path: Path) -> None:
    """Falling back to another channel's reference would produce a confident map of nothing
    but the difference between two illuminations."""
    config = PreprocessingConfig(width=16, height=16)
    train_ctx, infer_ctx = _contexts(tmp_path, config)

    model = PixelReferenceModel(PixelReferenceConfig(smoothing_sigma=1.0))
    model.fit(
        [
            ImageRecord(
                image_id=i, sample_id=i, channel="bright", path=_bright(tmp_path / f"b{i}.png", i)
            )
            for i in range(5)
        ],
        train_ctx,
    )

    stranger = [
        ImageRecord(image_id=7, sample_id=7, channel="dome", path=_dark(tmp_path / "x.png", 3))
    ]
    with pytest.raises(RuntimeError, match="no reference for channel dome"):
        model.predict(stranger, infer_ctx)


def test_the_per_channel_bank_round_trips_through_save_and_load(tmp_path: Path) -> None:
    config = PreprocessingConfig(width=16, height=16)
    train_ctx, infer_ctx = _contexts(tmp_path, config)

    model = PixelReferenceModel(PixelReferenceConfig(smoothing_sigma=1.0))
    model.fit(_two_channel_train(tmp_path), train_ctx)
    model.save(tmp_path)

    restored = PixelReferenceModel(PixelReferenceConfig(smoothing_sigma=1.0))
    restored.load(tmp_path)

    assert set(restored._references) == {"bright", "dark"}
    probe = [
        ImageRecord(
            image_id=901, sample_id=901, channel="dark", path=_dark(tmp_path / "hd.png", 42)
        )
    ]
    assert restored.predict(probe, infer_ctx)[0].score == pytest.approx(
        model.predict(probe, infer_ctx)[0].score
    )


def test_dataset_scope_is_unchanged_by_the_channel_split(tmp_path: Path) -> None:
    """The old behaviour has to still be available and still be exactly the old behaviour."""
    config = PreprocessingConfig(width=16, height=16)
    train_ctx, _ = _contexts(tmp_path, config)
    train = _two_channel_train(tmp_path)

    model = PixelReferenceModel(
        PixelReferenceConfig(smoothing_sigma=1.0, reference_scope=ReferenceScope.DATASET)
    )
    model.fit(train, train_ctx)

    assert set(model._references) == {""}


def test_exporting_onnx_refuses_a_multi_channel_bank_by_name(tmp_path: Path) -> None:
    """One static graph carries one reference tensor. Refused in the plugin, so no route
    has to branch on a method's own configuration."""
    config = PreprocessingConfig(width=16, height=16)
    train_ctx, _ = _contexts(tmp_path, config)

    model = PixelReferenceModel(PixelReferenceConfig(smoothing_sigma=1.0))
    model.fit(_two_channel_train(tmp_path), train_ctx)

    with pytest.raises(ValueError, match="reference_scope=channel"):
        model.export_onnx(tmp_path / "m.onnx", config)
