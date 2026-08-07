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
)
from anomaly_lab.models.device import resolve_device
from anomaly_lab.models.diagnostics import (
    DiagnosticError,
    DiagnosticKind,
    DiagnosticScope,
    DiagnosticWriter,
    load_index,
)
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


def test_every_method_sees_the_same_shape_regardless_of_source_size(tmp_path: Path) -> None:
    """The whole point of the bridge: two differently sized files become one shape."""
    small = write_image(tmp_path / "small.png", size=(11, 7))
    large = write_image(tmp_path / "large.png", size=(97, 53))
    config = PreprocessingConfig(width=32, height=24)

    assert load_array(small, config).shape == (24, 32, 3)
    assert load_array(large, config).shape == (24, 32, 3)


def test_pixels_arrive_as_float32_in_zero_to_one(tmp_path: Path) -> None:
    array = load_array(write_image(tmp_path / "a.png"), PreprocessingConfig())
    assert array.dtype == np.float32
    assert float(array.min()) >= 0.0 and float(array.max()) <= 1.0


def test_grayscale_mode_produces_one_channel(tmp_path: Path) -> None:
    config = PreprocessingConfig(width=16, height=16, color=ColorMode.GRAYSCALE)
    assert load_array(write_image(tmp_path / "g.png", mode="L"), config).shape == (16, 16, 1)


def test_a_grayscale_file_still_expands_to_three_under_rgb(tmp_path: Path) -> None:
    """Pretrained backbones expect three channels; a mono dataset must not be a special case."""
    config = PreprocessingConfig(width=16, height=16, color=ColorMode.RGB)
    assert load_array(write_image(tmp_path / "g.png", mode="L"), config).shape == (16, 16, 3)


def test_to_chw_gives_torch_the_layout_it_wants(tmp_path: Path) -> None:
    array = load_array(write_image(tmp_path / "a.png"), PreprocessingConfig(width=16, height=8))
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


# ----------------------------------------------------------------- registry


def test_every_registered_method_describes_itself_without_importing_torch() -> None:
    """`describe_all` is called whenever the picker opens; it must stay cheap."""
    described = describe_all()
    keys = {entry.key for entry in described}
    assert {"pixel_reference", "efficientad_anomalib"} <= keys

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
    from anomaly_lab.models.pixel_reference import _evenly_spaced

    assert _evenly_spaced(4, 10) == [0, 1, 2, 3]
    chosen = _evenly_spaced(100, 5)
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
