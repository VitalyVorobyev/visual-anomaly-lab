"""`subspace_ad`'s portable graph: one static graph, the same numbers as the Python path.

Hermetic like `test_dl_subspace_ad_plugin.py`: a seeded random ViT-S, no download, a 112-pixel
frame (an 8x8 grid at patch 14, 7x7 at patch 16). Parity is a claim about the arithmetic of a
fitted subspace, and a random encoder exercises every operation a pretrained one does. The
tolerance is the one the plugin declares, which is the bound the export-parity gate
predeclared for a deep frozen backbone (docs/measurements.md).

`portable_formats` stays empty here on purpose: a format is declared only after the gate has
passed on real pixels, so these tests pin the graph, not the declaration.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

pytest.importorskip("torch")
pytest.importorskip("timm")
pytest.importorskip("onnxruntime")

from anomaly_lab.deployment.export import _fixture
from anomaly_lab.deployment.parity import compare_outputs
from anomaly_lab.deployment.protocol import SupportsOnnxExport
from anomaly_lab.deployment.schema import TensorScore
from anomaly_lab.map_files import read_map
from anomaly_lab.models.base import (
    Device,
    ImageRecord,
    InferContext,
    NullReporter,
    TrainContext,
)
from anomaly_lab.models.diagnostics import DiagnosticWriter
from anomaly_lab.models.dino_backbone import DinoBackbone, LayerWindow
from anomaly_lab.models.preprocessing import (
    ColorMode,
    PreprocessingConfig,
    load_array,
    to_chw,
)
from anomaly_lab.models.subspace_ad import SubspaceAdConfig, SubspaceAdModel

SIZE = 112


def _image(path: Path, seed: int, *, stamp: bool = False) -> Path:
    rng = np.random.default_rng(seed)
    values = np.clip(
        np.linspace(104, 116, SIZE)[:, None]
        + np.linspace(0, 6, SIZE)[None, :]
        + rng.normal(0.0, 3.0, size=(SIZE, SIZE)),
        0,
        255,
    )
    if stamp:
        values[42:70, 42:70] = 255.0
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(values.astype(np.uint8), mode="L").convert("RGB").save(path)
    return path


def _contexts(root: Path, color: ColorMode) -> tuple[TrainContext, InferContext]:
    artifacts = root / "artifacts"
    cache = root / "cache"
    artifacts.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    preprocessing = PreprocessingConfig(width=SIZE, height=SIZE, color=color)
    shared: dict[str, Any] = {
        "artifact_dir": artifacts,
        "cache_dir": cache,
        "preprocessing": preprocessing,
        "device": Device.CPU,
        "diagnostics": DiagnosticWriter(artifacts / "diagnostics"),
    }
    return (
        TrainContext(reporter=NullReporter(), **shared),
        InferContext(reporter=NullReporter(), **shared),
    )


def _fitted(
    root: Path,
    *,
    color: ColorMode = ColorMode.RGB,
    channels: tuple[str | None, ...] = (None,),
    **overrides: Any,
) -> tuple[SubspaceAdModel, InferContext, list[ImageRecord]]:
    base: dict[str, Any] = {
        "backbone": DinoBackbone.DINOV2_VIT_S14,
        "layers": LayerWindow.UPPER_HALF,
        "pretrained_backbone": False,
        "allow_downloads": False,
        "rotations": 2,
        "max_fit_images": 3,
        "smoothing_sigma": 2.0,
        # Large enough that the tail is several patches, so the graph's TopK and mean are
        # exercised rather than a single max.
        "tail_fraction": 0.1,
        "seed": 0,
    }
    base.update(overrides)
    config = SubspaceAdConfig(**base)
    train_ctx, infer_ctx = _contexts(root, color)
    train = [
        ImageRecord(
            image_id=index * len(channels) + offset,
            sample_id=index,
            path=_image(root / f"n{index}-{offset}.png", index * 7 + offset),
            channel=channel,
        )
        for index in range(3)
        for offset, channel in enumerate(channels)
    ]
    model = SubspaceAdModel(config)
    model.fit(train, train_ctx)
    probe = [
        ImageRecord(image_id=100, sample_id=100, path=_image(root / "probe-n.png", 100)),
        ImageRecord(
            image_id=101, sample_id=101, path=_image(root / "probe-d.png", 101, stamp=True)
        ),
    ]
    return model, infer_ctx, probe


def _prepared(record: ImageRecord, preprocessing: PreprocessingConfig) -> np.ndarray:
    """What a portable host is handed: the prepared frame as `[0, 1]` NCHW float32."""
    return np.ascontiguousarray(
        to_chw(load_array(record.path, preprocessing))[np.newaxis], dtype=np.float32
    )


def _run(graph: Path, input_name: str, tensor: np.ndarray) -> list[np.ndarray]:
    ort: Any = importlib.import_module("onnxruntime")
    session = ort.InferenceSession(str(graph), providers=["CPUExecutionProvider"])
    return [np.asarray(value) for value in session.run(None, {input_name: tensor})]


def test_the_graph_is_implemented_and_the_format_is_not_yet_declared() -> None:
    assert SubspaceAdModel.capabilities().portable_formats == []
    assert isinstance(SubspaceAdModel(SubspaceAdConfig()), SupportsOnnxExport)


@pytest.mark.parametrize(
    ("backbone", "color", "sigma"),
    [
        (DinoBackbone.DINOV2_VIT_S14, ColorMode.RGB, 2.0),
        (DinoBackbone.DINOV2_VIT_S14, ColorMode.GRAYSCALE, 2.0),
        (DinoBackbone.DINOV2_VIT_S14, ColorMode.RGB, 0.0),
        (DinoBackbone.DINOV3_VIT_S16, ColorMode.RGB, 2.0),
    ],
)
def test_the_graph_matches_the_python_path_on_prepared_pixels(
    tmp_path: Path, backbone: DinoBackbone, color: ColorMode, sigma: float
) -> None:
    """Map and score, within the declared tolerance, on a normal, a stamped and a ramp input.

    Grayscale is where a shape bug would hide: the bundle consumes the one plane the
    experiment froze and replicates it as its own first operation. `sigma = 0` takes the
    unsmoothed branch, whose operator is the bare bilinear resample. DINOv3 has no absolute
    position table to pin; its rotary embedding is built in the graph.
    """
    model, infer_ctx, probe = _fitted(
        tmp_path, color=color, backbone=backbone, smoothing_sigma=sigma
    )
    graph = tmp_path / "subspace.onnx"
    contract = model.export_onnx(graph, infer_ctx.preprocessing)
    assert isinstance(contract.score, TensorScore)

    preprocessing = infer_ctx.preprocessing
    tensors = [_prepared(record, preprocessing) for record in probe]
    tensors.append(_fixture(preprocessing.channels, SIZE, SIZE))
    for tensor in tensors:
        assert tensor.shape == (1, preprocessing.channels, SIZE, SIZE)
        expected_map, expected_score = model.portable_reference(tensor)
        actual_map, actual_score = _run(graph, contract.input_name, tensor)
        assert actual_map.shape == (1, 1, SIZE, SIZE)
        reading = compare_outputs(
            expected_map,
            expected_score,
            actual_map.squeeze(),
            float(actual_score.squeeze()),
            absolute_tolerance=contract.absolute_tolerance,
            relative_tolerance=contract.relative_tolerance,
        )
        assert reading.passed, (reading, expected_score)


def test_the_python_reference_is_what_predict_stores(tmp_path: Path) -> None:
    """Parity against `portable_reference` means nothing unless that is the method."""
    model, infer_ctx, probe = _fitted(tmp_path)
    predictions = model.predict(probe, infer_ctx)

    for record, prediction in zip(probe, predictions, strict=True):
        assert prediction.anomaly_map is not None
        reference_map, reference_score = model.portable_reference(
            _prepared(record, infer_ctx.preprocessing)
        )
        np.testing.assert_allclose(
            reference_map, read_map(prediction.anomaly_map), atol=1e-6, rtol=1e-6
        )
        assert reference_score == pytest.approx(prediction.score, rel=1e-9, abs=1e-9)


def test_the_blur_reaches_the_exported_map_and_not_the_score(tmp_path: Path) -> None:
    model, infer_ctx, probe = _fitted(tmp_path, smoothing_sigma=4.0)
    contract = model.export_onnx(tmp_path / "blurred.onnx", infer_ctx.preprocessing)
    tensor = _prepared(probe[1], infer_ctx.preprocessing)
    blurred_map, blurred_score = _run(tmp_path / "blurred.onnx", contract.input_name, tensor)

    model.config = model.config.model_copy(update={"smoothing_sigma": 0.0})
    model.export_onnx(tmp_path / "sharp.onnx", infer_ctx.preprocessing)
    sharp_map, sharp_score = _run(tmp_path / "sharp.onnx", contract.input_name, tensor)

    assert float(np.abs(blurred_map - sharp_map).max()) > 1e-3
    assert float(blurred_score.squeeze()) == pytest.approx(float(sharp_score.squeeze()), rel=1e-6)


def test_a_fit_over_several_channels_is_refused_by_name(tmp_path: Path) -> None:
    """One graph carries one subspace; the bundle has no channel input to choose with."""
    model, infer_ctx, _ = _fitted(tmp_path, channels=("bright", "dark"))
    with pytest.raises(ValueError, match=r"2 subspaces, one per channel \(bright, dark\)"):
        model.export_onnx(tmp_path / "multi.onnx", infer_ctx.preprocessing)
    assert not (tmp_path / "multi.onnx").exists()


def test_a_single_named_channel_exports(tmp_path: Path) -> None:
    model, infer_ctx, _ = _fitted(tmp_path, channels=("bright",))
    model.export_onnx(tmp_path / "one.onnx", infer_ctx.preprocessing)
    assert (tmp_path / "one.onnx").stat().st_size > 0


def test_export_refuses_a_frame_the_subspace_was_not_fitted_on(tmp_path: Path) -> None:
    model, _, _ = _fitted(tmp_path)
    with pytest.raises(ValueError, match="8x8 token grid"):
        model.export_onnx(tmp_path / "wrong.onnx", PreprocessingConfig(width=126, height=126))


def test_export_before_fitting_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="no fitted subspace"):
        SubspaceAdModel(SubspaceAdConfig()).export_onnx(
            tmp_path / "none.onnx", PreprocessingConfig(width=SIZE, height=SIZE)
        )


def test_a_loaded_fit_exports_the_same_graph(tmp_path: Path) -> None:
    """The export job loads the stored fit before exporting; the loaded fit must agree."""
    model, infer_ctx, probe = _fitted(tmp_path)
    stored = tmp_path / "stored"
    stored.mkdir()
    model.save(stored)
    loaded = SubspaceAdModel(model.config)
    loaded.load(stored)
    contract = loaded.export_onnx(tmp_path / "loaded.onnx", infer_ctx.preprocessing)

    tensor = _prepared(probe[1], infer_ctx.preprocessing)
    expected_map, expected_score = model.portable_reference(tensor)
    actual_map, actual_score = _run(tmp_path / "loaded.onnx", contract.input_name, tensor)
    reading = compare_outputs(
        expected_map,
        expected_score,
        actual_map.squeeze(),
        float(actual_score.squeeze()),
        absolute_tolerance=contract.absolute_tolerance,
        relative_tolerance=contract.relative_tolerance,
    )
    assert reading.passed, reading
