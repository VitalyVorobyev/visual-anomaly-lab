"""`dinomaly_custom`'s portable export: one graph, the same numbers as the Python path.

Hermetic like `test_dl_dinomaly_custom.py`: a seeded random ViT-S/14, no download, a 112-pixel
frame (an 8x8 token grid). A few training steps are enough — parity is a claim about the
arithmetic of a fitted network, and a decoder moved off its initialisation exercises every
operation the trained one does. The tolerance is the one the plugin declares, which is the
one the VisA export-parity gate predeclared and passed (docs/measurements.md).
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

from anomaly_lab.deployment.parity import compare_outputs
from anomaly_lab.deployment.protocol import SupportsOnnxExport
from anomaly_lab.deployment.schema import TensorScore
from anomaly_lab.map_files import read_map
from anomaly_lab.models.base import (
    Device,
    ImageRecord,
    InferContext,
    NullReporter,
    PortableFormat,
    TrainContext,
)
from anomaly_lab.models.diagnostics import DiagnosticWriter
from anomaly_lab.models.dino_backbone import (
    DinoBackbone,
    extract_layer_tokens,
    load_backbone,
    pin_frame,
)
from anomaly_lab.models.dinomaly_custom import (
    DinomalyCustomConfig,
    DinomalyCustomModel,
)
from anomaly_lab.models.preprocessing import (
    ColorMode,
    PreprocessingConfig,
    load_array,
    to_chw,
)

SIZE = 112
STEPS = 6


def _image(path: Path, seed: int, *, stamp: bool = False) -> Path:
    rng = np.random.default_rng(seed)
    values = np.clip(
        np.linspace(96, 128, SIZE)[:, None]
        + np.linspace(0, 8, SIZE)[None, :]
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
    root: Path, *, color: ColorMode = ColorMode.RGB, **overrides: Any
) -> tuple[DinomalyCustomModel, InferContext, list[ImageRecord]]:
    config = DinomalyCustomConfig(
        encoder=DinoBackbone.DINOV2_VIT_S14_REG4,
        max_steps=STEPS,
        decoder_depth=4,
        pretrained_encoder=False,
        allow_downloads=False,
        seed=3,
        **overrides,
    )
    train_ctx, infer_ctx = _contexts(root, color)
    train = [
        ImageRecord(image_id=index, sample_id=index, path=_image(root / f"n{index}.png", index))
        for index in range(3)
    ]
    model = DinomalyCustomModel(config)
    model.fit(train, train_ctx)
    probe = [
        ImageRecord(image_id=10, sample_id=10, path=_image(root / "probe-n.png", 10)),
        ImageRecord(image_id=11, sample_id=11, path=_image(root / "probe-d.png", 11, stamp=True)),
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


def test_the_capability_and_the_export_protocol_agree() -> None:
    assert DinomalyCustomModel.capabilities().portable_formats == [PortableFormat.ONNX]
    assert isinstance(DinomalyCustomModel(DinomalyCustomConfig()), SupportsOnnxExport)


@pytest.mark.parametrize("color", [ColorMode.RGB, ColorMode.GRAYSCALE])
def test_the_graph_matches_the_python_path_on_prepared_pixels(
    tmp_path: Path, color: ColorMode
) -> None:
    """Map and score, within the declared tolerance, on a normal and a stamped image.

    Grayscale is the case a shape bug would hide in: the bundle consumes the one plane the
    experiment froze and replicates it as its own first operation.
    """
    model, infer_ctx, probe = _fitted(tmp_path, color=color)
    contract = model.export_onnx(tmp_path / "dinomaly.onnx", infer_ctx.preprocessing)
    assert isinstance(contract.score, TensorScore)

    for record in probe:
        tensor = _prepared(record, infer_ctx.preprocessing)
        assert tensor.shape == (1, infer_ctx.preprocessing.channels, SIZE, SIZE)
        expected_map, expected_score = model.portable_reference(tensor)
        actual_map, actual_score = _run(tmp_path / "dinomaly.onnx", contract.input_name, tensor)
        reading = compare_outputs(
            expected_map,
            expected_score,
            actual_map.squeeze(),
            float(actual_score.squeeze()),
            absolute_tolerance=contract.absolute_tolerance,
            relative_tolerance=contract.relative_tolerance,
        )
        assert reading.passed, reading


def test_the_python_reference_is_what_predict_stores(tmp_path: Path) -> None:
    """Parity against `portable_reference` means nothing unless that is the method.

    `predict` standardises in numpy and `portable_reference` inside the module; both are
    exactly rounded elementwise float32, so on CPU the two agree far below the gate's
    tolerance — a drift between them would be a second map rule, not rounding.
    """
    model, infer_ctx, probe = _fitted(tmp_path, map_blur_sigma=2.0)
    predictions = model.predict(probe, infer_ctx)

    for record, prediction in zip(probe, predictions, strict=True):
        assert prediction.anomaly_map is not None
        stored_map = read_map(prediction.anomaly_map)
        reference_map, reference_score = model.portable_reference(
            _prepared(record, infer_ctx.preprocessing)
        )
        np.testing.assert_allclose(reference_map, stored_map, atol=1e-6, rtol=1e-6)
        assert reference_score == pytest.approx(prediction.score, abs=1e-6)


def test_a_blurred_map_is_exported_blurred_and_the_score_is_not(tmp_path: Path) -> None:
    """`map_blur_sigma` reaches the stored map only, in the graph exactly as in `predict`."""
    model, infer_ctx, probe = _fitted(tmp_path, map_blur_sigma=2.0)
    contract = model.export_onnx(tmp_path / "blurred.onnx", infer_ctx.preprocessing)
    tensor = _prepared(probe[1], infer_ctx.preprocessing)

    blurred_map, blurred_score = _run(tmp_path / "blurred.onnx", contract.input_name, tensor)
    model.config = model.config.model_copy(update={"map_blur_sigma": 0.0})
    model.export_onnx(tmp_path / "sharp.onnx", infer_ctx.preprocessing)
    sharp_map, sharp_score = _run(tmp_path / "sharp.onnx", contract.input_name, tensor)

    assert float(np.abs(blurred_map - sharp_map).max()) > 1e-4
    assert float(blurred_score.squeeze()) == pytest.approx(float(sharp_score.squeeze()), abs=1e-6)


def test_export_refuses_a_frame_the_network_was_not_fitted_on(tmp_path: Path) -> None:
    model, _, _ = _fitted(tmp_path)
    with pytest.raises(ValueError, match="fitted at 112x112"):
        model.export_onnx(tmp_path / "wrong.onnx", PreprocessingConfig(width=126, height=126))


def test_export_before_fitting_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="no fitted network"):
        DinomalyCustomModel(DinomalyCustomConfig()).export_onnx(
            tmp_path / "none.onnx", PreprocessingConfig(width=SIZE, height=SIZE)
        )


@pytest.mark.parametrize("rebound", [False, True])
def test_a_pinned_frame_is_the_dynamic_forward_exactly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, rebound: bool
) -> None:
    """`pin_frame` computes the forward's own resample once; `atol=0` is the claim.

    `rebound` stands in for anomalib, whose feature extractor rebinds timm's ViT resample
    process-wide: the pin has to follow whatever the forward it replaces would call.
    """
    vit: Any = importlib.import_module("timm.models.vision_transformer")
    import torch

    if rebound:
        original = vit.resample_abs_pos_embed

        def no_antialias(*args: Any, **kwargs: Any) -> Any:
            kwargs["antialias"] = False
            return original(*args, **kwargs)

        monkeypatch.setattr(vit, "resample_abs_pos_embed", no_antialias)

    encoder = load_backbone(
        DinoBackbone.DINOV2_VIT_S14_REG4,
        pretrained=False,
        allow_downloads=False,
        cache_dir=tmp_path,
        seed=2,
        method="test",
    ).eval()
    pinned = pin_frame(encoder, 8, 10)
    batch = torch.from_numpy(
        np.random.default_rng(1).normal(size=(1, 3, 112, 140)).astype(np.float32)
    )
    with torch.no_grad():
        dynamic = extract_layer_tokens(encoder, batch, (2, 9))
        fixed = extract_layer_tokens(pinned, batch, (2, 9))

    assert pinned is not encoder
    assert encoder.dynamic_img_size is True
    for left, right in zip(dynamic, fixed, strict=True):
        torch.testing.assert_close(left, right, atol=0.0, rtol=0.0)
