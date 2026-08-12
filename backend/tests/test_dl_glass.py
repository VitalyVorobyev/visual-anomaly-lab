"""GLASS plugin contract without network access or private images.

CI replaces only the asset-backed WRN-50 feature extractor with a tiny seeded CNN and
reduces the embedding width. Anomalib's Perlin synthesis, global perturbation and mining,
projection, discriminator, losses, optimizers and map rule remain real.
"""

from __future__ import annotations

import hashlib
from importlib import import_module
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

torch = pytest.importorskip("torch")
pytest.importorskip("anomalib")

from anomaly_lab.models.base import (  # noqa: E402
    Device,
    ImageRecord,
    InferContext,
    NullReporter,
    TrainContext,
)
from anomaly_lab.models.diagnostics import DiagnosticWriter  # noqa: E402
from anomaly_lab.models.glass_anomalib import (  # noqa: E402
    STATE_FILENAME,
    GlassAnomalibModel,
    GlassConfig,
)
from anomaly_lab.models.preprocessing import PreprocessingConfig  # noqa: E402

SIZE = 64
TorchModule: Any = torch.nn.Module


class TinyFeatureExtractor(TorchModule):  # type: ignore[misc]
    """Two real feature maps behind anomalib's extractor contract."""

    def __init__(
        self,
        *,
        backbone: str,
        layers: list[str],
        requires_grad: bool,
        **_: Any,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.layers = layers
        self.feature_extractor = torch.nn.Sequential(
            torch.nn.Conv2d(3, 8, kernel_size=3, stride=4, padding=1),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),
            torch.nn.LeakyReLU(0.2),
        )
        self.feature_extractor.requires_grad_(requires_grad)

    def forward(self, batch: Any) -> dict[str, Any]:
        layer2 = self.feature_extractor[1](self.feature_extractor[0](batch))
        layer3 = self.feature_extractor[3](self.feature_extractor[2](layer2))
        return {"layer2": layer2, "layer3": layer3}


class TinyGlass:
    """Lightning-shaped holder around the real, width-reduced GlassModel."""

    def __init__(self, **kwargs: Any) -> None:
        from anomalib.models.image.glass.torch_model import GlassModel

        selected = {
            key: value
            for key, value in kwargs.items()
            if key
            in {
                "input_shape",
                "anomaly_source_path",
                "backbone",
                "layers",
                "step",
                "svd",
                "mining",
            }
        }
        self.model = GlassModel(
            **selected,
            pretrain_embed_dim=24,
            target_embed_dim=24,
            discriminator_hidden=16,
        )


@pytest.fixture(autouse=True)
def no_network_backbone(monkeypatch: pytest.MonkeyPatch) -> None:
    import anomalib.models
    import anomalib.models.image.glass.torch_model as torch_model

    monkeypatch.setattr(torch_model, "TimmFeatureExtractor", TinyFeatureExtractor)
    monkeypatch.setattr(anomalib.models, "Glass", TinyGlass)


def _write(path: Path, seed: int) -> Path:
    rng = np.random.default_rng(seed)
    pixels = rng.integers(48, 208, size=(SIZE, SIZE, 3), dtype=np.uint8)
    Image.fromarray(pixels).save(path)
    return path


def _contexts(root: Path) -> tuple[TrainContext, InferContext]:
    artifacts = root / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    cache = root / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    preprocessing = PreprocessingConfig(width=SIZE, height=SIZE)
    return (
        TrainContext(
            artifact_dir=artifacts,
            cache_dir=cache,
            preprocessing=preprocessing,
            device=Device.CPU,
            reporter=NullReporter(),
            diagnostics=DiagnosticWriter(artifacts / "diagnostics"),
        ),
        InferContext(
            artifact_dir=artifacts,
            cache_dir=cache,
            preprocessing=preprocessing,
            device=Device.CPU,
            reporter=NullReporter(),
            diagnostics=DiagnosticWriter(artifacts / "diagnostics"),
        ),
    )


def _records(root: Path) -> list[ImageRecord]:
    return [
        ImageRecord(image_id=index, sample_id=index, path=_write(root / f"n{index}.png", index))
        for index in range(3)
    ]


def _digest_optimized(model: GlassAnomalibModel) -> str:
    assert model._model is not None
    digest = hashlib.sha256()
    for module in (model._model.projection, model._model.discriminator):
        for key, tensor in sorted(module.state_dict().items()):
            digest.update(key.encode())
            digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _fit(
    root: Path, seed: int = 0, steps: int = 2
) -> tuple[GlassAnomalibModel, TrainContext, InferContext]:
    root.mkdir(parents=True, exist_ok=True)
    train_ctx, infer_ctx = _contexts(root)
    model = GlassAnomalibModel(
        GlassConfig(
            max_steps=steps,
            center_images=2,
            center_refresh_steps=100,
            mining_steps=1,
            seed=seed,
        )
    )
    model.fit(_records(root), train_ctx)
    return model, train_ctx, infer_ctx


def test_same_seed_is_identical_and_a_different_seed_is_different(tmp_path: Path) -> None:
    first, _, _ = _fit(tmp_path / "first", seed=7)
    second, _, _ = _fit(tmp_path / "second", seed=7)
    different, _, _ = _fit(tmp_path / "different", seed=8)

    assert _digest_optimized(first) == _digest_optimized(second)
    assert _digest_optimized(first) != _digest_optimized(different)


def test_fit_save_load_predict_and_continue_are_one_contract(tmp_path: Path) -> None:
    model, train_ctx, infer_ctx = _fit(tmp_path / "run")
    probe = ImageRecord(
        image_id=99,
        sample_id=99,
        path=_write(tmp_path / "run" / "probe.png", 99),
    )
    before = model.predict([probe], infer_ctx)[0]
    model.save(train_ctx.artifact_dir)

    restored = GlassAnomalibModel(
        GlassConfig(
            max_steps=2,
            center_images=2,
            center_refresh_steps=100,
            mining_steps=1,
        )
    )
    restored.load(train_ctx.artifact_dir)
    after = restored.predict([probe], infer_ctx)[0]

    assert after.score == pytest.approx(before.score, abs=1e-7)
    assert after.anomaly_map is not None
    assert np.load(after.anomaly_map).shape == (SIZE, SIZE)
    assert restored.completed_steps() == 2

    restored.fit_more(_records(tmp_path / "run"), train_ctx, additional_steps=1)
    assert restored.completed_steps() == 3
    uninterrupted, _, _ = _fit(tmp_path / "uninterrupted", steps=3)
    assert _digest_optimized(restored) == _digest_optimized(uninterrupted)


def test_fitted_model_exports_map_and_graph_score_with_parity(tmp_path: Path) -> None:
    """GLASS's discriminator score travels in the graph beside its anomaly map."""
    model, _, infer_ctx = _fit(tmp_path / "run")
    destination = tmp_path / "glass.onnx"
    contract = model.export_onnx(destination, infer_ctx.preprocessing)
    fixture = np.linspace(0.0, 1.0, 3 * SIZE * SIZE, dtype=np.float32).reshape(1, 3, SIZE, SIZE)
    expected_map, expected_score = model.portable_reference(fixture)
    ort: Any = import_module("onnxruntime")
    session = ort.InferenceSession(str(destination), providers=["CPUExecutionProvider"])
    values = session.run(None, {contract.input_name: fixture})
    actual_map = np.asarray(values[0])[0, 0]
    actual_score = float(np.asarray(values[1])[0])

    assert contract.score.kind == "tensor"
    np.testing.assert_allclose(actual_map, expected_map, atol=2e-4, rtol=2e-4)
    assert abs(actual_score - expected_score) <= 2e-4


def test_load_refuses_changed_backbone_weights(tmp_path: Path) -> None:
    model, train_ctx, _ = _fit(tmp_path / "run", steps=1)
    model.save(train_ctx.artifact_dir)
    checkpoint = train_ctx.artifact_dir / STATE_FILENAME
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    payload["backbone_fingerprint"] = "0" * 64
    torch.save(payload, checkpoint)

    restored = GlassAnomalibModel(
        GlassConfig(
            max_steps=1,
            center_images=2,
            center_refresh_steps=100,
            mining_steps=1,
        )
    )
    with pytest.raises(RuntimeError, match="backbone weights changed"):
        restored.load(train_ctx.artifact_dir)
