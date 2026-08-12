"""Dinomaly plugin boundary without network access or a hidden image preprocessor.

The real pretrained encoder is resource- and operator-gated by
``scripts/dinomaly-smoke-test.py`` and quality-gated on public data.  CI replaces only
timm's asset-backed extractor with a tiny seeded patch projection; anomalib's bottleneck,
eight-layer decoder, hard-mining loss, StableAdamW, map rule and this plugin's whole pass
remain real.  That keeps these tests about our integration contract and avoids making a
public network service part of the test suite.
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
from anomaly_lab.models.dinomaly_anomalib import (  # noqa: E402
    STATE_FILENAME,
    DinomalyAnomalibModel,
    DinomalyConfig,
    _asset_environment,
)
from anomaly_lab.models.preprocessing import PreprocessingConfig  # noqa: E402

SIZE = 28
TorchModule: Any = torch.nn.Module


class TinyFeatureExtractor(TorchModule):  # type: ignore[misc]
    """The DINOv2 token contract, backed by one small seeded patch projection."""

    def __init__(self, *, layers: list[str], **_: Any) -> None:
        super().__init__()
        self.layers = layers
        self.patch_size = 14
        self.num_register_tokens = 0
        self.projection = torch.nn.Conv2d(3, 384, kernel_size=14, stride=14)

    def forward(self, batch: Any) -> dict[str, Any]:
        patches = self.projection(batch).flatten(2).transpose(1, 2)
        class_token = patches.mean(dim=1, keepdim=True)
        tokens = torch.cat((class_token, patches), dim=1)
        return dict.fromkeys(self.layers, tokens)


@pytest.fixture(autouse=True)
def no_network_encoder(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace the only component that resolves a downloaded asset."""
    import anomalib.models.image.dinomaly.torch_model as torch_model

    monkeypatch.setattr(torch_model, "TimmFeatureExtractor", TinyFeatureExtractor)


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


def _digest_trainable(model: DinomalyAnomalibModel) -> str:
    assert model._model is not None
    digest = hashlib.sha256()
    for module in (model._model.bottleneck, model._model.decoder):
        for key, tensor in sorted(module.state_dict().items()):
            digest.update(key.encode())
            digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _fit(
    root: Path, seed: int = 0, steps: int = 2
) -> tuple[DinomalyAnomalibModel, TrainContext, InferContext]:
    root.mkdir(parents=True, exist_ok=True)
    train_ctx, infer_ctx = _contexts(root)
    model = DinomalyAnomalibModel(DinomalyConfig(max_steps=steps, seed=seed))
    model.fit(_records(root), train_ctx)
    return model, train_ctx, infer_ctx


def test_same_seed_is_identical_and_a_different_seed_is_different(tmp_path: Path) -> None:
    first, _, _ = _fit(tmp_path / "first", seed=7)
    second, _, _ = _fit(tmp_path / "second", seed=7)
    different, _, _ = _fit(tmp_path / "different", seed=8)

    assert _digest_trainable(first) == _digest_trainable(second)
    assert _digest_trainable(first) != _digest_trainable(different)


def test_asset_policy_controls_the_live_huggingface_cache(tmp_path: Path) -> None:
    constants: Any = import_module("huggingface_hub.constants")

    original_cache = constants.HF_HUB_CACHE
    original_offline = constants.HF_HUB_OFFLINE
    with _asset_environment(tmp_path / "cache", allow_downloads=False):
        assert str(tmp_path / "cache" / "huggingface" / "hub") == constants.HF_HUB_CACHE
        assert constants.HF_HUB_OFFLINE is True

    assert original_cache == constants.HF_HUB_CACHE
    assert constants.HF_HUB_OFFLINE is original_offline


def test_fit_save_load_predict_and_continue_are_one_contract(tmp_path: Path) -> None:
    model, train_ctx, infer_ctx = _fit(tmp_path / "run")
    probe = ImageRecord(image_id=99, sample_id=99, path=_write(tmp_path / "run" / "p.png", 99))
    before = model.predict([probe], infer_ctx)[0]
    model.save(train_ctx.artifact_dir)

    restored = DinomalyAnomalibModel(DinomalyConfig(max_steps=2, seed=0))
    restored.load(train_ctx.artifact_dir)
    after = restored.predict([probe], infer_ctx)[0]

    assert after.score == pytest.approx(before.score, abs=1e-7)
    assert after.anomaly_map is not None
    assert np.load(after.anomaly_map).shape == (SIZE, SIZE)
    assert restored.completed_steps() == 2

    restored.fit_more(_records(tmp_path / "run"), train_ctx, additional_steps=1)
    assert restored.completed_steps() == 3
    uninterrupted, _, _ = _fit(tmp_path / "uninterrupted", steps=3)
    assert _digest_trainable(restored) == _digest_trainable(uninterrupted)


def test_load_refuses_changed_encoder_weights(tmp_path: Path) -> None:
    model, train_ctx, _ = _fit(tmp_path / "run")
    model.save(train_ctx.artifact_dir)
    checkpoint = train_ctx.artifact_dir / STATE_FILENAME
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    payload["encoder_fingerprint"] = "0" * 64
    torch.save(payload, checkpoint)

    restored = DinomalyAnomalibModel(DinomalyConfig(max_steps=2))
    with pytest.raises(RuntimeError, match="encoder weights changed"):
        restored.load(train_ctx.artifact_dir)
