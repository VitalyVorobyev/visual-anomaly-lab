"""Building an anomalib network does not change what a DINO method computes afterwards.

anomalib's `TimmFeatureExtractor`, whenever it builds a ViT, rebinds
`timm.models.vision_transformer.resample_abs_pos_embed` for the whole process, and a DINOv2
encoder's features move with it. `model_assets.timm_bindings_preserved` is the boundary our
anomalib plugins build inside; these tests pin that a DINO encoder's tokens are bit-identical
whichever came first, that the control without it does move them (so the pin can fail), and
that an anomalib method built inside it computes exactly what it computed bare.

Every network here is built with `pretrained=False`: the resample is a property of the
architecture and the frame, not of the weights.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("timm")
pytest.importorskip("anomalib")

from anomaly_lab.models.dino_backbone import (  # noqa: E402
    BACKBONES,
    DinoBackbone,
    extract_layer_tokens,
    load_backbone,
)
from anomaly_lab.models.model_assets import timm_bindings_preserved  # noqa: E402
from anomaly_lab.models.patchcore_anomalib import (  # noqa: E402
    PatchcoreAnomalibModel,
    PatchcoreConfig,
)

ENCODER = DinoBackbone.DINOV2_VIT_S14_REG4
BLOCKS = (2, 9)


@pytest.fixture
def vit(monkeypatch: pytest.MonkeyPatch) -> Any:
    """timm's ViT module, with its resample binding put back after the test whatever happens."""
    module: Any = importlib.import_module("timm.models.vision_transformer")
    monkeypatch.setattr(module, "resample_abs_pos_embed", module.resample_abs_pos_embed)
    return module


def _batch() -> Any:
    # 112x140 is an 8x10 grid, far from the table's 37x37, so the resample is a real downsample
    # and the antialias the rebinding drops is doing work.
    return torch.from_numpy(
        np.random.default_rng(1).normal(size=(1, 3, 112, 140)).astype(np.float32)
    )


def _encoder(tmp_path: Path) -> Any:
    return load_backbone(
        ENCODER,
        pretrained=False,
        allow_downloads=False,
        cache_dir=tmp_path,
        seed=2,
        method="test",
    )


def _tokens(encoder: Any) -> list[Any]:
    with torch.no_grad():
        return extract_layer_tokens(encoder, _batch(), BLOCKS)


def _build_anomalib_vit() -> Any:
    from anomalib.models.components.feature_extractors import TimmFeatureExtractor

    return TimmFeatureExtractor(
        BACKBONES[ENCODER].timm_name.split(".")[0],
        layers=["blocks.2"],
        pre_trained=False,
        output_fmt="NLC",
    )


def _assert_identical(left: list[Any], right: list[Any]) -> None:
    for a, b in zip(left, right, strict=True):
        torch.testing.assert_close(a, b, atol=0.0, rtol=0.0)


def test_without_the_guard_the_extractor_moves_dino_features(vit: Any, tmp_path: Path) -> None:
    """The control: the finding this file exists for is real, so the pins below can fail."""
    original = vit.resample_abs_pos_embed
    encoder = _encoder(tmp_path)
    before = _tokens(encoder)

    _build_anomalib_vit()

    assert vit.resample_abs_pos_embed is not original
    after = _tokens(encoder)
    assert any(not torch.equal(a, b) for a, b in zip(before, after, strict=True))


@pytest.mark.parametrize("order", ["dino_first", "anomalib_first"])
def test_dino_features_do_not_depend_on_an_extractor_built_before(
    vit: Any, tmp_path: Path, order: str
) -> None:
    original = vit.resample_abs_pos_embed
    reference = _tokens(_encoder(tmp_path))

    if order == "dino_first":
        encoder = _encoder(tmp_path)
        with timm_bindings_preserved():
            _build_anomalib_vit()
    else:
        with timm_bindings_preserved():
            _build_anomalib_vit()
        encoder = _encoder(tmp_path)

    assert vit.resample_abs_pos_embed is original
    _assert_identical(_tokens(encoder), reference)


def test_the_binding_is_restored_when_construction_fails(vit: Any) -> None:
    original = vit.resample_abs_pos_embed
    with pytest.raises(RuntimeError, match="boom"), timm_bindings_preserved():
        vit.resample_abs_pos_embed = lambda *args, **kwargs: None
        raise RuntimeError("boom")
    assert vit.resample_abs_pos_embed is original


def test_patchcore_built_inside_the_guard_computes_what_it_computed_bare(vit: Any) -> None:
    from anomalib.models.image.patchcore.torch_model import PatchcoreModel

    config = PatchcoreConfig(backbone="resnet18", pretrained_backbone=False, seed=0)
    original = vit.resample_abs_pos_embed
    guarded = PatchcoreAnomalibModel(config)._build_model("cpu")
    assert vit.resample_abs_pos_embed is original

    torch.manual_seed(config.seed)
    bare = PatchcoreModel(
        layers=list(config.layer_set.layers),
        backbone=config.backbone,
        pre_trained=False,
        num_neighbors=config.num_neighbors,
    ).eval()
    batch = torch.from_numpy(
        np.random.default_rng(3).normal(size=(2, 3, 64, 64)).astype(np.float32)
    )
    with torch.no_grad():
        left = guarded.feature_extractor(batch)
        right = bare.feature_extractor(batch)
    assert left.keys() == right.keys()
    for key in left:
        torch.testing.assert_close(left[key], right[key], atol=0.0, rtol=0.0)
