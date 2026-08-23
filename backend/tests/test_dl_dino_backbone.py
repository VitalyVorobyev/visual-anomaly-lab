"""The frozen encoder's contract, against real timm model definitions and no network.

Everything here builds its encoder with `pretrained=False`. That is not a shortcut around a
download: the properties being pinned — the token grid, the channel arithmetic of a
multi-layer read, and whether one seed gives one set of weights — are properties of the
*architecture* and of torch's random streams, and pretrained weights would make them slower
to check without making them stronger. What pretrained weights would add is a public network
service in the test suite, which is the trade `test_dl_dinomaly.py` also declines.

The one test that does ask for pretrained weights asks for them **offline**, against an
empty cache, because refusing to reach the network is the behaviour being tested.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("timm")

from anomaly_lab.models.coreset import gaussian_projection  # noqa: E402
from anomaly_lab.models.dino_backbone import (  # noqa: E402
    BACKBONES,
    DinoBackbone,
    FeatureLayers,
    backbone_fingerprint,
    extract_patch_features,
    load_backbone,
    patch_grid,
)

SIZE = 112
"""One prepared size divisible by both patch sizes — 14x8 and 16x7 — so the two families are
exercised on identical pixels, small enough that a forward pass is a fraction of a second."""

FAMILIES = (DinoBackbone.DINOV2_VIT_S14_REG4, DinoBackbone.DINOV3_VIT_S16)
"""One encoder from each family: the registered DINOv2 anomalib pins for Dinomaly, and the
ungated-shaped DINOv3 ViT-S. Both are ViT-S/12-deep, so what differs is the patch size."""


def _load(backbone: DinoBackbone, tmp_path: Path, *, seed: int = 0) -> Any:
    return load_backbone(
        backbone,
        pretrained=False,
        allow_downloads=False,
        cache_dir=tmp_path,
        seed=seed,
        method="test",
    )


# ------------------------------------------------------------------- the token grid


@pytest.mark.parametrize("backbone", FAMILIES)
def test_both_families_produce_the_grid_their_patch_size_implies(
    backbone: DinoBackbone, tmp_path: Path
) -> None:
    """The arithmetic `patch_grid` does without torch, checked against what timm does with it.

    A grid predicted by a pure function and a grid produced by a forward pass have to be the
    same number, or every memory figure computed at plan time is a plausible fiction.
    """
    model = _load(backbone, tmp_path)
    batch = torch.zeros(1, 3, SIZE, SIZE)

    features = extract_patch_features(model, batch, FeatureLayers.LAST.indices)

    expected = SIZE // BACKBONES[backbone].patch_size
    assert patch_grid(backbone, SIZE, SIZE) == (expected, expected)
    assert features.shape == (1, BACKBONES[backbone].embedding_dim, expected, expected)
    assert bool(torch.isfinite(features).all())


@pytest.mark.parametrize("backbone", FAMILIES)
def test_the_encoder_arrives_frozen_and_in_eval_mode(
    backbone: DinoBackbone, tmp_path: Path
) -> None:
    """Nothing here trains the encoder, and a stray `requires_grad` is how one starts to."""
    model = _load(backbone, tmp_path)

    assert model.training is False
    assert not any(parameter.requires_grad for parameter in model.parameters())


# --------------------------------------------------------------- the layer selection


def test_reading_two_layers_doubles_the_channel_width(tmp_path: Path) -> None:
    """Layers are concatenated, so the width is the sum — and the rest of the map is not."""
    model = _load(DinoBackbone.DINOV2_VIT_S14_REG4, tmp_path)
    batch = torch.zeros(1, 3, SIZE, SIZE)

    one = extract_patch_features(model, batch, FeatureLayers.LAST.indices)
    two = extract_patch_features(model, batch, FeatureLayers.LAST_TWO.indices)
    four = extract_patch_features(model, batch, FeatureLayers.LAST_FOUR.indices)

    assert two.shape[1] == 2 * one.shape[1]
    assert four.shape[1] == 4 * one.shape[1]
    assert two.shape[2:] == one.shape[2:] == four.shape[2:]


def test_each_layer_is_normalized_on_its_own_before_the_concatenation(tmp_path: Path) -> None:
    """The constraint the concatenation depends on, asserted rather than described.

    Every vector downstream is compared by Euclidean distance. If the layers were normalized
    after being joined, a block whose activations happen to be larger would carry more of
    every distance; normalized first, each layer contributes a unit vector and votes equally.
    So a two-layer feature has squared norm 2, not 1.
    """
    model = _load(DinoBackbone.DINOV2_VIT_S14_REG4, tmp_path)
    batch = torch.randn(1, 3, SIZE, SIZE)

    two = extract_patch_features(model, batch, FeatureLayers.MID_LATE.indices)

    norms = two.pow(2).sum(dim=1)
    assert torch.allclose(norms, torch.full_like(norms, 2.0), atol=1e-4)


# ------------------------------------------------------------------- the seed, M6 again


def test_one_seed_is_one_encoder_and_a_different_seed_is_a_different_one(
    tmp_path: Path,
) -> None:
    """Both directions, because pinning a seed only means something if changing it changes.

    An unpretrained encoder draws from torch's *global* stream, which is exactly the shape
    M6 found in EfficientAD's weight initialisation. Here the consequence is sharper than
    noise: the fingerprint is what a checkpoint stores to prove it is loading the weights it
    was fitted against, so an unseeded init would make every reload refuse a valid run.
    """
    backbone = DinoBackbone.DINOV2_VIT_S14
    first = backbone_fingerprint(_load(backbone, tmp_path, seed=7))
    again = backbone_fingerprint(_load(backbone, tmp_path, seed=7))
    other = backbone_fingerprint(_load(backbone, tmp_path, seed=8))

    assert first == again
    assert first != other
    assert len(first) == 64


# ------------------------------------------------------------------- the projection


def test_the_projection_is_one_matrix_per_seed_and_honours_its_width() -> None:
    torch.manual_seed(0)
    features = torch.randn(256, 64)

    first = gaussian_projection(features, 32, seed=3)
    again = gaussian_projection(features, 32, seed=3)
    other = gaussian_projection(features, 32, seed=4)

    assert first.shape == (256, 32)
    assert torch.equal(first, again)
    assert not torch.allclose(first, other)


def test_the_projection_is_not_drawn_from_the_global_stream() -> None:
    """Seeding a projection must not move a caller's weight initialisation, or vice versa.

    The reason it takes an explicit `torch.Generator`: a bank's projection and a method's
    init are two independent decisions, and routing both through `torch.manual_seed` makes
    each one silently depend on how many times the other was called first.
    """
    features = torch.randn(64, 16)

    torch.manual_seed(11)
    before = torch.randn(4)
    gaussian_projection(features, 8, seed=99)
    after = torch.randn(4)

    torch.manual_seed(11)
    assert torch.equal(before, torch.randn(4))
    assert torch.equal(after, torch.randn(4))


def test_the_projection_roughly_preserves_distance() -> None:
    """What the Johnson-Lindenstrauss scaling is for, checked as a property rather than a shape."""
    torch.manual_seed(0)
    features = torch.nn.functional.normalize(torch.randn(512, 256), dim=1)

    projected = gaussian_projection(features, 128, seed=0)

    original = torch.cdist(features[:64], features[:64])
    reduced = torch.cdist(projected[:64], projected[:64])
    mask = original > 0
    ratio = (reduced[mask] / original[mask]).mean()
    assert 0.9 < float(ratio) < 1.1


# ------------------------------------------------------------------ refusing the network


def test_pretrained_weights_offline_fail_by_name_rather_than_reaching_out(
    tmp_path: Path,
) -> None:
    """`allow_downloads=False` against an empty cache must name the asset and the knob.

    The failure has to be readable without opening huggingface_hub: which weights were
    wanted, and which switch would have allowed them.
    """
    backbone = DinoBackbone.DINOV2_VIT_S14
    with pytest.raises(RuntimeError) as failure:
        load_backbone(
            backbone,
            pretrained=True,
            allow_downloads=False,
            cache_dir=tmp_path / "cache",
            seed=0,
            method="dino_memory",
        )

    message = str(failure.value)
    assert BACKBONES[backbone].timm_name in message
    assert "allow_downloads" in message


def test_a_gated_backbone_says_how_to_get_access_and_what_the_ungated_ones_are(
    tmp_path: Path,
) -> None:
    """A 401 on licensed weights is a licence problem, and the message has to say so.

    The token itself is never read or printed — what the message carries is where to request
    approval, that an `HF_TOKEN` is needed afterwards, and the ungated encoders that need
    neither.
    """
    backbone = DinoBackbone.DINOV3_VIT_S16
    with pytest.raises(RuntimeError) as failure:
        load_backbone(
            backbone,
            pretrained=True,
            allow_downloads=False,
            cache_dir=tmp_path / "cache",
            seed=0,
            method="dino_memory",
        )

    message = str(failure.value)
    assert "huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m" in message
    assert "HF_TOKEN" in message
    assert "dinov2_vit_s14" in message
