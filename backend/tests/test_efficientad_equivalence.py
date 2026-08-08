"""`efficientad_custom`'s verified core, pinned against anomalib.

**This is not deference to anomalib.** ADR-0029 says anomalib is the baseline this method
is measured against, not the specification it must match, and the improvements in
`efficientad_custom.py` are opt-in fields precisely so they can depart from it. What these
tests protect is the thing that makes such a departure *legible*: with every improvement
switched off, our arithmetic is the published arithmetic, so a change in a number is
attributable to the change we made rather than to a difference nobody noticed.

Two pins, different in kind:

  * **The PDN pin is not a choice.** The pretrained teacher is a published file keyed
    `conv1.weight` … `convN.bias`. If our PDN is not the network those weights describe,
    we are loading a public asset into the wrong model — which does not raise, it just
    quietly costs accuracy. Checked by shape everywhere, and by output wherever the asset
    is actually on disk.
  * **The rest is the core regression net.** The autoencoder, the three losses and the map
    computation, verified once and then held.

Gated on anomalib because it is the reference. The method's own tests are gated on torch
alone — `efficientad_custom` deliberately does not depend on anomalib, and
`test_efficientad_custom.py` is what measures that.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("anomalib")

from anomalib.models.image.efficient_ad import torch_model as reference  # noqa: E402

from anomaly_lab.models.efficientad_nets import (  # noqa: E402
    TEACHER_OUT_CHANNELS,
    EfficientAdNet,
    ModelSize,
    PatchDescriptionNetwork,
)

SIZES: tuple[ModelSize, ...] = ("small", "medium")
INPUT = 256

_REFERENCE_PDN = {
    "small": reference.SmallPatchDescriptionNetwork,
    "medium": reference.MediumPatchDescriptionNetwork,
}


def _published_teacher(size: ModelSize) -> Path | None:
    """Where anomalib caches the pretrained teacher, if this machine has fetched it.

    CI never downloads it, so the output comparison that needs it skips there and the
    shape comparison — which catches the same class of mistake — runs everywhere.
    """
    from anomalib.utils.path import get_pretrained_weights_dir

    path = (
        get_pretrained_weights_dir()
        / "efficientad_pretrained_weights"
        / f"pretrained_teacher_{size}.pth"
    )
    return path if path.is_file() else None


def _paired(size: ModelSize) -> tuple[EfficientAdNet, Any]:
    """Our net and the reference, carrying identical weights.

    Copied across by name, so a structural difference surfaces here as a loud
    `load_state_dict` failure rather than as a small numeric discrepancy later.
    """
    torch.manual_seed(7)
    ours = EfficientAdNet(size=size)
    torch.manual_seed(7)
    theirs = reference.EfficientAdModel(model_size=reference.EfficientAdModelSize(size))
    ours.teacher.load_state_dict(theirs.teacher.state_dict())
    ours.student.load_state_dict(theirs.student.state_dict())
    ours.ae.load_state_dict(theirs.ae.state_dict())
    return ours, theirs


def _fit_statistics(ours: EfficientAdNet, theirs: Any) -> None:
    """Give both the same teacher statistics and map quantiles.

    Unfitted, both sides skip their normalization and the comparison would never reach the
    code that applies it — so the interesting path would go untested.
    """
    torch.manual_seed(11)
    mean = torch.randn(1, TEACHER_OUT_CHANNELS, 1, 1)
    std = torch.rand(1, TEACHER_OUT_CHANNELS, 1, 1) + 0.5
    ours.set_teacher_statistics(mean, std)
    theirs.mean_std.update({"mean": mean, "std": std})
    quantiles = {
        "qa_st": torch.tensor(0.2),
        "qb_st": torch.tensor(1.3),
        "qa_ae": torch.tensor(0.1),
        "qb_ae": torch.tensor(0.9),
    }
    ours.set_quantiles(quantiles)
    theirs.quantiles.update(quantiles)


@pytest.mark.parametrize("size", SIZES)
def test_our_pdn_is_the_network_the_published_weights_describe(size: ModelSize) -> None:
    """Same parameter names, same shapes — checked without downloading 40 MB.

    `load_state_dict` matches on names. A PDN whose layers are named or shaped differently
    would either refuse the published teacher outright or, worse, accept a subset of it.
    """
    ours = PatchDescriptionNetwork(TEACHER_OUT_CHANNELS, size)
    theirs = _REFERENCE_PDN[size](out_channels=TEACHER_OUT_CHANNELS)

    expected = {name: tuple(value.shape) for name, value in theirs.state_dict().items()}
    actual = {name: tuple(value.shape) for name, value in ours.state_dict().items()}
    assert actual == expected


@pytest.mark.parametrize("size", SIZES)
def test_the_published_teacher_loads_and_produces_the_same_features(size: ModelSize) -> None:
    """The pin that matters, wherever the asset is actually present.

    Skipped rather than downloaded: a test suite that fetches 40 MB is a test suite people
    turn off.
    """
    weights = _published_teacher(size)
    if weights is None:
        pytest.skip(f"the pretrained {size} teacher is not cached on this machine")

    stored = torch.load(weights, map_location="cpu", weights_only=True)
    ours = PatchDescriptionNetwork(TEACHER_OUT_CHANNELS, size).eval()
    theirs = _REFERENCE_PDN[size](out_channels=TEACHER_OUT_CHANNELS).eval()
    ours.load_state_dict(stored, strict=True)
    theirs.load_state_dict(stored, strict=True)

    torch.manual_seed(3)
    probe = torch.rand(2, 3, INPUT, INPUT)
    with torch.no_grad():
        torch.testing.assert_close(ours(probe), theirs(probe), rtol=0, atol=0)


@pytest.mark.parametrize("size", SIZES)
def test_both_branch_maps_match_the_reference(size: ModelSize) -> None:
    """The student-teacher and autoencoder maps, through normalization, at input scale."""
    ours, theirs = _paired(size)
    _fit_statistics(ours, theirs)
    ours.eval()
    theirs.eval()

    torch.manual_seed(3)
    probe = torch.rand(2, 3, INPUT, INPUT)
    with torch.no_grad():
        our_st, our_stae = ours.maps(probe, normalize=True)
        their_st, their_stae = theirs.get_maps(probe, normalize=True)

    torch.testing.assert_close(our_st, their_st, rtol=0, atol=0)
    torch.testing.assert_close(our_stae, their_stae, rtol=0, atol=0)


@pytest.mark.parametrize("size", SIZES)
def test_the_combined_map_and_score_match_the_reference(size: ModelSize) -> None:
    """With default weights and no score quantile, `score` is the published behaviour.

    The two knobs M6 measures — the branch weighting and the percentile reduction — have
    defaults that reproduce an even blend and a maximum, so an untouched run *is* the
    verified core. If that ever stops being true this test is what says so.
    """
    ours, theirs = _paired(size)
    _fit_statistics(ours, theirs)
    ours.eval()
    theirs.eval()

    torch.manual_seed(3)
    probe = torch.rand(2, 3, INPUT, INPUT)
    with torch.no_grad():
        combined, score = ours.score(probe)
        expected = theirs(probe)

    torch.testing.assert_close(combined, expected.anomaly_map, rtol=0, atol=0)
    # Ours is `(B,)` where the reference carries a redundant channel axis; comparing them
    # unflattened would broadcast into a `(B, B)` matrix and pass for the wrong reason.
    assert score.shape == (probe.shape[0],)
    torch.testing.assert_close(score, expected.pred_score.flatten(), rtol=0, atol=0)


def _silence_dropout(module: Any) -> None:
    """Turn the decoder's dropout off on either implementation.

    A loss comparison has to hold the stochastic inputs fixed, and dropout draws six masks
    per autoencoder forward. Aligning six draws across two implementations is exactly the
    test that passes once and then flakes; zeroing the rate compares the arithmetic, and
    `test_the_decoder_keeps_its_six_dropout_layers` separately holds the fact that the rate
    is 0.2 in real training.
    """
    from torch import nn

    for child in module.modules():
        if isinstance(child, nn.Dropout):
            child.p = 0.0


@pytest.mark.parametrize("size", SIZES)
def test_all_three_losses_match_the_reference(
    size: ModelSize, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Hard feature loss, autoencoder loss and student-autoencoder loss.

    Ours takes the augmented image as an argument, so the comparison fixes it and hands the
    same tensor to both sides — the reference draws its own inside `compute_losses`, and is
    monkeypatched here to return ours. That is the whole reason the signature differs:
    a pure loss is one a test can pin without choreographing two RNG streams.
    """
    ours, theirs = _paired(size)
    _fit_statistics(ours, theirs)
    ours.train()
    theirs.train()
    _silence_dropout(ours)
    _silence_dropout(theirs)

    torch.manual_seed(3)
    probe = torch.rand(2, 3, INPUT, INPUT)
    penalty = torch.rand(2, 3, INPUT, INPUT)
    augmented = torch.rand(2, 3, INPUT, INPUT)
    monkeypatch.setattr(
        reference.EfficientAdModel, "choose_random_aug_image", staticmethod(lambda _: augmented)
    )

    _, distance = ours.student_teacher_distance(probe)
    our_losses = ours.compute_losses(penalty, augmented, distance)
    their_losses = theirs.compute_losses(
        probe, penalty, theirs.compute_student_teacher_distance(probe)[1]
    )

    for name, actual, expected in zip(("st", "ae", "stae"), our_losses, their_losses, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0, msg=f"loss_{name}")


def test_the_decoder_keeps_its_six_dropout_layers() -> None:
    """Held separately, because the loss comparison above switches them off.

    Dropout is the autoencoder branch's only regularization. Without it the decoder learns
    the training set well enough that a logical anomaly reconstructs cleanly too — the
    branch keeps working and stops detecting, which no equivalence test would notice.
    """
    from torch import nn

    ours, _ = _paired("small")
    rates = [child.p for child in ours.ae.decoder.modules() if isinstance(child, nn.Dropout)]
    assert rates == [0.2] * 6


@pytest.mark.parametrize("name", ["adjust_brightness", "adjust_contrast", "adjust_saturation"])
@pytest.mark.parametrize("factor", [0.8, 0.95, 1.0, 1.2])
def test_our_augmentations_match_torchvision(name: str, factor: float) -> None:
    """This is what licenses dropping the torchvision dependency.

    The three adjustments are scalar blends against an ITU-R 601-2 luma, so writing them
    out is fifteen lines and removes a dependency `efficientad_custom` otherwise needs for
    nothing else. The claim that they are the *same* fifteen lines is checked here rather
    than asserted in a comment.
    """
    tv = pytest.importorskip("torchvision.transforms.functional")
    from anomaly_lab.models import efficientad_nets

    torch.manual_seed(1)
    image = torch.rand(2, 3, 32, 32)
    ours = getattr(efficientad_nets, name)(image, factor)
    theirs = getattr(tv, name)(image, factor)
    torch.testing.assert_close(ours, theirs, rtol=1e-6, atol=1e-6)


def test_turning_off_the_penalty_changes_the_student_loss_and_nothing_else() -> None:
    """The pretraining penalty is a real term, not a no-op we could quietly drop.

    The paper measures it at +0.4 AU-ROC. A toggle that changed nothing would make that
    ablation silently unmeasurable, which is the failure this catches.
    """
    ours, theirs = _paired("small")
    _fit_statistics(ours, theirs)
    ours.train()
    _silence_dropout(ours)

    torch.manual_seed(3)
    probe = torch.rand(1, 3, INPUT, INPUT)
    penalty = torch.rand(1, 3, INPUT, INPUT)
    augmented = torch.rand(1, 3, INPUT, INPUT)

    _, distance = ours.student_teacher_distance(probe)
    with_penalty = ours.compute_losses(penalty, augmented, distance)
    without = ours.compute_losses(penalty, augmented, distance, use_penalty=False)

    assert without[0] < with_penalty[0]
    torch.testing.assert_close(without[1], with_penalty[1], rtol=0, atol=0)
    torch.testing.assert_close(without[2], with_penalty[2], rtol=0, atol=0)
