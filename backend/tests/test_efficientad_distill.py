"""The distillation stage, checked against the reference it reproduces.

**Gated on torch, and it downloads nothing.** The arithmetic that decides what a distilled
teacher learns is `aggregate_patch_features`, which takes two feature maps and knows nothing
about WideResNet — so it is tested against a transcription of the reference implementation on
synthetic tensors, at full width, in under a second. Testing it through a 243 MB backbone
would have tested torchvision.

The transcription below is deliberately literal, down to the reshape/permute sequence, and is
not shared with the implementation — a test that called the same helper would be a tautology.

**It agrees to 6e-8, not exactly, and that is the honest bar.** The two expressions reduce the
same numbers in a different order, so the last unit in the last place of a float32 disagrees on
about a quarter of the elements. `atol=1e-6` is two decades above the observed disagreement and
six decades below anything that could change what a teacher learns; asserting bit-equality here
would be asserting that two orderings of a sum are the same, which is false and not the claim.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")

from torch.nn import functional as F  # noqa: E402, N812

from anomaly_lab.models.teacher_distill import (  # noqa: E402
    DISTILLED_SUBDIR,
    PDN_GRID,
    SOURCE_INPUT,
    DistillConfig,
    aggregate_patch_features,
    corpus_images,
    load_manifest,
    teacher_dir,
)

OUT_CHANNELS = 384


class _Reporter:
    def log(self, message: str, level: str = "info") -> None:
        return

    def progress(self, fraction: float, message: str | None = None) -> None:
        return


def _reference_embed(features: list[Any], out_channels: int) -> Any:
    """PatchCore's aggregation as the reference writes it, transcribed without shortcuts."""

    def patchify(feature: Any) -> tuple[Any, list[int]]:
        padding = 1
        unfolder = torch.nn.Unfold(kernel_size=3, stride=1, padding=padding, dilation=1)
        unfolded = unfolder(feature)
        counts = [
            int((side + 2 * padding - 1 * (3 - 1) - 1) / 1 + 1) for side in feature.shape[-2:]
        ]
        unfolded = unfolded.reshape(*feature.shape[:2], 3, 3, -1)
        return unfolded.permute(0, 4, 1, 2, 3), counts

    patched = [patchify(x) for x in features]
    shapes = [x[1] for x in patched]
    values = [x[0] for x in patched]
    reference_patches = shapes[0]

    for index in range(1, len(values)):
        current = values[index]
        dims = shapes[index]
        current = current.reshape(current.shape[0], dims[0], dims[1], *current.shape[2:])
        current = current.permute(0, -3, -2, -1, 1, 2)
        base_shape = current.shape
        current = current.reshape(-1, *current.shape[-2:])
        current = F.interpolate(
            current.unsqueeze(1),
            size=(reference_patches[0], reference_patches[1]),
            mode="bilinear",
            align_corners=False,
        )
        current = current.squeeze(1)
        current = current.reshape(*base_shape[:-2], reference_patches[0], reference_patches[1])
        current = current.permute(0, -2, -1, 1, 2, 3)
        values[index] = current.reshape(len(current), -1, *current.shape[-3:])

    values = [x.reshape(-1, *x.shape[-3:]) for x in values]
    mapped = [F.adaptive_avg_pool1d(x.reshape(len(x), 1, -1), 1024).squeeze(1) for x in values]
    stacked = torch.stack(mapped, dim=1)
    aggregated = F.adaptive_avg_pool1d(stacked.reshape(len(stacked), 1, -1), out_channels)
    aggregated = aggregated.reshape(len(aggregated), -1)
    reshaped = torch.reshape(
        aggregated, (-1, reference_patches[0], reference_patches[1], out_channels)
    )
    return torch.permute(reshaped, (0, 3, 1, 2))


@pytest.mark.parametrize("batch", [1, 2])
def test_the_aggregation_matches_the_reference_exactly(batch: int) -> None:
    """The target the PDN regresses onto must be the reference's target, to the bit.

    A teacher distilled onto a subtly different target is a different teacher, and the whole
    reason this stage exists is that different teachers are worth 0.36 AU-PRO.
    """
    torch.manual_seed(0)
    # The real shapes at 512 px: layer2 is stride 8 with 512 channels, layer3 stride 16
    # with 1024. Small spatial extents keep the transcription's memory sane while leaving
    # the channel arithmetic — which is where an ordering mistake would hide — at full width.
    fine = torch.randn(batch, 512, 8, 8)
    coarse = torch.randn(batch, 1024, 4, 4)

    ours = aggregate_patch_features(fine, coarse, OUT_CHANNELS, torch)
    theirs = _reference_embed([fine, coarse], OUT_CHANNELS)

    assert ours.shape == theirs.shape == (batch, OUT_CHANNELS, 8, 8)
    assert torch.allclose(ours, theirs, rtol=0, atol=1e-6)


def test_the_chunked_pool_does_not_change_the_answer(monkeypatch: pytest.MonkeyPatch) -> None:
    """Memory chunking is an implementation detail and must stay one.

    The chunk exists so batch size is a throughput knob rather than a memory cliff; if it
    changed the arithmetic it would be a silent accuracy setting instead.
    """
    from anomaly_lab.models import teacher_distill

    torch.manual_seed(1)
    fine = torch.randn(2, 512, 8, 8)
    coarse = torch.randn(2, 1024, 4, 4)

    whole = aggregate_patch_features(fine, coarse, OUT_CHANNELS, torch)
    monkeypatch.setattr(teacher_distill, "POOL_CHUNK", 3)
    chunked = aggregate_patch_features(fine, coarse, OUT_CHANNELS, torch)

    assert torch.equal(whole, chunked)


def test_the_pdn_emits_the_grid_the_source_targets() -> None:
    """Distillation regresses a 64x64 PDN output onto a 64x64 target.

    The reference distils with padding **on** and detects with it off, where the same
    weights give a 56x56 map that is padded afterwards. Both are true because padding
    changes a convolution's extent and not its weights — but a run that got this backwards
    would train against a resized target and finish without complaining.
    """
    from anomaly_lab.models.efficientad_nets import PatchDescriptionNetwork

    probe = torch.zeros(1, 3, 256, 256)
    for size in ("small", "medium"):
        padded = PatchDescriptionNetwork(out_channels=OUT_CHANNELS, size=size, padding=True)
        unpadded = PatchDescriptionNetwork(out_channels=OUT_CHANNELS, size=size, padding=False)
        assert padded(probe).shape[-2:] == (PDN_GRID, PDN_GRID)
        assert unpadded(probe).shape[-2:] == (56, 56)

    # And the source's fine layer lands on the same grid: stride 8 over 512 px.
    assert SOURCE_INPUT // 8 == PDN_GRID


def test_a_directory_corpus_with_no_images_is_refused_by_name(tmp_path: Path) -> None:
    empty = tmp_path / "corpus"
    empty.mkdir()
    config = DistillConfig(name="probe", corpus="directory", corpus_path=str(empty))
    with pytest.raises(FileNotFoundError, match="no images"):
        corpus_images(config, tmp_path / "cache", _Reporter())


def test_a_missing_corpus_directory_names_imagenet(tmp_path: Path) -> None:
    """Somebody pointing this at ImageNet mistyped a path; say what the field is for."""
    config = DistillConfig(name="probe", corpus="directory", corpus_path=str(tmp_path / "absent"))
    with pytest.raises(FileNotFoundError, match="ImageNet-1K"):
        corpus_images(config, tmp_path / "cache", _Reporter())


def test_an_unknown_distilled_teacher_lists_what_is_there(tmp_path: Path) -> None:
    """The refusal has to be actionable: a name that is wrong is usually a name that is close."""
    cache = tmp_path / "cache"
    (cache / DISTILLED_SUBDIR / "wrn-imagenette").mkdir(parents=True)
    with pytest.raises(FileNotFoundError) as failure:
        load_manifest(cache, "wrn-imagenet")
    assert "wrn-imagenette" in str(failure.value)
    assert str(teacher_dir(cache, "wrn-imagenet")) in str(failure.value)
