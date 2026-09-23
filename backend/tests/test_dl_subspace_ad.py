"""The encoder half of the SubspaceAD campaign, against real timm definitions and no network.

`pretrained=False` throughout, for the reason `test_dl_dino_backbone.py` gives: what is
pinned here is the *plumbing* -- that one forward pass serves every layer view, that a view
slices the blocks it asked for, that rotation masking removes exactly the invented corners,
and that a category runs end to end into well-formed rows. None of that is a property of the
weights, and downloading a gigabyte to check it would put a public network service in the
test suite.

The arithmetic those features flow into is checked without torch at all, in
`test_subspace_ad_math.py`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("timm")

from research.subspace_ad.benchmarks import CategorySplit, ScoredImage
from research.subspace_ad.campaign import CampaignSpec, run_category
from research.subspace_ad.features import (
    LAYER_BANDS,
    Aggregation,
    FeatureView,
    PatchEncoder,
    RotationFill,
    prepare,
    rotations,
    valid_patches,
)

from anomaly_lab.models.dino_backbone import BACKBONES, DinoBackbone

SIZE = 112
"""Divisible by 14 and by 16, so both families run on identical pixels -- the same choice
`test_dl_dino_backbone.py` makes, and the property that lets the campaign compare the two
families at one resolution rather than at two nearly-equal ones."""

FAMILIES = (DinoBackbone.DINOV2_VIT_S14_REG4, DinoBackbone.DINOV3_VIT_S16)


def _encoder(
    backbone: DinoBackbone, tmp_path: Path, views: tuple[FeatureView, ...]
) -> PatchEncoder:
    return PatchEncoder(
        backbone,
        size=SIZE,
        views=views,
        cache_dir=tmp_path,
        device="cpu",
        pretrained=False,
        allow_downloads=False,
    )


def _frames(count: int, seed: int = 0) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [rng.random((SIZE, SIZE, 3), dtype=np.float32) for _ in range(count)]


@pytest.mark.parametrize("backbone", FAMILIES)
def test_one_pass_serves_every_view_at_its_own_width(
    backbone: DinoBackbone, tmp_path: Path
) -> None:
    views = (
        FeatureView(LAYER_BANDS["last"]),
        FeatureView(LAYER_BANDS["final7"]),
        FeatureView(LAYER_BANDS["final7"], aggregation=Aggregation.CONCAT),
    )
    encoder = _encoder(backbone, tmp_path, views)
    dimension = BACKBONES[backbone].embedding_dim
    patches = encoder.patch_count

    features = encoder.encode(_frames(2))

    assert set(features) == {"last-mean", "final7-mean", "final7-concat"}
    assert features["last-mean"].shape == (2, patches, dimension)
    assert features["final7-mean"].shape == (2, patches, dimension)
    assert features["final7-concat"].shape == (2, patches, dimension * 7)
    assert encoder.patch_count == encoder.grid[0] * encoder.grid[1]


def test_the_union_of_every_views_blocks_is_requested_exactly_once(tmp_path: Path) -> None:
    """The layer axis is free only if overlapping views share one forward pass."""
    views = (
        FeatureView(LAYER_BANDS["last"]),
        FeatureView(LAYER_BANDS["last_four"]),
        FeatureView(LAYER_BANDS["final7"]),
    )
    encoder = _encoder(DinoBackbone.DINOV2_VIT_S14_REG4, tmp_path, views)

    assert encoder.indices == tuple(range(5, 12))
    assert len(encoder.indices) == len(set(encoder.indices))
    assert encoder.positions[11] == len(encoder.indices) - 1


def test_a_view_aggregates_the_blocks_it_asked_for_and_no_others() -> None:
    """Eq. 2 is an unweighted average over the chosen blocks, checked on a known stack.

    Against a synthetic `(B, L, D, P)` tensor rather than against an encoder, because an
    *untrained* ViT cannot express the difference: with timm's default initialization the
    residual branches contribute almost nothing, so every block output is within 1e-5 of
    every other once `norm=True` has put them all through the same final LayerNorm. A test
    that ran this through `pretrained=False` weights would pass just as happily on an
    implementation that ignored the layer list and returned one block.
    """
    import torch

    # Block b carries the constant value b, so any aggregation's answer is arithmetic.
    depth, dimension, patches = 12, 4, 3
    indices = [8, 9, 10, 11]
    stack = torch.stack(
        [torch.full((1, dimension, patches), float(index + 1)) for index in indices], dim=1
    )
    positions = {index: row for row, index in enumerate(indices)}

    mean = FeatureView(LAYER_BANDS["last_two"]).apply(stack, positions, depth)
    torch.testing.assert_close(mean, torch.full((1, dimension, patches), 11.5))

    concat = FeatureView(LAYER_BANDS["last_two"], aggregation=Aggregation.CONCAT)
    widened = concat.apply(stack, positions, depth)
    assert widened.shape == (1, dimension * 2, patches)
    torch.testing.assert_close(widened[0, :dimension, 0], torch.full((dimension,), 11.0))
    torch.testing.assert_close(widened[0, dimension:, 0], torch.full((dimension,), 12.0))

    # All four blocks, so a view that silently took only the last would answer 12.
    everything = FeatureView(LAYER_BANDS["last_four"]).apply(stack, positions, depth)
    torch.testing.assert_close(everything, torch.full((1, dimension, patches), 10.5))


def test_l2_normalizing_makes_every_layer_an_equal_vote_before_the_pool() -> None:
    """The convention `dino_memory` uses and SubspaceAD does not, held apart on purpose.

    A layer whose activations are ten times another's dominates a plain mean and contributes
    equally to a normalized one, which is the whole of the difference.
    """
    import torch

    depth, dimension, patches = 12, 4, 1
    indices = [10, 11]
    stack = torch.stack(
        [torch.full((1, dimension, patches), scale) for scale in (1.0, 10.0)], dim=1
    )
    positions = {index: row for row, index in enumerate(indices)}

    plain = FeatureView(LAYER_BANDS["last_two"]).apply(stack, positions, depth)
    normalized = FeatureView(LAYER_BANDS["last_two"], l2_normalize=True).apply(
        stack, positions, depth
    )

    torch.testing.assert_close(plain, torch.full((1, dimension, patches), 5.5))
    # Each layer becomes a unit vector, so both contribute 1/sqrt(dimension) per component.
    torch.testing.assert_close(normalized, torch.full((1, dimension, patches), 0.5))


def test_the_final_norm_convention_changes_the_intermediates(tmp_path: Path) -> None:
    """timm's `norm=True` applies the encoder's last LayerNorm to every block it returns."""
    views = (FeatureView(LAYER_BANDS["final7"]),)
    frames = _frames(1)
    normed = _encoder(DinoBackbone.DINOV2_VIT_S14_REG4, tmp_path, views)
    raw = PatchEncoder(
        DinoBackbone.DINOV2_VIT_S14_REG4,
        size=SIZE,
        views=views,
        cache_dir=tmp_path,
        device="cpu",
        pretrained=False,
        allow_downloads=False,
        final_norm=False,
    )
    assert not np.allclose(
        normed.encode(frames)["final7-mean"], raw.encode(frames)["final7-mean"], atol=1e-4
    )


def test_rotation_masking_removes_exactly_the_invented_corners() -> None:
    frame = np.ones((SIZE, SIZE, 3), dtype=np.float32)
    rng = np.random.default_rng(0)

    zeros = rotations(frame, count=3, rng=rng, fill=RotationFill.ZEROS)
    masked = rotations(frame, count=3, rng=np.random.default_rng(0), fill=RotationFill.MASKED)

    assert [array.shape for array, _ in zeros] == [(SIZE, SIZE, 3)] * 4
    assert all(valid is None for _, valid in zeros)
    # The unrotated original is never masked; every rotated copy is.
    assert masked[0][1] is None
    assert all(valid is not None for _, valid in masked[1:])

    grid = (SIZE // 14, SIZE // 14)
    keeps = [valid_patches(valid, grid, 14) for _, valid in masked[1:]]
    for keep in keeps:
        assert keep is not None
        assert 0 < int(keep.sum()) < grid[0] * grid[1]
    assert valid_patches(None, grid, 14) is None


def test_rotation_is_withheld_entirely_when_the_category_is_orientation_sensitive() -> None:
    frame = np.ones((SIZE, SIZE, 3), dtype=np.float32)
    assert (
        len(rotations(frame, count=0, rng=np.random.default_rng(0), fill=RotationFill.ZEROS)) == 1
    )


def _synthetic_split(tmp_path: Path) -> CategorySplit:
    """Normals that are smooth, anomalies that carry a block of noise.

    The direction matters. An earlier version made the normals noisy and the defect a
    uniform white square, which separated perfectly *backwards*: with a 8x8 token grid,
    rho=1% takes a single patch, so an image whose defect replaced a fifth of its noisy
    patches simply had fewer draws at the maximum and scored lower. The lesson is about the
    fixture rather than the method -- but it is the reason this one adds structure the
    subspace has never seen instead of removing structure it has.

    Unpretrained weights are still a deterministic function of the input, so a subspace
    fitted on smooth gradients does not reconstruct dense noise. What is being pinned is the
    sign of the score, not the discriminative power of random weights.
    """
    from PIL import Image

    rng = np.random.default_rng(5)
    root = tmp_path / "pictures"
    root.mkdir()
    ramp = np.linspace(40, 210, SIZE, dtype=np.float32)

    def smooth(offset: float) -> np.ndarray:
        field = np.clip(ramp[None, :] + ramp[:, None] * 0.25 + offset, 0, 255)
        return np.repeat(field[:, :, None], 3, axis=2).astype(np.uint8)

    def write(name: str, array: np.ndarray) -> Path:
        path = root / name
        Image.fromarray(array, mode="RGB").save(path)
        return path

    normals = [write(f"normal{index}.png", smooth(index * 3.0)) for index in range(4)]
    tests: list[ScoredImage] = [
        ScoredImage(path=normals[0], label=0, mask_path=None, defect="good"),
        ScoredImage(path=write("clean.png", smooth(1.5)), label=0, mask_path=None, defect="good"),
    ]
    for index in range(2):
        loud = smooth(index * 2.0)
        loud[28:84, 28:84] = rng.integers(0, 255, (56, 56, 3), dtype=np.uint8)
        mask = np.zeros((SIZE, SIZE), dtype=np.uint8)
        mask[28:84, 28:84] = 255
        mask_path = root / f"mask{index}.png"
        Image.fromarray(mask, mode="L").save(mask_path)
        tests.append(
            ScoredImage(
                path=write(f"defect{index}.png", loud),
                label=1,
                mask_path=mask_path,
                defect="noise_block",
            )
        )
    return CategorySplit(
        benchmark="synthetic",
        category="texture",
        fit_pool=tuple(normals),
        test_items=tuple(tests),
    )


def test_a_category_runs_end_to_end_into_well_formed_rows(tmp_path: Path) -> None:
    import io

    split = _synthetic_split(tmp_path)
    views = (FeatureView(LAYER_BANDS["last"]), FeatureView(LAYER_BANDS["final7"]))
    encoder = _encoder(DinoBackbone.DINOV2_VIT_S14_REG4, tmp_path, views)
    spec = CampaignSpec(
        views=views,
        shots=(1, 2),
        seeds=(0,),
        taus=(0.9, 0.99),
        rhos=(0.01, 0.1),
        augmentations=2,
        batch_size=3,
    )

    rows = run_category(encoder, split, spec, io.StringIO())

    # two views x two shot counts x one seed x two thresholds x two tail fractions
    assert len(rows) == 2 * 2 * 1 * 2 * 2
    for row in rows:
        assert row.test_images == 4
        assert row.anomalies == 2
        assert row.image_auroc is not None and 0.0 <= row.image_auroc <= 1.0
        assert row.pixel_auroc is not None and 0.0 <= row.pixel_auroc <= 1.0
        assert row.au_pro is not None
        assert 1 <= row.rank <= row.dimension
        assert row.fit_patches == row.shots * 3 * encoder.patch_count
        assert row.blocks and max(row.blocks) <= encoder.depth

    # The shot count is nested: a 2-shot fit sees exactly twice a 1-shot fit's patches.
    by_shots = {row.shots: row.fit_patches for row in rows}
    assert by_shots[2] == 2 * by_shots[1]


def test_the_planted_square_is_the_image_the_subspace_cannot_reconstruct(
    tmp_path: Path,
) -> None:
    """The end-to-end sign check: a defect must score above a normal, not merely differ."""
    import io

    split = _synthetic_split(tmp_path)
    views = (FeatureView(LAYER_BANDS["final7"]),)
    encoder = _encoder(DinoBackbone.DINOV2_VIT_S14_REG4, tmp_path, views)
    spec = CampaignSpec(
        views=views, shots=(2,), seeds=(0,), taus=(0.9,), rhos=(0.05,), augmentations=2
    )

    rows = run_category(encoder, split, spec, io.StringIO())

    assert len(rows) == 1
    assert rows[0].image_auroc == pytest.approx(1.0)
    assert rows[0].pixel_auroc is not None and rows[0].pixel_auroc > 0.5


def test_prepared_pixels_come_back_with_the_transform_that_made_them(tmp_path: Path) -> None:
    """The mask has to travel the same path, so the transform is not optional baggage."""
    from PIL import Image

    path = tmp_path / "wide.png"
    Image.new("RGB", (200, 100), color=(10, 20, 30)).save(path)

    prepared = prepare(path, SIZE)

    assert prepared.array.shape == (SIZE, SIZE, 3)
    assert prepared.array.dtype == np.float32
    assert 0.0 <= float(prepared.array.min()) <= float(prepared.array.max()) <= 1.0
    assert prepared.transform.source_width == 200
    assert prepared.transform.prepared_width == SIZE
    # A 2:1 source is letterboxed rather than squashed, which is what keeps a round defect
    # round and a pixel metric comparable across datasets of different aspect ratios.
    assert prepared.transform.pad_top > 0
