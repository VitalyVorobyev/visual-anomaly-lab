"""The sweep: one forward pass per image, and every other axis read off it.

The loop is written around the cost model rather than around the parameter list, which is
why it does not read like a set of nested `for` statements over the axes. Two things are
expensive -- loading an encoder and pushing an image through it -- so those sit in the
outer loops and everything else is arranged to happen inside one pass:

  * **The shot count is free on the fit side** because the draws are nested. Seed *s*
    chooses `max(shots)` normals once; the accumulator is snapshotted as the pass crosses
    each requested k, so a 1-, 2- and 4-shot fit share every forward they have in common.
  * **The layer window is free** because the encoder returns the union of every view's
    blocks from a single pass, and a view is a slice and a mean over what came back.
  * **tau is free** because the projection is computed once at the largest rank any
    threshold selects and every smaller rank is a column of the running sum.
  * **rho is free** because the image score is a prefix mean of the sorted patch scores.

What remains is `benchmarks x categories x resolutions x backbones` forward passes, and a
1.5 TFLOP tail of linear algebra that runs while the GPU is idle anyway.

**Results are appended as JSON lines, one per arm per category, as they are produced.** A
sweep runs for hours on a laptop that may be closed; a run that only writes at the end is a
run that has to start again. `--resume` skips the (backbone, resolution, category) groups
already present in the file, which is the unit the loop can restart on cleanly.
"""

from __future__ import annotations

import json
import sys
import time
import zlib
from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, TextIO

import numpy as np

from anomaly_lab.eval.metrics import average_precision, roc_auc
from anomaly_lab.eval.pixel import PixelAccumulator, connected_regions
from anomaly_lab.models.dino_backbone import BACKBONES, DinoBackbone
from anomaly_lab.models.preprocessing import load_mask
from anomaly_lab.regions.transform import SpatialTransform
from research.subspace_ad.benchmarks import CategorySplit, ScoredImage, load_splits
from research.subspace_ad.features import (
    FeatureView,
    PatchEncoder,
    RotationFill,
    prepare,
    rotations,
    valid_patches,
)
from research.subspace_ad.maps import pixel_map
from research.subspace_ad.subspace import (
    CovarianceAccumulator,
    SubspaceFit,
    fit_subspace,
    residual_basis,
    tail_value_at_risk,
)

GAUSSIAN_SIGMA = 4.0
"""The paper's localization blur, applied in the prepared frame."""


@dataclass(frozen=True)
class CampaignSpec:
    """Every axis of one sweep, and the protocol constants it holds fixed."""

    benchmarks: tuple[str, ...] = ("visa",)
    backbones: tuple[DinoBackbone, ...] = (DinoBackbone.DINOV2_VIT_B14,)
    resolutions: tuple[int, ...] = (672,)
    views: tuple[FeatureView, ...] = ()
    shots: tuple[int, ...] = (1, 2, 4)
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    taus: tuple[float, ...] = (0.95, 0.97, 0.99, 1.0)
    rhos: tuple[float, ...] = (0.01,)
    augmentations: int = 30
    rotation_fill: RotationFill = RotationFill.ZEROS
    final_norm: bool = True
    sigma: float = GAUSSIAN_SIGMA
    batch_size: int = 4
    categories: tuple[str, ...] = ()
    pixel_metrics: bool = True
    pixel_taus: tuple[float, ...] = ()
    """Which thresholds get pixel-level metrics. Empty means every one in `taus`. The axis
    exists because a pixel metric costs about as much as the forward pass that produced the
    map, while the image metrics it shares a map with are free."""

    def pixel_thresholds(self) -> tuple[float, ...]:
        return self.pixel_taus or self.taus


@dataclass(frozen=True)
class ArmRow:
    """One configuration's verdict on one category: the unit the report aggregates."""

    backbone: str
    resolution: int
    view: str
    benchmark: str
    category: str
    shots: int
    seed: int
    tau: float
    rho: float
    rank: int
    dimension: int
    fit_patches: int
    test_images: int
    anomalies: int
    image_auroc: float | None
    image_ap: float | None
    pixel_auroc: float | None
    au_pro: float | None
    blocks: list[int]
    rotation_fill: str
    final_norm: bool
    augmentations: int


@dataclass(frozen=True)
class _Fitted:
    """A fitted subspace with the ranks every threshold resolves to on it."""

    fit: SubspaceFit
    ranks: dict[float, int]

    @property
    def max_rank(self) -> int:
        return max(self.ranks.values())


@dataclass
class _ArmScores:
    """What one (view, shots, seed, tau) arm collects as the test set streams past."""

    images: dict[float, list[float]] = field(default_factory=dict)
    maps: list[np.ndarray] = field(default_factory=list)


def _stream_rng(seed: int, key: str, purpose: int) -> np.random.Generator:
    """A generator that depends on the seed, the category and what it is for.

    Three separate requirements, and a single `default_rng(seed)` meets none of them. The
    category has to enter or every category draws the same indices into differently ordered
    pools; `crc32` rather than `hash` because Python salts string hashing per process, so a
    resumed run would draw a different fit set from the same seed. And the purpose keeps the
    rotation angles on their own stream, so raising `augmentations` does not silently change
    which images a 4-shot arm was fitted to.
    """
    material = zlib.crc32(key.encode("utf-8"))
    return np.random.default_rng([seed, material, purpose])


def _batched(items: Sequence[Any], size: int) -> Iterator[Sequence[Any]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def _fit_arms(
    encoder: PatchEncoder,
    split: CategorySplit,
    spec: CampaignSpec,
    log: TextIO,
) -> dict[tuple[str, int, int], _Fitted]:
    """Fit one subspace per (view, shots, seed), sharing every forward pass they overlap on.

    The augmented copies of one normal image are encoded once and folded into the
    accumulators of every k that image belongs to, so the pass costs `max(shots)` images
    rather than `sum(shots)`.
    """
    shots = tuple(sorted(spec.shots))
    if len(split.fit_pool) < shots[-1]:
        msg = (
            f"{split.key} has {len(split.fit_pool)} normals available to fit from, which "
            f"cannot supply a {shots[-1]}-shot arm"
        )
        raise ValueError(msg)

    dimensions = encoder.dimensions()
    fitted: dict[tuple[str, int, int], _Fitted] = {}
    for seed in spec.seeds:
        draw_rng = _stream_rng(seed, split.key, 0)
        angle_rng = _stream_rng(seed, split.key, 1)
        chosen = draw_rng.permutation(len(split.fit_pool))[: shots[-1]]
        accumulators = {
            view.name: CovarianceAccumulator(dimensions[view.name]) for view in spec.views
        }
        for position, index in enumerate(chosen, start=1):
            path = split.fit_pool[int(index)]
            frame = prepare(path, encoder.size).array
            count = spec.augmentations if split.rotation_safe else 0
            copies = rotations(frame, count=count, rng=angle_rng, fill=spec.rotation_fill)
            for chunk in _batched(copies, spec.batch_size):
                features = encoder.encode([array for array, _ in chunk])
                keeps = [
                    valid_patches(valid, encoder.grid, encoder.spec.patch_size)
                    for _, valid in chunk
                ]
                for name, block in features.items():
                    for row, keep in enumerate(keeps):
                        patches = block[row] if keep is None else block[row][keep]
                        accumulators[name].add(patches)
            if position in shots:
                for name, accumulator in accumulators.items():
                    fitted[name, position, seed] = _resolve_ranks(accumulator, spec.taus)
                print(
                    f"      fit {split.key} seed={seed} k={position} "
                    f"patches={accumulators[spec.views[0].name].count}",
                    file=log,
                    flush=True,
                )
    return fitted


def _resolve_ranks(accumulator: CovarianceAccumulator, taus: Sequence[float]) -> _Fitted:
    """Eigendecompose once, then trim the basis to the largest rank any threshold needs.

    The trim matters at scale rather than in principle: a concatenated seven-layer view of a
    ViT-L is 7168-dimensional, and keeping every eigenvector of every arm's fit would hold
    two hundred megabytes per fit for components no threshold will ever reach.
    """
    fit = fit_subspace(accumulator)
    ranks = {tau: fit.rank_for(tau) for tau in taus}
    trimmed = replace(fit, components=fit.components[: max(ranks.values())])
    return _Fitted(fit=trimmed, ranks=ranks)


def _ground_truth(item: ScoredImage, size: int, transform: SpatialTransform) -> np.ndarray:
    """The mask in the prepared frame, projected through the image's own transform.

    A normal test image has no mask file and gets an all-false frame rather than being
    dropped. Excluding it would remove every true negative it contributes and quietly
    inflate the pixel false-positive rate's denominator out of the metric.
    """
    if item.mask_path is None:
        return np.zeros((size, size), dtype=bool)
    mask = load_mask(item.mask_path, size=(transform.source_width, transform.source_height))
    prepared: np.ndarray = transform.prepare_mask(mask)
    return prepared


def _score_arms(
    encoder: PatchEncoder,
    split: CategorySplit,
    spec: CampaignSpec,
    fitted: dict[tuple[str, int, int], _Fitted],
    log: TextIO,
) -> tuple[dict[tuple[str, int, int, float], _ArmScores], list[np.ndarray], list[int]]:
    """One pass over the test set that scores every arm, and prepares the truth once.

    The masks come back with it because they are the other thing that must not be recomputed
    per arm: projecting one through its transform and finding its connected components costs
    more than the whole of an arm's accumulation, and neither depends on the configuration
    being scored.
    """
    wanted = set(spec.pixel_thresholds())
    collected: dict[tuple[str, int, int, float], _ArmScores] = {}
    masks: list[np.ndarray] = []
    labels: list[int] = []
    started = time.monotonic()

    for chunk in _batched(split.test_items, spec.batch_size):
        prepared = [prepare(item.path, encoder.size) for item in chunk]
        features = encoder.encode([entry.array for entry in prepared])
        if spec.pixel_metrics and split.mask_count:
            masks.extend(
                _ground_truth(item, encoder.size, entry.transform)
                for item, entry in zip(chunk, prepared, strict=True)
            )
        labels.extend(item.label for item in chunk)

        for (name, shots, seed), entry in fitted.items():
            block = features[name]
            for row in range(block.shape[0]):
                basis = residual_basis(entry.fit, block[row], rank=entry.max_rank)
                for tau, rank in entry.ranks.items():
                    scores = basis.at_rank(rank)
                    arm = collected.setdefault((name, shots, seed, tau), _ArmScores())
                    for rho, value in tail_value_at_risk(scores, spec.rhos).items():
                        arm.images.setdefault(rho, []).append(value)
                    if spec.pixel_metrics and split.mask_count and tau in wanted:
                        arm.maps.append(scores.reshape(encoder.grid).astype(np.float32))

    elapsed = time.monotonic() - started
    per_image = elapsed / max(1, len(split.test_items)) * 1000
    print(
        f"      scored {split.key}: {len(split.test_items)} images, "
        f"{len(collected)} arms, {elapsed:.0f}s ({per_image:.0f} ms/image)",
        file=log,
        flush=True,
    )
    return collected, masks, labels


def _pixel_summary(
    maps: Sequence[np.ndarray],
    masks: Sequence[np.ndarray],
    regions: Sequence[Sequence[np.ndarray]],
    *,
    size: int,
    sigma: float,
) -> dict[str, float | int | None]:
    """Pixel metrics for one arm, over maps already computed on the token grid.

    The accumulator's range is taken from the *token* scores rather than from the smoothed
    frame. Upsampling cannot exceed the grid's extremes and blurring can only pull them in,
    so the bound is guaranteed to contain every value -- and taking it here avoids a second
    pass over a hundred megapixel frames just to find out where they lie.
    """
    lo = min(float(entry.min()) for entry in maps)
    hi = max(float(entry.max()) for entry in maps)
    accumulator = PixelAccumulator(vmin=lo, vmax=hi)
    for grid, mask, components in zip(maps, masks, regions, strict=True):
        accumulator.add(pixel_map(grid, (size, size), sigma=sigma), mask, regions=components)
    return accumulator.summary()


def run_category(
    encoder: PatchEncoder,
    split: CategorySplit,
    spec: CampaignSpec,
    log: TextIO,
) -> list[ArmRow]:
    """Every arm's row for one category, from one pass over its images."""
    fitted = _fit_arms(encoder, split, spec, log)
    collected, masks, labels = _score_arms(encoder, split, spec, fitted, log)
    truth = np.asarray(labels, dtype=np.int64)
    regions = [connected_regions(mask) for mask in masks]
    dimensions = encoder.dimensions()
    blocks = {view.name: list(view.band.blocks(encoder.depth)) for view in spec.views}

    rows: list[ArmRow] = []
    for (name, shots, seed, tau), arm in sorted(collected.items()):
        pixel: dict[str, float | int | None] = {}
        if arm.maps:
            pixel = _pixel_summary(arm.maps, masks, regions, size=encoder.size, sigma=spec.sigma)
        for rho, values in arm.images.items():
            scores = np.asarray(values, dtype=np.float64)
            rows.append(
                ArmRow(
                    backbone=encoder.backbone.value,
                    resolution=encoder.size,
                    view=name,
                    benchmark=split.benchmark,
                    category=split.category,
                    shots=shots,
                    seed=seed,
                    tau=tau,
                    rho=rho,
                    rank=fitted[name, shots, seed].ranks[tau],
                    dimension=dimensions[name],
                    fit_patches=fitted[name, shots, seed].fit.sample_count,
                    test_images=len(split.test_items),
                    anomalies=split.anomaly_count,
                    image_auroc=roc_auc(truth, scores),
                    image_ap=average_precision(truth, scores),
                    pixel_auroc=_as_float(pixel.get("pixel_roc_auc")),
                    au_pro=_as_float(pixel.get("au_pro")),
                    blocks=blocks[name],
                    rotation_fill=spec.rotation_fill.value,
                    final_norm=spec.final_norm,
                    augmentations=spec.augmentations if split.rotation_safe else 0,
                )
            )
    return rows


def _as_float(value: float | int | None) -> float | None:
    return None if value is None else float(value)


def _completed_groups(path: Path) -> set[tuple[str, int, str, str]]:
    """Which (backbone, resolution, benchmark, category) groups the file already holds."""
    if not path.is_file():
        return set()
    done: set[tuple[str, int, str, str]] = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            done.add((row["backbone"], row["resolution"], row["benchmark"], row["category"]))
    return done


def run_campaign(
    spec: CampaignSpec,
    *,
    datasets_dir: Path,
    cache_dir: Path,
    output: Path,
    resume: bool = False,
    log: TextIO = sys.stderr,
) -> int:
    """Run every arm and append its rows. Returns how many rows were written."""
    if not spec.views:
        msg = "a campaign needs at least one feature view"
        raise ValueError(msg)

    splits = [
        split
        for benchmark in spec.benchmarks
        for split in load_splits(datasets_dir, benchmark)
        if not spec.categories or split.category in spec.categories
    ]
    if not splits:
        msg = "no category matched the requested benchmarks and category filter"
        raise ValueError(msg)

    done = _completed_groups(output) if resume else set()
    output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    for backbone in spec.backbones:
        for size in spec.resolutions:
            pending = [
                split
                for split in splits
                if (backbone.value, size, split.benchmark, split.category) not in done
            ]
            if not pending:
                print(f"  {backbone.value} @ {size}px: already complete", file=log, flush=True)
                continue
            spent = time.monotonic()
            encoder = PatchEncoder(
                backbone,
                size=size,
                views=spec.views,
                cache_dir=cache_dir,
                final_norm=spec.final_norm,
            )
            print(
                f"  {backbone.value} @ {size}px on {encoder.device} "
                f"({encoder.device_reason}); grid {encoder.grid}, "
                f"blocks {[index + 1 for index in encoder.indices]} of {encoder.depth}",
                file=log,
                flush=True,
            )
            for split in pending:
                rows = run_category(encoder, split, spec, log)
                with output.open("a", encoding="utf-8") as handle:
                    for row in rows:
                        handle.write(json.dumps(asdict(row)) + "\n")
                written += len(rows)
            print(
                f"  {backbone.value} @ {size}px done in {(time.monotonic() - spent) / 60:.1f} min",
                file=log,
                flush=True,
            )
            del encoder
    return written


def backbone_summary() -> list[dict[str, Any]]:
    """What the menu offers, for a report that has to say what was actually swept."""
    return [
        {
            "key": entry.value,
            "timm_name": BACKBONES[entry].timm_name,
            "patch": BACKBONES[entry].patch_size,
            "dimension": BACKBONES[entry].embedding_dim,
            "depth": BACKBONES[entry].depth,
            "gated": BACKBONES[entry].gated,
        }
        for entry in DinoBackbone
    ]
