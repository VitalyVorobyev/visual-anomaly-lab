"""`dino_memory` — a frozen DINO backbone whose patch features *are* the model.

The sixth method, the second whose cost is a memory footprint rather than a step budget,
and the first in-house one built on the shared encoder table in `dino_backbone.py`. Nothing
is trained: a frozen DINOv2/DINOv3 encoder produces L2-normalized patch features for the
training normals, those features are held as a memory, and a test patch is scored by how far
it is from that memory. `patchcore_anomalib` does the same thing through anomalib with a
convolutional backbone; this does it with a self-supervised transformer, and — the part that
is genuinely new here — with a **choice of what the memory is**.

**Three scoring rules on one axis.** `scoring` is a single enum with three values rather
than two independent options, and that is the design rather than a simplification:

  * `global_knn` — one coreset bank over every position of every image. Position-blind, so a
    pattern that is normal *somewhere* is normal *everywhere*. PatchCore's rule, over
    transformer features.
  * `local_knn` — one small bank per patch position, searched over a `window_radius`
    neighbourhood. Registration-aware: a pattern that belongs at the top of the frame is an
    anomaly at the bottom, which a global bank explains away by construction.
  * `local_gaussian` — one shrunk Gaussian per patch position, scored by Mahalanobis
    distance. PaDiM's rule, over the same features; the memory is a distribution rather than
    a set of examples.

A layout x distance product (`global` x {knn, gaussian}, `local` x {knn, gaussian}) would
have an invalid cell: a *global* Gaussian over every patch of every image is one distribution
fitted to the union of everything the encoder ever sees, which is not a model of normality,
it is a model of the dataset's marginal. One axis with three named values says what can
actually be run, and a picker generated from the schema shows exactly those three.

**Bounded before it runs.** `plan_memory` is pure, torch-free and module-level, and its
`describe()` goes into the job log *before the first forward pass* — no probe is needed,
because the grid comes from `patch_grid`'s arithmetic and the feature width from the
backbone table times the number of layers read. The first real batch then verifies both and
refuses a disagreement, so the announced footprint is a measurement rather than a hope. This
is `patchcore_anomalib`'s discipline with the tension removed.

**Device placement is measured, not assumed** (`scripts/dino-memory-smoke-test.py`, ADR-0008):

  * the encoder forward wants the accelerator — about 2x on MPS — so it runs on `ctx.device`;
  * `topk` over a 100 000-wide row is ~7x *slower* on MPS and breaks exact ties differently,
    so neighbour selection is on the CPU and a neighbour index is never a cross-device
    identity;
  * batched Cholesky is 16x slower on MPS where it exists at all, and a near-singular
    covariance is exactly where float32 stops being enough, so the covariance fit is CPU
    float64;
  * the distance/einsum kernels and the map post-processing run on whichever device their
    tensors are already on.

Since every bank is CPU-resident, everything after the forward pass is on the CPU by
construction. That is the placement the verdicts ask for and it needs no branch.

**Channel count is data, never schema.** `channel_fusion=feature_concat` concatenates a
sample's per-channel feature vectors in a stable channel order and L2-normalizes the result,
which makes the score scale independent of how many channels a dataset has — the numeric
form of the rule in CLAUDE.md. `per_image` is the pooled default and scores every image on
its own. Both declare `channel_aware`, because the capability says the model *may* consume
channel metadata; the config field decides whether it does.

Heavy imports stay inside functions. This module is imported wherever the method picker is
opened, and a plan must not cost three seconds of torch.
"""

from __future__ import annotations

import time
from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass
from enum import StrEnum
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from anomaly_lab.models.base import (
    AnomalyModel,
    Availability,
    Capabilities,
    Device,
    ImageRecord,
    InferContext,
    Prediction,
    TrainContext,
    evenly_spaced,
    module_available,
)
from anomaly_lab.models.coreset import (
    gaussian_projection,
    greedy_select,
    johnson_lindenstrauss_dim,
)
from anomaly_lab.models.diagnostics import DiagnosticKind
from anomaly_lab.models.dino_backbone import (
    BACKBONES,
    DinoBackbone,
    FeatureLayers,
    backbone_fingerprint,
    extract_patch_features,
    load_backbone,
    patch_grid,
    validate_prepared_size,
)
from anomaly_lab.models.preprocessing import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    PreprocessingConfig,
    expand_planes,
    load_array,
    to_chw,
)
from anomaly_lab.schemas import API_MODEL_CONFIG

STATE_FILENAME = "dino_memory.pt"

CHECKPOINT_FORMAT = 1
"""There is no format 0. This method owns its checkpoint from the first commit.

It carries one of three bank payloads, the geometry, the channel order and a fingerprint of
the frozen encoder — but **not** the encoder's weights, which are somebody else's artifact
and are rebuilt by `load_backbone` from the app-managed cache.
"""

BYTES_PER_FLOAT32 = 4

UNASSIGNED = ""
"""Channel key for images belonging to no channel — the spelling the import commit already
uses (`datasets/commit.py`) and the one `pixel_reference` reads. A channel name is never
empty, so this can never collide with a real one."""

_PROJECTION_EPS = 0.9
"""anomalib's `SparseRandomProjection` operating point, named here rather than inline
because the projected width it implies is a factor in every greedy iteration's cost."""

_SELECTION_LOG_EVERY = 250
"""How often the coreset selection reports. Frequent enough that the bar moves, rare enough
that the log stays readable."""

_DISTANCE_CHUNK = 2048
"""Bank rows per distance block. A full (1024, 100000) float32 matrix is 409 MB in one
allocation, which is what chunking exists to avoid."""

_POSITION_LOG_EVERY = 64
"""How often the per-position covariance fit reports."""


class Scoring(StrEnum):
    """What the memory *is*, and therefore how a patch is scored.

    One axis with three values rather than a layout x distance product — the module
    docstring says why the fourth cell is not a method. A closed set for the reason
    `patchcore_anomalib.LayerSet` is one: the form is generated from this schema and renders
    an enum as a picker.
    """

    GLOBAL_KNN = "global_knn"
    LOCAL_KNN = "local_knn"
    LOCAL_GAUSSIAN = "local_gaussian"


class Shrinkage(StrEnum):
    """How a per-position covariance is made invertible.

    Both entries produce `(1 - lambda) S + lambda (tr S / d) I`; they differ in where lambda
    comes from. Ledoit-Wolf derives it from the samples, ridge takes it from the form.
    """

    LEDOIT_WOLF = "ledoit_wolf"
    RIDGE = "ridge"


class ChannelFusion(StrEnum):
    """How a multi-channel sample reaches the memory."""

    PER_IMAGE = "per_image"
    FEATURE_CONCAT = "feature_concat"


# ------------------------------------------------------------------------ the plan


@dataclass(frozen=True)
class MemoryPlan:
    """What a fit is about to cost, computed before any of it is spent.

    Deliberately **torch-free and pure** — plain integers in, a plan out — so the arithmetic
    that decides whether this machine survives the run is checked by the CI job that
    installs without the `dl` extra.

    The **unit of banking** is what most of these numbers are per, and it depends on
    `channel_fusion`. Under `per_image` a unit is one image. Under `feature_concat` a unit is
    one **sample**: its channel images are read, their per-channel features concatenated in
    channel order and normalized, and the result is *one* vector per position for the whole
    sample. `max_bank_images` therefore bounds units, and `images_used` reports the channel
    images that implies — `units_used x channels_fused` — so the number in the plan is the
    number of files the forward pass will actually open.
    """

    scoring: Scoring
    channel_fusion: ChannelFusion
    images_available: int
    images_used: int
    samples_available: int
    grid_rows: int
    grid_cols: int
    positions: int
    embedding_dim: int
    """Width of **one channel's** feature vector: the backbone's width times the number of
    layers read, because `extract_patch_features` concatenates the layers."""
    channels_fused: int
    fused_dim: int
    patches_per_image: int
    patches_kept_per_image: int
    candidates_kept: int
    coreset_size: int
    projection_dim: int
    per_position_images: int
    window_radius: int
    window_candidates: int
    reduced_dim: int
    samples_per_position: int
    sample_deficit: int
    """How many samples short of a full-rank covariance each position is, as a **field**
    rather than a sentence in a log line: `max(0, reduced_dim - samples_per_position + 1)`.
    A positive value is not a failure — it is the number that makes shrinkage load-bearing
    rather than cosmetic — but it has to be assertable, because a run that quietly inverts a
    rank-deficient covariance produces confident nonsense."""

    # ------------------------------------------------------------------ derived

    @property
    def units_available(self) -> int:
        """Banking units the training set offers — samples when fusing, images otherwise."""
        if self.channel_fusion is ChannelFusion.FEATURE_CONCAT:
            return self.samples_available
        return self.images_available

    @property
    def units_used(self) -> int:
        return self.images_used // max(self.channels_fused, 1)

    @property
    def candidate_bytes(self) -> int:
        """Float32 footprint of the vectors this fit reads into the memory.

        For `global_knn` and `local_knn` that is a real peak: the store and the bank are held
        at once. For `local_gaussian` the vectors are streamed into accumulators and this is
        the volume read rather than the volume held — `precision_bytes` is that mode's cost.
        """
        return self.candidates_kept * self.fused_dim * BYTES_PER_FLOAT32

    @property
    def bank_bytes(self) -> int:
        if self.scoring is Scoring.GLOBAL_KNN:
            return self.coreset_size * self.fused_dim * BYTES_PER_FLOAT32
        if self.scoring is Scoring.LOCAL_KNN:
            return self.positions * self.per_position_images * self.fused_dim * BYTES_PER_FLOAT32
        return self.positions * self.reduced_dim * BYTES_PER_FLOAT32

    @property
    def precision_bytes(self) -> int:
        """Stored float32 footprint of the per-position inverse covariances.

        Zero outside `local_gaussian`. It is quadratic in `reduced_dim`, which is the entire
        reason `mahalanobis_dims` exists: at a 768-wide fused vector and a 32x32 grid this is
        2.42 GB stored and twice that while the fit holds it in float64.
        """
        return self.positions * self.reduced_dim * self.reduced_dim * BYTES_PER_FLOAT32

    @property
    def unbounded_candidates(self) -> int:
        """What a run with no caps at all would have held — the number the caps avoid."""
        return self.units_available * self.patches_per_image

    @property
    def unbounded_bytes(self) -> int:
        return self.unbounded_candidates * self.fused_dim * BYTES_PER_FLOAT32

    @property
    def images_dropped(self) -> int:
        return self.images_available - self.images_used

    @property
    def patches_dropped_per_image(self) -> int:
        return self.patches_per_image - self.patches_kept_per_image

    def describe(self) -> str:
        """One line for the job log, said before the work rather than after it.

        It names the footprint **and** what was dropped, in both directions: what the caps
        kept and what an uncapped run would have held. A silent truncation reads as "this is
        all there was", and a footprint discovered afterwards is a machine that has already
        started swapping.
        """
        unit = "samples" if self.channel_fusion is ChannelFusion.FEATURE_CONCAT else "images"
        singular = unit[:-1]
        head = (
            f"dino_memory plan ({self.scoring.value}): {self.units_used}/{self.units_available} "
            f"{unit} on a {self.grid_rows}x{self.grid_cols} grid ({self.positions} positions), "
            f"{self.channels_fused}x{self.embedding_dim} = {self.fused_dim}-wide fused vectors"
        )

        if self.scoring is Scoring.GLOBAL_KNN:
            body = (
                f"; {self.patches_kept_per_image}/{self.patches_per_image} patches per "
                f"{singular} = {self.candidates_kept:,} candidates "
                f"({self.candidate_bytes / 1e6:.1f} MB), projected to {self.projection_dim} "
                f"dims, coreset {self.coreset_size:,} vectors ({self.bank_bytes / 1e6:.1f} MB)"
            )
        elif self.scoring is Scoring.LOCAL_KNN:
            body = (
                f"; K={self.per_position_images} per position = {self.candidates_kept:,} "
                f"vectors ({self.bank_bytes / 1e6:.1f} MB), window radius "
                f"{self.window_radius} gives {self.window_candidates} candidates per position"
            )
        else:
            body = (
                f"; {self.samples_per_position} samples per position, {self.fused_dim} -> "
                f"{self.reduced_dim} dims; means {self.bank_bytes / 1e6:.1f} MB, precision "
                f"{self.precision_bytes / 1e9:.2f} GB stored float32 and fitted in CPU float64"
            )
            if self.sample_deficit:
                body += (
                    f"; sample deficit {self.sample_deficit}, so the raw covariance is "
                    "singular and shrinkage is what makes it invertible"
                )

        parts = []
        if self.images_dropped:
            parts.append(f"{self.images_dropped} of {self.images_available} images dropped")
        if self.patches_dropped_per_image:
            parts.append(
                f"{self.patches_dropped_per_image} of {self.patches_per_image} patches per "
                f"{singular} dropped"
            )
        dropped = "; " + ", ".join(parts) if parts else "; nothing dropped"

        tail = (
            f"; uncapped this would have been {self.unbounded_candidates:,} vectors "
            f"({self.unbounded_bytes / 1e9:.2f} GB)"
        )
        return head + body + dropped + tail


def plan_memory(
    config: DinoMemoryConfig,
    *,
    samples_available: int,
    images_available: int,
    grid: tuple[int, int],
    embedding_dim: int,
    channels: tuple[str, ...],
) -> MemoryPlan:
    """Resolve every cap into one plan, without touching a file or a tensor.

    The caps compose in a fixed order, and the order keeps coverage honest: **units first,
    then patches within each surviving unit**. Dropping units to fit the vector cap would be
    the other way round and is worse, because patches inside one image are highly
    redundant — neighbouring transformer tokens attend to overlapping context — while two
    images differ by whatever the process actually varies. So a memory sees as many *images*
    as it is allowed, at whatever patch density that leaves.

    Under `feature_concat` a unit is a **sample**, not an image (see `MemoryPlan`). One
    sample contributes one fused vector per position however many channels it has, so
    `max_bank_images` bounds samples and `images_used` reports `units_used x channels_fused`
    — the number of files the forward pass opens. `channels` is the dataset's channel set;
    under `per_image` it is recorded but `channels_fused` stays 1, because that mode scores
    each channel image on its own and a fused vector is one channel wide.
    """
    rows, cols = grid
    if rows < 1 or cols < 1:
        msg = f"plan_memory needs a real patch grid; got {rows}x{cols}"
        raise ValueError(msg)
    if embedding_dim < 1:
        msg = f"plan_memory needs a real embedding width; got {embedding_dim}"
        raise ValueError(msg)
    if not channels:
        msg = "plan_memory needs at least one channel name; got an empty channel set"
        raise ValueError(msg)
    if images_available < 1 or samples_available < 1:
        msg = (
            "plan_memory needs at least one training image and one sample; got "
            f"images_available={images_available}, samples_available={samples_available}"
        )
        raise ValueError(msg)
    if not 0.0 < config.coreset_ratio <= 1.0:
        msg = f"coreset_ratio must be in (0, 1]; got {config.coreset_ratio}"
        raise ValueError(msg)

    fusing = config.channel_fusion is ChannelFusion.FEATURE_CONCAT
    channels_fused = len(channels) if fusing else 1
    fused_dim = embedding_dim * channels_fused
    positions = rows * cols
    units_available = samples_available if fusing else images_available
    units_used = max(1, min(units_available, config.max_bank_images))

    patches_kept = positions
    candidates_kept = 0
    coreset_size = 0
    projection_dim = 0
    per_position = 0
    window_radius = 0
    window_candidates = 0
    reduced_dim = 0
    samples_per_position = 0

    if config.scoring is Scoring.GLOBAL_KNN:
        # A vector cap below the unit count cannot be met by thinning patches — one patch per
        # unit is the floor — so it takes units off too. Degenerate, and reachable from the
        # form, and silently overshooting a stated cap is the one thing it must not do.
        units_used = min(units_used, max(1, config.max_candidate_vectors))
        if units_used * positions > config.max_candidate_vectors:
            patches_kept = max(1, config.max_candidate_vectors // units_used)
        candidates_kept = units_used * patches_kept
        # `int()` truncates, so a ratio that would select nothing selects one: an empty bank
        # is not a smaller bank, it is a model that cannot score.
        coreset_size = max(1, min(candidates_kept, int(candidates_kept * config.coreset_ratio)))
        # Clamped to the data's own width: a projection wider than the vectors it embeds is
        # cost with no distance benefit, and the JL bound depends only on the point count.
        projection_dim = min(
            fused_dim, johnson_lindenstrauss_dim(candidates_kept, eps=_PROJECTION_EPS)
        )
    else:
        per_position = min(units_used, config.per_position_images)
        candidates_kept = positions * per_position
        samples_per_position = per_position
        if config.scoring is Scoring.LOCAL_KNN:
            window_radius = config.window_radius
            window_candidates = (2 * window_radius + 1) ** 2 * per_position
        else:
            reduced_dim = (
                fused_dim
                if config.mahalanobis_dims == 0
                else min(config.mahalanobis_dims, fused_dim)
            )

    return MemoryPlan(
        scoring=config.scoring,
        channel_fusion=config.channel_fusion,
        images_available=images_available,
        images_used=units_used * channels_fused,
        samples_available=samples_available,
        grid_rows=rows,
        grid_cols=cols,
        positions=positions,
        embedding_dim=embedding_dim,
        channels_fused=channels_fused,
        fused_dim=fused_dim,
        patches_per_image=positions,
        patches_kept_per_image=patches_kept,
        candidates_kept=candidates_kept,
        coreset_size=coreset_size,
        projection_dim=projection_dim,
        per_position_images=per_position,
        window_radius=window_radius,
        window_candidates=window_candidates,
        reduced_dim=reduced_dim,
        samples_per_position=samples_per_position,
        sample_deficit=max(0, reduced_dim - samples_per_position + 1),
    )


# --------------------------------------------------------------- covariance conditioning


def condition_covariance(
    samples: np.ndarray,
    *,
    shrinkage: Shrinkage,
    ridge_epsilon: float,
) -> tuple[np.ndarray, float]:
    """One position's centred samples to an invertible covariance, and the lambda used.

    `samples` is `(n, d)` float64 and already centred. Both rules produce the same form —
    `(1 - lambda) S + lambda (tr S / d) I` — and differ only in where lambda comes from.

    **Ledoit-Wolf (2004)** derives it from the data: the optimal shrinkage toward a scaled
    identity under Frobenius loss, `lambda = b^2 / d^2` with `d^2` the distance from `S` to
    `mI` and `b^2` the sample-averaged variance of `S` itself, clipped to `[0, 1]` as the
    paper's estimator requires. With `n < d` — the ordinary case here, since a position sees
    `per_position_images` samples of `reduced_dim` — `S` is singular by construction and this
    is what makes the inverse exist at all.

    **Ridge is scale-aware on purpose**, and that is not a stylistic preference. An absolute
    `S + eps I` with `eps = 0.01` would swamp `S` entirely: these features are unit vectors,
    so a per-position covariance has a trace around `1/d` and entries several orders of
    magnitude below 0.01. The Mahalanobis distance would quietly become a Euclidean one that
    still runs, still produces maps, and no longer measures the shape of the distribution.
    Expressing the ridge *relative to* `tr(S)/d` keeps it a fraction of the data's own scale,
    so scaling the features scales the conditioned matrix by the same factor.
    """
    count, dim = samples.shape
    if count < 1 or dim < 1:
        msg = f"a covariance needs at least one sample and one dimension; got {samples.shape}"
        raise ValueError(msg)

    covariance = samples.T @ samples / float(count)
    mean_variance = float(np.trace(covariance)) / dim
    target = np.eye(dim, dtype=np.float64) * mean_variance

    if shrinkage is Shrinkage.RIDGE:
        lam = float(ridge_epsilon)
    else:
        dispersion = float(np.sum((covariance - target) ** 2)) / dim
        if dispersion <= 0.0:
            # `S` already *is* the scaled identity; every shrinkage toward it is a no-op, and
            # taking the target outright is the only well-defined answer.
            lam = 1.0
        else:
            outer_error = 0.0
            for row in samples:
                deviation = np.outer(row, row) - covariance
                outer_error += float(np.sum(deviation**2))
            noise = outer_error / (float(count) ** 2 * dim)
            lam = float(np.clip(min(noise, dispersion) / dispersion, 0.0, 1.0))

    conditioned = (1.0 - lam) * covariance + lam * target
    return conditioned, lam


def precision_from_covariance(covariance: np.ndarray) -> np.ndarray:
    """Invert one conditioned covariance through its Cholesky factor, in float64.

    `numpy.linalg` has no triangular solve, so the factor is inverted with the general solver
    against an identity and the precision is rebuilt as `L^-T L^-1`. That is still the
    Cholesky route rather than `inv`, which matters twice: it is numerically the respectable
    way to invert a symmetric positive-definite matrix, and it **raises** on a matrix that is
    not positive definite instead of returning a plausible inverse of a singular one.
    """
    dim = covariance.shape[0]
    factor = np.linalg.cholesky(covariance)
    inverse_factor = np.linalg.solve(factor, np.eye(dim, dtype=np.float64))
    return np.asarray(inverse_factor.T @ inverse_factor, dtype=np.float64)


# ---------------------------------------------------------------------------- config


class DinoMemoryConfig(BaseModel):
    """Hyperparameters. Every field here becomes a control on the experiment form."""

    model_config = API_MODEL_CONFIG

    backbone: DinoBackbone = Field(
        default=DinoBackbone.DINOV2_VIT_S14_REG4,
        description=(
            "Frozen encoder the patch features come from. The default is deliberately an "
            "ungated Apache-2.0 DINOv2 — the same registered encoder anomalib pins for "
            "Dinomaly, so a run against it compares methods rather than encoders. The two "
            "DINOv3 entries are licence-gated: access must be requested from Meta on "
            "Hugging Face and an approved HF_TOKEN must already be in the environment, "
            "which is why they are never the default."
        ),
    )
    layers: FeatureLayers = Field(
        default=FeatureLayers.LAST_TWO,
        description=(
            "Which transformer blocks the patch features are read from; the chosen blocks "
            "are each L2-normalized and concatenated, so this multiplies the feature width "
            "and therefore every footprint in the plan. Late blocks are more semantic and "
            "less localized; a nearest-neighbour memory usually wants them."
        ),
    )
    pretrained_backbone: bool = Field(
        default=True,
        description=(
            "Use the published self-supervised weights. Off gives a seeded random ViT, "
            "which needs no download and is a real question rather than a broken run: patch "
            "nearest-neighbour matching only needs *some* feature space, and an untrained "
            "transformer is one. This is what the hermetic tests use."
        ),
    )
    allow_downloads: bool = Field(
        default=True,
        description=(
            "Permit fetching the encoder weights. Turn off to make a missing encoder an "
            "error naming the asset and this switch, instead of a silent reach for the "
            "network."
        ),
    )
    scoring: Scoring = Field(
        default=Scoring.GLOBAL_KNN,
        description=(
            "What the memory is. 'global_knn' holds one coreset bank over every position of "
            "every image and is position-blind, so a pattern that is normal somewhere is "
            "normal everywhere. 'local_knn' holds a small bank per patch position and "
            "searches a window around it, so a pattern in the wrong place is an anomaly — "
            "which only means something on registered data. 'local_gaussian' fits one "
            "shrunk Gaussian per position and scores Mahalanobis distance."
        ),
    )
    window_radius: int = Field(
        default=1,
        ge=0,
        le=8,
        description=(
            "local_knn only: how many patch positions either side of a query position its "
            "bank neighbours may come from. 0 pins each patch to its own position exactly; "
            "larger values tolerate more registration slack and cost (2r+1)^2 times the "
            "comparisons."
        ),
    )
    num_neighbors: int = Field(
        default=1,
        ge=1,
        le=64,
        description=(
            "Both kNN modes: how many nearest bank vectors a patch score averages over. 1 is "
            "the plain nearest-neighbour distance; larger values are steadier and blunter."
        ),
    )
    coreset_ratio: float = Field(
        default=0.1,
        gt=0.0,
        le=1.0,
        description=(
            "global_knn only: fraction of the candidate pool kept in the bank. Both the "
            "selection time and the per-image inference cost are linear in the result."
        ),
    )
    max_candidate_vectors: int = Field(
        default=100_000,
        ge=16,
        le=2_000_000,
        description=(
            "global_knn only: ceiling on the patch vectors held before coreset selection. "
            "This is the cap that bounds memory and time — the greedy selection runs "
            "coreset_ratio x N times and each iteration touches all N rows, so its total "
            "cost is quadratic in this number."
        ),
    )
    max_bank_images: int = Field(
        default=256,
        ge=2,
        le=10_000,
        description=(
            "How many training units are pushed through the encoder, sampled evenly across "
            "the training set rather than taken from its front. A unit is one image under "
            "per_image fusion and one whole sample under feature_concat."
        ),
    )
    per_position_images: int = Field(
        default=64,
        ge=2,
        le=1024,
        description=(
            "Local modes only: how many units contribute a vector to each patch position. "
            "For local_gaussian this is the sample count every per-position covariance is "
            "fitted from, and the plan reports how far short of full rank that leaves it."
        ),
    )
    mahalanobis_dims: int = Field(
        default=128,
        ge=0,
        le=4096,
        description=(
            "local_gaussian only: how many feature dimensions each per-position Gaussian is "
            "fitted over, chosen as one seeded subset shared by every position. 0 means "
            "every dimension, which is quadratic: a 768-wide fused vector on a 32x32 grid "
            "is 2.42 GB of stored inverse covariances and twice that during the float64 fit."
        ),
    )
    shrinkage: Shrinkage = Field(
        default=Shrinkage.LEDOIT_WOLF,
        description=(
            "How each per-position covariance is made invertible. 'ledoit_wolf' derives the "
            "shrinkage from the samples themselves and needs no tuning; 'ridge' uses a fixed "
            "fraction instead, which is reproducible across positions but ignores how "
            "singular any one of them actually is."
        ),
    )
    ridge_epsilon: float = Field(
        default=0.01,
        gt=0.0,
        le=1.0,
        description=(
            "Ridge shrinkage strength, as a fraction of the position's own mean variance "
            "rather than an absolute value. On unit-norm features an absolute 0.01 would "
            "swamp the covariance and silently turn Mahalanobis distance into Euclidean."
        ),
    )
    channel_fusion: ChannelFusion = Field(
        default=ChannelFusion.PER_IMAGE,
        description=(
            "How a multi-channel sample reaches the memory. 'per_image' pools every channel "
            "image into one memory and scores each image on its own. 'feature_concat' joins "
            "a sample's per-channel vectors in a stable channel order into one vector per "
            "position, so a part is scored once across all its views and every sample must "
            "present the same channel set."
        ),
    )
    blur_sigma: float = Field(
        default=4.0,
        ge=0.0,
        le=32.0,
        description=(
            "Gaussian smoothing applied to the upsampled patch scores, in pixels. The "
            "unblurred map is emitted as a diagnostic so the cost of this is visible, and "
            "the image score is taken before it so the two are independent."
        ),
    )
    score_percentile: float = Field(
        default=100.0,
        ge=50.0,
        le=100.0,
        description=(
            "Which percentile of the patch-grid scores becomes the image score. Taken on "
            "the grid rather than the upsampled map, so blur_sigma cannot move it. The "
            "maximum (100) is the most sensitive and the least stable."
        ),
    )
    feature_batch_size: int = Field(
        default=8,
        ge=1,
        le=64,
        description="Images per encoder forward during the fit pass. Inference is per unit.",
    )
    seed: int = Field(
        default=0,
        description=(
            "Seeds the coreset start, the random projection, the Mahalanobis dimension "
            "subset and — when pretrained_backbone is off — the encoder initialisation. Two "
            "runs with one seed are one experiment."
        ),
    )


# ------------------------------------------------------------------------ channels


def channel_order(records: Sequence[ImageRecord]) -> tuple[str, ...]:
    """The dataset's channel names, sorted — the order a fused vector is concatenated in.

    Pure, torch-free and **stable under the order the records arrive in**, which is the whole
    point: a fused vector's meaning is positional, so if two runs of one dataset built it in
    two different orders their banks would be silently incomparable and every distance
    between them meaningless. A set plus `sorted` is the cheapest thing that cannot depend on
    a database's row order, a directory listing, or which split subset was queried first.

    Images belonging to no channel key on `UNASSIGNED`, so a single-view dataset is a
    one-entry order rather than a special case.
    """
    return tuple(sorted({record.channel or UNASSIGNED for record in records}))


# ------------------------------------------------------------------ feature extraction


def _normalized_batch(
    records: Sequence[ImageRecord],
    preprocessing: PreprocessingConfig,
    device: str,
) -> Any:
    """Load a batch through the shared bridge, then standardize it for the encoder.

    Two separate acts. `load_array` is the experiment's decision and is identical for every
    method; the plane count and the ImageNet statistics belong to *this* backbone, which is
    why they live on this side of the seam (`preprocessing.py`).
    """
    import torch

    stacked = np.stack(
        [expand_planes(to_chw(load_array(record.path, preprocessing)), 3) for record in records]
    )
    batch = torch.from_numpy(stacked).to(device)
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
    return (batch - mean) / std


def _image_features(
    model: Any,
    records: Sequence[ImageRecord],
    preprocessing: PreprocessingConfig,
    indices: tuple[int, ...],
    device: str,
) -> Any:
    """`(N, P, D)` float32 patch features on the **CPU**, one row per patch.

    The encoder forward happens on `device` — measured at about 2x on MPS — and the result
    comes straight back, because every bank this method builds is CPU-resident and every
    kernel that consumes one was measured to belong there.

    The final normalization is over the channel dimension of the whole concatenated feature,
    *after* `extract_patch_features` has already normalized each layer on its own. Both are
    load-bearing and they are not the same act: the per-layer step makes each block an equal
    vote, and this one makes every stored vector a unit vector — which is what lets every
    distance kernel below drop the norm terms and read `d^2 = 2 - 2 cos`.
    """
    import torch

    with torch.no_grad():
        batch = _normalized_batch(records, preprocessing, device)
        features = extract_patch_features(model, batch, indices)
        features = torch.nn.functional.normalize(features, p=2.0, dim=1)
        width = int(features.shape[1])
        flat = features.permute(0, 2, 3, 1).reshape(len(records), -1, width)
        return flat.float().cpu().contiguous()


def _fuse(block: Any) -> Any:
    """`(C, P, D)` per-channel features to one `(P, C*D)` unit vector per position.

    **This is "channel count is data, never schema", numerically.** Concatenating and then
    normalizing means each channel contributes `1/sqrt(C)` of the vector, so the squared
    distance between two fused vectors is the *average* of the per-channel squared distances
    rather than their sum. A four-channel sample therefore scores on the same scale as a
    two-channel one, and a one-channel group comes out exactly equal to `per_image` — not
    approximately, but through this same call with `C = 1`, where the normalization of an
    already-unit vector is the identity.
    """
    import torch

    count, positions, dim = block.shape
    fused = block.permute(1, 0, 2).reshape(positions, count * dim)
    return torch.nn.functional.normalize(fused, p=2.0, dim=1)


# ------------------------------------------------------------------- map post-processing


def _gaussian_kernel(sigma: float, device: str, dtype: Any) -> Any:
    """A 1-D Gaussian truncated at three sigma and normalized — half of a separable blur."""
    import torch

    radius = max(1, round(3.0 * sigma))
    offsets = torch.arange(-radius, radius + 1, dtype=dtype, device=device)
    weights = torch.exp(-offsets.pow(2) / (2.0 * sigma * sigma))
    return weights / weights.sum()


def _separable_blur(image: Any, sigma: float) -> Any:
    """Two depthwise passes rather than one square kernel: `2k` work instead of `k^2`.

    In-module rather than through torchvision, for the reason `pixel_reference.gaussian_blur`
    is in numpy: this is the only filtering the method needs, and a second imaging dependency
    for one convolution pair is not a trade worth making.
    """
    import torch.nn.functional as functional

    kernel = _gaussian_kernel(sigma, str(image.device), image.dtype)
    radius = (kernel.numel() - 1) // 2
    channels = int(image.shape[1])
    horizontal = kernel.view(1, 1, 1, -1).expand(channels, 1, 1, -1)
    vertical = kernel.view(1, 1, -1, 1).expand(channels, 1, -1, 1)
    padded = functional.pad(image, (radius, radius, 0, 0), mode="replicate")
    blurred = functional.conv2d(padded, horizontal, groups=channels)
    padded = functional.pad(blurred, (0, 0, radius, radius), mode="replicate")
    return functional.conv2d(padded, vertical, groups=channels)


# --------------------------------------------------------------------------- the model


class DinoMemoryModel(AnomalyModel):
    """A frozen DINO encoder plus one of three memories over its patch features."""

    title = "DINO patch memory"
    summary = (
        "A frozen DINOv2/DINOv3 backbone whose patch features are held as a memory of "
        "normal images and scored globally, per position, or as a per-position Gaussian. "
        "Nothing is trained; the memory is the model."
    )

    def __init__(self, config: DinoMemoryConfig) -> None:
        super().__init__(config)
        self.config = config
        self._encoder: Any = None
        self._plan: MemoryPlan | None = None
        self._grid: tuple[int, int] | None = None
        self._fingerprint: str | None = None
        self._cache_dir: Path | None = None
        self._channel_order: tuple[str, ...] = ()
        self._fitted_scoring: Scoring | None = None
        self._selection_seconds = 0.0

        # Exactly one of these is populated, decided by `scoring`.
        self._bank: Any = None
        self._bank_positions: Any = None
        self._position_bank: Any = None
        self._position_counts: Any = None
        self._gaussian_mean: Any = None
        self._gaussian_precision: Any = None
        self._gaussian_dims: Any = None
        self._gaussian_lambda: Any = None

    @classmethod
    def config_model(cls) -> type[BaseModel]:
        return DinoMemoryConfig

    @classmethod
    def capabilities(cls) -> Capabilities:
        return Capabilities(
            # A memory is *fitted* even though nothing is trained: there is a pass over the
            # normals whose output the checkpoint carries, which is what `requires_training`
            # actually gates. `patchcore_anomalib` is in the same position.
            requires_training=True,
            produces_anomaly_map=True,
            produces_diagnostics=True,
            # There are no steps to continue, so `SupportsResume` is deliberately not
            # implemented: a flag without the protocol behind it is a lie the train handler
            # would then have to catch.
            supports_resume=False,
            # True regardless of `channel_fusion`. The capability says the model *may*
            # consume channel metadata; the config field decides whether it does, and a
            # capability that flipped with configuration could not be read from the registry.
            channel_aware=True,
            dataset_specific=False,
            # Empty rather than conditional. `feature_concat` has no single-input graph at
            # all — a sample's channels are separate files — and a portable format that is
            # true for one configuration of a method is worse than an absent one, because
            # the export offer is made from the registry before any config is read.
            portable_formats=[],
            preferred_device=Device.MPS,
        )

    @classmethod
    def availability(cls) -> Availability:
        return module_available("torch", "dl", "DINO patch memory")

    # ------------------------------------------------------------------ grouping

    def _channel_name(self, record: ImageRecord) -> str:
        return record.channel or UNASSIGNED

    def _group(
        self,
        records: Sequence[ImageRecord],
        order: tuple[str, ...],
    ) -> list[list[ImageRecord]]:
        """Records to banking units, refusing a sample that cannot be fused, by name.

        Under `per_image` every record is its own unit and `order` is irrelevant. Under
        `feature_concat` a unit is a sample, and every sample must present exactly the fitted
        channel set — no fallback, for `pixel_reference`'s reason one level up: fusing a
        sample that is missing its dark-field view against a memory built with one would
        produce a confident vector of the wrong shape's worth of nothing.

        This is also where `experiments/diagnose.py`'s single-record path lands. It scores
        one image at a time, so a multi-channel `feature_concat` model refuses it *here*,
        with the channel refusal below, rather than several frames later with an index error.
        """
        if self.config.channel_fusion is ChannelFusion.PER_IMAGE:
            return [[record] for record in records]

        by_sample: dict[int, list[ImageRecord]] = {}
        for record in records:
            by_sample.setdefault(record.sample_id, []).append(record)

        groups: list[list[ImageRecord]] = []
        for sample_id in sorted(by_sample):
            present: dict[str, ImageRecord] = {}
            for record in by_sample[sample_id]:
                name = self._channel_name(record)
                if name in present:
                    raise RuntimeError(self._duplicate_message(sample_id, name))
                present[name] = record
            if set(present) != set(order):
                raise RuntimeError(self._channel_message(sample_id, tuple(present), order))
            groups.append([present[name] for name in order])
        return groups

    @staticmethod
    def _named(channels: Sequence[str]) -> str:
        return ", ".join(name or "unassigned" for name in sorted(channels)) or "no channels"

    def _duplicate_message(self, sample_id: int, name: str) -> str:
        return (
            f"dino_memory fuses one image per channel into a sample's vector, and sample "
            f"{sample_id} presents two images on channel {name or 'unassigned'}. Fix the "
            "channel assignment, or set channel_fusion=per_image to score each image alone."
        )

    def _channel_message(
        self,
        sample_id: int,
        present: tuple[str, ...],
        order: tuple[str, ...],
    ) -> str:
        missing = self._named(sorted(set(order) - set(present)))
        unexpected = self._named(sorted(set(present) - set(order)))
        detail = []
        if set(order) - set(present):
            detail.append(f"it is missing {missing}")
        if set(present) - set(order):
            detail.append(f"it presents {unexpected}, which this model was not fitted on")
        return (
            f"dino_memory under channel_fusion=feature_concat needs every sample to present "
            f"the same channel set. Sample {sample_id} has {self._named(present)}, expected "
            f"{self._named(order)}: {'; '.join(detail)}. Train on the channels this run "
            "scores, select them on the experiment, or set channel_fusion=per_image to score "
            "each channel image on its own."
        )

    # ------------------------------------------------------------------ construction

    def _build_encoder(self, device: str, cache_dir: Path) -> Any:
        """Resolve the frozen encoder through the shared table, seeded before construction."""
        encoder = load_backbone(
            self.config.backbone,
            pretrained=self.config.pretrained_backbone,
            allow_downloads=self.config.allow_downloads,
            cache_dir=cache_dir,
            seed=self.config.seed,
            method="dino_memory",
        )
        self._fingerprint = backbone_fingerprint(encoder)
        self._cache_dir = cache_dir
        return encoder.to(device)

    def _feature_batches(
        self,
        encoder: Any,
        groups: Sequence[Sequence[ImageRecord]],
        preprocessing: PreprocessingConfig,
        device: str,
    ) -> Iterator[tuple[int, Any]]:
        """Yield `(index, (P, fused_dim))` per unit, in order, batching the forward pass.

        Units are packed into a forward until `feature_batch_size` images are reached, so
        `per_image` still gets full batches and `feature_concat` never splits a sample across
        two forwards — which it must not, since the channels are concatenated afterwards.
        """
        indices = self.config.layers.indices
        budget = max(1, self.config.feature_batch_size)
        cursor = 0
        while cursor < len(groups):
            batch: list[Sequence[ImageRecord]] = []
            images: list[ImageRecord] = []
            while cursor < len(groups) and (
                not batch or len(images) + len(groups[cursor]) <= budget
            ):
                batch.append(groups[cursor])
                images.extend(groups[cursor])
                cursor += 1

            features = _image_features(encoder, images, preprocessing, indices, device)
            offset = 0
            first = cursor - len(batch)
            for step, group in enumerate(batch):
                block = features[offset : offset + len(group)]
                offset += len(group)
                yield first + step, _fuse(block)

    def _verify(self, plan: MemoryPlan, features: Any) -> None:
        """The announced footprint has to be the measured one, or the plan was a fiction.

        The plan is computed from `patch_grid`'s arithmetic and the backbone table, with no
        forward pass at all, which is what lets `describe()` be logged before the first one.
        The price of that is this check: the first real batch confirms the grid and the width,
        and a disagreement is a bug in the table rather than a number to quietly adopt.
        """
        positions = int(features.shape[0])
        width = int(features.shape[1])
        if positions != plan.positions or width != plan.fused_dim:
            msg = (
                f"dino_memory planned a {plan.grid_rows}x{plan.grid_cols} grid of "
                f"{plan.fused_dim}-wide vectors for {self.config.backbone.value} with "
                f"layers={self.config.layers.value}, but the encoder produced {positions} "
                f"positions of width {width}. The plan's footprints describe a run that is "
                "not this one; the backbone table and the layer indices disagree with timm."
            )
            raise RuntimeError(msg)

    # ------------------------------------------------------------------ training

    def fit(self, train: Sequence[ImageRecord], ctx: TrainContext) -> None:
        if not train:
            msg = "dino_memory was given no training images to build a memory from"
            raise RuntimeError(msg)

        validate_prepared_size(
            self.config.backbone, ctx.preprocessing.width, ctx.preprocessing.height
        )
        rows, cols = patch_grid(
            self.config.backbone, ctx.preprocessing.width, ctx.preprocessing.height
        )
        channels = channel_order(train)
        order = channels if self.config.channel_fusion is ChannelFusion.FEATURE_CONCAT else ()
        groups = self._group(train, order)

        spec = BACKBONES[self.config.backbone]
        plan = plan_memory(
            self.config,
            samples_available=len({record.sample_id for record in train}),
            images_available=len(train),
            grid=(rows, cols),
            embedding_dim=spec.embedding_dim * len(self.config.layers.indices),
            channels=channels,
        )
        if self.config.scoring is Scoring.LOCAL_GAUSSIAN and plan.samples_per_position < 2:
            msg = (
                "dino_memory's local_gaussian fits a covariance per patch position and this "
                f"training set offers {plan.samples_per_position} sample(s) per position. "
                "One sample has no covariance at all; use at least two units, or choose "
                "local_knn, which is well defined from a single example."
            )
            raise RuntimeError(msg)

        # Before the first forward pass, not after it — no probe is needed, because nothing
        # in the plan was measured. What follows is bounded by numbers the reader has
        # already seen.
        ctx.log(plan.describe())
        if plan.images_dropped:
            ctx.log(
                f"{plan.images_dropped} of {plan.images_available} training images are not "
                f"in the memory (max_bank_images={self.config.max_bank_images}), sampled "
                "evenly across the set rather than taken from its front"
            )
        if plan.patches_dropped_per_image:
            ctx.log(
                f"{plan.patches_dropped_per_image} of {plan.patches_per_image} patches per "
                f"unit are dropped (max_candidate_vectors="
                f"{self.config.max_candidate_vectors}), on a regular grid"
            )

        ctx.progress(0.02, f"loading {spec.timm_name} (downloads on first run only)")
        encoder = self._build_encoder(ctx.device.value, ctx.cache_dir)
        self._channel_order = order

        # **Nothing is fitted until the memory exists.** The three markers `predict` reads
        # are assigned after the pass rather than before it, so a cancelled or failed fit
        # leaves an object that refuses rather than one that answers from a half-filled
        # store. A store that is merely *partly* full is the worst available outcome here:
        # it scores, and every number it produces is wrong in a way nothing reports.
        chosen = [groups[index] for index in evenly_spaced(len(groups), max(plan.units_used, 1))]
        if self.config.scoring is Scoring.GLOBAL_KNN:
            self._fit_global(encoder, chosen, ctx, plan)
        elif self.config.scoring is Scoring.LOCAL_KNN:
            self._fit_local_knn(encoder, chosen, ctx, plan)
        else:
            self._fit_local_gaussian(encoder, chosen, ctx, plan)

        self._encoder = encoder
        self._plan = plan
        self._grid = (rows, cols)
        self._fitted_scoring = self.config.scoring
        self._emit_memory_table(ctx, plan)

    def _fit_global(
        self,
        encoder: Any,
        groups: Sequence[Sequence[ImageRecord]],
        ctx: TrainContext,
        plan: MemoryPlan,
    ) -> None:
        """One coreset bank over every kept patch of every kept unit — PatchCore's memory."""
        import torch

        keep = evenly_spaced(plan.patches_per_image, plan.patches_kept_per_image)
        keep_index = torch.tensor(keep, dtype=torch.long)
        # Where each kept patch sits, so the coverage diagnostic is a picture and not a
        # count. Deterministic and identical for every unit, so it is built once.
        grid_positions = torch.stack([keep_index // plan.grid_cols, keep_index % plan.grid_cols], 1)

        store = torch.empty(plan.candidates_kept, plan.fused_dim, dtype=torch.float32)
        positions = torch.empty(plan.candidates_kept, 2, dtype=torch.long)

        started = time.perf_counter()
        cursor = 0
        verified = False
        for index, fused in self._feature_batches(
            encoder, groups, ctx.preprocessing, ctx.device.value
        ):
            ctx.raise_if_cancelled()
            if not verified:
                self._verify(plan, fused)
                verified = True
            thinned = fused[keep_index]
            end = cursor + int(thinned.shape[0])
            store[cursor:end] = thinned
            positions[cursor:end] = grid_positions
            cursor = end
            ctx.progress(
                0.05 + 0.45 * (index + 1) / len(groups),
                f"embedded {index + 1}/{len(groups)} units",
            )

        elapsed = time.perf_counter() - started
        ctx.log(
            f"embedded {len(groups)} units in {elapsed:.1f}s "
            f"({elapsed / max(len(groups), 1) * 1000:.0f} ms/unit on {ctx.device.value}); "
            f"{cursor:,} candidate vectors held on cpu"
        )
        store = store[:cursor]
        positions = positions[:cursor]

        ctx.progress(0.52, "projecting the candidate pool")
        projected = gaussian_projection(store, plan.projection_dim, seed=self.config.seed)
        ctx.log(
            f"projected {cursor:,}x{plan.fused_dim} to {plan.projection_dim} dimensions for "
            f"the selection (eps={_PROJECTION_EPS})"
        )

        size = min(plan.coreset_size, cursor)
        generator = np.random.default_rng(self.config.seed)

        def observe(step: int, furthest: float) -> None:
            # Every iteration, which is the whole reason the loop is ours rather than a
            # library call: a selection that cannot be watched or stopped is a frozen job.
            ctx.raise_if_cancelled()
            if step % _SELECTION_LOG_EVERY == 0 or step == size - 1:
                ctx.metric("coreset_max_distance", furthest, step=step)
                ctx.progress(
                    0.55 + 0.35 * (step + 1) / max(size, 1),
                    f"selected {step + 1:,}/{size:,} bank vectors",
                )

        started = time.perf_counter()
        selected = greedy_select(
            projected,
            size,
            start=int(generator.integers(projected.shape[0])),
            observe=observe,
        )
        self._selection_seconds = time.perf_counter() - started
        self._bank = store[selected].contiguous()
        self._bank_positions = positions[selected].contiguous()

        ctx.metric("bank_vectors", float(self._bank.shape[0]))
        ctx.metric("bank_megabytes", float(self._bank.numel() * BYTES_PER_FLOAT32 / 1e6))
        ctx.log(
            f"memory bank: {self._bank.shape[0]:,} vectors of {plan.fused_dim} "
            f"({self._bank.numel() * BYTES_PER_FLOAT32 / 1e6:.1f} MB), selected in "
            f"{self._selection_seconds:.1f}s on cpu"
        )
        self._emit_coverage(ctx, self._bank_positions, plan)

    def _fit_local_knn(
        self,
        encoder: Any,
        groups: Sequence[Sequence[ImageRecord]],
        ctx: TrainContext,
        plan: MemoryPlan,
    ) -> None:
        """One small bank per patch position — the memory a registered dataset can use."""
        import torch

        keep = evenly_spaced(len(groups), plan.per_position_images)
        wanted = {index: slot for slot, index in enumerate(keep)}
        bank = torch.empty(
            plan.grid_rows,
            plan.grid_cols,
            plan.per_position_images,
            plan.fused_dim,
            dtype=torch.float32,
        )
        counts = torch.zeros(plan.grid_rows, plan.grid_cols, dtype=torch.float32)

        started = time.perf_counter()
        verified = False
        for index, fused in self._feature_batches(
            encoder, groups, ctx.preprocessing, ctx.device.value
        ):
            ctx.raise_if_cancelled()
            if not verified:
                self._verify(plan, fused)
                verified = True
            slot = wanted.get(index)
            if slot is None:
                continue
            bank[:, :, slot, :] = fused.reshape(plan.grid_rows, plan.grid_cols, plan.fused_dim)
            counts += 1.0
            ctx.progress(
                0.05 + 0.85 * (slot + 1) / len(keep),
                f"banked {slot + 1}/{len(keep)} units",
            )

        elapsed = time.perf_counter() - started
        self._position_bank = bank
        self._position_counts = counts
        ctx.metric("bank_vectors", float(plan.positions * plan.per_position_images))
        ctx.metric("bank_megabytes", float(plan.bank_bytes / 1e6))
        ctx.log(
            f"per-position bank: {plan.grid_rows}x{plan.grid_cols} positions x "
            f"{plan.per_position_images} vectors of {plan.fused_dim} "
            f"({plan.bank_bytes / 1e6:.1f} MB) in {elapsed:.1f}s"
        )
        self._emit_position_counts(ctx, counts)

    def _fit_local_gaussian(
        self,
        encoder: Any,
        groups: Sequence[Sequence[ImageRecord]],
        ctx: TrainContext,
        plan: MemoryPlan,
    ) -> None:
        """One shrunk Gaussian per patch position, fitted on the CPU in float64.

        The dtype and the device are both smoke-test verdicts rather than preferences:
        batched Cholesky is 16x slower on MPS where it exists at all, and a covariance fitted
        from fewer samples than dimensions is exactly where float32 stops being enough.

        The dimension subset is **one seeded draw shared by every position**. Drawing per
        position would make neighbouring positions incomparable for no benefit, and a subset
        rather than a projection keeps the stored means readable as feature dimensions.
        """
        import torch

        keep = evenly_spaced(len(groups), plan.samples_per_position)
        wanted = {index: slot for slot, index in enumerate(keep)}
        generator = np.random.default_rng(self.config.seed)
        if plan.reduced_dim >= plan.fused_dim:
            dims = np.arange(plan.fused_dim, dtype=np.int64)
        else:
            dims = np.sort(
                generator.choice(plan.fused_dim, size=plan.reduced_dim, replace=False)
            ).astype(np.int64)
        dim_index = torch.from_numpy(dims)

        samples = torch.empty(
            plan.positions, plan.samples_per_position, plan.reduced_dim, dtype=torch.float64
        )
        verified = False
        for index, fused in self._feature_batches(
            encoder, groups, ctx.preprocessing, ctx.device.value
        ):
            ctx.raise_if_cancelled()
            if not verified:
                self._verify(plan, fused)
                verified = True
            slot = wanted.get(index)
            if slot is None:
                continue
            samples[:, slot, :] = fused[:, dim_index].double()
            ctx.progress(
                0.05 + 0.45 * (slot + 1) / len(keep),
                f"embedded {slot + 1}/{len(keep)} units",
            )

        values = samples.numpy()
        means = values.mean(axis=1)
        centred = values - means[:, None, :]
        precision = np.empty((plan.positions, plan.reduced_dim, plan.reduced_dim), dtype=np.float32)
        lambdas = np.empty(plan.positions, dtype=np.float32)

        started = time.perf_counter()
        for position in range(plan.positions):
            ctx.raise_if_cancelled()
            conditioned, lam = condition_covariance(
                centred[position],
                shrinkage=self.config.shrinkage,
                ridge_epsilon=self.config.ridge_epsilon,
            )
            precision[position] = precision_from_covariance(conditioned).astype(np.float32)
            lambdas[position] = lam
            if position % _POSITION_LOG_EVERY == 0 or position == plan.positions - 1:
                ctx.progress(
                    0.55 + 0.4 * (position + 1) / plan.positions,
                    f"fitted {position + 1}/{plan.positions} positions",
                )

        elapsed = time.perf_counter() - started
        self._gaussian_mean = torch.from_numpy(means.astype(np.float32))
        self._gaussian_precision = torch.from_numpy(precision)
        self._gaussian_dims = dim_index
        self._gaussian_lambda = torch.from_numpy(
            lambdas.reshape(plan.grid_rows, plan.grid_cols).copy()
        )

        ctx.metric("shrinkage_lambda_median", float(np.median(lambdas)))
        ctx.metric("bank_megabytes", float((plan.bank_bytes + plan.precision_bytes) / 1e6))
        ctx.log(
            f"fitted {plan.positions} per-position Gaussians over {plan.reduced_dim} of "
            f"{plan.fused_dim} dimensions from {plan.samples_per_position} samples each in "
            f"{elapsed:.1f}s on cpu float64; {self.config.shrinkage.value} lambda median "
            f"{float(np.median(lambdas)):.3f}"
        )
        self._emit_shrinkage(ctx)

    # ------------------------------------------------------------------ diagnostics

    def _emit_memory_table(self, ctx: TrainContext, plan: MemoryPlan) -> None:
        """What the memory is made of and what it cost, written where a reader will find it.

        The same `{columns, rows}` shape `patchcore_anomalib._emit_bank_table` emits, under a
        key the diagnostics index already renders — so this reaches M4's Architecture tab
        with no new UI code at all, which is ADR-0018's whole claim.
        """
        spec = BACKBONES[self.config.backbone]
        rows: list[list[str]] = [
            ["scoring", self.config.scoring.value],
            ["backbone", f"{self.config.backbone.value} ({spec.timm_name})"],
            [
                "backbone weights",
                "published"
                if self.config.pretrained_backbone
                else f"random, seeded {self.config.seed}",
            ],
            ["layers", f"{self.config.layers.value} {list(self.config.layers.indices)}"],
            ["patch grid", f"{plan.grid_rows}x{plan.grid_cols}"],
            ["feature width", f"{plan.channels_fused} x {plan.embedding_dim} = {plan.fused_dim}"],
            ["channel fusion", self.config.channel_fusion.value],
            ["channel order", self._named(self._channel_order) if self._channel_order else "-"],
            ["units used", f"{plan.units_used} of {plan.units_available}"],
            ["images read", f"{plan.images_used} of {plan.images_available}"],
        ]
        if plan.scoring is Scoring.GLOBAL_KNN:
            rows.extend(
                [
                    [
                        "patches per unit",
                        f"{plan.patches_kept_per_image} of {plan.patches_per_image}",
                    ],
                    ["candidate vectors", f"{plan.candidates_kept:,}"],
                    ["candidate pool", f"{plan.candidate_bytes / 1e6:.1f} MB"],
                    ["projection width", str(plan.projection_dim)],
                    ["coreset ratio", f"{self.config.coreset_ratio:g}"],
                    ["bank vectors", f"{plan.coreset_size:,}"],
                    ["bank size", f"{plan.bank_bytes / 1e6:.1f} MB"],
                    ["selection time", f"{self._selection_seconds:.1f} s on cpu"],
                ]
            )
        elif plan.scoring is Scoring.LOCAL_KNN:
            rows.extend(
                [
                    ["vectors per position", str(plan.per_position_images)],
                    ["window radius", str(plan.window_radius)],
                    ["candidates per position", str(plan.window_candidates)],
                    ["bank vectors", f"{plan.candidates_kept:,}"],
                    ["bank size", f"{plan.bank_bytes / 1e6:.1f} MB"],
                ]
            )
        else:
            rows.extend(
                [
                    ["samples per position", str(plan.samples_per_position)],
                    ["mahalanobis dimensions", f"{plan.reduced_dim} of {plan.fused_dim}"],
                    ["sample deficit", str(plan.sample_deficit)],
                    ["shrinkage", f"{self.config.shrinkage.value}"],
                    ["means", f"{plan.bank_bytes / 1e6:.1f} MB"],
                    ["precision", f"{plan.precision_bytes / 1e6:.1f} MB"],
                ]
            )
        rows.append(["embedding device", ctx.device.value])
        rows.append(["uncapped pool would have been", f"{plan.unbounded_candidates:,} vectors"])

        ctx.emit_diagnostic(
            "memory_bank",
            "Memory",
            DiagnosticKind.TABLE,
            {"columns": ["quantity", "value"], "rows": rows},
            description=(
                "What this memory is made of and what it cost. The last row is what the same "
                "run would have held with every cap removed — the number the caps exist for."
            ),
        )

    def _emit_coverage(self, ctx: TrainContext, positions: Any, plan: MemoryPlan) -> None:
        """Where on the frame a global bank spent its budget.

        A coreset is chosen for diversity in feature space, and whether that concentrates on
        a few textured regions or spreads over the frame is a real property of the fit that
        no metric reports. Same key and kind as PatchCore's, so it renders identically.
        """
        if not ctx.diagnostics.enabled:
            return
        counts = np.zeros((plan.grid_rows, plan.grid_cols), dtype=np.float32)
        for row, col in positions.tolist():
            counts[row, col] += 1.0
        ctx.emit_diagnostic(
            "coreset_coverage",
            "Coreset coverage",
            DiagnosticKind.MAP,
            counts,
            description=(
                "How many bank vectors were selected from each patch position, over every "
                "contributing unit. Bright regions are where normal appearance varies most, "
                "and so where the bank spends its budget."
            ),
        )

    def _emit_position_counts(self, ctx: TrainContext, counts: Any) -> None:
        if not ctx.diagnostics.enabled:
            return
        ctx.emit_diagnostic(
            "position_counts",
            "Vectors per position",
            DiagnosticKind.MAP,
            counts.numpy().astype(np.float32),
            description=(
                "How many training vectors each patch position holds. A per-position bank "
                "fills every position from the same units, so a flat map is the fit working; "
                "anything else means a unit failed to contribute and is worth reading."
            ),
        )

    def _emit_shrinkage(self, ctx: TrainContext) -> None:
        if not ctx.diagnostics.enabled or self._gaussian_lambda is None:
            return
        ctx.emit_diagnostic(
            "shrinkage_lambda",
            "Shrinkage per position",
            DiagnosticKind.MAP,
            self._gaussian_lambda.numpy().astype(np.float32),
            description=(
                "How much each position's covariance had to be pulled toward a scaled "
                "identity to become invertible. High values mark positions whose normal "
                "appearance varies in fewer directions than the fit has dimensions, so their "
                "Mahalanobis distance is closer to a Euclidean one than the method intends."
            ),
        )

    # ------------------------------------------------------------------ scoring

    def _global_scores(self, query: Any) -> Any:
        """`(P,)` patch scores against the coreset bank, chunked.

        The expansion `|q|^2 + |b|^2 - 2 q b^T` rather than `torch.cdist`, which is anomalib's
        own choice in `euclidean_dist` and the smoke test's: the expansion is one matmul with
        a broadcast add, and both norms are exactly 1 here by construction, so this is the
        exact squared distance and not an approximation of it.

        The whole `(P, N)` matrix is never allocated — 409 MB at P=1024 and N=100 000 — so
        each chunk's own top-k is merged into a running one. Selection is on the **CPU**
        deliberately: `topk` over a 100 000-wide row measured about 7x slower on MPS, and
        breaks exact ties the other way, so a neighbour index is never a cross-device
        identity.
        """
        import torch

        bank = self._bank
        count = int(bank.shape[0])
        neighbors = min(self.config.num_neighbors, count)
        query_square = query.pow(2).sum(dim=1, keepdim=True)
        best: Any = None

        for start in range(0, count, _DISTANCE_CHUNK):
            block = bank[start : start + _DISTANCE_CHUNK]
            distances = (
                query_square + block.pow(2).sum(dim=1).unsqueeze(0) - 2.0 * (query @ block.T)
            )
            take = min(neighbors, int(block.shape[0]))
            values = distances.topk(k=take, largest=False, dim=1).values
            best = values if best is None else torch.cat([best, values], dim=1)
            if int(best.shape[1]) > neighbors:
                best = best.topk(k=neighbors, largest=False, dim=1).values

        return best.clamp_min(0.0).mean(dim=1)

    def _local_knn_scores(self, query: Any, plan: MemoryPlan) -> Any:
        """`(rows, cols)` patch scores against the per-position bank and its window.

        One aligned slice per offset rather than a gather. An `unfold` that materialized every
        window's neighbours at once would hold `(2r+1)^2 x P x K x D` floats — about 900 MB at
        r=1, P=1024, K=64 and a 384-wide feature, and 1.8 GB at 768 — where the offset loop
        holds one `(P, K, D)` view at a time and reuses the bank in place.

        The distance is `2 - 2 cos` because every vector is a unit vector: the norm terms are
        exactly 1 each, so dropping them is an identity rather than an approximation.
        """
        import torch

        rows, cols = plan.grid_rows, plan.grid_cols
        grid = query.reshape(rows, cols, plan.fused_dim)
        bank = self._position_bank
        neighbors = min(self.config.num_neighbors, int(bank.shape[2]))
        radius = self.config.window_radius

        best = torch.full((rows, cols), float("inf"), dtype=torch.float32)
        centre: Any = None

        for row_offset in range(-radius, radius + 1):
            for col_offset in range(-radius, radius + 1):
                qr0, qr1 = max(0, -row_offset), rows - max(0, row_offset)
                br0, br1 = max(0, row_offset), rows - max(0, -row_offset)
                qc0, qc1 = max(0, -col_offset), cols - max(0, col_offset)
                bc0, bc1 = max(0, col_offset), cols - max(0, -col_offset)
                if qr1 <= qr0 or qc1 <= qc0:
                    continue

                similarity = torch.einsum(
                    "rcd,rckd->rck",
                    grid[qr0:qr1, qc0:qc1, :],
                    bank[br0:br1, bc0:bc1, :, :],
                )
                distances = (2.0 - 2.0 * similarity).clamp_min(0.0)
                scored = distances.topk(k=neighbors, largest=False, dim=2).values.mean(dim=2)
                window = torch.full((rows, cols), float("inf"), dtype=torch.float32)
                window[qr0:qr1, qc0:qc1] = scored
                best = torch.minimum(best, window)
                if row_offset == 0 and col_offset == 0:
                    centre = window

        if centre is None:
            msg = "dino_memory's window loop produced no centre offset, which cannot happen"
            raise RuntimeError(msg)
        # A border patch has fewer valid offsets than an interior one, and the guard says it
        # is scored against its own position rather than against nothing. Redundant while
        # (0, 0) is always in the loop — and that is exactly why it is here: a later change
        # to the offset set must not silently leave a border patch at infinity.
        return torch.where(torch.isinf(best), centre, best)

    def _gaussian_scores(self, query: Any, plan: MemoryPlan) -> Any:
        """`(rows, cols)` Mahalanobis distances against the per-position Gaussians."""
        import torch

        reduced = query[:, self._gaussian_dims]
        delta = (reduced - self._gaussian_mean).reshape(
            plan.grid_rows, plan.grid_cols, plan.reduced_dim
        )
        precision = self._gaussian_precision.reshape(
            plan.grid_rows, plan.grid_cols, plan.reduced_dim, plan.reduced_dim
        )
        squared = torch.einsum("rcd,rcde,rce->rc", delta, precision, delta)
        return squared.clamp_min(0.0).sqrt()

    def _patch_scores(self, fused: Any, plan: MemoryPlan) -> Any:
        if self.config.scoring is Scoring.GLOBAL_KNN:
            return self._global_scores(fused).reshape(plan.grid_rows, plan.grid_cols)
        if self.config.scoring is Scoring.LOCAL_KNN:
            return self._local_knn_scores(fused, plan)
        return self._gaussian_scores(fused, plan)

    def _anomaly_map(self, scores: Any, size: tuple[int, int]) -> tuple[Any, Any]:
        """Patch grid to `(unblurred, final)` maps at the prepared size."""
        import torch.nn.functional as functional

        grid = scores.reshape(1, 1, *scores.shape)
        upsampled = functional.interpolate(grid, size=size, mode="bilinear", align_corners=False)
        if self.config.blur_sigma <= 0:
            return upsampled, upsampled
        return upsampled, _separable_blur(upsampled, self.config.blur_sigma)

    # ------------------------------------------------------------------ inference

    def predict(self, images: Sequence[ImageRecord], ctx: InferContext) -> list[Prediction]:
        import torch

        plan = self._plan
        if self._encoder is None or plan is None or self._fitted_scoring is None:
            msg = "dino_memory was asked to predict before it was fitted or loaded"
            raise RuntimeError(msg)
        if self._fitted_scoring is not self.config.scoring:
            msg = (
                f"this dino_memory instance is configured for scoring="
                f"{self.config.scoring.value} but the loaded memory is a "
                f"{self._fitted_scoring.value} one. The two are different models over the "
                "same features; refit, or load this checkpoint into a matching experiment."
            )
            raise RuntimeError(msg)
        validate_prepared_size(
            self.config.backbone, ctx.preprocessing.width, ctx.preprocessing.height
        )

        groups = self._group(images, self._channel_order)
        order = {record.image_id: index for index, record in enumerate(images)}
        predictions: list[Prediction | None] = [None] * len(images)
        size = (ctx.preprocessing.height, ctx.preprocessing.width)
        indices = self.config.layers.indices
        encoder = self._encoder.to(ctx.device.value)

        with torch.no_grad():
            for step, group in enumerate(groups):
                ctx.raise_if_cancelled()
                started = time.perf_counter()

                block = _image_features(
                    encoder, group, ctx.preprocessing, indices, ctx.device.value
                )
                fused = _fuse(block)
                scores = self._patch_scores(fused, plan)
                unblurred, blurred = self._anomaly_map(scores, size)

                grid_values = scores.numpy().astype(np.float32)
                # The score is taken on the **patch grid**, before upsampling and before the
                # blur, so `blur_sigma` cannot move it. A score that depended on the
                # smoothing would make the smoothing a scoring hyperparameter disguised as a
                # display one.
                score = float(np.percentile(grid_values, self.config.score_percentile))
                values = blurred[0, 0].numpy().astype(np.float32)
                raw_values = unblurred[0, 0].numpy().astype(np.float32)
                elapsed_ms = (time.perf_counter() - started) * 1000.0

                for record in group:
                    # One map per image even under feature_concat, where the map is the
                    # sample's. Each write goes through `write_map` on its own, because each
                    # image carries its own source-frame projection and sharing the array
                    # does not mean sharing the transform.
                    map_path = ctx.write_map(record.image_id, values)
                    ctx.emit_diagnostic(
                        "patch_scores",
                        "Patch scores",
                        DiagnosticKind.MAP,
                        grid_values,
                        image_id=record.image_id,
                        description=(
                            "Distance from each patch to the memory, at the encoder's own "
                            "resolution. This is what the method actually computed; the "
                            "anomaly map is this upsampled and smoothed."
                        ),
                    )
                    ctx.emit_diagnostic(
                        "map_unblurred",
                        "Anomaly map before smoothing",
                        DiagnosticKind.MAP,
                        raw_values,
                        image_id=record.image_id,
                        description=(
                            f"The upsampled patch scores without the Gaussian blur "
                            f"(blur_sigma={self.config.blur_sigma:g}), so the smoothing's "
                            "effect is visible rather than assumed."
                        ),
                    )
                    predictions[order[record.image_id]] = Prediction(
                        image_id=record.image_id,
                        score=score,
                        anomaly_map=map_path,
                        inference_ms=elapsed_ms,
                    )

                ctx.progress((step + 1) / len(groups), f"scored {step + 1}/{len(groups)} units")

        missing = [
            record.image_id for record, out in zip(images, predictions, strict=True) if out is None
        ]
        if missing:
            msg = f"dino_memory produced no prediction for image ids {missing}"
            raise RuntimeError(msg)
        return [prediction for prediction in predictions if prediction is not None]

    # ------------------------------------------------------------------ persistence

    def save(self, artifact_dir: Path) -> None:
        """Write the memory and the identity of everything it was built from.

        The encoder's weights are deliberately absent: they are somebody else's artifact,
        `load_backbone` rebuilds them from the app-managed cache, and a copy per experiment
        would multiply the disk cost of a comparison by the number of runs in it. What is
        stored instead is the fingerprint, which turns "the weights might have changed" from
        an unanswerable worry into a check.
        """
        import torch

        plan = self._plan
        if plan is None or self._fingerprint is None or self._cache_dir is None:
            msg = "dino_memory has nothing to save; it was never fitted"
            raise RuntimeError(msg)

        spec = BACKBONES[self.config.backbone]
        payload: dict[str, Any] = {
            "format": CHECKPOINT_FORMAT,
            "scoring": self.config.scoring.value,
            "backbone": self.config.backbone.value,
            "timm_name": spec.timm_name,
            "patch_size": spec.patch_size,
            "layers": self.config.layers.value,
            "layer_indices": list(self.config.layers.indices),
            "pretrained_backbone": self.config.pretrained_backbone,
            "backbone_fingerprint": self._fingerprint,
            "grid": list(self._grid) if self._grid else None,
            "embedding_dim": plan.embedding_dim,
            "fused_dim": plan.fused_dim,
            "channel_fusion": self.config.channel_fusion.value,
            "channel_order": list(self._channel_order),
            "plan": asdict(plan),
            "cache_dir": str(self._cache_dir),
            "selection_seconds": self._selection_seconds,
            "versions": {"timm": version("timm"), "torch": version("torch")},
        }
        if self.config.scoring is Scoring.GLOBAL_KNN:
            payload["global_bank"] = {
                "vectors": self._bank,
                "positions": self._bank_positions,
            }
        elif self.config.scoring is Scoring.LOCAL_KNN:
            payload["position_bank"] = {
                "vectors": self._position_bank,
                "counts": self._position_counts,
            }
        else:
            payload["position_gaussian"] = {
                "mean": self._gaussian_mean,
                "precision": self._gaussian_precision,
                "dims": self._gaussian_dims,
                "lambda": self._gaussian_lambda,
            }
        torch.save(payload, artifact_dir / STATE_FILENAME)

    def load(self, artifact_dir: Path) -> None:
        """Rebuild the encoder, restore the memory, and refuse anything that does not match.

        Every refusal here is **by name**, because the alternative in each case is the failure
        `methods.md` warns about for the EfficientAD teacher: a checkpoint that loads without
        complaint and produces a plausible, wrong number that nothing distinguishes from the
        run it is being compared against.
        """
        import torch

        stored = torch.load(artifact_dir / STATE_FILENAME, map_location="cpu", weights_only=False)

        if int(stored.get("format", 0)) != CHECKPOINT_FORMAT:
            msg = (
                f"unsupported dino_memory checkpoint format {stored.get('format')!r}; this "
                f"build reads format {CHECKPOINT_FORMAT}. The file was written by a "
                "different version of the plugin and its memory layout is not this one."
            )
            raise RuntimeError(msg)

        scoring = Scoring(str(stored["scoring"]))
        if scoring is not self.config.scoring:
            msg = (
                f"this dino_memory experiment is configured for scoring="
                f"{self.config.scoring.value} but the checkpoint holds a "
                f"{scoring.value} memory. The three scoring rules build different memories "
                "over the same features, so one cannot be read as another; refit, or open "
                "the experiment this checkpoint belongs to."
            )
            raise RuntimeError(msg)

        cache_dir = Path(stored["cache_dir"])
        encoder = self._build_encoder("cpu", cache_dir)
        expected = str(stored["backbone_fingerprint"])
        actual = str(self._fingerprint)
        if expected != actual:
            msg = (
                f"the encoder weights for {stored.get('backbone', self.config.backbone.value)!r} "
                f"are not the ones this experiment was fitted against: the checkpoint records "
                f"{expected[:12]} and the weights now resolving are {actual[:12]}. The memory "
                "was built in the old feature space, so distances against the new one are not "
                "comparable. Refit the experiment, or restore the original weights."
            )
            raise RuntimeError(msg)

        self._encoder = encoder
        self._fitted_scoring = scoring
        grid = stored.get("grid")
        self._grid = (int(grid[0]), int(grid[1])) if grid else None
        self._channel_order = tuple(str(name) for name in stored.get("channel_order", ()))
        self._selection_seconds = float(stored.get("selection_seconds", 0.0))
        self._plan = MemoryPlan(**stored["plan"])

        if scoring is Scoring.GLOBAL_KNN:
            self._bank = stored["global_bank"]["vectors"]
            self._bank_positions = stored["global_bank"]["positions"]
        elif scoring is Scoring.LOCAL_KNN:
            self._position_bank = stored["position_bank"]["vectors"]
            self._position_counts = stored["position_bank"]["counts"]
        else:
            gaussian = stored["position_gaussian"]
            self._gaussian_mean = gaussian["mean"]
            self._gaussian_precision = gaussian["precision"]
            self._gaussian_dims = gaussian["dims"]
            self._gaussian_lambda = gaussian["lambda"]
