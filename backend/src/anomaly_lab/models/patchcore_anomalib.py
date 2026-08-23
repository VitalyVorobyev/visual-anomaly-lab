"""`patchcore_anomalib` — PatchCore (arXiv:2106.08265) through Intel's anomalib.

The third method, and the first whose cost is a **memory footprint rather than a step
budget**. PatchCore does not train: it runs a frozen ImageNet backbone over the normals,
keeps every patch embedding, greedily selects a representative subset — the *coreset* — and
scores a test patch by its distance to the nearest vector in that memory bank. There are no
gradients, no epochs and nothing to resume.

**What comes from anomalib:** the backbone feature extraction, the multi-scale embedding,
`euclidean_dist`, `nearest_neighbors`, the paper's score reweighting, the anomaly-map
generator, and `SparseRandomProjection`. Everything that decides a number is anomalib's.

**What is ours:** the fit pass, the bounding, the greedy selection loop, and the reporting.

The selection loop is ours for the same reason a training loop that drives its torch module
directly instead of through anomalib's own runner is: `KCenterGreedy` is a `tqdm` loop that
returns when it is finished and not before. At library defaults on a full VisA class that is 92 160
iterations over a 5.66 GB store — ten minutes in which the job reports no progress and
cancellation does nothing. Every other long operation in this workbench stops within one
step, and a method that cannot is a hole in the application rather than a property of the
method. The greedy *rule* is not reimplemented so much as re-hosted: the projection and the
distance are still anomalib's, the iteration order is identical, and
`test_dl_patchcore.py::test_the_greedy_selection_matches_anomalib` pins that from the same
starting point the two produce the same coreset.

**The bank is bounded by two independent caps, and both are load-bearing.**

  * `max_bank_images` bounds how many normals are read and pushed through the backbone.
    That cost is linear in the training set and its value is not — a bank does not get
    better by seeing the same production batch twice.
  * `max_candidate_vectors` bounds the embedding store, and with it the selection. This is
    the cap that saves the machine: the greedy loop runs `coreset_ratio x N` times and each
    iteration touches all N rows, so **its total cost is quadratic in N**. Measured by
    `scripts/patchcore-smoke-test.py`: 1.2 s at N=25 000, 24.6 s at 100 000, 150.8 s at
    250 000, and roughly half an hour at the 921 600 a full class would generate.

Neither cap is a suggestion the code makes quietly. `plan_bank` computes the whole footprint
*before* the extraction pass and the numbers go into the job log, so the cost is known
before it is paid rather than discovered by a machine that has started swapping.

At the defaults — 512 images, a 100 000-vector candidate pool, `coreset_ratio` 0.1 — the
bank is about 10 000 vectors and 61 MB, which is close to what the paper's own recipe
produces (1% of ~921 600 is ~9 200) by a different route. Fitting takes about half a minute.

**The selection runs on the CPU even when the rest runs on MPS**, and that is measured
rather than assumed. The loop is one norm over a tall thin tensor, an argmax and a scatter —
too little arithmetic to cover per-iteration dispatch — so MPS loses it by a factor of
three (7.16 ms against 2.46 at N=100 000). The backbone forward wins on MPS by 2.8x and
stays there. Nothing in the application would have shown this: the run finishes and the bank
is correct, it is merely three times slower than it needed to be.

**PatchCore's backbone needs ImageNet normalization and anomalib does not apply it inside
the model.** `Patchcore.configure_pre_processor` puts it in the Lightning pre-processor,
which this wrapper does not use: the preprocessing bridge is what makes a comparison mean
anything, and adopting anomalib's own pre-processor would mean adopting its own resize and
normalization ahead of it, breaking the property that every method sees identical pixels
(`preprocessing.py`). `efficientad_custom` normalizes in `forward` and can therefore be
handed `load_array`'s `[0, 1]` array unchanged; PatchCore cannot, and feeding it unnormalized
pixels runs, produces maps, and quietly scores an off-distribution backbone. So it is applied
here, from the constants in `preprocessing.py`.

**The backbone weights are an input to the experiment** (`docs/architecture/methods.md`).
timm's pretrained `wide_resnet50_2` is the output of somebody's training run, resolved
through the HuggingFace hub, and it can change upstream. The checkpoint therefore stores a
fingerprint of the backbone rather than 260 MB of it, and `load` refuses a mismatch by
name — a checksum failure naming the backbone, never a silent change of weights between two
runs that read as comparable.
"""

from __future__ import annotations

import hashlib
import os
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from anomaly_lab.deployment.protocol import OnnxGraphContract
from anomaly_lab.deployment.schema import ScalarTensorSpec, TensorDtype, TensorScore
from anomaly_lab.models.base import (
    AnomalyModel,
    Availability,
    Capabilities,
    Device,
    ImageRecord,
    InferContext,
    PortableFormat,
    Prediction,
    TrainContext,
    evenly_spaced,
    module_available,
)
from anomaly_lab.models.diagnostics import DiagnosticKind
from anomaly_lab.models.feature_view import pca_to_rgb
from anomaly_lab.models.introspect import build_tree, collect
from anomaly_lab.models.preprocessing import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    ColorMode,
    PreprocessingConfig,
    expand_planes,
    load_array,
    to_chw,
)
from anomaly_lab.schemas import API_MODEL_CONFIG

STATE_FILENAME = "patchcore.pt"

CHECKPOINT_FORMAT = 1
"""There is no format 0. This method owns its checkpoint from the first commit.

It carries the bank, the identity needed to rebuild the module, the fitted geometry and the
backbone fingerprint — but **not** the backbone weights, which are 260 MB of somebody
else's artifact and are refetched by timm on load. The fingerprint is what makes that safe
rather than merely small.
"""

BYTES_PER_FLOAT32 = 4

_SELECTION_LOG_EVERY = 250
"""How often the selection reports. At the default 10 000 iterations that is 40 updates —
often enough that the progress bar visibly moves, rare enough that the log stays readable."""

_PROJECTION_EPS = 0.9
"""anomalib's own `SparseRandomProjection` setting, and the reason it is named here rather
than passed inline: the projected dimension it implies (284 for D=1536 at N=100 000) is a
factor in every greedy iteration's cost, so it belongs where the cost is discussed."""


class LayerSet(StrEnum):
    """Which backbone layers form the patch embedding.

    A closed set rather than a free `list[str]`, for two reasons. The form is generated from
    this schema and renders an enum as a picker, where a list of strings would be a text box
    that fails inside timm several minutes later. And the choice is genuinely a small menu:
    the depth of the features is the trade — shallow layers localize better and generalize
    worse — and these four cover it.
    """

    LAYER1_LAYER2 = "layer1+layer2"
    LAYER2_LAYER3 = "layer2+layer3"
    LAYER3_LAYER4 = "layer3+layer4"
    LAYER2_LAYER3_LAYER4 = "layer2+layer3+layer4"

    @property
    def layers(self) -> tuple[str, ...]:
        return tuple(self.value.split("+"))


@dataclass(frozen=True)
class BankPlan:
    """What a fit is about to cost, computed before any of it is spent.

    Deliberately **torch-free and pure** — plain integers in, a plan out — the same split
    `introspect.build_tree` has from `introspect.collect`, and for the same reason: the
    arithmetic that decides whether this machine survives the run is tested in the CI job
    that installs without the `dl` extra.
    """

    images_available: int
    images_used: int
    patches_per_image: int
    patches_kept_per_image: int
    embedding_dim: int
    candidates_generated: int
    candidates_kept: int
    coreset_size: int

    @property
    def candidate_bytes(self) -> int:
        return self.candidates_kept * self.embedding_dim * BYTES_PER_FLOAT32

    @property
    def bank_bytes(self) -> int:
        return self.coreset_size * self.embedding_dim * BYTES_PER_FLOAT32

    @property
    def unbounded_candidates(self) -> int:
        """What a run with no caps at all would have held — the number the caps avoid."""
        return self.images_available * self.patches_per_image

    @property
    def images_dropped(self) -> int:
        return self.images_available - self.images_used

    @property
    def patches_dropped_per_image(self) -> int:
        return self.patches_per_image - self.patches_kept_per_image

    def describe(self) -> str:
        """One line for the job log, said before the work rather than after it."""
        return (
            f"memory bank plan: {self.images_used}/{self.images_available} images x "
            f"{self.patches_kept_per_image}/{self.patches_per_image} patches = "
            f"{self.candidates_kept:,} candidates ({self.candidate_bytes / 1e6:.0f} MB), "
            f"coreset {self.coreset_size:,} vectors ({self.bank_bytes / 1e6:.0f} MB); "
            f"uncapped this would have been {self.unbounded_candidates:,} candidates "
            f"({self.unbounded_candidates * self.embedding_dim * BYTES_PER_FLOAT32 / 1e9:.2f} GB)"
        )


def plan_bank(
    *,
    images_available: int,
    patches_per_image: int,
    embedding_dim: int,
    max_bank_images: int,
    max_candidate_vectors: int,
    coreset_ratio: float,
) -> BankPlan:
    """Resolve the two caps into one plan, without touching a file or a tensor.

    The caps compose in a fixed order, and it is the order that keeps coverage honest:
    **images first, then patches within each surviving image**. Dropping images to fit the
    vector cap would be the other way round and is worse, because patches inside one image
    are highly redundant — neighbouring positions overlap through the 3x3 pooling — while
    two images differ by whatever the process actually varies. So a bank sees as many
    *images* as it is allowed, at whatever patch density that leaves.
    """
    if patches_per_image < 1 or embedding_dim < 1:
        msg = (
            "plan_bank needs a real patch grid and embedding width; got "
            f"patches_per_image={patches_per_image}, embedding_dim={embedding_dim}"
        )
        raise ValueError(msg)
    if not 0.0 < coreset_ratio <= 1.0:
        msg = f"coreset_ratio must be in (0, 1]; got {coreset_ratio}"
        raise ValueError(msg)

    images_used = min(images_available, max_bank_images)
    # A cap below the image count cannot be met by thinning patches — one patch per image is
    # the floor — so it takes images off the end too. Degenerate, but a configuration the
    # form can produce, and silently overshooting the stated cap is the one thing it must
    # not do.
    images_used = min(images_used, max(1, max_candidate_vectors))

    if images_used * patches_per_image > max_candidate_vectors:
        patches_kept = max(1, max_candidate_vectors // images_used)
    else:
        patches_kept = patches_per_image

    candidates_kept = images_used * patches_kept
    # `int()` truncates, so a ratio that would select nothing selects one: an empty bank is
    # not a smaller bank, it is a model that cannot score.
    coreset_size = max(1, min(candidates_kept, int(candidates_kept * coreset_ratio)))

    return BankPlan(
        images_available=images_available,
        images_used=images_used,
        patches_per_image=patches_per_image,
        patches_kept_per_image=patches_kept,
        embedding_dim=embedding_dim,
        candidates_generated=images_used * patches_per_image,
        candidates_kept=candidates_kept,
        coreset_size=coreset_size,
    )


class PatchcoreConfig(BaseModel):
    """Hyperparameters. Every field here becomes a control on the experiment form."""

    model_config = API_MODEL_CONFIG

    backbone: str = Field(
        default="wide_resnet50_2",
        description=(
            "timm backbone the patch features come from. The paper uses wide_resnet50_2; "
            "resnet18 is roughly a quarter of the embedding width and a much cheaper "
            "ablation. Any timm model with named residual layers will resolve."
        ),
    )
    layer_set: LayerSet = Field(
        default=LayerSet.LAYER2_LAYER3,
        description=(
            "Which backbone layers are concatenated into each patch embedding. Shallower "
            "layers localize better and carry less semantics; the paper uses layer2+layer3."
        ),
    )
    pretrained_backbone: bool = Field(
        default=True,
        description=(
            "Use ImageNet-pretrained backbone weights. Off gives a seeded random backbone, "
            "which needs no download and is a real question rather than a broken run: "
            "patch nearest-neighbour matching does not require the features to be trained."
        ),
    )
    allow_downloads: bool = Field(
        default=True,
        description=(
            "Permit fetching the pretrained backbone weights. Turn off to make a missing "
            "backbone an error naming it instead of a silent reach for the network."
        ),
    )
    coreset_ratio: float = Field(
        default=0.1,
        gt=0.0,
        le=1.0,
        description=(
            "Fraction of the candidate pool kept in the memory bank. Both the fit time and "
            "the per-image inference cost are linear in the result."
        ),
    )
    num_neighbors: int = Field(
        default=9,
        ge=1,
        le=64,
        description=(
            "Neighbours used by the paper's score reweighting. 1 skips the reweighting "
            "entirely and takes the plain maximum patch distance."
        ),
    )
    max_bank_images: int = Field(
        default=512,
        ge=1,
        le=10_000,
        description=(
            "How many training normals are pushed through the backbone. Sampled evenly "
            "across the training set rather than taking the first N. At about 7 ms per "
            "image on MPS this bound is cheap; it exists so the pass stays linear-bounded."
        ),
    )
    max_candidate_vectors: int = Field(
        default=100_000,
        ge=16,
        le=2_000_000,
        description=(
            "Ceiling on the patch embeddings held before coreset selection. This is the "
            "cap that bounds memory and time: selection cost is quadratic in this number "
            "— about 25 s at 100000, 150 s at 250000 and half an hour at a full class."
        ),
    )
    blur_sigma: int = Field(
        default=4,
        ge=0,
        le=32,
        description=(
            "Gaussian smoothing applied to the upsampled patch scores, in pixels. The "
            "unblurred map is emitted as a diagnostic so the cost of this is visible."
        ),
    )
    feature_batch_size: int = Field(
        default=8,
        ge=1,
        le=64,
        description="Batch size for the feature-extraction pass. Inference is per image.",
    )
    seed: int = Field(
        default=0,
        description=(
            "Seeds the coreset's starting point, the random projection and — when "
            "pretrained_backbone is off — the backbone initialisation. Two runs with one "
            "seed are one experiment."
        ),
    )


@contextmanager
def _downloads_refused(backbone: str) -> Iterator[None]:
    """Make a missing backbone an error naming it, rather than a silent fetch.

    timm resolves pretrained weights through the HuggingFace hub, which has its own offline
    switch. Setting it is better than reimplementing the cache lookup — the trade this
    codebase makes everywhere else — and it turns a network reach into a local failure that
    is then translated into a sentence saying which knob to turn.
    """
    previous = os.environ.get("HF_HUB_OFFLINE")
    os.environ["HF_HUB_OFFLINE"] = "1"
    try:
        yield
    except Exception as exc:
        msg = (
            f"PatchCore needs pretrained weights for backbone {backbone!r} and "
            "allow_downloads is off, and they are not in the local cache. Turn "
            "allow_downloads back on for one run, or set pretrained_backbone off to use a "
            f"seeded random backbone. The underlying failure was: {exc}"
        )
        raise RuntimeError(msg) from exc
    finally:
        if previous is None:
            os.environ.pop("HF_HUB_OFFLINE", None)
        else:
            os.environ["HF_HUB_OFFLINE"] = previous


def greedy_select(
    features: Any,
    size: int,
    *,
    start: int,
    observe: Callable[[int, float], None] | None = None,
) -> Any:
    """k-center-greedy over already-projected features. Anomalib's rule, our loop.

    Each step takes the point furthest from everything chosen so far, which is
    `KCenterGreedy.select_coreset_idxs` exactly: update the running per-point minimum with
    the newest centre, take the argmax of that minimum, zero the point just taken so it
    cannot be taken twice, record it. Anomalib keeps its minima as `(n, 1)` and starts them
    at `None`; this keeps them as `(n,)` starting at infinity, which is the same function.

    `start` is a parameter rather than drawn here, for two reasons. The caller owns its
    randomness — a seed in this workbench has to mean something (M6) — and the equivalence
    test can then hand both implementations the same starting point, so what it compares is
    the greedy rule rather than two draws from a shared global stream.

    `observe` is called **once per iteration**, before the work. It is where cancellation
    and progress live, and it is the entire reason this function exists rather than a call
    to anomalib's: a selection that cannot be watched or stopped is a job that freezes.
    """
    import torch

    count = int(features.shape[0])
    if not 0 <= start < count:
        msg = f"coreset start index {start} is outside the candidate pool of {count}"
        raise ValueError(msg)
    size = max(1, min(size, count))

    minimum = torch.full((count,), float("inf"), dtype=features.dtype)
    selected = torch.empty(size, dtype=torch.long)
    index = start
    selected[0] = index

    for step in range(1, size):
        if observe is not None:
            observe(step, float(minimum.max()) if step > 1 else float("inf"))
        distances = torch.linalg.norm(features - features[index], ord=2, dim=1)
        minimum = torch.minimum(minimum, distances)
        index = int(torch.argmax(minimum))
        minimum[index] = 0.0
        selected[step] = index

    return selected


def _fingerprint(state: dict[str, Any]) -> str:
    """A stable digest of the backbone's weights.

    Sorted by key so it does not depend on `state_dict` ordering, and over the raw bytes so
    it does not depend on how they print. What it is *for* is stated in the module
    docstring: the weights are an input to the experiment, so a run has to be able to say
    that the ones it is loading are the ones it was fitted against.
    """
    digest = hashlib.sha256()
    for key in sorted(state):
        tensor = state[key]
        digest.update(key.encode("utf-8"))
        digest.update(np.ascontiguousarray(tensor.detach().cpu().numpy()).tobytes())
    return digest.hexdigest()


class PatchcoreAnomalibModel(AnomalyModel):
    """PatchCore, with anomalib supplying the arithmetic and this class supplying the pass."""

    title = "PatchCore (anomalib)"
    summary = (
        "A coreset memory bank of patch features from a frozen ImageNet backbone, scored "
        "by nearest-neighbour distance. Nothing is trained; the bank is the model."
    )

    def __init__(self, config: PatchcoreConfig) -> None:
        super().__init__(config)
        self.config = config
        self._model: Any = None
        self._plan: BankPlan | None = None
        self._grid: tuple[int, int] | None = None
        self._fingerprint: str | None = None
        self._selection_seconds: float = 0.0

    @classmethod
    def config_model(cls) -> type[BaseModel]:
        return PatchcoreConfig

    @classmethod
    def capabilities(cls) -> Capabilities:
        return Capabilities(
            requires_training=True,
            produces_anomaly_map=True,
            produces_diagnostics=True,
            # There are no steps to continue. PatchCore builds a bank and is either fitted
            # or not — `pixel_reference`'s situation, and exactly why `SupportsResume` is a
            # Protocol rather than two more abstract methods nobody can honestly implement.
            supports_resume=False,
            channel_aware=False,
            dataset_specific=False,
            portable_formats=[PortableFormat.ONNX],
            preferred_device=Device.MPS,
        )

    @classmethod
    def availability(cls) -> Availability:
        return module_available("anomalib", "dl", "PatchCore")

    # ---------------------------------------------------------------- construction

    def _build_model(self, device: str) -> Any:
        """Instantiate anomalib's `PatchcoreModel`, seeded, with our own map generator.

        The seed is set immediately before construction because an unpretrained backbone is
        initialised from torch's **global** stream — the bug M6 found in EfficientAD, where
        `seed` controlled the training order over different initial weights. Here it would
        be worse than noise: the fingerprint written at save time would never match the one
        computed at load time, and every reload would refuse a checkpoint that was fine.
        """
        import torch
        from anomalib.models.image.patchcore.anomaly_map import AnomalyMapGenerator
        from anomalib.models.image.patchcore.torch_model import PatchcoreModel

        torch.manual_seed(self.config.seed)
        layers = list(self.config.layer_set.layers)
        if self.config.pretrained_backbone and not self.config.allow_downloads:
            with _downloads_refused(self.config.backbone):
                model = PatchcoreModel(
                    layers=layers,
                    backbone=self.config.backbone,
                    pre_trained=True,
                    num_neighbors=self.config.num_neighbors,
                )
        else:
            model = PatchcoreModel(
                layers=layers,
                backbone=self.config.backbone,
                pre_trained=self.config.pretrained_backbone,
                num_neighbors=self.config.num_neighbors,
            )

        # anomalib fixes sigma at 4 in `PatchcoreModel.__init__`. Replacing the generator is
        # how `blur_sigma` becomes a field rather than a constant nobody can ablate.
        model.anomaly_map_generator = AnomalyMapGenerator(sigma=self.config.blur_sigma)
        return model.to(device).eval()

    def _normalized_batch(self, records: Sequence[ImageRecord], ctx: Any, device: str) -> Any:
        """Load a batch through the shared bridge, then standardize it for the backbone.

        Two separate acts, and the module docstring says why they must stay separate:
        `load_array` is the experiment's decision and is identical for every method, while
        the plane count and the ImageNet statistics belong to this backbone.

        The expansion is explicit rather than left to broadcasting. A `(B, 1, H, W)` batch
        minus a `(1, 3, 1, 1)` mean broadcasts to the same numbers this now computes, so
        grayscale runs were never wrong here — but they were right by accident, and an
        accident that produces correct output is the kind that survives until the shapes
        change.
        """
        import torch

        stacked = np.stack(
            [
                expand_planes(to_chw(load_array(record.path, ctx.preprocessing)), 3)
                for record in records
            ]
        )
        batch = torch.from_numpy(stacked).to(device)
        mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
        return (batch - mean) / std

    def _embed(self, model: Any, batch: Any) -> Any:
        """One batch of images to `(B, D, H, W)` patch embeddings, through anomalib's own path.

        `PatchcoreModel.forward` is deliberately not called. In training mode it appends to
        an internal `embedding_store` — unbounded, which is the thing this plugin exists to
        avoid — and in eval mode it needs a bank that does not exist yet. The three steps it
        would have taken are taken here instead, and each of them is still anomalib's.
        """
        import torch

        with torch.no_grad():
            features = model.feature_extractor(batch)
            pooled = {layer: model.feature_pooler(value) for layer, value in features.items()}
            return model.generate_embedding(pooled)

    # ---------------------------------------------------------------- training

    def fit(self, train: Sequence[ImageRecord], ctx: TrainContext) -> None:
        if not train:
            msg = "patchcore_anomalib was given no training images to build a bank from"
            raise RuntimeError(msg)

        device = ctx.device.value
        model = self._build_model(device)
        generator = np.random.default_rng(self.config.seed)

        chosen = evenly_spaced(len(train), self.config.max_bank_images)
        records = [train[index] for index in chosen]

        # The grid and the width are *read from a forward pass*, never derived from an
        # assumed stride. Every number in the plan below is a product of the three, and two
        # of them are properties of a backbone this config can change.
        ctx.progress(0.02, "measuring the patch grid")
        probe = self._embed(model, self._normalized_batch(records[:1], ctx, device))
        rows, cols = int(probe.shape[-2]), int(probe.shape[-1])
        dim = int(probe.shape[1])
        del probe

        plan = plan_bank(
            images_available=len(train),
            patches_per_image=rows * cols,
            embedding_dim=dim,
            max_bank_images=self.config.max_bank_images,
            max_candidate_vectors=self.config.max_candidate_vectors,
            coreset_ratio=self.config.coreset_ratio,
        )
        records = records[: plan.images_used]
        self._plan = plan
        self._grid = (rows, cols)
        ctx.log(f"patch grid {rows}x{cols}, embedding width {dim}")
        # Before the pass, not after it. A footprint discovered afterwards is a machine that
        # has already started swapping.
        ctx.log(plan.describe())
        if plan.images_dropped:
            ctx.log(
                f"{plan.images_dropped} of {plan.images_available} training images are not "
                f"in the bank (max_bank_images={self.config.max_bank_images}), sampled "
                "evenly across the set rather than taken from its front"
            )
        if plan.patches_dropped_per_image:
            ctx.log(
                f"{plan.patches_dropped_per_image} of {plan.patches_per_image} patches per "
                f"image are dropped (max_candidate_vectors="
                f"{self.config.max_candidate_vectors}), on a regular grid"
            )

        store, positions = self._extract(model, records, ctx, plan, device)
        ctx.raise_if_cancelled()

        self._emit_architecture(ctx, model, dim)
        self._emit_embedding_view(ctx, model, records[0], device, rows, cols)

        selected = self._select(store, plan, ctx, generator)
        bank = store[selected].contiguous()
        model.memory_bank = bank.to(device)
        self._model = model
        self._fingerprint = _fingerprint(model.feature_extractor.state_dict())

        ctx.metric("bank_vectors", float(bank.shape[0]))
        ctx.metric("bank_megabytes", float(bank.numel() * BYTES_PER_FLOAT32 / 1e6))
        ctx.log(
            f"memory bank: {bank.shape[0]:,} vectors of {dim} "
            f"({bank.numel() * BYTES_PER_FLOAT32 / 1e6:.0f} MB), selected in "
            f"{self._selection_seconds:.0f}s"
        )

        self._emit_bank_table(ctx, plan, device)
        self._emit_coverage(ctx, positions[selected], rows, cols)

    def _extract(
        self,
        model: Any,
        records: Sequence[ImageRecord],
        ctx: TrainContext,
        plan: BankPlan,
        device: str,
    ) -> tuple[Any, Any]:
        """Fill the candidate store, bounded, and remember where each vector came from.

        The store is preallocated at the planned size and lives on the **CPU**: it is the
        largest thing this method holds, the selection that consumes it runs on the CPU, and
        keeping it off the accelerator leaves the accelerator for the backbone.

        Patches are thinned with `evenly_spaced` over the flattened grid, which lands on a
        regular spatial lattice — uniform coverage of the frame, which is what a bank wants,
        and deterministic, so a seed is not needed for it.
        """
        import torch

        keep = evenly_spaced(plan.patches_per_image, plan.patches_kept_per_image)
        keep_index = torch.tensor(keep, dtype=torch.long)
        # Only the width is needed, to turn a flat patch index back into a grid position.
        cols = self._grid[1] if self._grid else 1
        # Where each kept patch sits on the grid, so the coverage diagnostic can be a
        # picture rather than a count. Same order as `keep`, reused for every image.
        grid_positions = torch.stack([keep_index // cols, keep_index % cols], dim=1)

        store = torch.empty(plan.candidates_kept, plan.embedding_dim, dtype=torch.float32)
        positions = torch.empty(plan.candidates_kept, 2, dtype=torch.long)

        started = time.perf_counter()
        cursor = 0
        for start in range(0, len(records), self.config.feature_batch_size):
            ctx.raise_if_cancelled()
            chunk = records[start : start + self.config.feature_batch_size]
            embedding = self._embed(model, self._normalized_batch(chunk, ctx, device))
            # `(B, D, H, W)` to one row per patch, then thinned. Reshaping before selecting
            # keeps this identical to `PatchcoreModel.reshape_embedding`, which is what the
            # bank is compared against at inference.
            flat = embedding.permute(0, 2, 3, 1).reshape(len(chunk), -1, plan.embedding_dim)
            thinned = flat[:, keep_index, :].reshape(-1, plan.embedding_dim).cpu()

            end = cursor + thinned.shape[0]
            store[cursor:end] = thinned
            positions[cursor:end] = grid_positions.repeat(len(chunk), 1)
            cursor = end

            ctx.progress(
                0.05 + 0.45 * (start + len(chunk)) / len(records),
                f"embedded {start + len(chunk)}/{len(records)} images",
            )

        elapsed = time.perf_counter() - started
        ctx.log(
            f"embedded {len(records)} images in {elapsed:.0f}s "
            f"({elapsed / max(len(records), 1) * 1000:.0f} ms/image on {device}); "
            f"{cursor:,} candidate vectors held on cpu"
        )
        return store[:cursor], positions[:cursor]

    def _select(
        self,
        store: Any,
        plan: BankPlan,
        ctx: TrainContext,
        generator: np.random.Generator,
    ) -> Any:
        """k-center-greedy, ours, over anomalib's projection and anomalib's distance.

        The rule is exactly `KCenterGreedy.select_coreset_idxs`: project, pick a random
        start, then repeatedly take the point furthest from everything chosen so far. The
        iteration order is identical — update the running minimum with the current centre,
        argmax, zero the new pick, append — which is what makes the equivalence test able to
        assert the same indices rather than merely a similar bank.

        What is different is that this one can be watched and stopped. Anomalib's returns
        when it is done; at defaults that is 10 000 iterations and about 25 seconds, and at
        a raised cap it is half an hour of a job that reports nothing and ignores cancel.
        """
        import torch
        from anomalib.models.components.dimensionality_reduction import SparseRandomProjection

        ctx.progress(0.52, "projecting the candidate pool")
        # **Two** random streams, and missing either one makes `seed` a lie.
        #
        # `SparseRandomProjection` draws its non-zero pattern through scikit-learn's
        # `sample_without_replacement`, whose `random_state` defaults to `None` — numpy's
        # global RNG, which `torch.manual_seed` does not touch — and its signs and counts
        # from torch's. anomalib constructs it with no `random_state`, so **its own coreset
        # is not reproducible**: two runs of one configuration select different banks, and
        # nothing says so. That is M6's finding again in a different library, and it is the
        # thing that would have put noise under every number this method produces.
        torch.manual_seed(self.config.seed)
        projector = SparseRandomProjection(eps=_PROJECTION_EPS, random_state=self.config.seed)
        projector.fit(store)
        features = projector.transform(store)
        ctx.log(
            f"projected {plan.candidates_kept:,}x{plan.embedding_dim} to "
            f"{int(features.shape[1])} dimensions for the selection (eps={_PROJECTION_EPS})"
        )

        size = plan.coreset_size

        def observe(step: int, furthest: float) -> None:
            # Every iteration, which is the whole point of owning this loop. At the measured
            # 2.46 ms per iteration a cancel lands in single-digit milliseconds.
            ctx.raise_if_cancelled()
            if step % _SELECTION_LOG_EVERY == 0 or step == size - 1:
                ctx.metric("coreset_max_distance", furthest, step=step)
                ctx.progress(
                    0.55 + 0.35 * (step + 1) / size,
                    f"selected {step + 1:,}/{size:,} bank vectors",
                )

        started = time.perf_counter()
        selected = greedy_select(
            features,
            size,
            start=int(generator.integers(features.shape[0])),
            observe=observe,
        )
        self._selection_seconds = time.perf_counter() - started
        ctx.log(
            f"coreset selection took {self._selection_seconds:.0f}s for {size:,} vectors "
            f"({self._selection_seconds / max(size, 1) * 1000:.2f} ms/iteration on cpu — "
            "measured 3x faster there than on mps, where the per-iteration dispatch "
            "dominates a loop this small)"
        )
        return selected

    # ---------------------------------------------------------------- diagnostics

    def _emit_architecture(self, ctx: TrainContext, model: Any, dim: int) -> None:
        """The real module hierarchy, from a dry forward pass through the shared helper.

        **The edge list stays empty, deliberately.** `introspect` records nodes finely and
        leaves wiring to the caller precisely so a view never draws what it did not measure,
        and PatchCore has nothing branch-level to draw: it is one backbone forward followed
        by a nearest-neighbour search, and the search is not a module. EfficientAD's three
        branches produced two honest edges; here two would have to be invented.

        What the tab does show, and what is easy to get wrong from the outside, is that
        **the backbone is pruned to the selected layers rather than merely read from**.
        `TimmFeatureExtractor` builds with `features_only`, so under `layer2+layer3` there
        is no `layer4` at all — every node in the tree executes, and `total_parameters` is
        the real inference cost rather than the published size of the named architecture.
        """
        if not ctx.diagnostics.enabled:
            return

        import torch

        probe = torch.zeros(
            1, 3, ctx.preprocessing.height, ctx.preprocessing.width, device=ctx.device.value
        )
        records = collect(model.feature_extractor, probe, prefix="feature_extractor.")
        payload = build_tree(records)
        payload["total_parameters"] = int(sum(p.numel() for p in model.parameters()))

        ctx.emit_diagnostic(
            "architecture",
            "Model architecture",
            DiagnosticKind.GRAPH,
            payload,
            description=(
                f"The frozen backbone, with every module's real shapes from a dry forward "
                f"pass. Patch embeddings are {dim} wide. The backbone is pruned to the "
                "selected layers, so the parameter count is what inference actually costs "
                "rather than the published size of the named architecture. The coreset "
                "search is not a module and is not drawn."
            ),
        )

    def _emit_embedding_view(
        self,
        ctx: TrainContext,
        model: Any,
        record: ImageRecord,
        device: str,
        rows: int,
        cols: int,
    ) -> None:
        """What the backbone sees on one training image — PatchCore's equivalent of the
        teacher inspector, and the same two kinds, so it renders on the same screen."""
        if not ctx.diagnostics.enabled:
            return

        embedding = self._embed(model, self._normalized_batch([record], ctx, device))
        features = embedding[0].detach().cpu().numpy().astype(np.float32)

        ctx.emit_diagnostic(
            "embedding_pca",
            "Patch embedding (PCA composite)",
            DiagnosticKind.IMAGE,
            pca_to_rgb(features),
            description=(
                f"Three leading principal components of the {features.shape[0]} concatenated "
                f"backbone channels, on the {rows}x{cols} patch grid. False colour."
            ),
        )
        ctx.emit_diagnostic(
            "embedding_magnitude",
            "Patch embedding magnitude",
            DiagnosticKind.MAP,
            np.linalg.norm(features, axis=0).astype(np.float32),
            description="Where the backbone responds most strongly on a normal image.",
        )

    def _emit_bank_table(self, ctx: TrainContext, plan: BankPlan, device: str) -> None:
        """The memory-bank configuration, written down where a reader will find it.

        This table *is* M7's first exit criterion. A bank that fits is only half the claim;
        the other half is that a run says which images, how many patches from each, what was
        dropped by which cap, and what the whole thing cost.
        """
        rows = [
            ["backbone", self.config.backbone],
            ["backbone weights", self._weights_identity()],
            ["layers", "+".join(self.config.layer_set.layers)],
            ["patch grid", f"{self._grid[0]}x{self._grid[1]}" if self._grid else "-"],
            ["embedding width", str(plan.embedding_dim)],
            ["images used", f"{plan.images_used} of {plan.images_available}"],
            ["patches per image", f"{plan.patches_kept_per_image} of {plan.patches_per_image}"],
            ["candidate vectors", f"{plan.candidates_kept:,}"],
            ["candidate pool", f"{plan.candidate_bytes / 1e6:.0f} MB"],
            ["coreset ratio", f"{self.config.coreset_ratio:g}"],
            ["bank vectors", f"{plan.coreset_size:,}"],
            ["bank size", f"{plan.bank_bytes / 1e6:.0f} MB"],
            ["selection time", f"{self._selection_seconds:.0f} s on cpu"],
            ["embedding device", device],
            ["uncapped pool would have been", f"{plan.unbounded_candidates:,} vectors"],
        ]
        ctx.emit_diagnostic(
            "memory_bank",
            "Memory bank",
            DiagnosticKind.TABLE,
            {"columns": ["quantity", "value"], "rows": rows},
            description=(
                "What the bank is made of and what it cost. The last row is what the same "
                "run would have held with both caps removed — the selection is quadratic "
                "in it, so it is the number the caps exist for."
            ),
        )

    def _weights_identity(self) -> str:
        """Which pretrained weights this is, as far as timm will say.

        Best effort on purpose: the identity is worth recording and is not worth failing a
        run over, so an unfamiliar backbone reports what it can rather than raising.
        """
        if not self.config.pretrained_backbone:
            return f"random, seeded {self.config.seed}"
        extractor = getattr(self._model, "feature_extractor", None) if self._model else None
        inner = getattr(extractor, "feature_extractor", None)
        config = getattr(inner, "pretrained_cfg", None)
        if isinstance(config, dict):
            tag = config.get("hf_hub_id") or config.get("url") or config.get("tag")
            if tag:
                return str(tag)
        return "timm default pretrained tag"

    def _emit_coverage(self, ctx: TrainContext, positions: Any, rows: int, cols: int) -> None:
        """Where on the frame the bank spent its budget.

        A count per patch position, which answers a question the metrics cannot: a coreset
        is chosen for diversity in feature space, and whether that concentrates on a few
        textured regions or spreads over the frame is a real property of the fit. Falls out
        of bookkeeping the extraction pass was already doing.
        """
        if not ctx.diagnostics.enabled or rows < 1 or cols < 1:
            return

        counts = np.zeros((rows, cols), dtype=np.float32)
        for row, col in positions.tolist():
            counts[row, col] += 1.0

        ctx.emit_diagnostic(
            "coreset_coverage",
            "Coreset coverage",
            DiagnosticKind.MAP,
            counts,
            description=(
                "How many bank vectors were selected from each patch position, over every "
                "contributing image. Bright regions are where normal appearance varies "
                "most, and so where the bank spends its budget."
            ),
        )

    # ---------------------------------------------------------------- inference

    def predict(self, images: Sequence[ImageRecord], ctx: InferContext) -> list[Prediction]:
        import torch
        import torch.nn.functional as functional

        if self._model is None:
            msg = "patchcore_anomalib was asked to predict before it was fitted or loaded"
            raise RuntimeError(msg)

        device = ctx.device.value
        model = self._model.to(device)
        model.eval()
        if model.memory_bank.numel() == 0:
            msg = "patchcore_anomalib has an empty memory bank; the fit did not complete"
            raise RuntimeError(msg)

        size = (ctx.preprocessing.height, ctx.preprocessing.width)
        predictions: list[Prediction] = []

        for index, record in enumerate(images):
            ctx.raise_if_cancelled()
            started = time.perf_counter()

            embedding = self._embed(model, self._normalized_batch([record], ctx, device))
            rows, cols = int(embedding.shape[-2]), int(embedding.shape[-1])
            flat = model.reshape_embedding(embedding)

            with torch.no_grad():
                # The eval path taken step by step rather than through `forward`, so the
                # patch scores are available *once*. Calling `forward` and then recomputing
                # them for the diagnostic is the double-compute that costs the EfficientAD
                # wrapper 46.7 ms against our 25.8 — the same mistake, one method later.
                patch_scores, locations = model.nearest_neighbors(flat, n_neighbors=1)
                patch_scores = patch_scores.reshape((1, -1))
                locations = locations.reshape((1, -1))
                score = float(model.compute_anomaly_score(patch_scores, locations, flat)[0])
                grid = patch_scores.reshape((1, 1, rows, cols))
                anomaly_map = model.anomaly_map_generator(grid, size)
                unblurred = functional.interpolate(grid, size=size)

            values = anomaly_map[0].squeeze().detach().cpu().numpy().astype(np.float32)
            map_path = ctx.write_map(record.image_id, values)
            elapsed_ms = (time.perf_counter() - started) * 1000.0

            ctx.emit_diagnostic(
                "patch_scores",
                "Patch distances",
                DiagnosticKind.MAP,
                grid[0, 0].detach().cpu().numpy().astype(np.float32),
                image_id=record.image_id,
                description=(
                    "Distance from each patch to its nearest bank vector, at the backbone's "
                    "own resolution. This is what PatchCore actually computed; the anomaly "
                    "map is this upsampled and smoothed."
                ),
            )
            ctx.emit_diagnostic(
                "map_unblurred",
                "Anomaly map before smoothing",
                DiagnosticKind.MAP,
                unblurred[0, 0].detach().cpu().numpy().astype(np.float32),
                image_id=record.image_id,
                description=(
                    f"The upsampled patch distances without the Gaussian blur "
                    f"(blur_sigma={self.config.blur_sigma}), so the smoothing's effect is "
                    "visible rather than assumed."
                ),
            )

            predictions.append(
                Prediction(
                    image_id=record.image_id,
                    score=score,
                    anomaly_map=map_path,
                    inference_ms=elapsed_ms,
                )
            )
            ctx.progress((index + 1) / len(images), f"scored {index + 1}/{len(images)}")

        return predictions

    # -------------------------------------------------------------- deployment

    def _portable_module(self, preprocessing: PreprocessingConfig) -> Any:
        """Build the static-shape inference graph from anomalib's fitted components."""
        import torch

        if self._model is None or self._model.memory_bank.numel() == 0:
            raise RuntimeError("patchcore_anomalib has no fitted memory bank to export")
        model = self._model.to("cpu").eval()
        size = (preprocessing.height, preprocessing.width)

        # In the torch-free install ``torch`` is intentionally absent and mypy resolves
        # this function-local lazy import as Any. Runtime export still requires the dl
        # extra; the ignore preserves that boundary without importing torch at registry
        # load time.
        class PortablePatchcore(torch.nn.Module):  # type: ignore[misc,unused-ignore]
            def __init__(self) -> None:
                super().__init__()
                self.model = model
                self.register_buffer("input_mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
                self.register_buffer("input_std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

            def forward(self, image: Any) -> tuple[Any, Any]:
                if image.shape[1] == 1:
                    image = image.expand(-1, 3, -1, -1)
                batch = (image - self.input_mean) / self.input_std
                features = self.model.feature_extractor(batch)
                pooled = {
                    layer: self.model.feature_pooler(value) for layer, value in features.items()
                }
                embedding = self.model.generate_embedding(pooled)
                rows, cols = embedding.shape[-2:]
                flat = self.model.reshape_embedding(embedding)
                patch_scores, locations = self.model.nearest_neighbors(flat, n_neighbors=1)
                patch_scores = patch_scores.reshape((1, -1))
                locations = locations.reshape((1, -1))
                score = self.model.compute_anomaly_score(patch_scores, locations, flat)
                grid = patch_scores.reshape((1, 1, rows, cols))
                anomaly_map = self.model.anomaly_map_generator(grid, size)
                return anomaly_map, score

        return PortablePatchcore().eval()

    def export_onnx(
        self,
        destination: Path,
        preprocessing: PreprocessingConfig,
    ) -> OnnxGraphContract:
        """Export feature extraction, bank lookup, map generation and paper score."""
        import torch

        module = self._portable_module(preprocessing)
        fixture = torch.zeros(
            1,
            preprocessing.channels,
            preprocessing.height,
            preprocessing.width,
            dtype=torch.float32,
        )
        input_name = "image"
        map_name = "anomaly_map"
        score_name = "score"
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.onnx.export(
            module,
            (fixture,),
            destination,
            input_names=[input_name],
            output_names=[map_name, score_name],
            opset_version=18,
            dynamo=False,
        )
        return OnnxGraphContract(
            opset=18,
            input_name=input_name,
            output_name=map_name,
            score=TensorScore(tensor=ScalarTensorSpec(name=score_name, dtype=TensorDtype.FLOAT32)),
            absolute_tolerance=1e-4,
            relative_tolerance=1e-4,
        )

    def portable_reference(self, input_nchw: np.ndarray) -> tuple[np.ndarray, float]:
        """Run the exact fitted deployment path on an already-prepared tensor."""
        import torch

        preprocessing = PreprocessingConfig(
            width=int(input_nchw.shape[3]),
            height=int(input_nchw.shape[2]),
            color=ColorMode.GRAYSCALE if input_nchw.shape[1] == 1 else ColorMode.RGB,
        )
        with torch.no_grad():
            anomaly_map, score = self._portable_module(preprocessing)(torch.from_numpy(input_nchw))
        return anomaly_map[0, 0].numpy().astype(np.float32), float(score[0])

    # ---------------------------------------------------------------- persistence

    def save(self, artifact_dir: Path) -> None:
        """Write the bank and the identity of everything it was built from.

        The backbone's 260 MB are deliberately absent. They are somebody else's artifact,
        timm refetches them from a cache that is already on this machine, and storing a copy
        per experiment would multiply the disk cost of a comparison by the number of runs in
        it. What is stored instead is the fingerprint, which is what turns "the weights might
        have changed" from an unanswerable worry into a check.
        """
        import torch

        if self._model is None or self._plan is None:
            msg = "patchcore_anomalib has nothing to save; it was never fitted"
            raise RuntimeError(msg)

        payload: dict[str, Any] = {
            "format": CHECKPOINT_FORMAT,
            "memory_bank": self._model.memory_bank.detach().cpu(),
            "backbone": self.config.backbone,
            "layers": list(self.config.layer_set.layers),
            "num_neighbors": self.config.num_neighbors,
            "blur_sigma": self.config.blur_sigma,
            "pretrained_backbone": self.config.pretrained_backbone,
            "seed": self.config.seed,
            "backbone_fingerprint": self._fingerprint,
            "grid": list(self._grid) if self._grid else None,
            "embedding_dim": self._plan.embedding_dim,
            "coreset_size": self._plan.coreset_size,
            "selection_seconds": self._selection_seconds,
        }
        torch.save(payload, artifact_dir / STATE_FILENAME)

    def load(self, artifact_dir: Path) -> None:
        """Rebuild the module, restore the bank, and check the weights are the same ones.

        The fingerprint mismatch is refused **by name**, because the alternative is the
        failure `methods.md` warns about for the EfficientAD teacher: a backbone that
        changed upstream loads without complaint and produces a plausible, wrong number that
        nothing distinguishes from the run it is being compared against.
        """
        import torch

        stored = torch.load(artifact_dir / STATE_FILENAME, map_location="cpu", weights_only=False)
        model = self._build_model("cpu")

        expected = stored.get("backbone_fingerprint")
        actual = _fingerprint(model.feature_extractor.state_dict())
        if expected is not None and expected != actual:
            msg = (
                f"the backbone weights for {stored.get('backbone', self.config.backbone)!r} "
                "are not the ones this experiment was fitted against: the checkpoint records "
                f"{expected[:12]} and the weights now resolving are {actual[:12]}. The memory "
                "bank was selected in the old feature space and distances against the new one "
                "are not comparable. Refit the experiment, or restore the original weights."
            )
            raise RuntimeError(msg)

        model.memory_bank = stored["memory_bank"]
        self._model = model
        self._fingerprint = actual
        grid = stored.get("grid")
        self._grid = (int(grid[0]), int(grid[1])) if grid else None
        self._selection_seconds = float(stored.get("selection_seconds", 0.0))
