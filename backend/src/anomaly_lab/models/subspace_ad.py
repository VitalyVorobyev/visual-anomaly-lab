"""SubspaceAD: what normal patches span, and how far a patch falls outside it.

Training-free in the sense that matters — **nothing is optimized**. The encoder is frozen,
the fit is one covariance and one eigendecomposition, and a "fitted model" is a mean vector
plus an orthonormal basis. On a handful of normal images it finishes in seconds, and there
is no loss curve to watch. That is the point: this is the strong few-shot baseline a trained
method has to beat, and it costs a minute to find out.

**Three numbers and a window.** Patch tokens are mean-pooled over a band of transformer
blocks; PCA is fitted to the normals; a patch scores the squared length of the part of
itself that the leading subspace cannot reconstruct. `variance` (the paper's tau) decides
how much of the spectrum counts as normal, `tail_fraction` (rho) decides how much of the map
the image score averages, and `layers` decides which blocks are pooled.

**The window is a fraction of the encoder's depth, not a count of layers.** That is the one
finding this plugin exists to carry. "Layers 22-28 of 40" is two instructions in one
sentence, and they coincide only on a forty-block encoder; measured at twelve and
twenty-four blocks, the fixed count loses 6.5 points on ViT-S, 1.5 on ViT-B and nothing on
ViT-L, so a method that hard-codes seven layers is silently choosing a different window on
every backbone it is offered — and choosing worst on the small fast encoder a user reaches
for first. `dino_backbone.LayerBand` is where that is expressed; the default here is the
window that won at both depths.

**The arithmetic is shared with the sweep that chose these defaults** (ADR-0038).
`subspace` and `score_map` hold the covariance, the residual, the TVaR and the pixel map,
and the campaign in `backend/research/` imports the same two modules. A number the campaign
measured and a number a run inside the workbench measures therefore differ by the
experiment, not by a second implementation of the same equations. What this module adds is
the plugin: channels, artifacts, diagnostics, and a bound on how much of the training set a
fit is allowed to look at.

**Every default is a measured verdict, and `docs/measurements.md` records the protocol.**
The one axis that is *not* the campaign's is `max_fit_images`: the sweep is a few-shot
protocol and measured one, two and four normals, while a workbench experiment usually has
hundreds. More normals can only improve a covariance's conditioning, so the cap here is
about cost, not about accuracy.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
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
    Prediction,
    TrainContext,
    evenly_spaced,
    module_available,
)
from anomaly_lab.models.diagnostics import DiagnosticKind
from anomaly_lab.models.dino_backbone import (
    BACKBONES,
    DinoBackbone,
    LayerWindow,
    backbone_fingerprint,
    load_backbone,
    native_frame,
    patch_grid,
    patch_multiple,
    validate_prepared_size,
)
from anomaly_lab.models.preprocessing import (
    PreprocessingConfig,
    expand_planes,
    load_array,
    to_chw,
)
from anomaly_lab.models.score_map import (
    pixel_map,
    resample_operator,
    separable_operator,
    upsample_bilinear,
)
from anomaly_lab.models.subspace import (
    CovarianceAccumulator,
    RotationFill,
    SubspaceFit,
    fit_subspace,
    residual_basis,
    rotations,
    tail_value_at_risk,
    valid_patches,
)
from anomaly_lab.schemas import API_MODEL_CONFIG

STATE_FILENAME = "subspace_ad.npz"
MANIFEST_FILENAME = "subspace_ad.json"
CHECKPOINT_FORMAT = 1

UNASSIGNED = ""
"""Key for an image belonging to no channel, so a single-view dataset is a one-entry
mapping rather than a branch. The same convention `pixel_reference` uses."""

PORTABLE_OPSET = 18

PORTABLE_TOLERANCE = 1e-4
"""What the export job holds the graph to on its fixture — `atol = rtol` on the map, `atol` on
the score. The bound the export-parity gate predeclared for a deep frozen backbone
(docs/measurements.md). The graph widens to float64 where the Python path does, so the
residual's cancellation is not what spends it."""

BATCH_FRAMES = 4
"""Frames per forward pass. The campaign's figure, kept because a fit's rotated copies and
a run's test images are the same shape of work, and a batch that fits one fits the other."""


class SubspaceAdConfig(BaseModel):
    """Hyperparameters. Every field here becomes a control on the experiment form."""

    model_config = API_MODEL_CONFIG

    backbone: DinoBackbone = Field(
        json_schema_extra={"x-primary": True},
        default=DinoBackbone.DINOV2_VIT_L14,
        description=(
            "Frozen encoder the patch features come from. ViT-L is the default because it "
            "is never worse and it is ungated Apache-2.0 DINOv2 — but its margin is "
            "benchmark-dependent: decisive on VisA (+0.025 image AUROC over ViT-B, winning "
            "11 categories of 12) and nothing at all on MVTec (-0.0002, a tie), for roughly "
            "twice the compute. Try ViT-B first on data that resembles MVTec, and while an "
            "experiment is still being shaped. "
            "The DINOv3 entries are licence-gated: access must be requested from Meta on "
            "Hugging Face and an approved HF_TOKEN must already be in the environment."
        ),
    )
    layers: LayerWindow = Field(
        json_schema_extra={"x-primary": True},
        default=LayerWindow.UPPER_HALF,
        description=(
            "Which band of transformer blocks the patch features are pooled over, as a "
            "position in the encoder's depth rather than a count of layers. 'upper_half' "
            "won at both depths measured and won by more on the deeper one; 'last' — "
            "pooling the final block alone, which is what most frozen-feature methods do — "
            "was the worst of nine windows on a 24-block encoder. 'mid_band' and 'mid7' are "
            "the two readings of the paper's own 'layers 22-28 of 40'."
        ),
    )
    variance: float = Field(
        json_schema_extra={"x-primary": True},
        default=0.99,
        gt=0.0,
        le=1.0,
        description=(
            "How much of the fitted spectrum counts as normal: the subspace keeps the "
            "leading components carrying this fraction of the total variance. 1.0 keeps "
            "every direction, which makes the residual a number minus itself and destroys "
            "the signal entirely — it is a real setting that measures that collapse, not a "
            "safe maximum."
        ),
    )
    tail_fraction: float = Field(
        json_schema_extra={"x-primary": True},
        default=0.002,
        gt=0.0,
        le=1.0,
        description=(
            "The image score is the mean of the top fraction of patch scores. Small values "
            "find a small defect that a mean over the whole map would drown; large values "
            "are steadier on a diffuse one. The optimum fell with encoder capacity — 0.01 "
            "at ViT-S, 0.005 at ViT-B, 0.002 at ViT-L — so it wants revisiting with the "
            "backbone. At least one patch is always taken."
        ),
    )
    rotations: int = Field(
        default=30,
        ge=0,
        le=256,
        description=(
            "Random rotations added per training image, which is how a few-shot fit gets "
            "enough patches for a covariance. Set 0 for a part whose orientation carries "
            "meaning — the paper excludes one MVTec category on exactly this ground — "
            "because a rotated copy of it is not a normal example."
        ),
    )
    rotation_fill: RotationFill = Field(
        default=RotationFill.ZEROS,
        description=(
            "What the corners a rotation invents contribute. 'zeros' lets the black corners "
            "into the subspace as a direction of normal variation no test image will show; "
            "'masked' excludes every patch not backed entirely by real pixels. The sweep "
            "held this at 'zeros', which is what a literal reading of the paper does, so "
            "'masked' is the option that has not been measured."
        ),
    )
    max_fit_images: int = Field(
        default=16,
        ge=1,
        le=512,
        description=(
            "How many training images the fit looks at, sampled evenly across the set "
            "rather than taken from its front. Each one costs (1 + rotations) forward "
            "passes, which is the whole cost of a fit; the covariance itself is free."
        ),
    )
    smoothing_sigma: float = Field(
        default=4.0,
        ge=0.0,
        le=32.0,
        description=(
            "Gaussian blur applied to the anomaly map after it is upsampled to the prepared "
            "frame, in pixels. The paper's value. Zero leaves the upsampled map unsmoothed, "
            "which is the honest way to see what the encoder actually resolved. It cannot "
            "move the image score, which is taken on the patch grid before any of this."
        ),
    )
    pretrained_backbone: bool = Field(
        default=True,
        description=(
            "Use the published self-supervised weights. Off gives a seeded random ViT, "
            "which needs no download and is still a real question rather than a broken "
            "run: a subspace of normal appearance needs *some* feature space. This is what "
            "the hermetic tests use."
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
    seed: int = Field(
        default=0,
        ge=0,
        description=(
            "Seeds the rotation angles and, when pretrained_backbone is off, the encoder's "
            "own initialization. Two fits of one split with this fixed are identical."
        ),
    )


@dataclass
class _Fitted:
    """One channel's normal subspace, and what it was built from."""

    fit: SubspaceFit
    images: int

    def rank_for(self, variance: float) -> int:
        return self.fit.rank_for(variance)


def _frame(record: ImageRecord, preprocessing: PreprocessingConfig) -> np.ndarray:
    """One prepared image as `(H, W, 3)` float32 in `[0, 1]`.

    Through `load_array`, because preprocessing is a property of the experiment and a method
    that decodes its own pixels makes every comparison against it partly a measurement of
    its resize. The plane expansion is on this side of that seam for `expand_planes`' own
    reason: a three-channel first layer applied to a grey image is not a different image.
    """
    planes = expand_planes(to_chw(load_array(record.path, preprocessing)), 3)
    return np.ascontiguousarray(planes.transpose(1, 2, 0))


def _batched(items: Sequence[Any], size: int) -> Iterator[Sequence[Any]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


class SubspaceAdModel(AnomalyModel):
    """A PCA of normal patch appearance, and the residual against it."""

    title = "SubspaceAD (frozen encoder)"
    summary = (
        "PCA of frozen DINO patch features over a few normal images; a patch scores the "
        "part of itself the normal subspace cannot reconstruct. Nothing is trained."
    )

    def __init__(self, config: SubspaceAdConfig) -> None:
        super().__init__(config)
        self.config = config
        self._fits: dict[str, _Fitted] = {}
        self._grid: tuple[int, int] | None = None
        self._encoder: Any | None = None
        self._stats: tuple[tuple[float, ...], tuple[float, ...]] | None = None
        self._fingerprint: str | None = None
        self._cache_dir: Path | None = None

    @classmethod
    def config_model(cls) -> type[BaseModel]:
        return SubspaceAdConfig

    @classmethod
    def native_size(cls, config: BaseModel) -> tuple[int, int]:
        """672 px square: the shipped defaults are the sweep's leading arm, ViT-L/14 at 672 px.

        16 divides 672 too (docs/measurements.md).
        """
        if not isinstance(config, SubspaceAdConfig):
            raise TypeError(f"expected SubspaceAdConfig, got {type(config).__name__}")
        return native_frame(config.backbone, 672)

    @classmethod
    def size_multiple(cls, config: BaseModel) -> int:
        if not isinstance(config, SubspaceAdConfig):
            raise TypeError(f"expected SubspaceAdConfig, got {type(config).__name__}")
        return patch_multiple(config.backbone)

    @classmethod
    def check_input(cls, config: BaseModel, preprocessing: PreprocessingConfig) -> None:
        if not isinstance(config, SubspaceAdConfig):
            raise TypeError(f"expected SubspaceAdConfig, got {type(config).__name__}")
        validate_prepared_size(config.backbone, preprocessing.width, preprocessing.height)

    @classmethod
    def capabilities(cls) -> Capabilities:
        return Capabilities(
            requires_training=True,
            produces_anomaly_map=True,
            produces_diagnostics=True,
            # One subspace per channel. A grouped multi-view dataset holds views that do not
            # look alike at all — a bright-field and a dark-field frame of one part share no
            # normal appearance — and pooling them into one covariance would fit a subspace
            # spanning both, against which neither is far from normal.
            channel_aware=True,
            supports_resume=False,
            # The plugin writes a graph (`export_onnx`) and its parity reference, but a format
            # is declared only once `scripts/export-parity-gate.py` has passed on a real fit
            # and its verdict is in docs/measurements.md. Until then the export job refuses.
            portable_formats=[],
            preferred_device=Device.MPS,
        )

    @classmethod
    def availability(cls) -> Availability:
        return module_available("torch", "dl", "SubspaceAD's frozen encoder")

    # ------------------------------------------------------------------ encoder

    def _channel(self, record: ImageRecord) -> str:
        return record.channel or UNASSIGNED

    def _build_encoder(self, device: str, cache_dir: Path) -> Any:
        """Resolve the frozen encoder through the shared table, seeded before construction."""
        encoder = load_backbone(
            self.config.backbone,
            pretrained=self.config.pretrained_backbone,
            allow_downloads=self.config.allow_downloads,
            cache_dir=cache_dir,
            seed=self.config.seed,
            method="subspace_ad",
        )
        self._fingerprint = backbone_fingerprint(encoder)
        self._cache_dir = cache_dir
        # The checkpoint's own statistics rather than a hard-coded ImageNet triple.
        # Standardizing pixels for a pretrained network is a property of *that network*, and
        # a future entry in `BACKBONES` trained under different statistics would otherwise
        # be fed the wrong ones in silence.
        config = encoder.pretrained_cfg
        self._stats = (tuple(config["mean"]), tuple(config["std"]))
        return encoder.to(device)

    def _encode(self, frames: Sequence[np.ndarray], device: str) -> np.ndarray:
        """`(B, P, D)` float32 patch features, mean-pooled over the configured band.

        A plain mean over the band, with no per-layer L2 normalization — the paper's Eq. 2,
        and deliberately *not* what `dino_backbone.extract_patch_features` does for a
        nearest-neighbour bank. Rescaling a layer before averaging silently reweights the
        window, and the window is the axis this method is most sensitive to.
        """
        import torch

        if self._encoder is None or self._stats is None:
            msg = "subspace_ad has no encoder; it was neither fitted nor loaded"
            raise RuntimeError(msg)
        mean_values, std_values = self._stats
        batch = np.stack([np.ascontiguousarray(frame.transpose(2, 0, 1)) for frame in frames])
        tensor = torch.from_numpy(batch).to(device)
        mean = torch.tensor(mean_values, dtype=torch.float32, device=device).view(1, 3, 1, 1)
        std = torch.tensor(std_values, dtype=torch.float32, device=device).view(1, 3, 1, 1)
        tensor = (tensor - mean) / std
        indices = list(self.config.layers.indices(BACKBONES[self.config.backbone].depth))
        with torch.no_grad():
            maps = self._encoder.forward_intermediates(
                tensor,
                indices=indices,
                norm=True,
                output_fmt="NCHW",
                intermediates_only=True,
            )
            pooled = torch.stack(maps, dim=1).flatten(3).mean(dim=1)
            features: np.ndarray = pooled.transpose(1, 2).float().cpu().numpy()
        return features

    # ------------------------------------------------------------------ fitting

    def fit(self, train: Sequence[ImageRecord], ctx: TrainContext) -> None:
        if not train:
            msg = "subspace_ad was given no training images to fit a subspace from"
            raise RuntimeError(msg)

        validate_prepared_size(
            self.config.backbone, ctx.preprocessing.width, ctx.preprocessing.height
        )
        grid = patch_grid(self.config.backbone, ctx.preprocessing.width, ctx.preprocessing.height)
        spec = BACKBONES[self.config.backbone]
        depth = spec.depth
        blocks = self.config.layers.blocks(depth)

        by_channel: dict[str, list[ImageRecord]] = {}
        for record in train:
            by_channel.setdefault(self._channel(record), []).append(record)

        ctx.log(
            f"{self.config.layers.value} resolves to blocks "
            f"{blocks[0]}-{blocks[-1]} of {depth} on {self.config.backbone.value}, pooled to "
            f"{spec.embedding_dim} dimensions over a {grid[0]}x{grid[1]} token grid"
        )
        ctx.progress(0.02, f"loading {spec.timm_name} (downloads on first run only)")
        self._encoder = self._build_encoder(ctx.device.value, ctx.cache_dir)

        copies = 1 + self.config.rotations
        fits: dict[str, _Fitted] = {}
        units = sum(
            min(len(records), self.config.max_fit_images) for records in by_channel.values()
        )
        done = 0
        for channel, records in sorted(by_channel.items()):
            chosen = [
                records[index] for index in evenly_spaced(len(records), self.config.max_fit_images)
            ]
            if len(chosen) < len(records):
                ctx.log(
                    f"channel {channel or 'unassigned'}: fitting on {len(chosen)} of "
                    f"{len(records)} training images (max_fit_images="
                    f"{self.config.max_fit_images}), sampled evenly across the set rather "
                    "than taken from its front"
                )
            accumulator = CovarianceAccumulator(spec.embedding_dim)
            angles = np.random.default_rng(self.config.seed)
            for record in chosen:
                ctx.raise_if_cancelled()
                augmented = rotations(
                    _frame(record, ctx.preprocessing),
                    count=self.config.rotations,
                    rng=angles,
                    fill=self.config.rotation_fill,
                )
                for chunk in _batched(augmented, BATCH_FRAMES):
                    features = self._encode([array for array, _ in chunk], ctx.device.value)
                    for row, (_, valid) in enumerate(chunk):
                        keep = valid_patches(valid, grid, spec.patch_size)
                        patches = features[row] if keep is None else features[row][keep]
                        accumulator.add(patches)
                done += 1
                ctx.progress(
                    0.05 + 0.9 * done / max(units, 1),
                    f"fitted {done}/{units} images ({copies} frames each)",
                )
            if accumulator.count < 2:
                msg = (
                    f"channel {channel or 'unassigned'} contributed "
                    f"{accumulator.count} usable patch(es), which is not enough for a "
                    "covariance. Under rotation_fill=masked a frame whose corners are all "
                    "invented can contribute nothing; try rotation_fill=zeros, fewer "
                    "rotations, or a larger prepared size."
                )
                raise RuntimeError(msg)
            fits[channel] = _Fitted(fit=fit_subspace(accumulator), images=len(chosen))

        # Nothing is fitted until every channel is, so a cancelled fit leaves an object that
        # refuses rather than one that answers for the channels it happened to reach.
        self._fits = fits
        self._grid = grid
        self._emit_spectrum(ctx)
        ctx.progress(1.0, "fitted")

    def _emit_spectrum(self, ctx: TrainContext) -> None:
        """One row per channel: what the subspace kept, and what it cost to keep it."""
        rows = []
        for channel, fitted in sorted(self._fits.items()):
            rank = fitted.rank_for(self.config.variance)
            spectrum = np.clip(fitted.fit.eigenvalues, 0.0, None)
            total = float(spectrum.sum())
            explained = float(spectrum[:rank].sum() / total) if total > 0.0 else 0.0
            rows.append(
                [
                    channel or "unassigned",
                    fitted.images,
                    fitted.fit.sample_count,
                    fitted.fit.dimension,
                    rank,
                    round(explained, 4),
                    round(rank / max(fitted.fit.dimension, 1), 4),
                ]
            )
        ctx.emit_diagnostic(
            "subspace",
            "Fitted subspace",
            DiagnosticKind.TABLE,
            {
                "columns": [
                    "channel",
                    "images",
                    "patches",
                    "dimension",
                    "rank",
                    "explained",
                    "rank fraction",
                ],
                "rows": rows,
            },
            description=(
                f"The normal subspace at variance={self.config.variance:g}. 'rank' is how "
                "many of the encoder's dimensions carry that fraction of the patch "
                "variance; everything outside them is what a patch is scored against, so a "
                "rank close to the full dimension means almost nothing is left to be "
                "anomalous in."
            ),
        )

    # ------------------------------------------------------------------ inference

    def predict(self, images: Sequence[ImageRecord], ctx: InferContext) -> list[Prediction]:
        if not self._fits or self._grid is None:
            msg = "subspace_ad was asked to predict before it was fitted or loaded"
            raise RuntimeError(msg)
        validate_prepared_size(
            self.config.backbone, ctx.preprocessing.width, ctx.preprocessing.height
        )
        grid = patch_grid(self.config.backbone, ctx.preprocessing.width, ctx.preprocessing.height)
        if grid != self._grid:
            msg = (
                f"this subspace_ad model was fitted on a {self._grid[0]}x{self._grid[1]} "
                f"token grid and is being asked to score a {grid[0]}x{grid[1]} one. The "
                "subspace is over patch features, not over positions, so the vectors are "
                "still comparable — but the prepared size changed under the experiment, "
                "which it is not allowed to do; open the experiment this fit belongs to."
            )
            raise RuntimeError(msg)

        if self._encoder is None:
            msg = "subspace_ad restored a fit but no encoder; load() was not called"
            raise RuntimeError(msg)
        self._encoder = self._encoder.to(ctx.device.value)
        size = (ctx.preprocessing.height, ctx.preprocessing.width)
        ranks = {
            channel: fitted.rank_for(self.config.variance) for channel, fitted in self._fits.items()
        }
        predictions: list[Prediction] = []

        for chunk in _batched(list(images), BATCH_FRAMES):
            ctx.raise_if_cancelled()
            started = time.perf_counter()
            features = self._encode(
                [_frame(record, ctx.preprocessing) for record in chunk], ctx.device.value
            )
            elapsed_ms = (time.perf_counter() - started) * 1000.0 / len(chunk)

            for row, record in enumerate(chunk):
                channel = self._channel(record)
                fitted = self._fits.get(channel)
                if fitted is None:
                    raise RuntimeError(self._channel_message(record, channel))
                tokens, values, score = self._score(
                    fitted, features[row], rank=ranks[channel], grid=grid, size=size
                )
                map_path = ctx.write_map(record.image_id, values)
                ctx.emit_diagnostic(
                    "patch_residual",
                    "Patch residual",
                    DiagnosticKind.MAP,
                    tokens,
                    image_id=record.image_id,
                    description=(
                        f"Squared distance from each patch to the normal subspace at "
                        f"rank {ranks[channel]}, at the encoder's own resolution. This is "
                        "what the method computed; the anomaly map is this upsampled"
                        + (
                            f" and smoothed at sigma {self.config.smoothing_sigma:g}"
                            if self.config.smoothing_sigma > 0.0
                            else " and left unsmoothed"
                        )
                        + ", and the image score is the mean of its top "
                        + f"{self.config.tail_fraction:g}."
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
            ctx.progress(
                len(predictions) / max(len(images), 1),
                f"scored {len(predictions)}/{len(images)} images",
            )
        return predictions

    def _score(
        self,
        fitted: _Fitted,
        features: np.ndarray,
        *,
        rank: int,
        grid: tuple[int, int],
        size: tuple[int, int],
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """One image's patch residual, pixel map and image score from its `(P, D)` features.

        The one scoring path: `predict` stores what this returns, and `portable_reference`
        hands it to the export's parity check, so the graph is compared with the method.
        """
        basis = residual_basis(fitted.fit, features, rank=rank)
        scores = basis.at_rank(rank)
        # The image score is taken on the **patch grid**, before the upsample and before
        # the blur, so `smoothing_sigma` cannot move it. A score that depended on the
        # smoothing would make a display decision a scoring one.
        score = tail_value_at_risk(scores, [self.config.tail_fraction])[self.config.tail_fraction]
        tokens = scores.reshape(grid).astype(np.float32)
        # `pixel_map` refuses a non-positive sigma, and rightly: a Gaussian of width zero is
        # not a blur, it is the absence of one, and a blur helper that silently accepted it
        # would hide a misconfigured sweep. The absence is meaningful here, though — it is
        # how a reader sees the raw upsample — so the plugin spends the branch rather than
        # asking the helper to be vague.
        values = (
            pixel_map(tokens, size, sigma=self.config.smoothing_sigma)
            if self.config.smoothing_sigma > 0.0
            else upsample_bilinear(tokens, size)
        )
        return tokens, values, score

    # ------------------------------------------------------------------ deployment

    def _single_fit(self) -> _Fitted:
        """The one subspace a portable graph can carry, or a refusal that names the others.

        A bundle is one static graph over one prepared frame, and its contract has no
        channel input to branch on. A fit over several channels holds one subspace per
        channel, so it is refused here — inside the plugin, never pre-checked in a route,
        which is how `pixel_reference` refuses a per-channel reference too (ADR-0007).
        """
        if not self._fits or self._grid is None or self._encoder is None:
            msg = "subspace_ad has no fitted subspace to export; fit or load it first"
            raise RuntimeError(msg)
        if len(self._fits) > 1:
            named = ", ".join(sorted(name or "unassigned" for name in self._fits))
            msg = (
                f"subspace_ad fitted {len(self._fits)} subspaces, one per channel ({named}), "
                "and one ONNX graph carries exactly one: the bundle contract has no channel "
                "input to choose between them. Export a run whose channel selection leaves "
                "a single channel."
            )
            raise ValueError(msg)
        return next(iter(self._fits.values()))

    def _check_frame(self, width: int, height: int) -> tuple[int, int]:
        grid = patch_grid(self.config.backbone, width, height)
        if grid != self._grid:
            fitted = "x".join(str(side) for side in self._grid or ())
            msg = (
                f"this subspace_ad model was fitted on a {fitted} token grid and is asked for "
                f"{width}x{height}, a {grid[0]}x{grid[1]} one"
            )
            raise ValueError(msg)
        return grid

    def _portable_module(self, preprocessing: PreprocessingConfig) -> Any:
        """The fitted subspace on CPU, wrapped as prepared pixels in, map and score out."""
        from anomaly_lab.models.dino_backbone import pin_frame
        from anomaly_lab.models.subspace_portable import PortableSubspace

        fitted = self._single_fit()
        grid = self._check_frame(preprocessing.width, preprocessing.height)
        if self._stats is None or self._encoder is None:
            msg = "subspace_ad has no encoder; it was neither fitted nor loaded"
            raise RuntimeError(msg)
        rank = fitted.rank_for(self.config.variance)
        sigma = self.config.smoothing_sigma
        height, width = preprocessing.height, preprocessing.width
        # The map is linear in the token grid, so the upsample and the blur are one constant
        # matrix per axis — the very operators `pixel_map` applies, built from the same
        # primitives, rather than a second implementation of either in graph operators.
        if sigma > 0.0:
            rows = separable_operator(grid[0], height, sigma)
            columns = separable_operator(grid[1], width, sigma)
        else:
            rows = resample_operator(grid[0], height)
            columns = resample_operator(grid[1], width)
        self._encoder = self._encoder.to("cpu").eval()
        mean_values, std_values = self._stats
        return PortableSubspace(
            pin_frame(self._encoder, grid[0], grid[1]),
            indices=self.config.layers.indices(BACKBONES[self.config.backbone].depth),
            mean=mean_values,
            std=std_values,
            centre=fitted.fit.mean,
            components=fitted.fit.components[:rank],
            grid=grid,
            # `tail_value_at_risk`'s own count, fixed by the grid: at least one patch.
            tail_count=max(1, round(self.config.tail_fraction * grid[0] * grid[1])),
            rows=rows,
            columns=columns,
        ).eval()

    def export_onnx(
        self,
        destination: Path,
        preprocessing: PreprocessingConfig,
    ) -> OnnxGraphContract:
        """Export encoder, residual, tail mean and map as one static graph.

        The score is a named graph output: it is the tail mean of the *patch grid*, which no
        host reducer over the emitted pixel map could reproduce. The rank `variance` selects,
        the tail count and the upsample-and-blur operators are constants of the graph, so a
        bundle answers for the configuration it was exported under.
        """
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
            opset_version=PORTABLE_OPSET,
            dynamo=False,
        )
        return OnnxGraphContract(
            opset=PORTABLE_OPSET,
            input_name=input_name,
            output_name=map_name,
            score=TensorScore(tensor=ScalarTensorSpec(name=score_name, dtype=TensorDtype.FLOAT32)),
            absolute_tolerance=PORTABLE_TOLERANCE,
            relative_tolerance=PORTABLE_TOLERANCE,
        )

    def portable_reference(self, input_nchw: np.ndarray) -> tuple[np.ndarray, float]:
        """`predict`'s own arithmetic, on CPU, over an already-prepared parity tensor."""
        fitted = self._single_fit()
        height, width = int(input_nchw.shape[2]), int(input_nchw.shape[3])
        grid = self._check_frame(width, height)
        if self._encoder is None:
            msg = "subspace_ad has no encoder; it was neither fitted nor loaded"
            raise RuntimeError(msg)
        self._encoder = self._encoder.to("cpu")
        planes = expand_planes(np.asarray(input_nchw[0], dtype=np.float32), 3)
        features = self._encode([np.ascontiguousarray(planes.transpose(1, 2, 0))], "cpu")[0]
        _, values, score = self._score(
            fitted,
            features,
            rank=fitted.rank_for(self.config.variance),
            grid=grid,
            size=(height, width),
        )
        return np.asarray(values, dtype=np.float32), float(score)

    def _channel_message(self, record: ImageRecord, channel: str) -> str:
        known = ", ".join(name or "unassigned" for name in sorted(self._fits)) or "no channels"
        return (
            f"subspace_ad fits one subspace per channel and image {record.image_id} is on "
            f"channel {channel or 'unassigned'}, which this model was not fitted on. It "
            f"holds {known}. Train on the channels this run scores, or select the fitted "
            "ones on the experiment."
        )

    # ------------------------------------------------------------------ persistence

    def save(self, artifact_dir: Path) -> None:
        """Write the subspaces and the identity of everything they were built from.

        Numpy rather than `torch.save`: the state is four arrays per channel and reading it
        back must not need the optional `dl` extra any more than computing it did.

        **The whole spectrum and the whole basis are kept, not the part `variance` selects.**
        Because the components are orthonormal, a score at a smaller rank is a subtraction
        against a column that is already there — so a stored fit answers at every threshold,
        and changing `variance` costs a rescore rather than a refit. That is the same
        identity that made the campaign's tau axis free, and it would be a shame to store the
        one arrangement of the fit that throws it away.
        """
        if not self._fits or self._fingerprint is None or self._cache_dir is None:
            msg = "subspace_ad has nothing to save; it was never fitted"
            raise RuntimeError(msg)

        spec = BACKBONES[self.config.backbone]
        channels = sorted(self._fits)
        # Stacked by channel rather than one entry per channel per array, because every
        # channel's subspace has the same shape — one encoder, one embedding width — and
        # three named arrays read back without a naming convention to honour.
        np.savez(
            artifact_dir / STATE_FILENAME,
            means=np.stack([self._fits[channel].fit.mean for channel in channels]),
            components=np.stack([self._fits[channel].fit.components for channel in channels]),
            eigenvalues=np.stack([self._fits[channel].fit.eigenvalues for channel in channels]),
        )

        manifest = {
            "format": CHECKPOINT_FORMAT,
            "backbone": self.config.backbone.value,
            "timm_name": spec.timm_name,
            "patch_size": spec.patch_size,
            "depth": spec.depth,
            "layers": self.config.layers.value,
            "blocks": list(self.config.layers.blocks(spec.depth)),
            "pretrained_backbone": self.config.pretrained_backbone,
            "backbone_fingerprint": self._fingerprint,
            "cache_dir": str(self._cache_dir),
            "grid": list(self._grid) if self._grid else None,
            "channels": channels,
            "images": [self._fits[channel].images for channel in channels],
            "samples": [self._fits[channel].fit.sample_count for channel in channels],
        }
        (artifact_dir / MANIFEST_FILENAME).write_text(json.dumps(manifest, indent=2) + "\n")

    def load(self, artifact_dir: Path) -> None:
        """Restore the subspaces, and refuse anything that does not match, by name.

        Every refusal here has the same shape of alternative: a checkpoint that loads without
        complaint and produces a plausible, wrong number that nothing distinguishes from the
        run it is being compared against.
        """
        manifest = json.loads((artifact_dir / MANIFEST_FILENAME).read_text())
        if int(manifest.get("format", 0)) != CHECKPOINT_FORMAT:
            msg = (
                f"unsupported subspace_ad checkpoint format {manifest.get('format')!r}; this "
                f"build reads format {CHECKPOINT_FORMAT}."
            )
            raise RuntimeError(msg)
        stored_layers = str(manifest["layers"])
        if stored_layers != self.config.layers.value:
            msg = (
                f"this subspace_ad experiment is configured for layers="
                f"{self.config.layers.value} but the checkpoint was fitted on "
                f"{stored_layers}. The subspace is a basis of *those* features; refit, or "
                "open the experiment this checkpoint belongs to."
            )
            raise RuntimeError(msg)

        cache_dir = Path(manifest["cache_dir"])
        encoder = self._build_encoder("cpu", cache_dir)
        expected = str(manifest["backbone_fingerprint"])
        actual = str(self._fingerprint)
        if expected != actual:
            msg = (
                f"the encoder weights for {manifest.get('backbone', self.config.backbone.value)!r} "
                f"are not the ones this experiment was fitted against: the checkpoint records "
                f"{expected[:12]} and the weights now resolving are {actual[:12]}. The "
                "subspace was fitted in the old feature space, so a residual against the new "
                "one means nothing. Refit, or restore the original weights."
            )
            raise RuntimeError(msg)

        stored = np.load(artifact_dir / STATE_FILENAME)
        channels = [str(name) for name in manifest["channels"]]
        samples = [int(value) for value in manifest["samples"]]
        images = [int(value) for value in manifest["images"]]
        self._fits = {
            channel: _Fitted(
                fit=SubspaceFit(
                    mean=stored["means"][index],
                    components=stored["components"][index],
                    eigenvalues=stored["eigenvalues"][index],
                    sample_count=samples[index],
                ),
                images=images[index],
            )
            for index, channel in enumerate(channels)
        }
        self._encoder = encoder
        grid = manifest.get("grid")
        self._grid = (int(grid[0]), int(grid[1])) if grid else None
