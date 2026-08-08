"""The EfficientAD networks (arXiv:2303.14535), ours.

This is the architecture and the arithmetic of `efficientad_custom`, with no plugin
concerns in it: no context, no artifacts, no progress. The plugin next door owns those,
and keeps its own module scope torch-free so the registry stays lazy — which is why this
file exists separately rather than at the top of that one.

**anomalib is the baseline, not the specification.** The networks here are written from
the paper, and their outputs are pinned against anomalib's in `test_efficientad_equivalence.py`
with every improvement switched off. That pin is not deference: it is what makes a later
divergence *measurable*, because a change in a number can then be attributed to the change
we made rather than to a difference we never noticed we had.

Two of those pins are different in kind, and the difference matters:

  * **The PDN pin is permanent and is not a choice.** The pretrained teacher is a published
    file whose keys are `conv1.weight` … `convN.bias`. Our PDN must be the network those
    weights describe, or we are loading a public asset into the wrong model — a bug that
    would show up as a mediocre AUROC and nothing else. That is why one parameterised class
    covers both sizes: the *layer names* are the contract.
  * **The rest is a bring-up pin.** The autoencoder, the losses and the map computation are
    checked against anomalib once, to prove we read the paper correctly, and then stay as a
    regression net for the verified core. Improvements are opt-in arguments whose defaults
    reproduce it exactly.

**Deliberate departures, all of them small and all of them stated:**

  * `mean_std` and `quantiles` are **buffers with an explicit `fitted` flag**, not
    `ParameterDict`s. They are statistics, not parameters — nothing should be able to reach
    them through `model.parameters()` — and "has this been fitted?" is a fact worth
    recording rather than inferring from the values being non-zero, which reads a
    legitimately all-zero mean as unfitted.
  * **Hard-example mining is deterministic.** The reference subsamples with a random
    permutation before `torch.quantile` to stay under that operator's 2**24-element limit.
    At 256 px the distance tensor is 1.2M elements and the subsample never happens, so this
    is dormant nondeterminism in the loss path rather than a live behaviour; an evenly
    spaced stride costs nothing and removes it.
"""

from __future__ import annotations

import math
from typing import Literal

import torch
from torch import nn
from torch.nn import functional

ModelSize = Literal["small", "medium"]
ScoreReduction = Literal["max", "top_k_mean"]

TEACHER_OUT_CHANNELS = 384
"""The published teachers' output width. Both sizes distil to 384 channels."""

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

QUANTILE_ELEMENT_LIMIT = 2**24
"""`torch.quantile` refuses more than this many elements, on every backend."""

MINIMUM_INPUT = 256
"""The smallest input the autoencoder can take, and it is the *encoder* that decides it.

`enconv6` has an 8x8 kernel and sees the image after five stride-2 convolutions, so it
needs `size / 32 >= 8`. Below that, the failure is a `RuntimeError` from `conv2d` naming a
padded input size — several layers away from anything a reader could act on. The plugin
refuses the configuration by name instead; this constant is where the number comes from.
"""

STD_FLOOR = 1e-6
"""Smallest teacher channel deviation used as a divisor. See `set_teacher_statistics`."""

QUANTILE_SPREAD_FLOOR = 1e-12
"""Smallest `qb - qa` used as a divisor. See `set_quantiles`."""

INPUT_MULTIPLE = 64
"""The decoder's upsample ladder is written in terms of `size // 64`, `size // 32` … so a
size that is not a multiple of 64 silently lands the reconstruction on a different grid
from the student's output, and the two are then compared after an implicit resize."""


def imagenet_normalize(batch: torch.Tensor) -> torch.Tensor:
    """Standardize with ImageNet statistics, which is what the teacher was trained under.

    Applied inside each network rather than in the preprocessing bridge, deliberately.
    `preprocessing.py` decides what pixels every *method* sees so that a comparison is not
    partly a measurement of the resize; this is a detail of what this particular network
    does with them, and folding it into the shared bridge would push one method's
    normalization onto `pixel_reference`.

    **A single-channel batch is expanded to three, explicitly.** A grayscale experiment
    hands this `(N, 1, H, W)`, and subtracting a `(1, 3, 1, 1)` mean from it broadcasts to
    `(N, 3, H, W)` on its own — so this method already works on grayscale, by accident,
    through a rule nobody wrote down. Measured: the expanded and the broadcast paths agree
    to `0.0` through the whole PDN. Writing the expansion out changes no number and makes
    the behaviour a decision that can be tested, which is the difference between a
    supported configuration and one that happens not to crash.
    """
    if batch.shape[1] == 1:
        batch = batch.expand(-1, 3, -1, -1)
    mean = torch.tensor(IMAGENET_MEAN, device=batch.device, dtype=batch.dtype)
    std = torch.tensor(IMAGENET_STD, device=batch.device, dtype=batch.dtype)
    return (batch - mean[None, :, None, None]) / std[None, :, None, None]


class PatchDescriptionNetwork(nn.Module):
    """The PDN, in both published widths.

    One class rather than the reference's two, because the two differ only in their layer
    table and the pretrained weights key on layer *names*: `conv1` … `conv4` for small,
    `conv1` … `conv6` for medium. Splitting them into separate classes makes that contract
    invisible; keeping the names generated from one table makes it the thing you read first.

    A teacher and a student are the same network at different widths — the student emits
    `2 * out_channels` so its first half can chase the teacher while its second half chases
    the autoencoder. That is the whole trick behind detecting logical anomalies with a
    network this shallow.
    """

    def __init__(self, out_channels: int, size: ModelSize = "small", padding: bool = False) -> None:
        super().__init__()
        pad = 1 if padding else 0
        # (in, out, kernel, padding-in-units-of-`pad`), in execution order.
        table: list[tuple[int, int, int, int]] = (
            [(3, 128, 4, 3), (128, 256, 4, 3), (256, 256, 3, 1), (256, out_channels, 4, 0)]
            if size == "small"
            else [
                (3, 256, 4, 3),
                (256, 512, 4, 3),
                (512, 512, 1, 0),
                (512, 512, 3, 1),
                (512, out_channels, 4, 0),
                (out_channels, out_channels, 1, 0),
            ]
        )
        for index, (in_ch, out_ch, kernel, kernel_pad) in enumerate(table, start=1):
            self.add_module(
                f"conv{index}",
                nn.Conv2d(in_ch, out_ch, kernel_size=kernel, padding=kernel_pad * pad),
            )
        self.avgpool1 = nn.AvgPool2d(kernel_size=2, stride=2, padding=pad)
        self.avgpool2 = nn.AvgPool2d(kernel_size=2, stride=2, padding=pad)
        self._convolutions = len(table)
        # Pooling happens after conv1 and conv2 in both sizes; the final convolution is
        # linear, every earlier one is followed by ReLU.
        self._pool_after = {1: self.avgpool1, 2: self.avgpool2}

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        out = imagenet_normalize(batch)
        for index in range(1, self._convolutions + 1):
            convolution = getattr(self, f"conv{index}")
            out = convolution(out)
            if index < self._convolutions:
                out = functional.relu(out)
            pool = self._pool_after.get(index)
            if pool is not None:
                out = pool(out)
        return out


class Encoder(nn.Module):
    """The autoencoder's contracting half: 256x256 down to a 64-vector.

    Six convolutions, five of them stride 2, and a final 8x8 kernel that collapses what is
    left to 1x1. That last kernel is the entire reason `MINIMUM_INPUT` is 256.
    """

    def __init__(self) -> None:
        super().__init__()
        self.enconv1 = nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1)
        self.enconv2 = nn.Conv2d(32, 32, kernel_size=4, stride=2, padding=1)
        self.enconv3 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.enconv4 = nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1)
        self.enconv5 = nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1)
        self.enconv6 = nn.Conv2d(64, 64, kernel_size=8, stride=1, padding=0)

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        out = batch
        for index in range(1, 6):
            out = functional.relu(getattr(self, f"enconv{index}")(out))
        bottleneck: torch.Tensor = self.enconv6(out)
        return bottleneck


class Decoder(nn.Module):
    """The expanding half, back up to the teacher's feature grid.

    Upsample-then-convolve rather than transposed convolution, which is what keeps the
    reconstruction free of the checkerboard artefacts a transposed stack produces — and
    those artefacts would be indistinguishable from a fine-grained anomaly. The dropout is
    the only regularization the branch has; without it the decoder learns the training set
    well enough that a logical anomaly reconstructs cleanly too.

    The ladder of intermediate sizes is written in terms of the *input image* size, so the
    output lands exactly on the PDN's output grid with no resize between them.
    """

    def __init__(self, out_channels: int, padding: bool = False) -> None:
        super().__init__()
        self.padding = padding
        for index in range(1, 7):
            self.add_module(f"deconv{index}", nn.Conv2d(64, 64, kernel_size=4, padding=2))
            self.add_module(f"dropout{index}", nn.Dropout(p=0.2))
        self.deconv7 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.deconv8 = nn.Conv2d(64, out_channels, kernel_size=3, padding=1)

    def forward(self, batch: torch.Tensor, image_size: tuple[int, int]) -> torch.Tensor:
        height, width = image_size
        # The last step matches the PDN's own output size: an unpadded PDN loses 8 pixels
        # of context, and the reconstruction has to lose the same 8 or the two feature
        # grids being subtracted are not the same grid.
        last = (
            (math.ceil(height / 4), math.ceil(width / 4))
            if self.padding
            else (math.ceil(height / 4) - 8, math.ceil(width / 4) - 8)
        )
        ladder = [
            (height // 64 - 1, width // 64 - 1),
            (height // 32, width // 32),
            (height // 16 - 1, width // 16 - 1),
            (height // 8, width // 8),
            (height // 4 - 1, width // 4 - 1),
            (height // 2 - 1, width // 2 - 1),
            last,
        ]
        out = batch
        for index, size in enumerate(ladder, start=1):
            out = functional.interpolate(out, size=size, mode="bilinear")
            out = functional.relu(getattr(self, f"deconv{index}")(out))
            dropout = getattr(self, f"dropout{index}", None)
            if dropout is not None:
                out = dropout(out)
        reconstruction: torch.Tensor = self.deconv8(out)
        return reconstruction


class AutoEncoder(nn.Module):
    """The global branch: reconstruct the teacher's features from a 64-number bottleneck.

    The bottleneck is the point. A student with the teacher's receptive field can copy it
    patch by patch and never notice that the *arrangement* is wrong; forcing the whole
    image through 64 numbers means the branch can only reproduce arrangements it has seen.
    """

    def __init__(self, out_channels: int, padding: bool = False) -> None:
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder(out_channels, padding)

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """The target size is read off the input rather than passed in.

        The reference takes an `image_size` argument, and every one of its callers passes
        `batch.shape[-2:]` — flexibility that exists only to be got wrong. Dropping it also
        means `introspect.probe_arguments` has nothing to supply for this branch, so the
        architecture diagnostic works here with no signature special case at all.
        """
        latent = self.encoder(imagenet_normalize(batch))
        reconstruction: torch.Tensor = self.decoder(latent, batch.shape[-2:])
        return reconstruction


def quantile_of(values: torch.Tensor, q: float) -> torch.Tensor:
    """`torch.quantile` with the operator's element limit handled deterministically.

    Above 2**24 elements the operator refuses outright. The reference draws a random
    permutation and keeps a prefix; an evenly spaced stride estimates the same quantile,
    costs less, and does not consume the global RNG stream — which is what makes a training
    step reproducible from a seed rather than merely similar.
    """
    flat = values.flatten()
    if flat.numel() > QUANTILE_ELEMENT_LIMIT:
        stride = math.ceil(flat.numel() / QUANTILE_ELEMENT_LIMIT)
        flat = flat[::stride]
    return torch.quantile(flat, q)


LUMA_WEIGHTS = (0.2989, 0.587, 0.114)
"""ITU-R 601-2 luma coefficients, which is what torchvision's grayscale conversion uses."""


def _luma(image: torch.Tensor) -> torch.Tensor:
    weights = torch.tensor(LUMA_WEIGHTS, device=image.device, dtype=image.dtype)
    return (image * weights[None, :, None, None]).sum(dim=1, keepdim=True)


def _blend(image: torch.Tensor, other: torch.Tensor, factor: float) -> torch.Tensor:
    return (factor * image + (1.0 - factor) * other).clamp(0.0, 1.0)


def adjust_brightness(image: torch.Tensor, factor: float) -> torch.Tensor:
    return (image * factor).clamp(0.0, 1.0)


def adjust_contrast(image: torch.Tensor, factor: float) -> torch.Tensor:
    mean = _luma(image).mean(dim=(-3, -2, -1), keepdim=True)
    return _blend(image, mean, factor)


def adjust_saturation(image: torch.Tensor, factor: float) -> torch.Tensor:
    return _blend(image, _luma(image), factor)


AUGMENTATIONS = (adjust_brightness, adjust_contrast, adjust_saturation)


def augment(image: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    """One of brightness, contrast or saturation, scaled by U(0.8, 1.2).

    The autoencoder and the student's second half are trained on an *augmented* copy, so
    what they learn to agree on is the image's content rather than its exposure.

    **Driven by a generator we own, not by the global torch RNG.** The reference draws from
    the global stream, which means an augmentation anywhere else in the process — or a
    library that draws a random number between two steps — silently changes the sequence.
    A workbench whose whole point is "did this change move the number?" cannot afford a
    stochastic input it does not control, and it is also what lets the training image
    order, the augmentation and the penalty order all resume exactly (ADR-0025).

    Written out rather than taken from torchvision, which is then a dependency this method
    does not need — three scalar blends against an ITU-R 601-2 luma. `test_efficientad_
    custom.py` pins them against torchvision's, which is what licenses dropping it.
    """
    low, high = 0.8, 1.2
    factor = low + (high - low) * float(torch.rand(1, generator=generator).item())
    index = int(torch.randint(0, len(AUGMENTATIONS), (1,), generator=generator).item())
    return AUGMENTATIONS[index](image, factor)


class EfficientAdNet(nn.Module):
    """Teacher, student and autoencoder, with the statistics that make their outputs
    comparable.

    The three branches are wired together only here and in the losses — not in any module's
    `forward` — which is why `models/introspect.py` can record every node but the plugin
    has to state the edges itself (ADR-0024).
    """

    def __init__(
        self,
        size: ModelSize = "small",
        teacher_out_channels: int = TEACHER_OUT_CHANNELS,
        padding: bool = False,
        pad_maps: bool = True,
    ) -> None:
        super().__init__()
        self.size = size
        self.teacher_out_channels = teacher_out_channels
        self.pad_maps = pad_maps
        self.teacher = PatchDescriptionNetwork(teacher_out_channels, size, padding).eval()
        self.student = PatchDescriptionNetwork(teacher_out_channels * 2, size, padding)
        self.ae = AutoEncoder(teacher_out_channels, padding)

        # Buffers, not parameters: these are measurements of the data, and an optimizer
        # walking `parameters()` must not be able to reach them. The flags are recorded
        # rather than inferred, so a fitted-but-all-zero statistic is not read as unfitted.
        #
        # Each is annotated before it is registered, which is torch's own idiom for the
        # purpose: `register_buffer` is typed as returning `Tensor | Module | None`, so
        # without the annotation every arithmetic use of one of these fails to type-check.
        self.teacher_mean: torch.Tensor
        self.teacher_std: torch.Tensor
        self.stats_fitted: torch.Tensor
        self.qa_st: torch.Tensor
        self.qb_st: torch.Tensor
        self.qa_ae: torch.Tensor
        self.qb_ae: torch.Tensor
        self.quantiles_fitted: torch.Tensor
        self.register_buffer("teacher_mean", torch.zeros(1, teacher_out_channels, 1, 1))
        self.register_buffer("teacher_std", torch.ones(1, teacher_out_channels, 1, 1))
        self.register_buffer("stats_fitted", torch.zeros((), dtype=torch.bool))
        for name in ("qa_st", "qb_st", "qa_ae", "qb_ae"):
            self.register_buffer(name, torch.zeros(()))
        self.register_buffer("quantiles_fitted", torch.zeros((), dtype=torch.bool))

    # ------------------------------------------------------------------ statistics

    def set_teacher_statistics(self, mean: torch.Tensor, std: torch.Tensor) -> int:
        """Record the channel statistics that put teacher and student on one scale.

        Returns how many channels had their standard deviation floored, so the caller can
        say so in the job log. A channel with no variance across the training set — a
        teacher feature that never fires on this dataset — divides every later comparison
        by zero, and the reference's `sqrt(E[x^2] - E[x]^2)` can go slightly negative from
        float32 cancellation and produce a `NaN` std that then poisons every score in the
        run. Neither raises. Both are silent, and a floor is cheaper than either.
        """
        std = torch.nan_to_num(std.to(self.teacher_std), nan=STD_FLOOR)
        floored = int((std < STD_FLOOR).sum())
        self.teacher_mean.copy_(mean.to(self.teacher_mean))
        self.teacher_std.copy_(std.clamp_min(STD_FLOOR))
        self.stats_fitted.fill_(True)
        return floored

    def set_quantiles(self, values: dict[str, torch.Tensor]) -> list[str]:
        """Record the four map-normalization quantiles.

        Their *ratio* across the two branches is what reorders images, so this is not the
        display convenience it looks like — see the plugin's calibration step.

        Returns the names of any branch whose two quantiles collapsed onto one value, which
        happens when the calibration images produce a near-constant map. The reference
        divides by `qb - qa` unguarded and yields `inf` maps that render as a solid block;
        the denominator is floored here and the caller reports it.
        """
        for name, value in values.items():
            getattr(self, name).copy_(value.to(getattr(self, name)))
        self.quantiles_fitted.fill_(True)
        pairs = (("st", self.qa_st, self.qb_st), ("ae", self.qa_ae, self.qb_ae))
        return [branch for branch, low, high in pairs if float(high - low) < QUANTILE_SPREAD_FLOOR]

    # ------------------------------------------------------------------ forward paths

    def teacher_features(self, batch: torch.Tensor) -> torch.Tensor:
        """The teacher's output, standardized once the statistics have been fitted."""
        with torch.no_grad():
            features: torch.Tensor = self.teacher(batch)
            if bool(self.stats_fitted):
                features = (features - self.teacher_mean) / self.teacher_std
        return features

    def student_teacher_distance(self, batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """The student's full output, and its squared error against the teacher."""
        teacher_output = self.teacher_features(batch)
        student_output = self.student(batch)
        distance = (teacher_output - student_output[:, : self.teacher_out_channels]) ** 2
        return student_output, distance

    def compute_losses(
        self,
        penalty: torch.Tensor,
        augmented: torch.Tensor,
        distance_st: torch.Tensor,
        *,
        hard_quantile: float = 0.999,
        use_penalty: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """The three training losses, in the paper's decomposition.

        **A pure function of its inputs.** The augmented image is drawn by the caller and
        passed in, where the reference draws it inside. That keeps every random decision in
        the training loop, where the generator lives, and it makes this comparable against
        the reference without any RNG choreography — which is the difference between an
        equivalence test that holds and one that passes once and then flakes.

        `hard_quantile` is the hard feature loss: backpropagate only through the elements
        the student is worst at, so the easy 99.9% of a normal image stops dominating the
        gradient once it is already fitted. `use_penalty` is the pretraining penalty, which
        pushes the student's output toward zero on *unrelated* natural images so it cannot
        simply learn the teacher's function everywhere and lose its capacity to be
        surprised. Both are exposed because the paper measures them (+1.0 and +0.4 AU-ROC)
        and so can we — a component that is silently dead shows up as a zero delta and as
        nothing else.
        """
        hard = quantile_of(distance_st, hard_quantile)
        loss_hard = torch.mean(distance_st[distance_st >= hard])
        if use_penalty:
            penalty_output = self.student(penalty)[:, : self.teacher_out_channels]
            loss_st = loss_hard + torch.mean(penalty_output**2)
        else:
            loss_st = loss_hard

        reconstruction = self.ae(augmented)
        teacher_augmented = self.teacher_features(augmented)
        student_augmented = self.student(augmented)[:, self.teacher_out_channels :]

        loss_ae = torch.mean((teacher_augmented - reconstruction) ** 2)
        loss_stae = torch.mean((reconstruction - student_augmented) ** 2)
        return loss_st, loss_ae, loss_stae

    def compute_maps(
        self,
        batch: torch.Tensor,
        student_output: torch.Tensor,
        distance_st: torch.Tensor,
        *,
        normalize: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The two branch maps at input resolution: local error, and global disagreement.

        Kept apart rather than combined here. An anomaly the student missed and an anomaly
        the autoencoder missed mean different things about what went wrong, and the
        combined map that gets stored averages that distinction away — which is exactly
        what the per-image diagnostics exist to give back (ADR-0018).
        """
        image_size = (batch.shape[-2], batch.shape[-1])
        with torch.no_grad():
            reconstruction = self.ae(batch)
            map_st = torch.mean(distance_st, dim=1, keepdim=True)
            map_stae = torch.mean(
                (reconstruction - student_output[:, self.teacher_out_channels :]) ** 2,
                dim=1,
                keepdim=True,
            )

        if self.pad_maps:
            # An unpadded PDN sees 8 fewer pixels in each direction; padding the map back
            # out puts an anomaly at the same place it is in the source image, rather than
            # 4 pixels off after the upsample.
            map_st = functional.pad(map_st, (4, 4, 4, 4))
            map_stae = functional.pad(map_stae, (4, 4, 4, 4))
        map_st = functional.interpolate(map_st, size=image_size, mode="bilinear")
        map_stae = functional.interpolate(map_stae, size=image_size, mode="bilinear")

        if normalize and bool(self.quantiles_fitted):
            spread_st = (self.qb_st - self.qa_st).clamp_min(QUANTILE_SPREAD_FLOOR)
            spread_ae = (self.qb_ae - self.qa_ae).clamp_min(QUANTILE_SPREAD_FLOOR)
            map_st = 0.1 * (map_st - self.qa_st) / spread_st
            map_stae = 0.1 * (map_stae - self.qa_ae) / spread_ae
        return map_st, map_stae

    def maps(
        self, batch: torch.Tensor, *, normalize: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Both branch maps for a batch, from pixels."""
        student_output, distance_st = self.student_teacher_distance(batch)
        return self.compute_maps(batch, student_output, distance_st, normalize=normalize)

    def score(
        self,
        batch: torch.Tensor,
        *,
        weights: tuple[float, float] = (0.5, 0.5),
        reduction: ScoreReduction = "max",
        top_k: int = 64,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """`(combined map, per-image score)`.

        `weights` and `reduction` are the two knobs the milestone measures. Their defaults
        — an even blend and the maximum — are the published behaviour, so a run that
        touches neither is the verified core.

        A maximum is one pixel's opinion, and `pixel_reference` already reduces its map by
        a high percentile for exactly that reason. The alternative offered here is the mean
        of the `top_k` pixels rather than a percentile, and the difference matters: at
        256x256 the 99.5th percentile is the top 328 pixels, an 18x18 region, and plenty of
        real defects are smaller than that — a percentile would suppress exactly the
        anomalies this method is best at. A small `top_k` averages away a lone hot pixel
        without averaging away a small defect. Which of the two wins is a measurement, not
        an argument; the field exists so the measurement can be made.
        """
        map_st, map_stae = self.maps(batch, normalize=True)
        combined = weights[0] * map_st + weights[1] * map_stae
        flat = combined.flatten(start_dim=1)
        if reduction == "max":
            return combined, flat.amax(dim=1)
        count = max(1, min(top_k, flat.shape[1]))
        return combined, flat.topk(count, dim=1).values.mean(dim=1)
