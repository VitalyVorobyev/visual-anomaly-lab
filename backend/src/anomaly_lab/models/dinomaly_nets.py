"""The architecture and the arithmetic behind `dinomaly_custom`, and nothing about jobs.

The split `efficientad_custom`/`efficientad_nets` established: the plugin next door owns the
configuration, the plan, the bounded pass and the checkpoint, and this module owns the
`nn.Module`s, the loss, the optimizer and the map rule. Nothing here imports anomalib, and
nothing here knows that a job system exists.

**What is ported and what is ours.** The mechanism is Dinomaly's (Guo et al., MIT-licensed
reference; Intel's anomalib carries an Apache-2.0 implementation of the same paper) and the
code is written here from that mechanism rather than wrapped:

  * a **frozen** ViT encoder supplies eight intermediate token sequences;
  * those eight are averaged into one sequence and pushed through a **bottleneck MLP** with
    input dropout — the narrowing that stops the decoder from simply copying;
  * a stack of **decoder blocks with linear attention** reconstructs the sequence. Linear
    attention is the paper's own choice and not an efficiency shortcut: softmax attention
    focuses too well, and a decoder that can focus learns to reconstruct an anomaly too;
  * encoder and decoder outputs are fused into **two groups** and compared by cosine
    distance, per patch, which is the anomaly map;
  * training minimises a **global** cosine distance while a backward hook shrinks the
    gradient on the easy points — "hard mining" that down-weights rather than up-weights,
    following the reference's own naming.

**Two shapes here look like bugs and are not.** `LinearAttention` computes `self.scale` and
never uses it, because the ELU feature map is normalised by `z` instead; and the loss's
scalar term is a cosine over each batch item's *entire flattened* feature map, while the
per-point distances only ever reach the gradient through the hook. Both are the reference's,
kept so that a later parity gate against `dinomaly_anomalib` measures the method rather than
a divergence introduced here.

Every symbol in this module needs torch. It is imported only from inside `dinomaly_custom`'s
functions, so the registry stays lazy.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Sequence
from typing import Any

import torch
import torch.nn.functional as functional
from torch import Tensor, nn

from anomaly_lab.models.dino_backbone import extract_layer_tokens

LAYER_NORM_EPS = 1e-8
"""The reference's decoder LayerNorm epsilon. Unusually small, and load-bearing for a
like-for-like comparison: torch's default 1e-5 changes every activation in the stack."""

MLP_RATIO = 4
"""Hidden width of every MLP, as a multiple of the embedding width."""

EASY_GRADIENT_FACTOR = 0.1
"""What the gradient of a well-reconstructed point is multiplied by. The reference's value."""

HARD_MINING_ANNEAL_STEPS = 1_000
"""How long the mined fraction takes to reach its configured value.

Mining from step zero would freeze the decoder's gradient onto whatever the *untrained*
network happened to find hard, which is noise. The reference anneals instead, and the anneal
is on the fraction rather than on the factor."""

SCORE_RESIZE = 256
"""Side length the map is resampled to before the image score is read off it.

A fixed square, independent of the prepared size, and it is the reference's. It makes the
score's neighbourhood — the blur below, and the top-one-percent count — the same number of
map cells whatever the experiment's input size is, which is defensible; it also means the
score is read from a resampling of the map rather than from the map, which is a wart. Kept
because the point of this implementation is to be measurable against the wrapper, and the
wrapper inherits it."""

SCORE_BLUR_KERNEL = 5
SCORE_BLUR_SIGMA = 4.0
"""The reference's score-path smoothing: a 5-tap Gaussian at sigma 4, so heavily truncated
that the kernel is nearly flat. Reproduced exactly rather than tidied."""

SCORE_TOP_FRACTION = 0.01
"""Fraction of the smoothed map averaged into the image score — the reference's top 1%."""


# ------------------------------------------------------------------------- layers


class FeedForward(nn.Module):
    """The two-layer MLP that is both the bottleneck and every decoder block's feed-forward.

    One class for both, because they differ in exactly two ways and neither justifies a
    second module: the bottleneck drops its *input* before the first projection (the decoder
    block does not), and the two are configured with different widths. Biases are off
    throughout, which is the reference's choice and not an oversight — the LayerNorm in front
    of each decoder MLP already carries one.
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        *,
        dropout: float = 0.0,
        input_dropout: bool = False,
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features, bias=False)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=False)
        self.drop = nn.Dropout(dropout)
        self.input_dropout = input_dropout

    def forward(self, tokens: Tensor) -> Tensor:
        if self.input_dropout:
            tokens = self.drop(tokens)
        tokens = self.drop(self.act(self.fc1(tokens)))
        projected: Tensor = self.drop(self.fc2(tokens))
        return projected


class LinearAttention(nn.Module):
    """Softmax-free attention over the token sequence, with an ELU feature map.

    `phi(q) . (phi(k)^T v)` normalised by `phi(q) . sum(phi(k))`, with `phi(x) = ELU(x) + 1`
    keeping both factors positive. Associativity is what makes it linear in sequence length,
    but that is a side benefit here. Dinomaly's argument is about *capacity*: softmax
    attention concentrates on the tokens most like the query, so a decoder built from it
    reconstructs an anomalous patch from itself and the anomaly disappears from the residual.
    Attention that cannot focus is attention that has to generalise.

    `scale` is computed and deliberately unused — the reference carries the same dead
    parameter, and removing it here would leave one more difference to explain at the gate.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        *,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        if dim % num_heads:
            msg = f"a {dim}-wide token sequence does not split into {num_heads} heads"
            raise ValueError(msg)
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, tokens: Tensor) -> Tensor:
        batch, length, dim = tokens.shape
        qkv = (
            self.qkv(tokens)
            .reshape(batch, length, 3, self.num_heads, dim // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        query, key, value = qkv[0], qkv[1], qkv[2]
        query = functional.elu(query) + 1.0
        key = functional.elu(key) + 1.0

        context = torch.matmul(key.transpose(-2, -1), value)
        key_sum = key.sum(dim=-2, keepdim=True)
        normaliser = 1.0 / torch.sum(query * key_sum, dim=-1, keepdim=True)
        attended = torch.matmul(query, context) * normaliser

        merged = attended.transpose(1, 2).reshape(batch, length, dim)
        projected: Tensor = self.proj_drop(self.proj(merged))
        return projected


class DecoderBlock(nn.Module):
    """Pre-norm transformer block: linear attention, then feed-forward, both residual."""

    def __init__(self, dim: int, num_heads: int, *, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=LAYER_NORM_EPS)
        self.attn = LinearAttention(dim, num_heads, proj_drop=dropout)
        self.norm2 = nn.LayerNorm(dim, eps=LAYER_NORM_EPS)
        self.mlp = FeedForward(dim, dim * MLP_RATIO, dim, dropout=dropout)

    def forward(self, tokens: Tensor) -> Tensor:
        tokens = tokens + self.attn(self.norm1(tokens))
        residual: Tensor = tokens + self.mlp(self.norm2(tokens))
        return residual


# -------------------------------------------------------------------------- the net


def _fuse(features: Sequence[Tensor]) -> Tensor:
    """Element-wise mean of several token sequences — the reference's only fusion."""
    return torch.stack(list(features), dim=1).mean(dim=1)


class DinomalyNet(nn.Module):
    """Frozen encoder, trainable bottleneck and decoder, and the two fused feature groups.

    The encoder is held as a submodule so that `.to(device)` and `.eval()` reach it, but it
    is **never** in `trainable_parameters()` and its tokens are read under `no_grad`. Only
    the bottleneck and the decoder are optimised; that is what "frozen" means here and it is
    asserted rather than assumed, because a stray `requires_grad` would silently put 22
    million encoder parameters into the optimizer's first param group.
    """

    def __init__(
        self,
        encoder: nn.Module,
        *,
        embedding_dim: int,
        num_heads: int,
        num_prefix_tokens: int,
        patch_size: int,
        decoder_depth: int,
        target_layers: tuple[int, ...],
        encoder_groups: tuple[tuple[int, ...], ...],
        decoder_groups: tuple[tuple[int, ...], ...],
        dropout: float,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.embedding_dim = embedding_dim
        self.num_prefix_tokens = num_prefix_tokens
        self.patch_size = patch_size
        self.target_layers = target_layers
        self.encoder_groups = encoder_groups
        self.decoder_groups = decoder_groups

        self.bottleneck = FeedForward(
            embedding_dim,
            embedding_dim * MLP_RATIO,
            embedding_dim,
            dropout=dropout,
            input_dropout=True,
        )
        # Dropout reaches the bottleneck and not the decoder blocks. That asymmetry is the
        # reference's and it is deliberate: the bottleneck is the narrowing that stops the
        # decoder from learning the identity, so it is where the regularisation belongs.
        self.decoder = nn.ModuleList(
            DecoderBlock(embedding_dim, num_heads, dropout=0.0) for _ in range(decoder_depth)
        )

    # ---------------------------------------------------------------- parameters

    def trainable_parameters(self) -> list[nn.Parameter]:
        """Exactly the bottleneck and the decoder, in a stable order.

        Taken from the two modules by name rather than by filtering `self.parameters()` on
        `requires_grad`. The filter would give the same answer today and would quietly start
        including the encoder the first time something un-froze it.
        """
        return [*self.bottleneck.parameters(), *self.decoder.parameters()]

    def trainable_state(self) -> dict[str, dict[str, Any]]:
        return {
            "bottleneck": self.bottleneck.state_dict(),
            "decoder": self.decoder.state_dict(),
        }

    def load_trainable_state(self, state: dict[str, dict[str, Any]]) -> None:
        self.bottleneck.load_state_dict(state["bottleneck"])
        self.decoder.load_state_dict(state["decoder"])

    # ---------------------------------------------------------------- the forward

    def features(self, batch: Tensor) -> tuple[list[Tensor], list[Tensor]]:
        """`(encoder_groups, decoder_groups)` as `(B, D, rows, cols)` spatial maps.

        The order is the mechanism and worth stating: the decoder's outputs are **reversed**
        before grouping, so the *deepest* decoder block is paired with the *shallowest*
        encoder layers. A decoder is an inverted encoder, and grouping it the other way round
        would compare the two ends of two stacks that mean opposite things.
        """
        rows = batch.shape[2] // self.patch_size
        cols = batch.shape[3] // self.patch_size

        with torch.no_grad():
            encoder_tokens = extract_layer_tokens(self.encoder, batch, self.target_layers)

        tokens = self.bottleneck(_fuse(encoder_tokens))
        decoder_tokens: list[Tensor] = []
        for block in self.decoder:
            tokens = block(tokens)
            decoder_tokens.append(tokens)
        decoder_tokens.reverse()

        encoded = [_fuse([encoder_tokens[i] for i in group]) for group in self.encoder_groups]
        decoded = [_fuse([decoder_tokens[i] for i in group]) for group in self.decoder_groups]
        return (
            [self._to_spatial(item, rows, cols) for item in encoded],
            [self._to_spatial(item, rows, cols) for item in decoded],
        )

    def _to_spatial(self, tokens: Tensor, rows: int, cols: int) -> Tensor:
        patches = tokens[:, self.num_prefix_tokens :, :]
        batch = patches.shape[0]
        return patches.permute(0, 2, 1).reshape(batch, -1, rows, cols).contiguous()


# ---------------------------------------------------------------------------- loss


def mined_fraction(step: int, final: float) -> float:
    """The fraction of hardest points kept at full gradient at `step`.

    Annealed **downwards** from 1.0, which is the reference's schedule written the way this
    plugin's config names it: the reference stores `p`, the fraction of *easy* points to
    down-weight, ramps it up from 0 to `p_final` over 1000 steps, and mines `1 - p`. So a
    run starts with every point at full gradient and narrows to `final` by step 1000.
    """
    if step < 0:
        msg = f"training step cannot be negative; got {step}"
        raise ValueError(msg)
    down_weighted = min((1.0 - final) * step / HARD_MINING_ANNEAL_STEPS, 1.0 - final)
    return 1.0 - down_weighted


def hard_mined_cosine_loss(
    encoded: Sequence[Tensor],
    decoded: Sequence[Tensor],
    *,
    step: int,
    fraction: float,
    factor: float = EASY_GRADIENT_FACTOR,
) -> Tensor:
    """Mean cosine distance per group, with the easy points' gradient shrunk by `factor`.

    Two things happen and only one of them is the number that gets logged.

    The **value** is a cosine distance between each batch item's whole flattened feature map
    and its reconstruction — one scalar per group, averaged. It is not the mean of the
    per-point distances, and reading it as one is the easiest mistake to make here.

    The **gradient** is where the mining lives. A backward hook multiplies the gradient at
    every point whose distance falls below the `fraction`-quantile threshold by `factor`, so
    the decoder spends its capacity on what it cannot yet reproduce. That is the point of the
    method: a decoder that reconstructs everything perfectly reconstructs defects too, and
    the anomaly map goes flat. The threshold is recomputed each step under `no_grad`.

    Encoder features are detached, so nothing here can reach the frozen backbone.
    """
    if not encoded or len(encoded) != len(decoded):
        msg = (
            "the hard-mined loss needs the same non-zero number of encoder and decoder "
            f"groups; got {len(encoded)} and {len(decoded)}"
        )
        raise ValueError(msg)
    if not 0.0 < fraction <= 1.0:
        msg = f"the mined fraction must lie in (0, 1]; got {fraction}"
        raise ValueError(msg)

    total = torch.zeros((), device=encoded[0].device, dtype=encoded[0].dtype)
    for source, target in zip(encoded, decoded, strict=True):
        frozen = source.detach()
        with torch.no_grad():
            point_distance = 1.0 - functional.cosine_similarity(frozen, target, dim=1)
            point_distance = point_distance.unsqueeze(1)
            keep = max(1, int(point_distance.numel() * fraction))
            threshold = torch.topk(point_distance.reshape(-1), k=keep).values[-1]

        flat_source = frozen.reshape(frozen.shape[0], -1)
        flat_target = target.reshape(target.shape[0], -1)
        total = total + torch.mean(1.0 - functional.cosine_similarity(flat_source, flat_target))

        easy = point_distance < threshold
        # `Tensor.register_hook` is untyped in torch's stubs and absent altogether in the
        # torch-free install, so the receiver is widened rather than silenced twice over.
        hooked: Any = target
        hooked.register_hook(_shrink_gradient(easy, factor))

    return total / len(encoded)


def _shrink_gradient(easy: Tensor, factor: float) -> Callable[[Tensor], Tensor]:
    """A backward hook that scales the gradient at the well-reconstructed points."""

    def hook(gradient: Tensor) -> Tensor:
        mask = easy.expand_as(gradient)
        scaled = gradient.clone()
        scaled[mask] = scaled[mask] * factor
        return scaled

    return hook


# ------------------------------------------------------------------------- the map


def gaussian_kernel1d(kernel_size: int, sigma: float, *, device: Any, dtype: Any) -> Tensor:
    """One normalised 1-D Gaussian tap set, centred on an odd-length window."""
    if kernel_size < 1 or kernel_size % 2 == 0:
        msg = f"a centred Gaussian kernel needs an odd positive length; got {kernel_size}"
        raise ValueError(msg)
    radius = kernel_size // 2
    offsets = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    weights = torch.exp(-offsets.pow(2) / (2.0 * sigma * sigma))
    return weights / weights.sum()


def gaussian_blur(image: Tensor, *, kernel_size: int, sigma: float) -> Tensor:
    """A separable Gaussian with reflect padding — `2k` work rather than `k²`.

    In-module rather than through torchvision or kornia, for the reason
    `pixel_reference.gaussian_blur` is in numpy: this is the only filtering the method needs.
    Reflect padding and the exact tap set match what the reference's kornia-backed blur
    produces, so the score path is comparable term by term.
    """
    channels = int(image.shape[1])
    taps = gaussian_kernel1d(kernel_size, sigma, device=image.device, dtype=image.dtype)
    radius = kernel_size // 2
    horizontal = taps.view(1, 1, 1, -1).expand(channels, 1, 1, -1)
    vertical = taps.view(1, 1, -1, 1).expand(channels, 1, -1, 1)
    padded = functional.pad(image, (radius, radius, 0, 0), mode="reflect")
    blurred = functional.conv2d(padded, horizontal, groups=channels)
    padded = functional.pad(blurred, (0, 0, radius, radius), mode="reflect")
    return functional.conv2d(padded, vertical, groups=channels)


def group_maps(
    encoded: Sequence[Tensor],
    decoded: Sequence[Tensor],
    size: tuple[int, int],
) -> list[Tensor]:
    """One `(B, 1, H, W)` cosine-distance map per fused group, at the prepared size.

    `align_corners=True` here and `False` on the score resample below: both are the
    reference's, and the inconsistency is its own rather than a transcription slip.
    """
    maps = []
    for source, target in zip(encoded, decoded, strict=True):
        distance = (1.0 - functional.cosine_similarity(source, target, dim=1)).unsqueeze(1)
        maps.append(
            functional.interpolate(distance, size=size, mode="bilinear", align_corners=True)
        )
    return maps


def image_score(anomaly_map: Tensor) -> Tensor:
    """One score per batch item: the mean of the smoothed map's hottest one percent.

    The map is resampled to a fixed square first, then blurred, then read — see
    `SCORE_RESIZE`. A `max` would be one pixel's opinion; a mean over the whole map would be
    dominated by the intact majority of any real part.
    """
    resized = functional.interpolate(
        anomaly_map, size=SCORE_RESIZE, mode="bilinear", align_corners=False
    )
    smoothed = gaussian_blur(resized, kernel_size=SCORE_BLUR_KERNEL, sigma=SCORE_BLUR_SIGMA)
    flat = smoothed.flatten(1)
    keep = max(1, int(flat.shape[1] * SCORE_TOP_FRACTION))
    return torch.sort(flat, dim=1, descending=True).values[:, :keep].mean(dim=1)


# --------------------------------------------------------------------- the optimizer


class StableAdamW(torch.optim.Optimizer):
    """AdamW with the update, not the gradient, clipped — arXiv:2304.13013.

    Ordinary gradient clipping asks how large the gradient is; StableAdamW asks how large the
    *step* it produces is, by taking the RMS of `grad / denom` — the ratio Adam actually
    applies — and dividing the step size down when that exceeds `clip_threshold`. A gradient
    spike that Adam's own second moment already absorbs therefore costs nothing, while one it
    does not is damped. That property is why the Dinomaly recipe uses it, and why this is
    ported rather than replaced with `torch.optim.AdamW`: at 2e-3 on a freshly initialised
    decoder the difference is whether the first hundred steps diverge.

    Decoupled weight decay is applied to the parameter directly, before the moments are
    updated, which is where the reference applies it.
    """

    def __init__(
        self,
        params: Iterable[Any],
        *,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
        amsgrad: bool = False,
        clip_threshold: float = 1.0,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"learning rate must not be negative; got {lr}")
        if eps < 0.0:
            raise ValueError(f"epsilon must not be negative; got {eps}")
        if not 0.0 <= betas[0] < 1.0 or not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"both betas must lie in [0, 1); got {betas}")
        if weight_decay < 0.0:
            raise ValueError(f"weight decay must not be negative; got {weight_decay}")
        if clip_threshold <= 0.0:
            raise ValueError(f"the clip threshold must be positive; got {clip_threshold}")
        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
            "amsgrad": amsgrad,
            "clip_threshold": clip_threshold,
        }
        super().__init__(params, defaults)

    def __setstate__(self, state: dict[str, Any]) -> None:
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("amsgrad", False)
            group.setdefault("clip_threshold", 1.0)

    @staticmethod
    def _rms(tensor: Tensor) -> Tensor:
        root_mean_square: Tensor = tensor.norm(2) / (tensor.numel() ** 0.5)
        return root_mean_square

    def step(self, closure: Callable[[], float] | None = None) -> float | None:  # type: ignore[override]
        """One update. Every write goes through `.data`, so no `no_grad` scope is needed."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            amsgrad = group["amsgrad"]
            for parameter in group["params"]:
                gradient = parameter.grad
                if gradient is None:
                    continue
                if gradient.is_sparse:
                    msg = "StableAdamW does not support sparse gradients"
                    raise RuntimeError(msg)

                parameter.data.mul_(1 - group["lr"] * group["weight_decay"])
                state = self.state[parameter]
                if not state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(parameter)
                    state["exp_avg_sq"] = torch.zeros_like(parameter)
                    if amsgrad:
                        state["max_exp_avg_sq"] = torch.zeros_like(parameter)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                state["step"] += 1
                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]

                exp_avg.mul_(beta1).add_(gradient, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(gradient, gradient, value=1 - beta2)
                if amsgrad:
                    maximum = state["max_exp_avg_sq"]
                    torch.max(maximum, exp_avg_sq, out=maximum)
                    denominator = (maximum.sqrt() / math.sqrt(bias_correction2)).add_(group["eps"])
                else:
                    denominator = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(
                        group["eps"]
                    )

                # The clip looks at the update Adam is about to make, not at the gradient.
                update_rms = self._rms(gradient / denominator)
                damping = max(1.0, float(update_rms) / group["clip_threshold"])
                step_size = group["lr"] / bias_correction1 / damping
                parameter.data.addcdiv_(exp_avg, denominator, value=-step_size)

        return loss
