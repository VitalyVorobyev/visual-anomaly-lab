#!/usr/bin/env -S uv run --project backend --extra dl python
"""Which device should an in-house DINO memory-bank method run each of its kernels on?

ADR-0008 requires a standalone probe **run first**, before any wrapper is written against a
new library. `scripts/mps-smoke-test.py` has already paid for itself once and
`scripts/patchcore-smoke-test.py` produced the two caps that plugin is bounded by. This is
the same discipline aimed at the method that comes next: a frozen DINO backbone whose patch
features feed a nearest-neighbour bank and a per-position Mahalanobis fit.

That method is five kernels, and **nothing in the application would reveal which device
each of them wants**. A run finishes and the numbers are correct whichever device it used;
it is merely some multiple slower, or it raises deep inside a loop because MPS has no Metal
kernel for a batched Cholesky. Both of those are findings that belong in a file before they
belong in a plugin.

Nine stages, cheapest first:

    1. do both encoder families instantiate at all (unpretrained — no download)
    2. does `forward_intermediates` run at 448x448 on CPU and on MPS, at the right grid
    3. the chunked distance expansion, MPS against CPU, and how far apart they land
    4. the top-k over that distance matrix, MPS against CPU
    5. the per-position kNN einsum
    6. the per-position Mahalanobis einsum
    7. batched Cholesky and `cholesky_solve` — the one MPS may simply not have
    8. bilinear upsampling and a separable Gaussian blur, the map post-processing
    9. what the encoder forward costs per image, both families, both devices

Run it with no arguments:

    ./scripts/dino-memory-smoke-test.py

Every stage is measured on **both** devices and none of them is required to pass on MPS.
Exit code 0 means every kernel has at least one device that computes it correctly, and the
per-stage verdict names which — a recorded fact about this machine rather than a device
uncertainty carried into the plugin. A non-zero exit means a kernel works nowhere, which is
a finding about the method's design rather than about its configuration.

Nothing here downloads weights: every model is built with `pretrained=False`, because what
is being measured is arithmetic and the arithmetic does not care what the weights are.
"""

from __future__ import annotations

import platform
import sys
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

RULE = "-" * 78

DINOV2 = "vit_small_patch14_reg4_dinov2.lvd142m"
"""The registered DINOv2 encoder anomalib pins for Dinomaly — /14, so 448 gives a 32x32 grid."""

DINOV3 = "vit_small_patch16_dinov3.lvd1689m"
"""DINOv3 ViT-S — /16, so the same 448 gives 28x28. Both resolve without a download here."""

PREPARED = 448
"""One prepared size that divides by both 14 and 16, which is why the plugin's two families
can be compared on identical pixels rather than on two resizes."""

QUERY_PATCHES = 1024
"""A 32x32 grid: one image's worth of query positions, which is what inference holds."""

FEATURE_DIM = 768
"""Two L2-normalized ViT-S layers concatenated. The width the distance kernels see."""

BANK_VECTORS = 100_000
"""`patchcore-smoke-test.py` measured this as the pool a class produces under its caps."""

CHUNK = 2048
"""Bank rows per distance chunk. The whole (1024, 100000) matrix is 409 MB in float32;
building it in one allocation is what a chunked expansion exists to avoid."""

NEIGHBORS = 9
"""PatchCore's k, kept so the two methods' inference costs are read on the same axis."""

POSITIONS = 1024
COVARIANCE_DIM = 128
"""One Mahalanobis fit per patch position, over a projected feature. 128 keeps a 1024-batch
of covariance matrices at 64 MB in float32 rather than 2.4 GB at the full 768."""


@dataclass
class Stage:
    name: str
    ok: bool
    detail: str
    rows: list[str] = field(default_factory=list)
    verdict: str = ""


def _run(name: str, body: Callable[[], Stage]) -> Stage:
    """Run one stage, turning any exception into a reported failure rather than a crash."""
    try:
        return body()
    except Exception as exc:  # noqa: BLE001 - reporting the failure *is* the job here
        first_line = str(exc).strip().splitlines()[0] if str(exc).strip() else repr(exc)
        return Stage(name=name, ok=False, detail=f"{type(exc).__name__}: {first_line}")


def _devices() -> list[str]:
    """Which devices to measure on. CPU always; MPS when this machine has it."""
    import torch

    if torch.backends.mps.is_available():
        return ["mps", "cpu"]
    return ["cpu"]


def _sync(device: str) -> None:
    import torch

    if device == "mps":
        torch.mps.synchronize()


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[len(ordered) // 2]


def _timed(device: str, call: Callable[[], Any], repeats: int = 3) -> tuple[Any, float]:
    """Median seconds over `repeats`, discarding the first — it pays for kernel compilation."""
    result: Any = None
    timings: list[float] = []
    for index in range(repeats + 1):
        start = time.perf_counter()
        result = call()
        _sync(device)
        if index > 0:
            timings.append(time.perf_counter() - start)
    return result, _median(timings)


def _max_delta(left: Any, right: Any) -> float:
    """Largest absolute disagreement between a device's answer and the CPU's."""
    import torch

    return float(torch.max(torch.abs(left.detach().cpu().float() - right.detach().cpu().float())))


# ------------------------------------------------------- stage 1: the model definitions


def stage_definitions() -> Stage:
    """Do both families exist in this timm, and what do they say about themselves?

    Unpretrained on purpose. A model definition that resolves is a different fact from
    weights that download, and only the first one is needed before the arithmetic below.
    """
    import timm

    rows: list[str] = []
    for name in (DINOV2, DINOV3):
        model = timm.create_model(name, pretrained=False, num_classes=0, dynamic_img_size=True)
        model.eval()
        parameters = sum(p.numel() for p in model.parameters())
        rows.append(
            f"    {name:<42} embed {model.embed_dim}   depth {len(model.blocks)}   "
            f"patch {model.patch_embed.patch_size[0]}   {parameters / 1e6:.1f}M params"
        )
        del model

    return Stage(
        name="model definitions",
        ok=True,
        detail=f"timm {timm.__version__}, both families instantiate with no download",
        rows=rows,
        verdict="both families are available unpretrained",
    )


# --------------------------------------------------- stage 2: the intermediate features


def stage_intermediates() -> Stage:
    """`forward_intermediates` at the prepared size, on every device, at the right grid.

    The grid is *read* rather than derived from an assumed stride, for the reason
    `patchcore-smoke-test.py` reads its own: every memory number downstream is a product of
    it, and it is a property of the encoder rather than a constant worth writing down.
    """
    import timm
    import torch

    rows: list[str] = []
    expected = {DINOV2: (PREPARED // 14, PREPARED // 14), DINOV3: (PREPARED // 16, PREPARED // 16)}
    mismatched: list[str] = []
    non_finite: list[str] = []

    for name in (DINOV2, DINOV3):
        model = timm.create_model(name, pretrained=False, num_classes=0, dynamic_img_size=True)
        model.eval()
        for device in _devices():
            on_device = model.to(device)
            batch = torch.randn(1, 3, PREPARED, PREPARED, device=device)
            with torch.no_grad():
                maps, seconds = _timed(
                    device,
                    lambda m=on_device, b=batch: m.forward_intermediates(  # type: ignore[misc]
                        b,
                        indices=[-2, -1],
                        norm=True,
                        output_fmt="NCHW",
                        intermediates_only=True,
                    ),
                )
            grid = (int(maps[0].shape[-2]), int(maps[0].shape[-1]))
            finite = all(bool(torch.isfinite(feature).all()) for feature in maps)
            if grid != expected[name]:
                mismatched.append(f"{name} on {device}: {grid} != {expected[name]}")
            if not finite:
                non_finite.append(f"{name} on {device}")
            rows.append(
                f"    {device:>4}  {name.split('.')[0]:<38} grid {grid[0]}x{grid[1]}   "
                f"{len(maps)} layers x {int(maps[0].shape[1])} ch   "
                f"{seconds * 1000:7.1f} ms   finite {finite}"
            )
            del batch, maps
        model.to("cpu")
        del model
        rows.append("")

    if mismatched:
        msg = "patch grid is not what the patch size implies: " + "; ".join(mismatched)
        raise RuntimeError(msg)
    if non_finite:
        msg = "non-finite intermediates from: " + ", ".join(non_finite)
        raise RuntimeError(msg)

    return Stage(
        name="forward_intermediates",
        ok=True,
        detail=f"{PREPARED}x{PREPARED}, indices [-2, -1], norm=True, NCHW",
        rows=rows,
        verdict="grids are 32x32 (/14) and 28x28 (/16); every device returns finite features",
    )


# ------------------------------------------------------- stage 3: the distance expansion


def _chunked_distances(query: Any, bank: Any) -> Any:
    """`||x||^2 + ||y||^2 - 2 x y^T`, one bank chunk at a time.

    The expansion rather than `cdist` because the bank's squared norms are computed once and
    reused for every image, which is the whole reason a bank is worth precomputing. Chunked
    because the full matrix is the largest thing inference allocates and a chunk bounds it.
    """
    import torch

    query_norm = query.pow(2).sum(dim=1, keepdim=True)
    out = torch.empty(query.shape[0], bank.shape[0], device=query.device, dtype=query.dtype)
    for start in range(0, bank.shape[0], CHUNK):
        block = bank[start : start + CHUNK]
        block_norm = block.pow(2).sum(dim=1).unsqueeze(0)
        out[:, start : start + block.shape[0]] = (
            query_norm + block_norm - 2.0 * (query @ block.T)
        ).clamp_min_(0.0)
    return out


def stage_distances() -> Stage:
    """The inner loop of every query, measured on both devices and compared between them.

    The comparison matters more than either timing. The expansion is a subtraction of two
    large similar numbers, so it loses precision where `cdist` would not, and a device that
    reorders the accumulation lands somewhere slightly different. How far apart is a number
    the plugin's tolerances have to be set from rather than guessed at.
    """
    import torch

    torch.manual_seed(0)
    query_cpu = torch.randn(QUERY_PATCHES, FEATURE_DIM)
    bank_cpu = torch.randn(BANK_VECTORS, FEATURE_DIM)
    query_cpu = torch.nn.functional.normalize(query_cpu, dim=1)
    bank_cpu = torch.nn.functional.normalize(bank_cpu, dim=1)

    rows: list[str] = []
    reference: Any = None
    matrix_mb = QUERY_PATCHES * BANK_VECTORS * 4 / 1e6

    # CPU first, so the reference the other devices are compared against already exists.
    for device in reversed(_devices()):
        query = query_cpu.to(device)
        bank = bank_cpu.to(device)
        distances, seconds = _timed(device, lambda q=query, b=bank: _chunked_distances(q, b))
        if device == "cpu":
            reference = distances.detach().cpu()
            delta = "reference"
        else:
            delta = f"max |delta| vs cpu {_max_delta(distances, reference):.2e}"
        rows.append(
            f"    {device:>4}: {seconds * 1000:8.1f} ms   {matrix_mb:.0f} MB result   {delta}"
        )
        del query, bank, distances

    return Stage(
        name="chunked distance expansion",
        ok=True,
        detail=(
            f"query {QUERY_PATCHES}x{FEATURE_DIM} vs bank {BANK_VECTORS:,}x{FEATURE_DIM}, "
            f"float32, chunk {CHUNK}"
        ),
        rows=rows,
        verdict="both devices compute the expansion; the delta is what a tolerance is set from",
    )


# ---------------------------------------------------------------- stage 4: the top-k


def stage_topk() -> Stage:
    """`topk` over the whole distance matrix — small arithmetic, very large tensor.

    Correctness first and timing second, and correctness here has two halves that have to be
    separated. The *values* are the score; the *indices* are an identity, and PatchCore's
    reweighting reads both. A device whose values differ is computing a different method. A
    device that merely orders equal values differently is not — but if the indices are used
    as an identity across a device boundary, it will look like one.

    The matrix is deliberately tie-rich: `torch.rand` in float32 puts genuinely equal values
    in a 100 000-wide row often enough that the tie-break shows up at this size, which is
    exactly the case a smoke test should be arranging rather than hoping to avoid.
    """
    import torch

    torch.manual_seed(1)
    matrix_cpu = torch.rand(QUERY_PATCHES, BANK_VECTORS)
    total = QUERY_PATCHES * NEIGHBORS

    rows: list[str] = []
    reference_values: Any = None
    reference_indices: Any = None
    values_agree = True
    disagreements = 0
    ties = 0
    cpu_seconds = 0.0
    mps_seconds = 0.0

    for device in reversed(_devices()):  # cpu first, so the reference exists
        matrix = matrix_cpu.to(device)
        (values, indices), seconds = _timed(
            device, lambda m=matrix: m.topk(k=NEIGHBORS, largest=False, dim=1)
        )
        if device == "cpu":
            cpu_seconds = seconds
            reference_values, reference_indices = values, indices.detach().cpu()
            rows.append(f"    {device:>4}: {seconds * 1000:8.1f} ms   reference")
        else:
            mps_seconds = seconds
            delta = _max_delta(values, reference_values)
            differing = indices.detach().cpu() != reference_indices
            disagreements = int(differing.sum())
            # A disagreement at a position whose *value* still matches the reference is a
            # tie broken the other way, not a different neighbour set.
            ties = int((differing & (values.detach().cpu() == reference_values)).sum())
            values_agree = delta == 0.0
            rows.append(
                f"    {device:>4}: {seconds * 1000:8.1f} ms   max |delta| {delta:.2e}   "
                f"index disagreements {disagreements}/{total} "
                f"({ties} of them exact ties)"
            )
        del matrix, values, indices

    if not values_agree:
        verdict = (
            "select neighbours on the CPU: MPS returns different distances, so the two "
            "devices are not computing one method"
        )
    else:
        if not mps_seconds:
            placement = "the CPU is the only device this machine has for it"
        elif mps_seconds > cpu_seconds:
            placement = (
                f"take the top-k on the CPU — MPS is {mps_seconds / cpu_seconds:.0f}x "
                "slower here, a reduction with too little arithmetic to cover its dispatch"
            )
        else:
            placement = f"take the top-k on MPS — {cpu_seconds / mps_seconds:.1f}x faster"
        exactness = (
            f"; MPS is value-exact but broke {disagreements}/{total} ties the other way, so "
            "a neighbour index is never a cross-device identity"
            if disagreements
            else "; every device agrees on the values and the indices"
        )
        verdict = placement + exactness

    return Stage(
        name="top-k neighbour selection",
        ok=True,
        detail=f"k={NEIGHBORS}, largest=False, over ({QUERY_PATCHES}, {BANK_VECTORS:,})",
        rows=rows,
        verdict=verdict,
    )


# ------------------------------------------------------------ stage 5: the kNN einsum


def stage_knn_einsum() -> Stage:
    """`pd,pkd->pk`: each position against its own K neighbours.

    This is what a *spatially indexed* bank costs, where PatchCore's flat search is one
    matmul against everything. It is a batched inner product with a batch of P, and whether
    MPS covers that shape's dispatch is exactly the kind of thing that is cheap to measure
    now and expensive to discover inside a scoring loop.
    """
    import torch

    torch.manual_seed(2)
    neighbors_k = 64
    query_cpu = torch.randn(POSITIONS, FEATURE_DIM)
    neighbours_cpu = torch.randn(POSITIONS, neighbors_k, FEATURE_DIM)

    rows: list[str] = []
    reference: Any = None
    for device in reversed(_devices()):
        query = query_cpu.to(device)
        neighbours = neighbours_cpu.to(device)
        scores, seconds = _timed(
            device,
            lambda q=query, n=neighbours: torch.einsum("pd,pkd->pk", q, n),
        )
        if device == "cpu":
            reference = scores
            rows.append(f"    {device:>4}: {seconds * 1000:8.2f} ms   reference")
        else:
            rows.append(
                f"    {device:>4}: {seconds * 1000:8.2f} ms   "
                f"max |delta| vs cpu {_max_delta(scores, reference):.2e}"
            )
        del query, neighbours, scores

    return Stage(
        name="per-position kNN einsum",
        ok=True,
        detail=f"pd,pkd->pk at P={POSITIONS}, K={neighbors_k}, D={FEATURE_DIM}",
        rows=rows,
        verdict="the batched inner product runs on every device",
    )


# ---------------------------------------------------- stage 6: the Mahalanobis einsum


def stage_mahalanobis_einsum() -> Stage:
    """`pd,pde,pe->p`: one quadratic form per patch position.

    The scoring half of a per-position Gaussian fit, once the inverse covariance exists.
    Stage 7 is about whether that inverse can be produced at all; this is about whether
    using it is cheap.
    """
    import torch

    torch.manual_seed(3)
    residual_cpu = torch.randn(POSITIONS, COVARIANCE_DIM)
    precision_cpu = torch.randn(POSITIONS, COVARIANCE_DIM, COVARIANCE_DIM)

    rows: list[str] = []
    reference: Any = None
    for device in reversed(_devices()):
        residual = residual_cpu.to(device)
        precision = precision_cpu.to(device)
        scores, seconds = _timed(
            device,
            lambda r=residual, p=precision: torch.einsum("pd,pde,pe->p", r, p, r),
        )
        if device == "cpu":
            reference = scores
            rows.append(f"    {device:>4}: {seconds * 1000:8.2f} ms   reference")
        else:
            rows.append(
                f"    {device:>4}: {seconds * 1000:8.2f} ms   "
                f"max |delta| vs cpu {_max_delta(scores, reference):.2e}"
            )
        del residual, precision, scores

    return Stage(
        name="per-position Mahalanobis einsum",
        ok=True,
        detail=f"pd,pde,pe->p at P={POSITIONS}, D={COVARIANCE_DIM}",
        rows=rows,
        verdict="the quadratic form runs on every device",
    )


# ------------------------------------------------------------- stage 7: the covariance


def _spd_batch(device: str, dtype: Any) -> Any:
    """A batch of genuinely positive-definite matrices, built the way a fit builds them."""
    import torch

    torch.manual_seed(4)
    samples = torch.randn(POSITIONS, COVARIANCE_DIM * 2, COVARIANCE_DIM)
    covariance = samples.transpose(1, 2) @ samples / (COVARIANCE_DIM * 2)
    covariance = covariance + torch.eye(COVARIANCE_DIM) * 1e-3
    return covariance.to(device=device, dtype=dtype)


def stage_cholesky() -> Stage:
    """Batched `cholesky` and `cholesky_solve` — the one kernel MPS may simply not have.

    A per-position Gaussian fit needs an inverse covariance per position, and the numerically
    respectable way to get one is a Cholesky factor rather than `inverse`. MPS's LAPACK
    coverage is the thinnest part of the backend, so this stage is written to *report* a
    failure as the finding it is rather than to fall over on it, and the CPU path is run in
    float64 because a near-singular covariance is where float32 stops being enough.
    """
    import torch

    rows: list[str] = []
    mps_seconds: float | None = None
    mps_note = "does not carry this kernel"

    for device in _devices():
        if device == "cpu":
            continue
        try:
            covariance = _spd_batch(device, torch.float32)
            residual = torch.randn(POSITIONS, COVARIANCE_DIM, 1, device=device)
            (factor, solved), seconds = _timed(
                device,
                lambda c=covariance, r=residual: _cholesky_pair(c, r),
            )
            finite = bool(torch.isfinite(factor).all() and torch.isfinite(solved).all())
            if finite:
                mps_seconds = seconds
            else:
                mps_note = "produced non-finite output"
            rows.append(
                f"    {device:>4} float32: {seconds * 1000:8.1f} ms   finite {finite}"
                + ("" if finite else "   (produced non-finite output)")
            )
            del covariance, residual, factor, solved
        except Exception as exc:  # noqa: BLE001 - a missing kernel is the result, not a crash
            mps_note = f"raised {type(exc).__name__}"
            rows.append(
                f"    {device:>4} float32: {type(exc).__name__}: {str(exc).splitlines()[0]}"
            )

    covariance = _spd_batch("cpu", torch.float64)
    residual = torch.randn(POSITIONS, COVARIANCE_DIM, 1, dtype=torch.float64)
    (factor, solved), cpu_seconds = _timed(
        "cpu", lambda c=covariance, r=residual: _cholesky_pair(c, r)
    )
    reconstruction = _max_delta(factor @ factor.transpose(1, 2), covariance)
    rows.append(
        f"     cpu float64: {cpu_seconds * 1000:8.1f} ms   finite True   "
        f"max |LL^T - A| {reconstruction:.2e}"
    )
    if not bool(torch.isfinite(solved).all()):
        msg = "the CPU float64 Cholesky produced non-finite output; there is no fallback left"
        raise RuntimeError(msg)

    if mps_seconds is None:
        verdict = f"fit the covariance on the CPU in float64 — MPS {mps_note}"
    elif mps_seconds > cpu_seconds:
        verdict = (
            f"fit the covariance on the CPU in float64: MPS runs it but "
            f"{mps_seconds / cpu_seconds:.0f}x slower, and float64 is the dtype a "
            "near-singular covariance needs anyway"
        )
    else:
        verdict = (
            f"MPS float32 is {cpu_seconds / mps_seconds:.1f}x faster, but the fit happens "
            "once per run — prefer CPU float64 unless the fit becomes the bottleneck"
        )
    return Stage(
        name="batched Cholesky + solve",
        ok=True,
        detail=f"({POSITIONS}, {COVARIANCE_DIM}, {COVARIANCE_DIM}) SPD batch",
        rows=rows,
        verdict=verdict,
    )


def _cholesky_pair(covariance: Any, residual: Any) -> tuple[Any, Any]:
    import torch

    factor = torch.linalg.cholesky(covariance)
    return factor, torch.cholesky_solve(residual, factor)


# ------------------------------------------------------- stage 8: the map post-processing


def _gaussian_kernel(sigma: float, device: str) -> Any:
    """A 1D Gaussian, truncated at four sigma and normalized — the separable half of a blur."""
    import torch

    radius = max(1, int(4.0 * sigma))
    positions = torch.arange(-radius, radius + 1, dtype=torch.float32, device=device)
    weights = torch.exp(-positions.pow(2) / (2.0 * sigma * sigma))
    return weights / weights.sum()


def _separable_blur(image: Any, kernel: Any) -> Any:
    """Two depthwise passes rather than one square kernel: `2k` work instead of `k^2`."""
    import torch.nn.functional as functional

    channels = image.shape[1]
    radius = (kernel.numel() - 1) // 2
    horizontal = kernel.view(1, 1, 1, -1).expand(channels, 1, 1, -1)
    vertical = kernel.view(1, 1, -1, 1).expand(channels, 1, -1, 1)
    padded = functional.pad(image, (radius, radius, 0, 0), mode="reflect")
    blurred = functional.conv2d(padded, horizontal, groups=channels)
    padded = functional.pad(blurred, (0, 0, radius, radius), mode="reflect")
    return functional.conv2d(padded, vertical, groups=channels)


def stage_map_postprocessing() -> Stage:
    """Upsampling a patch grid to the frame, then smoothing it.

    Every method here ends the same way and it is the cheapest part of the pass, which is
    precisely why it is worth confirming rather than assuming: an operator with no Metal
    kernel *here* would force a device transfer per image for arithmetic that costs
    microseconds.
    """
    import torch
    import torch.nn.functional as functional

    rows: list[str] = []
    grid = PREPARED // 14
    reference: Any = None

    for device in reversed(_devices()):
        torch.manual_seed(5)
        patch_map = torch.rand(1, 1, grid, grid).to(device)
        kernel = _gaussian_kernel(4.0, device)
        upsampled, up_seconds = _timed(
            device,
            lambda m=patch_map: functional.interpolate(
                m, size=(PREPARED, PREPARED), mode="bilinear", align_corners=False
            ),
        )
        blurred, blur_seconds = _timed(device, lambda u=upsampled, k=kernel: _separable_blur(u, k))
        if device == "cpu":
            reference = blurred
            rows.append(
                f"    {device:>4}: interpolate {up_seconds * 1000:6.2f} ms   "
                f"blur {blur_seconds * 1000:6.2f} ms   reference"
            )
        else:
            rows.append(
                f"    {device:>4}: interpolate {up_seconds * 1000:6.2f} ms   "
                f"blur {blur_seconds * 1000:6.2f} ms   "
                f"max |delta| vs cpu {_max_delta(blurred, reference):.2e}"
            )
        del patch_map, upsampled, blurred

    return Stage(
        name="map upsampling + separable blur",
        ok=True,
        detail=f"{grid}x{grid} -> {PREPARED}x{PREPARED} bilinear, depthwise Gaussian sigma 4",
        rows=rows,
        verdict="the map post-processing runs wherever the map already is",
    )


# ------------------------------------------------------------------- stage 9: the timing


def stage_timing() -> Stage:
    """What the encoder forward actually costs, per image, on each device.

    Unpretrained weights are fine here: a matmul takes the same time whatever numbers are in
    it. Batch 4 because that is the shape a bank-building pass uses, and the per-image figure
    is what a run's log will be quoting.
    """
    import timm
    import torch

    rows: list[str] = [
        f"    {'encoder':<40} {'device':>6}  {'ms/batch-4':>11}  {'ms/image':>9}  speedup"
    ]
    for name in (DINOV2, DINOV3):
        model = timm.create_model(name, pretrained=False, num_classes=0, dynamic_img_size=True)
        model.eval()
        seconds: dict[str, float] = {}
        for device in _devices():
            on_device = model.to(device)
            batch = torch.randn(4, 3, PREPARED, PREPARED, device=device)
            with torch.no_grad():
                _, elapsed = _timed(
                    device,
                    lambda m=on_device, b=batch: m.forward_intermediates(  # type: ignore[misc]
                        b,
                        indices=[-2, -1],
                        norm=True,
                        output_fmt="NCHW",
                        intermediates_only=True,
                    ),
                )
            seconds[device] = elapsed
            del batch
        for device, elapsed in seconds.items():
            speedup = (
                f"{seconds['cpu'] / elapsed:.1f}x"
                if device != "cpu" and seconds.get("cpu")
                else "-"
            )
            rows.append(
                f"    {name.split('.')[0]:<40} {device:>6}  {elapsed * 1000:11.1f}  "
                f"{elapsed * 1000 / 4:9.1f}  {speedup:>7}"
            )
        model.to("cpu")
        del model

    return Stage(
        name="encoder forward timing",
        ok=True,
        detail=f"{PREPARED}x{PREPARED}, batch 4, indices [-2, -1], unpretrained weights",
        rows=rows,
        verdict="the encoder forward is the stage that wants the accelerator",
    )


STAGES: list[tuple[str, Callable[[], Stage]]] = [
    ("model definitions", stage_definitions),
    ("forward_intermediates", stage_intermediates),
    ("chunked distance expansion", stage_distances),
    ("top-k neighbour selection", stage_topk),
    ("per-position kNN einsum", stage_knn_einsum),
    ("per-position Mahalanobis einsum", stage_mahalanobis_einsum),
    ("batched Cholesky + solve", stage_cholesky),
    ("map upsampling + separable blur", stage_map_postprocessing),
    ("encoder forward timing", stage_timing),
]

FATAL = {"model definitions", "forward_intermediates"}
"""Without an encoder that runs, every later stage measures arithmetic nothing will feed."""


def main() -> int:
    import torch

    print(RULE)
    print("DINO memory-bank smoke test — run before writing the method (ADR-0008)")
    print(f"torch {torch.__version__} on {platform.machine()} / macOS {platform.mac_ver()[0]}")
    print(f"MPS available: {torch.backends.mps.is_available()}")
    print(RULE)

    results: list[Stage] = []
    for name, body in STAGES:
        stage = _run(name, body)
        results.append(stage)
        mark = "ok  " if stage.ok else "FAIL"
        print(f"[{mark}] {name}: {stage.detail}")
        for row in stage.rows:
            print(row)
        if stage.ok and stage.verdict:
            print(f"    -> {stage.verdict}")
        print()
        if not stage.ok and name in FATAL:
            print(RULE)
            print("VERDICT: the encoder does not run here. There is nothing to place.")
            traceback.print_exc(file=sys.stderr)
            return 1

    print(RULE)
    print("summary")
    print(RULE)
    for stage in results:
        mark = "ok  " if stage.ok else "FAIL"
        print(f"  [{mark}] {stage.name:<34} {stage.verdict or stage.detail}")
    print(RULE)

    failed = [stage.name for stage in results if not stage.ok]
    if failed:
        print(
            f"VERDICT: {len(failed)} kernel(s) compute correctly on no device: "
            f"{', '.join(failed)}"
        )
        print("That is a finding about the method's design, not about its configuration.")
        return 1
    print("VERDICT: every kernel has a device. Place each one from the verdicts above.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
