#!/usr/bin/env -S uv run --project backend --extra dl python
"""What does PatchCore actually cost on this Mac? Measure it before writing the wrapper.

ADR-0008 requires a standalone probe **run first**, and `scripts/mps-smoke-test.py` has
already paid for itself once. This is the same discipline applied to a method whose cost is
not a step budget but a memory footprint: PatchCore holds every training patch in one
tensor, then runs a greedy selection whose loop count *is* the size of the memory bank.

Left at anomalib's defaults, at 256x256 with `wide_resnet50_2` and `layer2+layer3`, a full
VisA class means a 5.66 GB embedding store and 92 160 Python iterations each computing an
L2 norm over 921 600 rows. The plugin bounds both. **The bounds should be numbers this
script measured, not numbers somebody guessed** — which is what it exists to produce.

Five stages, cheapest first:

    1. does the timm backbone run on MPS at all, and what grid does it really produce
    2. what does anomalib's SparseRandomProjection cost, and what dimension does it pick
    3. what does one greedy iteration cost, on MPS and on CPU, as N grows
    4. what does one image's nearest-neighbour search cost against a realistic bank
    5. how much resident memory did all of that take

Run it with no arguments:

    ./scripts/patchcore-smoke-test.py

Stage 1 downloads the backbone weights on a cold cache (~260 MB for `wide_resnet50_2`).
Exit code 0 means MPS is usable for PatchCore. A non-zero exit is a finding, and
`preferred_device` or the greedy loop's device should be reconsidered rather than the
failure worked around silently.
"""

from __future__ import annotations

import platform
import resource
import sys
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

RULE = "-" * 78

BACKBONE = "wide_resnet50_2"
LAYERS = ("layer2", "layer3")
IMAGE_SIZE = (256, 256)

# The three candidate-pool sizes the plugin's `max_candidate_vectors` default sits among.
# 100k is the intended landing point; 25k and 250k bracket it so the scaling is visible
# rather than extrapolated from one point.
POOL_SIZES = (25_000, 100_000, 250_000)

GREEDY_ITERATIONS = 60
"""Enough to time an iteration past warm-up, few enough that the probe stays a probe.

The real loop runs `coreset_ratio * N` times, so the reported per-iteration cost is what
gets multiplied out — the point of the stage is the multiplier, not a full selection.
"""

BANK_VECTORS = 10_000
"""What the plugin's defaults produce: 0.1 of a 100k candidate pool."""


@dataclass
class Stage:
    name: str
    ok: bool
    detail: str
    rows: list[str] = field(default_factory=list)


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


# ------------------------------------------------------------------ stage 1: the backbone


def stage_backbone() -> Stage:
    """The real patch grid and embedding dimension, from a real forward pass.

    Read rather than derived from an assumed stride: the whole memory calculation is
    `images x grid x dim x 4`, and every one of those three factors is a property of the
    backbone and the input size rather than a constant anybody should be writing down.
    """
    import torch
    from anomalib.models.components.feature_extractors import TimmFeatureExtractor

    rows: list[str] = []
    grid: tuple[int, int] | None = None
    dim: int | None = None

    for device in _devices():
        extractor = TimmFeatureExtractor(
            backbone=BACKBONE, pre_trained=True, layers=list(LAYERS)
        ).eval()
        extractor = extractor.to(device)
        probe = torch.randn(1, 3, *IMAGE_SIZE, device=device)

        timings: list[float] = []
        features: dict[str, Any] = {}
        for index in range(4):
            start = time.perf_counter()
            with torch.no_grad():
                features = extractor(probe)
            _sync(device)
            # The first pass pays for lazy kernel compilation and is not representative.
            if index > 0:
                timings.append(time.perf_counter() - start)

        shallow = features[LAYERS[0]]
        grid = (int(shallow.shape[-2]), int(shallow.shape[-1]))
        dim = sum(int(features[layer].shape[1]) for layer in LAYERS)
        per_image_mb = grid[0] * grid[1] * dim * 4 / 1e6
        rows.append(
            f"    {device:>4}: {_median(timings) * 1000:7.1f} ms/image   "
            f"grid {grid[0]}x{grid[1]}   dim {dim}   {per_image_mb:.2f} MB/image"
        )

    if grid is None or dim is None:  # pragma: no cover - _devices() is never empty
        msg = "no device produced a forward pass"
        raise RuntimeError(msg)

    patches = grid[0] * grid[1]
    rows.append("")
    rows.append(f"    a 900-image class would generate {patches * 900:,} patches "
                f"= {patches * 900 * dim * 4 / 1e9:.2f} GB")
    return Stage(
        name="backbone forward",
        ok=True,
        detail=f"{BACKBONE} / {'+'.join(LAYERS)} at {IMAGE_SIZE[0]}x{IMAGE_SIZE[1]}",
        rows=rows,
    )


# -------------------------------------------------------------- stage 2: the projection


def stage_projection() -> Stage:
    """What anomalib's SparseRandomProjection costs, and the dimension it picks.

    The dimension matters more than the time: every greedy iteration is an L2 norm over
    `N x n_components`, so this number is a factor in stage 3's multiplier.
    """
    import torch
    from anomalib.models.components.dimensionality_reduction import SparseRandomProjection

    rows: list[str] = []
    n = 100_000
    dim = 1536

    for device in _devices():
        embedding = torch.randn(n, dim, device=device)
        projector = SparseRandomProjection(eps=0.9)

        start = time.perf_counter()
        projector.fit(embedding)
        features = projector.transform(embedding)
        _sync(device)
        elapsed = time.perf_counter() - start

        components = int(features.shape[1])
        rows.append(
            f"    {device:>4}: {elapsed:6.2f} s   {dim} -> {components} dims   "
            f"projected store {n * components * 4 / 1e6:.0f} MB"
        )
        del embedding, features, projector

    return Stage(
        name="sparse random projection",
        ok=True,
        detail=f"N={n:,}, D={dim} (eps=0.9)",
        rows=rows,
    )


# ------------------------------------------------------------- stage 3: the greedy loop


def _greedy_iteration_cost(device: str, n: int, components: int) -> float:
    """Median seconds for one k-center-greedy iteration — the plugin's actual inner loop.

    Written here exactly as the plugin will write it, so this measures the code that will
    run rather than a proxy for it. A norm over every row, a running minimum, an argmax,
    and zeroing the picked row.
    """
    import torch

    features = torch.randn(n, components, device=device)
    min_distances = torch.full((n,), float("inf"), device=device)
    index = torch.zeros((), dtype=torch.long, device=device)

    timings: list[float] = []
    for step in range(GREEDY_ITERATIONS + 1):
        start = time.perf_counter()
        distances = torch.linalg.norm(features - features[index], ord=2, dim=1)
        min_distances = torch.minimum(min_distances, distances)
        index = torch.argmax(min_distances)
        min_distances[index] = 0.0
        _sync(device)
        if step > 0:
            timings.append(time.perf_counter() - start)

    del features, min_distances
    return _median(timings)


def stage_greedy() -> Stage:
    """The number that sets `max_candidate_vectors`.

    The selection runs `coreset_ratio * N` times, so its total cost is **quadratic in N**.
    That is the cliff the plugin's second cap exists for, and reading it off a straight
    line drawn through three points is the only honest way to pick where the cap goes.
    """
    # What stage 2 measures `SparseRandomProjection(eps=0.9)` picking for D=1536 at this
    # N. Hardcoded so this stage is independent of stage 2 having run, and kept in step
    # with it by hand rather than by a global.
    components = 284
    rows: list[str] = []

    for device in _devices():
        for n in POOL_SIZES:
            try:
                per_iteration = _greedy_iteration_cost(device, n, components)
            except Exception as exc:  # noqa: BLE001 - a device that cannot hold it is a result
                rows.append(f"    {device:>4}  N={n:>7,}: {type(exc).__name__}")
                continue
            # A coreset at the default ratio, which is how many iterations actually run.
            selection = per_iteration * n * 0.1
            rows.append(
                f"    {device:>4}  N={n:>7,}: {per_iteration * 1000:6.2f} ms/iteration   "
                f"-> {int(n * 0.1):>6,} iterations = {selection:6.1f} s"
            )
        rows.append("")

    return Stage(
        name="greedy selection",
        ok=True,
        detail=f"projected to {components} dims, coreset_ratio=0.1",
        rows=rows,
    )


# ---------------------------------------------------------------- stage 4: the NN search


def stage_search() -> Stage:
    """One image's inference cost against a realistic bank.

    PatchCore's inference is a dense distance matrix between an image's patches and the
    whole bank. It scales with the bank, which is the other side of the sizing trade: a
    bank small enough to build is also a bank fast enough to query.
    """
    import torch
    from anomalib.models.image.patchcore.torch_model import PatchcoreModel

    rows: list[str] = []
    patches = 1024
    dim = 1536

    for device in _devices():
        bank = torch.randn(BANK_VECTORS, dim, device=device)
        query = torch.randn(patches, dim, device=device)

        timings: list[float] = []
        for index in range(4):
            start = time.perf_counter()
            distances = PatchcoreModel.euclidean_dist(query, bank)
            distances.topk(k=9, largest=False, dim=1)
            _sync(device)
            if index > 0:
                timings.append(time.perf_counter() - start)

        rows.append(
            f"    {device:>4}: {_median(timings) * 1000:6.1f} ms/image   "
            f"bank {BANK_VECTORS:,} x {dim} = {BANK_VECTORS * dim * 4 / 1e6:.0f} MB"
        )
        del bank, query

    return Stage(
        name="nearest-neighbour search",
        ok=True,
        detail=f"{patches} query patches, k=9",
        rows=rows,
    )


# ------------------------------------------------------------------- stage 5: the memory


def stage_memory() -> Stage:
    """Peak resident memory for everything above.

    Not a precise attribution to any one stage — it is the high-water mark of the process,
    which is the number that decides whether the machine swaps.
    """
    peak_bytes = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return Stage(
        name="peak resident memory",
        ok=True,
        detail=f"{peak_bytes / 1e9:.2f} GB high-water mark for this process",
    )


STAGES: list[tuple[str, Callable[[], Stage]]] = [
    ("backbone forward", stage_backbone),
    ("sparse random projection", stage_projection),
    ("greedy selection", stage_greedy),
    ("nearest-neighbour search", stage_search),
    ("peak resident memory", stage_memory),
]


def main() -> int:
    import torch

    print(RULE)
    print("PatchCore smoke test — run before writing the wrapper (ADR-0008)")
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
        # Every later stage assumes the backbone runs; without it the rest is noise.
        if not stage.ok and name == "backbone forward":
            print(RULE)
            print("VERDICT: the backbone does not run. Reconsider preferred_device.")
            traceback.print_exc(file=sys.stderr)
            return 1

    print(RULE)
    failed = [stage.name for stage in results if not stage.ok]
    if failed:
        print(f"VERDICT: {len(failed)} stage(s) failed: {', '.join(failed)}")
        return 1
    print("VERDICT: MPS is usable for PatchCore. Set the plugin defaults from the numbers above.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
