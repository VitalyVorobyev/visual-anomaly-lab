"""Coreset selection and random projection, with no library attached.

`greedy_select` began life inside `patchcore_anomalib.py` and is torch-only: it never
touched anomalib, and the module docstring there explains at length why the *loop* is ours
while the *rule* stays the paper's. That makes it the one piece of that plugin a second
method can reuse without importing anomalib, which an in-house method must not do — so it
lives here, and `patchcore_anomalib` imports it back. Nothing about it changed in the move.

Beside it sit the two pieces a bank needs before a selection can run at all. PatchCore gets
them from anomalib's `SparseRandomProjection`; an in-house method cannot, so the same job is
done here from `torch` alone:

  * `johnson_lindenstrauss_dim` says how few dimensions still preserve the distances the
    greedy rule compares, which is the number that sets the cost of every iteration;
  * `gaussian_projection` is the dense classical construction of that embedding — one
    seeded normal matrix — where anomalib's is the sparse variant. Both satisfy the same
    lemma; the dense one is a matmul and needs no scikit-learn, and so no second random
    stream to forget to seed (M6).

Heavy imports stay inside functions. This module is imported by plan-time code that runs in
the API process, where a three-second torch import would be paid for nothing.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any


def johnson_lindenstrauss_dim(count: int, *, eps: float) -> int:
    """The smallest projected width that keeps pairwise distances within `eps`.

    The classical Johnson-Lindenstrauss bound, `4 ln(n) / (eps^2/2 - eps^3/3)`, truncated
    the way scikit-learn's `johnson_lindenstrauss_min_dim` truncates it and clamped to at
    least one dimension. What it bounds is *relative distortion*: after projection, every
    pairwise distance among the `count` points is within a factor `1 +- eps` of the
    original, with high probability, and the bound depends on the number of points alone —
    never on how wide they started.

    `eps = 0.9` is PatchCore's operating point (anomalib's `SparseRandomProjection` default,
    named as `_PROJECTION_EPS` in `patchcore_anomalib`). It is a loose tolerance chosen on
    purpose: k-center-greedy needs the *ordering* of distances rather than their values, and
    at eps 0.9 a 1536-wide embedding of 100 000 patches projects to 284 dimensions — a
    fivefold cut in the cost of every one of the loop's iterations.
    """
    if count < 1:
        msg = f"a projection bound needs at least one point; got {count}"
        raise ValueError(msg)
    if not 0.0 < eps < 1.0:
        msg = f"eps must lie in (0, 1); got {eps}"
        raise ValueError(msg)
    denominator = eps**2 / 2.0 - eps**3 / 3.0
    return max(1, int(4.0 * math.log(count) / denominator))


def gaussian_projection(features: Any, out_dim: int, *, seed: int) -> Any:
    """Project `(n, d)` features to `(n, out_dim)` through one seeded normal matrix.

    The dense Johnson-Lindenstrauss construction: entries drawn `N(0, 1)` and scaled by
    `1/sqrt(out_dim)`, which is what makes the projection distance-preserving in expectation
    rather than merely smaller.

    The matrix is drawn on the **CPU** from an explicit `torch.Generator` and then moved to
    wherever the features are. Both halves of that are deliberate. An explicit generator
    keeps the draw out of torch's global stream, so seeding here cannot perturb a caller's
    weight initialisation and a caller's seeding cannot perturb this; drawing on the CPU
    makes one seed give one matrix regardless of the device the features happen to live on,
    which is the difference between a reproducible bank and a bank that is reproducible per
    machine.
    """
    import torch

    dim = int(features.shape[1])
    if out_dim < 1:
        msg = f"a projection needs at least one output dimension; got {out_dim}"
        raise ValueError(msg)

    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    matrix = torch.randn(dim, out_dim, generator=generator, dtype=torch.float32)
    matrix = matrix / math.sqrt(out_dim)
    return features @ matrix.to(device=features.device, dtype=features.dtype)


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
