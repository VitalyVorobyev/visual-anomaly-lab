#!/usr/bin/env -S uv run --project backend --extra dl python
"""Does this Mac actually train EfficientAD on MPS? Answer that before writing a wrapper.

ADR-0008 requires this check to exist as a **standalone script run first**, not as a
discovery made halfway through an integration. MPS has real gaps: an operator with no
Metal kernel raises at the moment it is reached, which — inside a training loop — looks
like a bug in the code that called it rather than a missing kernel.

The script escalates deliberately, cheapest first, and reports the exact operator that
fails rather than a traceback the reader has to interpret:

    1. is MPS built and available at all
    2. do the elementary ops used by every model work
    3. do the ops EfficientAD specifically leans on work
    4. does anomalib's EfficientAd module do a forward and a backward pass
    5. how long does one training step take, against the same step on CPU

Run it with no arguments:

    ./scripts/mps-smoke-test.py

Exit code 0 means the device named in the summary is usable for M3. A non-zero exit is a
finding, and the wrapper's `preferred_device` should be reconsidered rather than the
failure worked around silently.
"""

from __future__ import annotations

import platform
import sys
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass

RULE = "-" * 72


@dataclass
class Check:
    name: str
    ok: bool
    detail: str


def _run(name: str, body: Callable[[], str]) -> Check:
    """Run one check, turning any exception into a reported failure rather than a crash."""
    try:
        return Check(name=name, ok=True, detail=body())
    except Exception as exc:  # noqa: BLE001 - reporting the failure *is* the job here
        first_line = str(exc).strip().splitlines()[0] if str(exc).strip() else repr(exc)
        return Check(name=name, ok=False, detail=f"{type(exc).__name__}: {first_line}")


def check_availability() -> str:
    import torch

    if not torch.backends.mps.is_built():
        msg = "this torch build has no MPS backend compiled in"
        raise RuntimeError(msg)
    if not torch.backends.mps.is_available():
        msg = "MPS is built but unavailable (needs macOS 12.3+ on Apple Silicon)"
        raise RuntimeError(msg)
    return f"torch {torch.__version__} on {platform.machine()} / macOS {platform.mac_ver()[0]}"


def check_elementary() -> str:
    """Matmul, reduction and autograd — if these fail nothing else is worth trying."""
    import torch

    a = torch.randn(256, 256, device="mps", requires_grad=True)
    b = torch.randn(256, 256, device="mps")
    loss = (a @ b).pow(2).mean()
    loss.backward()
    if a.grad is None:
        msg = "backward produced no gradient"
        raise RuntimeError(msg)
    return "matmul, pow, mean and backward all resolved"


def check_efficientad_ops() -> str:
    """The specific operators EfficientAD's PDN, autoencoder and loss reach for.

    Named individually because "MPS failed" is not actionable and "avg_pool2d has no
    Metal kernel" is.
    """
    import torch
    import torch.nn.functional as functional

    x = torch.randn(2, 3, 256, 256, device="mps")
    ops: list[tuple[str, Callable[[], torch.Tensor]]] = [
        ("conv2d", lambda: functional.conv2d(x, torch.randn(8, 3, 4, 4, device="mps"))),
        ("avg_pool2d", lambda: functional.avg_pool2d(x, kernel_size=2, stride=2)),
        ("max_pool2d", lambda: functional.max_pool2d(x, kernel_size=2, stride=2)),
        (
            "interpolate/bilinear",
            lambda: functional.interpolate(x, size=(64, 64), mode="bilinear"),
        ),
        ("pad/reflect", lambda: functional.pad(x, (2, 2, 2, 2), mode="reflect")),
        ("quantile", lambda: torch.quantile(x.flatten()[:16384], 0.9)),
        ("std_mean", lambda: torch.std_mean(x)[0]),
        ("conv_transpose2d", lambda: functional.conv_transpose2d(
            x, torch.randn(3, 8, 2, 2, device="mps"), stride=2)),
    ]

    failures: list[str] = []
    for label, call in ops:
        try:
            call()
            torch.mps.synchronize()
        except Exception as exc:  # noqa: BLE001 - collect every gap, do not stop at the first
            failures.append(f"{label} ({type(exc).__name__})")

    if failures:
        msg = "no Metal kernel for: " + ", ".join(failures)
        raise RuntimeError(msg)
    return f"all {len(ops)} operators resolved"


def check_anomalib_forward() -> str:
    """A real EfficientAd module, a real forward, a real backward.

    The penalty batch is **not optional**. `batch_imagenet` defaults to `None` in the
    signature, but training dereferences it unconditionally — passing nothing fails with
    `'NoneType' object has no attribute 'device'`, which reads like a bug in the caller.
    This check therefore always supplies one, and the wrapper treats the penalty set as a
    hard requirement of training rather than an enhancement.
    """
    import torch
    from anomalib.models import EfficientAd

    model = EfficientAd()
    torch_model = model.model.to("mps")
    torch_model.train()

    batch = torch.randn(1, 3, 256, 256, device="mps")
    penalty = torch.randn(1, 3, 256, 256, device="mps")
    output = torch_model(batch, batch_imagenet=penalty)

    # anomalib 2.x returns a loss tuple in train mode and an InferenceBatch in eval mode.
    # Both shapes are accepted here; what is being tested is that the graph runs at all.
    if isinstance(output, tuple):
        loss = sum(term for term in output if torch.is_tensor(term))
    else:
        loss = output.pred_score.sum() if hasattr(output, "pred_score") else output.sum()

    if torch.is_tensor(loss) and loss.requires_grad:
        loss.backward()
    torch.mps.synchronize()

    parameters = sum(p.numel() for p in torch_model.parameters())
    return f"EfficientAd forward+backward on MPS, {parameters / 1e6:.1f}M parameters"


def _time_step(device: str, steps: int = 3) -> float:
    """Median seconds for one forward+backward at 256x256, batch 1."""
    import torch
    from anomalib.models import EfficientAd

    torch_model = EfficientAd().model.to(device)
    torch_model.train()
    optimizer = torch.optim.Adam(torch_model.parameters(), lr=1e-4)
    batch = torch.randn(1, 3, 256, 256, device=device)
    penalty = torch.randn(1, 3, 256, 256, device=device)

    timings: list[float] = []
    for index in range(steps + 1):
        start = time.perf_counter()
        optimizer.zero_grad()
        output = torch_model(batch, batch_imagenet=penalty)
        loss = (
            sum(term for term in output if torch.is_tensor(term))
            if isinstance(output, tuple)
            else output.sum()
        )
        if torch.is_tensor(loss) and loss.requires_grad:
            loss.backward()
            optimizer.step()
        if device == "mps":
            torch.mps.synchronize()
        # The first step pays for lazy kernel compilation and is not representative.
        if index > 0:
            timings.append(time.perf_counter() - start)

    timings.sort()
    return timings[len(timings) // 2]


def check_timing() -> str:
    mps_seconds = _time_step("mps")
    cpu_seconds = _time_step("cpu")
    speedup = cpu_seconds / mps_seconds if mps_seconds > 0 else float("nan")
    return (
        f"one train step: MPS {mps_seconds * 1000:.0f} ms, "
        f"CPU {cpu_seconds * 1000:.0f} ms ({speedup:.1f}x)"
    )


CHECKS: list[tuple[str, Callable[[], str]]] = [
    ("MPS availability", check_availability),
    ("elementary ops + autograd", check_elementary),
    ("EfficientAD operator coverage", check_efficientad_ops),
    ("anomalib EfficientAd forward/backward", check_anomalib_forward),
    ("training step timing", check_timing),
]


def main() -> int:
    print(RULE)
    print("MPS smoke test — run before trusting preferred_device='mps' (ADR-0008)")
    print(RULE)

    results: list[Check] = []
    for name, body in CHECKS:
        check = _run(name, body)
        results.append(check)
        mark = "ok  " if check.ok else "FAIL"
        print(f"[{mark}] {name}: {check.detail}")
        # Every later check assumes the device works, so a failure here makes the
        # remaining output noise rather than information.
        if not check.ok and name in {"MPS availability", "elementary ops + autograd"}:
            print(RULE)
            print("VERDICT: MPS unusable. Set preferred_device='cpu' and record why.")
            traceback.print_exc(file=sys.stderr)
            return 1

    print(RULE)
    failed = [check.name for check in results if not check.ok]
    if failed:
        print(f"VERDICT: MPS partially usable — {len(failed)} check(s) failed: {', '.join(failed)}")
        print("Decide per failure whether to fall back to CPU or pin the operator down.")
        return 1
    print("VERDICT: MPS is usable for EfficientAD training.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
