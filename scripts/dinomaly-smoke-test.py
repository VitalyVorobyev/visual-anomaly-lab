#!/usr/bin/env -S uv run --project backend --extra dl python
"""Measure whether anomalib's Dinomaly is a credible Apple-Silicon reference.

This is an integration probe, not application code.  It runs before the plugin exists so
the plugin's defaults and bounds are consequences of measurements rather than guesses.
Each device/size leg runs in a fresh subprocess: peak RSS is otherwise a high-water mark
from every model that happened to run before it.

The first candidate is intentionally the small registered DINOv2 encoder.  Dinomaly's
default base encoder is a quality target, but promoting it before measuring the small
variant would make a large memory commitment without knowing whether the operator graph
works on MPS at all.

Run the paper-shaped gate with:

    ./scripts/dinomaly-smoke-test.py

The first run may download the public DINOv2 weights through timm/Hugging Face.  The
summary prints an SHA-256 digest over the resolved encoder state, plus exact package
versions, so that the downloaded asset is recorded as an input rather than an invisible
environment detail.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import resource
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any

ENCODER = "vit_small_patch14_reg4_dinov2"
DECODER_DEPTH = 8
SIZES = (196, 392)
RULE = "-" * 82


@dataclass(frozen=True)
class Measurement:
    device: str
    size: int
    construct_seconds: float
    train_ms: float
    infer_ms: float
    loss: float
    total_parameters: int
    trainable_parameters: int
    checkpoint_bytes: int
    peak_rss_bytes: int
    mps_allocated_bytes: int | None
    mps_driver_bytes: int | None
    encoder_sha256: str


def _sync(device: str) -> None:
    import torch

    if device == "mps":
        torch.mps.synchronize()


def _tensor_digest(state: dict[str, Any]) -> str:
    """Stable digest over tensor names, shapes, dtypes and bytes."""
    digest = hashlib.sha256()
    for key in sorted(state):
        tensor = state[key].detach().cpu().contiguous()
        digest.update(key.encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _peak_rss_bytes() -> int:
    """Normalize ``ru_maxrss``: bytes on Darwin, KiB on Linux."""
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if platform.system() == "Darwin" else value * 1024


def _worker(device: str, size: int) -> Measurement:
    import torch
    from anomalib.models import Dinomaly
    from anomalib.models.image.dinomaly.components import StableAdamW

    if size % 14:
        raise ValueError(
            f"Dinomaly input size must be divisible by patch size 14; got {size}"
        )

    # Seed immediately before construction: anomalib initialises the trainable bottleneck
    # and decoder from torch's global stream.
    torch.manual_seed(0)
    started = time.perf_counter()
    wrapper = Dinomaly(
        encoder_name=ENCODER,
        decoder_depth=DECODER_DEPTH,
        pre_processor=False,
        post_processor=False,
        evaluator=False,
        visualizer=False,
    )
    model = wrapper.model.to(device)
    construct_seconds = time.perf_counter() - started

    trainable = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer = StableAdamW(
        [{"params": trainable}],
        lr=2e-3,
        betas=(0.9, 0.999),
        weight_decay=1e-4,
        amsgrad=True,
        eps=1e-8,
    )

    # Generate on CPU so both devices receive byte-identical synthetic pixels.
    generator = torch.Generator(device="cpu").manual_seed(1)
    batch = torch.randn(1, 3, size, size, generator=generator).to(device)

    train_timings: list[float] = []
    loss_value = float("nan")
    model.train()
    for step in range(3):
        started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        loss = model(batch, global_step=step)
        loss.backward()
        optimizer.step()
        _sync(device)
        if step:
            train_timings.append(time.perf_counter() - started)
        loss_value = float(loss.detach().cpu())

    infer_timings: list[float] = []
    model.eval()
    with torch.inference_mode():
        for step in range(4):
            started = time.perf_counter()
            prediction = model(batch)
            _sync(device)
            if step:
                infer_timings.append(time.perf_counter() - started)
    if prediction.anomaly_map is None or prediction.anomaly_map.shape[-2:] != (
        size,
        size,
    ):
        raise RuntimeError("Dinomaly did not return a prepared-frame anomaly map")

    # The plugin checkpoint will own only the trainable reconstruction network and
    # optimizer state.  Encoder weights are external and are verified by the digest.
    payload = {
        "bottleneck": model.bottleneck.state_dict(),
        "decoder": model.decoder.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    checkpoint = Path(os.environ.get("TMPDIR", "/tmp")) / (
        f"dinomaly-smoke-{os.getpid()}-{device}-{size}.pt"
    )
    torch.save(payload, checkpoint)
    checkpoint_bytes = checkpoint.stat().st_size
    checkpoint.unlink()

    mps_allocated = None
    mps_driver = None
    if device == "mps":
        mps_allocated = int(torch.mps.current_allocated_memory())
        mps_driver = int(torch.mps.driver_allocated_memory())

    return Measurement(
        device=device,
        size=size,
        construct_seconds=construct_seconds,
        train_ms=statistics.median(train_timings) * 1000,
        infer_ms=statistics.median(infer_timings) * 1000,
        loss=loss_value,
        total_parameters=sum(parameter.numel() for parameter in model.parameters()),
        trainable_parameters=sum(parameter.numel() for parameter in trainable),
        checkpoint_bytes=checkpoint_bytes,
        peak_rss_bytes=_peak_rss_bytes(),
        mps_allocated_bytes=mps_allocated,
        mps_driver_bytes=mps_driver,
        encoder_sha256=_tensor_digest(model.encoder.state_dict()),
    )


def _run_worker(device: str, size: int) -> Measurement:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        device,
        str(size),
    ]
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.returncode:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"{device} at {size}px failed:\n{detail}")
    line = completed.stdout.strip().splitlines()[-1]
    return Measurement(**json.loads(line))


def _gib(value: int | None) -> str:
    return "-" if value is None else f"{value / 1024**3:.2f} GiB"


def _main() -> int:
    import torch

    devices = ["cpu"]
    if torch.backends.mps.is_available():
        devices.insert(0, "mps")

    print(RULE)
    print("Dinomaly resource gate — small DINOv2, depth 8, batch 1")
    print(RULE)
    print(
        f"python {platform.python_version()} | torch {torch.__version__} | "
        f"anomalib {version('anomalib')} | timm {version('timm')}"
    )
    print(f"{platform.machine()} | macOS {platform.mac_ver()[0]} | encoder {ENCODER}")
    print()

    results: list[Measurement] = []
    failed = False
    for size in SIZES:
        for device in devices:
            try:
                result = _run_worker(device, size)
            except Exception as exc:  # noqa: BLE001 - every failed leg belongs in the report
                failed = True
                print(f"[FAIL] {device:>3} {size}px: {exc}")
                continue
            results.append(result)
            print(
                f"[ ok ] {device:>3} {size}px | train {result.train_ms:7.1f} ms | "
                f"infer {result.infer_ms:6.1f} ms | RSS {_gib(result.peak_rss_bytes)} | "
                f"MPS driver {_gib(result.mps_driver_bytes)}"
            )

    print()
    if results:
        first = results[0]
        print(
            f"parameters: {first.total_parameters / 1e6:.2f}M total, "
            f"{first.trainable_parameters / 1e6:.2f}M trainable"
        )
        print(
            f"checkpoint after one optimizer step: {first.checkpoint_bytes / 1e6:.1f} MB"
        )
        print(f"encoder SHA-256: {first.encoder_sha256}")

    if failed:
        print("\nResult: NOT READY — at least one required device/size leg failed.")
        return 1
    print("\nResult: COMPATIBLE — resource numbers still decide the plugin defaults.")
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", nargs=2, metavar=("DEVICE", "SIZE"))
    return parser.parse_args()


if __name__ == "__main__":
    arguments = _parse_args()
    if arguments.worker:
        measurement = _worker(arguments.worker[0], int(arguments.worker[1]))
        print(json.dumps(asdict(measurement), sort_keys=True))
        raise SystemExit(0)
    raise SystemExit(_main())
