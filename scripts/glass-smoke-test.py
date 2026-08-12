#!/usr/bin/env -S uv run --project backend --extra dl python
"""Measure whether anomalib's GLASS is a credible Apple-Silicon reference.

This gate precedes the application plugin.  It measures the complete GLASS training
mechanism: frozen ImageNet features, trainable projection and discriminator, image-space
Perlin synthesis, feature-space Gaussian synthesis, the bounded gradient-ascent mining
loop, backward/optimizer steps, and anomaly-map inference.

Every device/size leg runs in a fresh subprocess so peak RSS and MPS high-water marks do
not leak between profiles.  Synthetic source pixels are generated on the CPU from one
seed.  The first update and inference pass are warm-up and are not included in medians.

The resource gate deliberately uses GLASS's built-in Perlin-only local synthesis.  The
optional DTD texture corpus is a separate public asset whose value must be decided by the
public-data quality gate; omitting it here proves the core algorithm has no hidden
download beyond the fingerprinted ImageNet backbone.

Run:

    ./scripts/glass-smoke-test.py
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

BACKBONE = "wide_resnet50_2"
PROFILES = ((144, 1), (288, 1), (288, 8))
MINING_STEPS = 20
SEED = 20260812
RULE = "-" * 82


@dataclass(frozen=True)
class Measurement:
    device: str
    size: int
    batch_size: int
    construct_seconds: float
    center_ms: float
    train_ms: float
    infer_ms: float
    loss: float
    total_parameters: int
    optimized_parameters: int
    checkpoint_bytes: int
    peak_rss_bytes: int
    mps_allocated_bytes: int | None
    mps_driver_bytes: int | None
    backbone_sha256: str


def _sync(device: str) -> None:
    import torch

    if device == "mps":
        torch.mps.synchronize()


def _tensor_digest(state: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    for key in sorted(state):
        tensor = state[key].detach().cpu().contiguous()
        digest.update(key.encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if platform.system() == "Darwin" else value * 1024


def _normalise(batch: Any, device: str) -> Any:
    import torch

    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    return (batch - mean) / std


def _features(model: Any, images: Any) -> Any:
    embeddings = model.generate_embeddings(images, evaluation=True)[0]
    projected = model.projection(embeddings)
    return projected[0] if isinstance(projected, (tuple, list)) else projected


def _worker(device: str, size: int, batch_size: int) -> Measurement:
    import torch
    from anomalib.models import Glass

    torch.manual_seed(SEED)
    started = time.perf_counter()
    wrapper = Glass(
        input_shape=(size, size),
        anomaly_source_path=None,
        backbone=BACKBONE,
        pre_trained=True,
        step=MINING_STEPS,
        pre_processor=False,
        post_processor=False,
        evaluator=False,
        visualizer=False,
    )
    model = wrapper.model.to(device)
    construct_seconds = time.perf_counter() - started

    discriminator_optimizer = torch.optim.AdamW(
        model.discriminator.parameters(), lr=2e-4
    )
    projection_optimizer = torch.optim.Adam(
        model.projection.parameters(), lr=1e-4, weight_decay=1e-5
    )

    generator = torch.Generator(device="cpu").manual_seed(SEED + 1)
    center_batch = _normalise(
        torch.rand(max(2, batch_size), 3, size, size, generator=generator).to(device),
        device,
    )
    train_batch = _normalise(
        torch.rand(batch_size, 3, size, size, generator=generator).to(device), device
    )

    model.eval()
    _sync(device)
    started = time.perf_counter()
    with torch.no_grad():
        center_features = _features(model, center_batch)
        model.center = center_features.reshape(
            center_batch.shape[0], -1, center_features.shape[-1]
        ).mean(dim=0)
    _sync(device)
    center_ms = (time.perf_counter() - started) * 1000

    timings: list[float] = []
    loss_value = float("nan")
    model.train()
    for update in range(3):
        discriminator_optimizer.zero_grad(set_to_none=True)
        projection_optimizer.zero_grad(set_to_none=True)
        started = time.perf_counter()
        losses = model(train_batch)
        loss = losses[-1]
        loss.backward()
        projection_optimizer.step()
        discriminator_optimizer.step()
        _sync(device)
        if update:
            timings.append(time.perf_counter() - started)
        loss_value = float(loss.detach().cpu())

    infer_timings: list[float] = []
    model.eval()
    with torch.inference_mode():
        for index in range(4):
            started = time.perf_counter()
            prediction = model(train_batch)
            _sync(device)
            if index:
                infer_timings.append(time.perf_counter() - started)
    if prediction.anomaly_map is None or prediction.anomaly_map.shape[-2:] != (
        size,
        size,
    ):
        raise RuntimeError("GLASS did not return a prepared-frame anomaly map")

    payload = {
        "projection": model.projection.state_dict(),
        "discriminator": model.discriminator.state_dict(),
        "center": model.center,
        "discriminator_optimizer": discriminator_optimizer.state_dict(),
        "projection_optimizer": projection_optimizer.state_dict(),
    }
    checkpoint = Path(os.environ.get("TMPDIR", "/tmp")) / (
        f"glass-smoke-{os.getpid()}-{device}-{size}-{batch_size}.pt"
    )
    torch.save(payload, checkpoint)
    checkpoint_bytes = checkpoint.stat().st_size
    checkpoint.unlink()

    mps_allocated = None
    mps_driver = None
    if device == "mps":
        mps_allocated = int(torch.mps.current_allocated_memory())
        mps_driver = int(torch.mps.driver_allocated_memory())

    backbone = model.forward_modules["feature_aggregator"].feature_extractor
    optimized = [
        *model.projection.parameters(),
        *model.discriminator.parameters(),
    ]
    return Measurement(
        device=device,
        size=size,
        batch_size=batch_size,
        construct_seconds=construct_seconds,
        center_ms=center_ms,
        train_ms=statistics.median(timings) * 1000,
        infer_ms=statistics.median(infer_timings) * 1000,
        loss=loss_value,
        total_parameters=sum(parameter.numel() for parameter in model.parameters()),
        optimized_parameters=sum(parameter.numel() for parameter in optimized),
        checkpoint_bytes=checkpoint_bytes,
        peak_rss_bytes=_peak_rss_bytes(),
        mps_allocated_bytes=mps_allocated,
        mps_driver_bytes=mps_driver,
        backbone_sha256=_tensor_digest(backbone.state_dict()),
    )


def _run_worker(device: str, size: int, batch_size: int) -> Measurement:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        device,
        str(size),
        str(batch_size),
    ]
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.returncode:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(
            f"{device} at {size}px, batch {batch_size} failed:\n{detail}"
        )
    return Measurement(**json.loads(completed.stdout.strip().splitlines()[-1]))


def _gib(value: int | None) -> str:
    return "-" if value is None else f"{value / 1024**3:.2f} GiB"


def _main() -> int:
    import torch
    from anomalib.models.image.glass.lightning_model import DTD_DOWNLOAD_INFO

    devices = ["cpu"]
    if torch.backends.mps.is_available():
        devices.insert(0, "mps")

    print(RULE)
    print(f"GLASS resource gate — {BACKBONE}, {MINING_STEPS} mining steps")
    print(RULE)
    print(
        f"python {platform.python_version()} | torch {torch.__version__} | "
        f"anomalib {version('anomalib')} | timm {version('timm')}"
    )
    print(f"{platform.machine()} | macOS {platform.mac_ver()[0]}")
    print(
        "optional DTD asset: "
        f"{DTD_DOWNLOAD_INFO.name}, SHA-256 {DTD_DOWNLOAD_INFO.hashsum}"
    )
    print()

    results: list[Measurement] = []
    failed = False
    for size, batch_size in PROFILES:
        for device in devices:
            try:
                result = _run_worker(device, size, batch_size)
            except Exception as exc:  # noqa: BLE001 - failed legs belong in the report
                failed = True
                print(f"[FAIL] {device:>3} {size}px b{batch_size}: {exc}")
                continue
            results.append(result)
            print(
                f"[ ok ] {device:>3} {size}px b{batch_size} | "
                f"center {result.center_ms:7.1f} ms | "
                f"train {result.train_ms:7.1f} ms | infer {result.infer_ms:6.1f} ms | "
                f"RSS {_gib(result.peak_rss_bytes)} | MPS driver {_gib(result.mps_driver_bytes)}"
            )

    print()
    if results:
        first = results[0]
        print(
            f"parameters: {first.total_parameters / 1e6:.2f}M total, "
            f"{first.optimized_parameters / 1e6:.2f}M optimized"
        )
        print(f"resumable payload: {first.checkpoint_bytes / 1e6:.1f} MB")
        print(f"backbone SHA-256: {first.backbone_sha256}")

    if failed:
        print("\nResult: NOT READY — at least one required device/size leg failed.")
        return 1
    print("\nResult: COMPATIBLE — public quality still decides defaults and DTD use.")
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", nargs=3, metavar=("DEVICE", "SIZE", "BATCH_SIZE"))
    return parser.parse_args()


if __name__ == "__main__":
    arguments = _parse_args()
    if arguments.worker:
        measurement = _worker(
            arguments.worker[0], int(arguments.worker[1]), int(arguments.worker[2])
        )
        print(json.dumps(asdict(measurement), sort_keys=True))
        raise SystemExit(0)
    raise SystemExit(_main())
