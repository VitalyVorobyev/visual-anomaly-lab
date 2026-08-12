#!/usr/bin/env -S uv run --project backend --extra dl python
"""Measure whether anomalib's AnomalyVFM is a credible Apple-Silicon reference.

AnomalyVFM is zero-shot at application time, but it loads a complete adapted RADIO
checkpoint.  This gate makes that external asset and the inference cost explicit before
an application plugin is written.  Every device/size leg runs in a fresh subprocess so
model construction and peak-memory numbers are comparable.

The parent resolves the pinned public asset once.  Workers then run with Hugging Face in
offline mode, proving that a cached experiment does not quietly depend on the network.

Run:

    ./scripts/anomalyvfm-smoke-test.py
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

REPOSITORY = Path(__file__).resolve().parent.parent
ASSET_REPOSITORY = "MaticFuc/anomalyvfm_radio"
ASSET_FILENAME = "model.safetensors"
ASSET_REVISION = "17654e763c8fae5ae1c44e2ec421a427783d6196"
ASSET_SHA256 = "50a219ba436ed656ad3c0405f9e81df8ad00b2e715c98be66d7e2edb62a83a37"
ASSET_BYTES = 1_421_491_228
PROFILES = (256, 512, 768)
SEED = 20260812
RULE = "-" * 82


@dataclass(frozen=True)
class Measurement:
    device: str
    size: int
    construct_seconds: float
    infer_ms: float
    total_parameters: int
    peak_rss_bytes: int
    mps_allocated_bytes: int | None
    mps_driver_bytes: int | None
    score_shape: list[int]
    map_shape: list[int]


def _peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if platform.system() == "Darwin" else value * 1024


def _sync(device: str) -> None:
    if device == "mps":
        import torch

        torch.mps.synchronize()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_asset(cache_dir: Path, *, offline: bool) -> Path:
    from huggingface_hub import hf_hub_download

    path = Path(
        hf_hub_download(
            repo_id=ASSET_REPOSITORY,
            filename=ASSET_FILENAME,
            revision=ASSET_REVISION,
            cache_dir=cache_dir / "hub",
            local_files_only=offline,
        )
    )
    if path.stat().st_size != ASSET_BYTES:
        raise RuntimeError(
            f"AnomalyVFM asset has {path.stat().st_size} bytes; expected {ASSET_BYTES}"
        )
    actual = _sha256(path)
    if actual != ASSET_SHA256:
        raise RuntimeError(
            f"AnomalyVFM asset SHA-256 is {actual}; expected {ASSET_SHA256}"
        )
    return path


def _worker(device: str, size: int) -> Measurement:
    import torch
    from anomalib.models.image.anomalyvfm.torch_model import AnomalyVFMModel

    torch.manual_seed(SEED)
    started = time.perf_counter()
    model = AnomalyVFMModel().eval().to(device)
    construct_seconds = time.perf_counter() - started

    generator = torch.Generator(device="cpu").manual_seed(SEED + 1)
    image = torch.rand(1, 3, size, size, generator=generator).to(device)
    timings: list[float] = []
    with torch.inference_mode():
        for iteration in range(4):
            started = time.perf_counter()
            score, anomaly_map = model(image)
            _sync(device)
            if iteration:
                timings.append(time.perf_counter() - started)
    if tuple(anomaly_map.shape) != (1, 1, size, size):
        raise RuntimeError(
            f"AnomalyVFM returned map shape {tuple(anomaly_map.shape)} at {size}px"
        )

    mps_allocated = None
    mps_driver = None
    if device == "mps":
        mps_allocated = int(torch.mps.current_allocated_memory())
        mps_driver = int(torch.mps.driver_allocated_memory())
    return Measurement(
        device=device,
        size=size,
        construct_seconds=construct_seconds,
        infer_ms=statistics.median(timings) * 1000,
        total_parameters=sum(parameter.numel() for parameter in model.parameters()),
        peak_rss_bytes=_peak_rss_bytes(),
        mps_allocated_bytes=mps_allocated,
        mps_driver_bytes=mps_driver,
        score_shape=list(score.shape),
        map_shape=list(anomaly_map.shape),
    )


def _run_worker(device: str, size: int, cache_dir: Path) -> Measurement:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        device,
        str(size),
        "--cache-dir",
        str(cache_dir),
    ]
    environment = dict(os.environ)
    environment["HF_HOME"] = str(cache_dir)
    environment["HF_HUB_OFFLINE"] = "1"
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    if completed.returncode:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"{device} at {size}px failed:\n{detail}")
    return Measurement(**json.loads(completed.stdout.strip().splitlines()[-1]))


def _gib(value: int | None) -> str:
    return "-" if value is None else f"{value / 1024**3:.2f} GiB"


def _main(cache_dir: Path, *, offline: bool) -> int:
    import torch

    cache_dir.mkdir(parents=True, exist_ok=True)
    asset = _resolve_asset(cache_dir, offline=offline)
    devices = ["cpu"]
    if torch.backends.mps.is_available():
        devices.insert(0, "mps")

    print(RULE)
    print("AnomalyVFM resource gate — adapted RADIO, batch 1")
    print(RULE)
    print(
        f"python {platform.python_version()} | torch {torch.__version__} | "
        f"anomalib {version('anomalib')}"
    )
    print(f"{platform.machine()} | macOS {platform.mac_ver()[0]}")
    print(f"asset: {ASSET_REPOSITORY}@{ASSET_REVISION}/{ASSET_FILENAME}")
    print(f"asset bytes: {asset.stat().st_size:,} | SHA-256: {ASSET_SHA256}")
    print()

    results: list[Measurement] = []
    failed = False
    for size in PROFILES:
        for device in devices:
            try:
                result = _run_worker(device, size, cache_dir)
            except Exception as exc:  # noqa: BLE001 - every failed leg belongs in the report
                failed = True
                print(f"[FAIL] {device:>3} {size}px: {exc}")
                continue
            results.append(result)
            print(
                f"[ ok ] {device:>3} {size}px | load {result.construct_seconds:5.2f}s | "
                f"infer {result.infer_ms:7.1f} ms | RSS {_gib(result.peak_rss_bytes)} | "
                f"MPS driver {_gib(result.mps_driver_bytes)}"
            )

    print()
    if results:
        print(f"parameters: {results[0].total_parameters / 1e6:.2f}M")
    if failed:
        print("\nResult: NOT READY — at least one required device/size leg failed.")
        return 1
    print("\nResult: COMPATIBLE — resource numbers still decide the plugin defaults.")
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=REPOSITORY / "data" / "model-cache" / "huggingface",
    )
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--worker", nargs=2, metavar=("DEVICE", "SIZE"))
    return parser.parse_args()


if __name__ == "__main__":
    arguments = _parse_args()
    if arguments.worker:
        os.environ["HF_HOME"] = str(arguments.cache_dir)
        os.environ["HF_HUB_OFFLINE"] = "1"
        measurement = _worker(arguments.worker[0], int(arguments.worker[1]))
        print(json.dumps(asdict(measurement), sort_keys=True))
        raise SystemExit(0)
    raise SystemExit(_main(arguments.cache_dir, offline=arguments.offline))
