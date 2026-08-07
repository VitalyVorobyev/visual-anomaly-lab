"""Resolving a compute device, and saying out loud which one was used.

A plugin declares a `preferred_device`. That is a preference, not a fact: MPS is
unavailable on an Intel Mac, absent from a checkout without the optional deep-learning
group, and — as the standalone smoke test in `scripts/mps-smoke-test.py` exists to find —
occasionally missing a kernel a model needs.

Resolution therefore always succeeds, always falls back to CPU, and always returns the
reason. The reason is logged into the job's own log, so an experiment that took twenty
minutes instead of five explains itself months later without a rerun.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass

from anomaly_lab.models.base import Device


@dataclass(frozen=True)
class ResolvedDevice:
    device: Device
    reason: str

    @property
    def fell_back(self) -> bool:
        return self.device is Device.CPU and "preferred" not in self.reason


def resolve_device(preferred: Device) -> ResolvedDevice:
    """Pick a device, never raising, and explain the pick."""
    if preferred is Device.CPU:
        return ResolvedDevice(Device.CPU, "cpu is this method's preferred device")

    if importlib.util.find_spec("torch") is None:
        return ResolvedDevice(
            Device.CPU,
            f"{preferred.value} was preferred but torch is not installed; using cpu",
        )

    import torch

    if preferred is Device.MPS:
        if torch.backends.mps.is_available():
            return ResolvedDevice(Device.MPS, "mps is available and preferred")
        built = torch.backends.mps.is_built()
        detail = "the backend is not built into this torch" if not built else "no MPS device"
        return ResolvedDevice(Device.CPU, f"mps was preferred but {detail}; using cpu")

    if preferred is Device.CUDA:
        if torch.cuda.is_available():
            return ResolvedDevice(Device.CUDA, "cuda is available and preferred")
        return ResolvedDevice(Device.CPU, "cuda was preferred but is unavailable; using cpu")

    return ResolvedDevice(Device.CPU, "unrecognized preference; using cpu")
