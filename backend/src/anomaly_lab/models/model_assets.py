"""Shared policy for external model weights.

Pretrained weights are experiment inputs, not ambient process state.  Plugins resolve
them through app-managed storage, may refuse network access, and persist a stable tensor
fingerprint rather than copying a large frozen backbone into every run.
"""

from __future__ import annotations

import contextlib
import hashlib
import os
from collections.abc import Iterator
from importlib import import_module
from pathlib import Path
from typing import Any


@contextlib.contextmanager
def huggingface_environment(
    cache_dir: Path,
    *,
    allow_downloads: bool,
    method: str,
    asset: str,
) -> Iterator[None]:
    """Resolve Hugging Face assets inside app-managed storage.

    ``huggingface_hub`` snapshots environment variables at import time.  A resident
    worker may already have imported it for another method, so both the environment and
    the live constants are scoped and restored.
    """
    hub_cache = cache_dir / "huggingface" / "hub"
    hub_cache.mkdir(parents=True, exist_ok=True)
    updates = {"HF_HUB_CACHE": str(hub_cache)}
    if not allow_downloads:
        updates["HF_HUB_OFFLINE"] = "1"
    previous = {key: os.environ.get(key) for key in updates}
    os.environ.update(updates)
    hub_constants: Any = None
    constant_previous: dict[str, Any] = {}
    try:
        hub_constants = import_module("huggingface_hub.constants")
        constant_previous["HF_HUB_CACHE"] = hub_constants.HF_HUB_CACHE
        hub_constants.HF_HUB_CACHE = str(hub_cache)
        if not allow_downloads:
            constant_previous["HF_HUB_OFFLINE"] = hub_constants.HF_HUB_OFFLINE
            hub_constants.HF_HUB_OFFLINE = True
        yield
    except Exception as exc:
        if not allow_downloads:
            raise RuntimeError(
                f"{method} needs pretrained weights for {asset!r}, downloads are "
                f"disabled, and the app cache at {hub_cache} could not supply them. "
                f"Turn allow_downloads on for one run. The underlying failure was: {exc}"
            ) from exc
        raise
    finally:
        if hub_constants is not None:
            for key, value in constant_previous.items():
                setattr(hub_constants, key, value)
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def fingerprint_state(state: dict[str, Any]) -> str:
    """Stable SHA-256 over names, shapes, dtypes and raw tensor bytes."""
    digest = hashlib.sha256()
    for key in sorted(state):
        tensor = state[key].detach().cpu().contiguous()
        digest.update(key.encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()
