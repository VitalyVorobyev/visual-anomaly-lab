"""Resolution, verification and persisted external overrides for model assets."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from anomaly_lab.config import Settings
from anomaly_lab.model_assets.catalog import ModelAssetSpec

STATE_FILENAME = "sources.json"


@dataclass(frozen=True)
class ResolvedAsset:
    path: Path
    source: str
    ready: bool
    size: int | None
    reason: str | None = None


def managed_path(settings: Settings, spec: ModelAssetSpec) -> Path:
    return settings.model_assets_dir / spec.key / spec.filename


def resolve_asset(settings: Settings, spec: ModelAssetSpec) -> ResolvedAsset:
    override = _read_overrides(settings).get(spec.key)
    path = _absolute(Path(override)) if override else managed_path(settings, spec)
    return _resolve_specific(path, spec, "external" if override else "managed")


def _absolute(path: Path) -> Path:
    """Absolute and normalised, with symlinks kept.

    A Hugging Face snapshot is a directory of symlinks into content-addressed blobs; its
    companions are found beside the *link*, so resolving it would lose them.
    """
    return Path(os.path.normpath(path.expanduser().absolute()))


def set_external_source(settings: Settings, spec: ModelAssetSpec, path: Path) -> ResolvedAsset:
    resolved = _absolute(path)
    candidate = _resolve_specific(resolved, spec, "external")
    if not candidate.ready:
        raise ValueError(candidate.reason or "asset is not valid")
    overrides = _read_overrides(settings)
    overrides[spec.key] = str(resolved)
    _write_overrides(settings, overrides)
    return candidate


def clear_external_source(settings: Settings, spec: ModelAssetSpec) -> None:
    overrides = _read_overrides(settings)
    if overrides.pop(spec.key, None) is not None:
        _write_overrides(settings, overrides)


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def asset_files(spec: ModelAssetSpec, path: Path) -> list[tuple[Path, int, str]]:
    """Every file the asset is, as `(path, size, sha256)`: the main file, then its companions."""
    return [
        (path, spec.expected_size, spec.sha256),
        *((path.parent / entry.filename, entry.size, entry.sha256) for entry in spec.companions),
    ]


def _resolve_specific(path: Path, spec: ModelAssetSpec, source: str) -> ResolvedAsset:
    size: int | None = None
    for index, (candidate, expected_size, expected_digest) in enumerate(asset_files(spec, path)):
        # The main file names itself in no reason: "missing" is what the catalogue's
        # status reads. A companion does, so a half-copied directory says what it lacks.
        label = "" if index == 0 else f"{candidate.name}: "
        try:
            stat = candidate.stat()
        except FileNotFoundError:
            return ResolvedAsset(path, source, False, size, f"{label}missing")
        if not candidate.is_file():
            return ResolvedAsset(path, source, False, size, f"{label}not a regular file")
        if index == 0:
            size = stat.st_size
        if stat.st_size != expected_size:
            return ResolvedAsset(
                path,
                source,
                False,
                size,
                f"{label}expected {expected_size} bytes, found {stat.st_size}",
            )
        if _digest_for_stat(str(candidate), stat.st_size, stat.st_mtime_ns) != expected_digest:
            return ResolvedAsset(path, source, False, size, f"{label}SHA-256 mismatch")
    return ResolvedAsset(path, source, True, size)


def _state_path(settings: Settings) -> Path:
    return settings.model_assets_dir / STATE_FILENAME


@lru_cache(maxsize=64)
def _digest_for_stat(path: str, size: int, modified_ns: int) -> str:
    """Avoid re-hashing a 40 MB asset on every catalog poll.

    Size and mtime are part of the key. They are not treated as proof: the first read of
    every distinct file state still computes the catalogued SHA-256.
    """
    del size, modified_ns
    return sha256_file(Path(path))


def _read_overrides(settings: Settings) -> dict[str, str]:
    path = _state_path(settings)
    try:
        payload: Any = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}
    if not isinstance(payload, dict):
        return {}
    sources = payload.get("external_sources", {})
    if not isinstance(sources, dict):
        return {}
    return {str(key): str(value) for key, value in sources.items() if isinstance(value, str)}


def _write_overrides(settings: Settings, overrides: dict[str, str]) -> None:
    path = _state_path(settings)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps({"version": 1, "external_sources": overrides}, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
