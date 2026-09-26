"""Explore's scratch maps: written by the resident, read by the API process, kept for nobody.

A map is stored at **grid resolution** — 32x32 cells, a few kilobytes — together with the
frame's `SpatialTransform` and the patch size, as one `.npz`. Reading it upsamples the grid
into the explore frame and projects that into the source frame through the transform, so
the PNG the stage stacks is at the source image's own size and registered with it by
construction, exactly as a run's anomaly map is.

The directory is bounded (`MAX_STORED_MAPS`, oldest removed first): a map exists to be drawn
once, and a session of clicking should not leave a thousand files behind it.

Numpy, Pillow and pydantic only — the API process renders these and must not import torch.
"""

from __future__ import annotations

import re
import uuid
import zipfile
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import numpy as np
from PIL import Image

from anomaly_lab.config import Settings
from anomaly_lab.regions.transform import SpatialTransform

EXPLORE_SUBDIR = "explore"
MAX_STORED_MAPS = 24
MAP_ID = re.compile(r"^[0-9a-f]{32}$")


class MapKind(StrEnum):
    VALUES = "values"
    """A float grid in `[0, 1]`: similarity."""
    LABELS = "labels"
    """A uint8 grid, 0 off the image and `1..k` over it: clusters."""
    RGB = "rgb"
    """A `(rows, cols, 3)` float grid in `[0, 1]`: false colour."""


@dataclass(frozen=True)
class StoredGrid:
    kind: MapKind
    grid: np.ndarray
    transform: SpatialTransform
    patch: int


def explore_dir(settings: Settings) -> Path:
    return settings.data_dir / EXPLORE_SUBDIR


def map_path(directory: Path, map_id: str) -> Path:
    if not MAP_ID.fullmatch(map_id):
        msg = f"{map_id!r} is not an explore map id"
        raise ValueError(msg)
    return directory / f"{map_id}.npz"


def write_grid(
    directory: Path,
    kind: MapKind,
    grid: np.ndarray,
    transform: SpatialTransform,
    patch: int,
    *,
    keep: int = MAX_STORED_MAPS,
) -> str:
    """Store one grid and return its id; renamed into place, so a reader never sees half."""
    directory.mkdir(parents=True, exist_ok=True)
    map_id = uuid.uuid4().hex
    target = map_path(directory, map_id)
    temporary = directory / f"{map_id}.tmp.npz"
    arrays: dict[str, np.ndarray] = {
        "kind": np.frombuffer(kind.value.encode("utf-8"), dtype=np.uint8),
        "grid": np.ascontiguousarray(grid),
        "transform": np.frombuffer(transform.model_dump_json().encode("utf-8"), dtype=np.uint8),
        "patch": np.asarray(patch, dtype=np.int32),
    }
    try:
        np.savez_compressed(temporary, **arrays)  # type: ignore[arg-type]
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    prune(directory, keep=keep)
    return map_id


def prune(directory: Path, *, keep: int = MAX_STORED_MAPS) -> int:
    """Remove all but the newest `keep` maps; returns how many went."""
    stored = sorted(
        (path for path in directory.glob("*.npz") if MAP_ID.fullmatch(path.stem)),
        key=lambda path: path.stat().st_mtime_ns,
        reverse=True,
    )
    for path in stored[keep:]:
        path.unlink(missing_ok=True)
    return max(0, len(stored) - keep)


def read_grid(path: Path) -> StoredGrid:
    """One stored grid; `OSError` when gone, `ValueError` when not an explore map."""
    try:
        with np.load(path, allow_pickle=False) as stored:
            kind = MapKind(stored["kind"].tobytes().decode("utf-8"))
            grid = np.asarray(stored["grid"])
            transform = SpatialTransform.model_validate_json(stored["transform"].tobytes())
            patch = int(stored["patch"])
    except (zipfile.BadZipFile, KeyError, EOFError) as exc:
        raise ValueError(f"{path.name} is not an explore map") from exc
    return StoredGrid(kind=kind, grid=grid, transform=transform, patch=patch)


def _upsample(plane: np.ndarray, patch: int, resample: Image.Resampling, mode: str) -> np.ndarray:
    rows, cols = plane.shape
    image = Image.fromarray(plane, mode=mode).resize((cols * patch, rows * patch), resample)
    return np.asarray(image)


def to_source(stored: StoredGrid) -> np.ndarray:
    """The grid in the source frame: float `(H, W)`, uint8 `(H, W)` or float `(H, W, 3)`.

    Values and colour are bilinear between patch centres — a patch is a sample, not a tile —
    and labels are nearest, because a cluster index is a name. Pixels the frame never covered
    are NaN for values, 0 for labels and black for colour.
    """
    transform, patch = stored.transform, stored.patch
    if stored.kind is MapKind.LABELS:
        frame = _upsample(stored.grid.astype(np.uint8), patch, Image.Resampling.NEAREST, "L")
        return transform.project_labels(frame, fill=0)
    if stored.kind is MapKind.VALUES:
        frame = _upsample(stored.grid.astype(np.float32), patch, Image.Resampling.BILINEAR, "F")
        return transform.project_map(frame)
    planes = [
        transform.project_map(
            _upsample(
                np.ascontiguousarray(stored.grid[..., channel], dtype=np.float32),
                patch,
                Image.Resampling.BILINEAR,
                "F",
            )
        )
        for channel in range(3)
    ]
    return np.asarray(np.nan_to_num(np.stack(planes, axis=-1), nan=0.0), dtype=np.float32)


def mask_of(
    stored: StoredGrid, *, threshold: float | None = None, cluster: int | None = None
) -> np.ndarray:
    """A boolean source-frame mask: similarity at or above `threshold`, or one cluster."""
    source = to_source(stored)
    if stored.kind is MapKind.VALUES and threshold is not None:
        with np.errstate(invalid="ignore"):
            return np.asarray(np.nan_to_num(source, nan=-1.0) >= threshold)
    if stored.kind is MapKind.LABELS and cluster is not None:
        return np.asarray(source == cluster)
    msg = "a mask needs a threshold on a similarity map or a cluster on a cluster map"
    raise ValueError(msg)
