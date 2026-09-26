"""Explore's scratch maps: written by the resident, read by the API process, kept for nobody.

A map is stored at **grid resolution** — 32x32 cells, a few kilobytes — together with the
frame's `SpatialTransform` and the patch size, as one `.npz`. Reading it upsamples the grid
into the explore frame and projects that into the source frame through the transform, so
the PNG the stage stacks is at the source image's own size and registered with it by
construction, exactly as a run's anomaly map is.

Instance masks are the one exception to grid resolution: a text prompt's answer is a stack of
binary masks already at the source image's size, stored bit-packed with an identity transform,
and drawn as a label map whose label `i` is the `i`th instance by score.

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
    RGB = "rgb"
    """A `(rows, cols, 3)` float grid in `[0, 1]`: false colour."""
    CLUSTERS = "clusters"
    """A `(rows, cols, k)` float grid, each cell's soft assignment to k clusters, drawn as
    the argmax of the interpolated planes (`grid.cluster_affinity`)."""
    INSTANCES = "instances"
    """An `(n, H, W)` boolean stack at the source size, best-scoring instance first."""


@dataclass(frozen=True)
class StoredGrid:
    kind: MapKind
    grid: np.ndarray
    transform: SpatialTransform
    patch: int
    value_range: tuple[float, float] | None = None
    """For values: the range the map is coloured over (`grid.display_range`)."""


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
    value_range: tuple[float, float] | None = None,
    keep: int = MAX_STORED_MAPS,
) -> str:
    """Store one grid and return its id; renamed into place, so a reader never sees half."""
    directory.mkdir(parents=True, exist_ok=True)
    map_id = uuid.uuid4().hex
    target = map_path(directory, map_id)
    temporary = directory / f"{map_id}.tmp.npz"
    stored = (
        np.packbits(np.asarray(grid, dtype=np.bool_), axis=-1)
        if kind is MapKind.INSTANCES
        else np.ascontiguousarray(grid)
    )
    arrays: dict[str, np.ndarray] = {
        "kind": np.frombuffer(kind.value.encode("utf-8"), dtype=np.uint8),
        "grid": stored,
        "transform": np.frombuffer(transform.model_dump_json().encode("utf-8"), dtype=np.uint8),
        "patch": np.asarray(patch, dtype=np.int32),
    }
    if value_range is not None:
        arrays["range"] = np.asarray(value_range, dtype=np.float64)
    if kind is MapKind.INSTANCES:
        arrays["shape"] = np.asarray(grid.shape, dtype=np.int64)
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
            value_range = (
                (float(stored["range"][0]), float(stored["range"][1]))
                if "range" in stored.files
                else None
            )
            if kind is MapKind.INSTANCES:
                count, height, width = (int(value) for value in stored["shape"])
                grid = np.unpackbits(grid, axis=-1, count=width).astype(np.bool_)
                grid = grid.reshape(count, height, width)
    except (zipfile.BadZipFile, KeyError, EOFError) as exc:
        raise ValueError(f"{path.name} is not an explore map") from exc
    return StoredGrid(
        kind=kind, grid=grid, transform=transform, patch=patch, value_range=value_range
    )


def _upsample(plane: np.ndarray, patch: int, resample: Image.Resampling, mode: str) -> np.ndarray:
    rows, cols = plane.shape
    image = Image.fromarray(plane, mode=mode).resize((cols * patch, rows * patch), resample)
    return np.asarray(image)


def to_source(stored: StoredGrid) -> np.ndarray:
    """The grid in the source frame: float `(H, W)`, uint8 `(H, W)` or float `(H, W, 3)`.

    Values and colour are bilinear between patch centres — a patch is a sample, not a tile —
    and clusters are each cluster's affinity plane bilinear, then the argmax per frame pixel,
    as labels `1..k`: a cluster index is a name and cannot be interpolated, but how strongly
    a patch belongs to a cluster can, so the boundary falls where two clusters are equally
    likely rather than on a patch edge. Pixels the frame never covered are NaN for values,
    0 for clusters and black for colour.
    """
    transform, patch = stored.transform, stored.patch
    if stored.kind is MapKind.INSTANCES:
        return instance_labels(stored.grid)
    if stored.kind is MapKind.CLUSTERS:
        affinity = np.stack(
            [
                _upsample(
                    np.ascontiguousarray(stored.grid[..., cluster], dtype=np.float32),
                    patch,
                    Image.Resampling.BILINEAR,
                    "F",
                )
                for cluster in range(stored.grid.shape[-1])
            ]
        )
        frame = (np.argmax(affinity, axis=0) + 1).astype(np.uint8)
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


def instance_labels(masks: np.ndarray) -> np.ndarray:
    """An `(n, H, W)` stack as one uint8 label map: `i` where instance `i` (1-based) lies.

    Instances may overlap. The stack is best first, so it is painted in reverse and the
    better-scoring instance is the one a shared pixel shows.
    """
    count = masks.shape[0]
    labels = np.zeros(masks.shape[1:], dtype=np.uint8)
    for index in range(count - 1, -1, -1):
        labels[masks[index]] = index + 1
    return labels


def instance_mask(stored: StoredGrid, instance: int) -> np.ndarray:
    """One instance's whole mask, including where a better instance overlaps it."""
    if stored.kind is not MapKind.INSTANCES:
        msg = "only an instance map has instances"
        raise ValueError(msg)
    if not 1 <= instance <= stored.grid.shape[0]:
        msg = f"instance {instance} is not one of this map's {stored.grid.shape[0]}"
        raise ValueError(msg)
    return np.asarray(stored.grid[instance - 1], dtype=np.bool_)


def mask_of(
    stored: StoredGrid,
    *,
    threshold: float | None = None,
    cluster: int | None = None,
    instance: int | None = None,
) -> np.ndarray:
    """A boolean source-frame mask: similarity at or above `threshold`, one cluster, or one
    instance."""
    if instance is not None:
        return instance_mask(stored, instance)
    source = to_source(stored)
    if stored.kind is MapKind.VALUES and threshold is not None:
        with np.errstate(invalid="ignore"):
            return np.asarray(np.nan_to_num(source, nan=-1.0) >= threshold)
    if stored.kind is MapKind.CLUSTERS and cluster is not None:
        return np.asarray(source == cluster)
    msg = (
        "a mask needs a threshold on a similarity map, a cluster on a cluster map or an "
        "instance on an instance map"
    )
    raise ValueError(msg)
