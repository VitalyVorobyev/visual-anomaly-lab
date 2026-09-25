"""How an anomaly map is kept on disk, and the one function that reads one back.

A map is stored in the frame its method computed it in — the prepared frame — together with
the pinned `SpatialTransform` that places it in its source image, as one compressed `.npz`
(`map`, float32; `transform`, the transform's JSON as bytes). `read_map` projects it to
source pixels, which is the only frame any consumer sees: projection is deterministic, so
the array read back is bit-identical to the one projected at write time, where the run's
display range and the map's peak were taken. That form is a twenty-sixth of a projected
float32 source map (docs/measurements.md, "Anomaly-map storage").

A map written without a transform keeps its array as-is, and a map written before this format
— a float32 `.npy`, already in the source frame — reads unchanged.

Numpy, Pillow and pydantic only: the evaluation layer reads maps through this, and must not
need torch.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np

from anomaly_lab.regions.transform import SpatialTransform

MAP_SUFFIX = ".npz"
LEGACY_MAP_SUFFIX = ".npy"


def map_file(maps_dir: Path, image_id: int) -> Path:
    """Where one image's map is written under `maps_dir`."""
    return maps_dir / f"{image_id}{MAP_SUFFIX}"


def stored_map_file(maps_dir: Path, image_id: int) -> Path:
    """The file holding one image's map under `maps_dir`, in whichever format it was written.

    For a reader that addresses a map by image rather than through `ImageResult.map_path`.
    The current format wins; a legacy `.npy` is returned only when it is the one present.
    """
    current = map_file(maps_dir, image_id)
    legacy = maps_dir / f"{image_id}{LEGACY_MAP_SUFFIX}"
    return legacy if not current.is_file() and legacy.is_file() else current


def write_map_file(path: Path, values: np.ndarray, transform: SpatialTransform | None) -> None:
    """Persist one 2-D float32 map, in the prepared frame when `transform` places it.

    Written beside and then renamed over the destination, so a reader never sees half a
    file. A legacy `.npy` of the same image is removed: re-scoring a run written before this
    format would otherwise leave a six-megabyte file nothing references.
    """
    arrays: dict[str, np.ndarray] = {"map": np.ascontiguousarray(values, dtype=np.float32)}
    if transform is not None:
        encoded = transform.model_dump_json().encode("utf-8")
        arrays["transform"] = np.frombuffer(encoded, dtype=np.uint8)
    temporary = path.with_name(f"{path.stem}.tmp{MAP_SUFFIX}")
    try:
        np.savez_compressed(temporary, **arrays)  # type: ignore[arg-type]
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    path.with_suffix(LEGACY_MAP_SUFFIX).unlink(missing_ok=True)


def read_map(path: str | Path) -> np.ndarray:
    """One stored map as a float32 array in its source frame; uncovered pixels are NaN.

    Raises `OSError` when the file is gone and `ValueError` when it is not a map, which are
    the two failures every caller already turns into an absent map.
    """
    target = Path(path)
    if target.suffix != MAP_SUFFIX:
        return np.asarray(np.load(target, allow_pickle=False), dtype=np.float32)
    try:
        with np.load(target, allow_pickle=False) as stored:
            values = np.asarray(stored["map"], dtype=np.float32)
            encoded = stored["transform"].tobytes() if "transform" in stored.files else None
    except (zipfile.BadZipFile, KeyError, EOFError) as exc:
        raise ValueError(f"{target.name} is not a stored anomaly map") from exc
    if encoded is None:
        return values
    return SpatialTransform.model_validate_json(encoded).project_map(values)
