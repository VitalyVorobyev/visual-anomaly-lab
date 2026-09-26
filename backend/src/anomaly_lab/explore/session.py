"""One frozen encoder inside the resident worker, and the one image it last encoded.

The resident is keyed by backbone, so the encoder is loaded once per backbone; the image's
patch grid is cached the way `MobileSamSession` caches an embedding, so the first click on an
image pays for a forward pass and every later click on it — another point, another K, another
mode — is numpy on a cached `(32, 32, D)` array.

**Pixels come through the shared bridge, from the source image.** Explore looks at the
photograph a person is browsing, not at any region profile's crop, so the source is
contain-resized into the fixed frame (`grid.frame_transform`), written to the scratch
directory as a PNG, and read back through `models/preprocessing.load_array` — the same
bridge every method is made to see through (`image_patch_features` calls it).

Torch is imported inside functions only: `store` and `grid` are torch-free and the API
process imports them.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image as PILImage

from anomaly_lab.config import Settings
from anomaly_lab.db.connection import connection
from anomaly_lab.db.repositories import images as images_repo
from anomaly_lab.explore.grid import (
    DEFAULT_CLUSTERS,
    LAYERS,
    MAX_CLUSTERS,
    MAX_POINTS,
    MIN_CLUSTERS,
    CellWindow,
    ExploreMode,
    cluster_grid,
    covered_cells,
    frame_side,
    frame_transform,
    pca_grid,
    similarity,
    source_cell,
)
from anomaly_lab.explore.store import MapKind, explore_dir, write_grid
from anomaly_lab.media import decode
from anomaly_lab.models.base import ImageRecord
from anomaly_lab.models.dino_backbone import (
    BACKBONES,
    DinoBackbone,
    FrozenEncoder,
    image_patch_features,
)
from anomaly_lab.models.preprocessing import ColorMode, PreprocessingConfig
from anomaly_lab.regions.transform import SpatialTransform


class ExploreError(RuntimeError):
    """A request that cannot be answered, for a reason the reader can act on."""


@dataclass(frozen=True)
class Encoded:
    image_id: int
    features: np.ndarray
    """`(rows, cols, D)` float32 unit vectors."""
    transform: SpatialTransform
    window: CellWindow


class ExploreSession:
    def __init__(self, settings: Settings, backbone: DinoBackbone) -> None:
        self.settings = settings
        self.backbone = backbone
        self.patch = BACKBONES[backbone].patch_size
        self.side = frame_side(backbone)
        self.device = _preferred_device()
        self.scratch = explore_dir(settings)
        self._encoder = FrozenEncoder(
            backbone,
            pretrained=True,
            allow_downloads=True,
            seed=0,
            method="feature_explorer",
        )
        self._model: Any = self._encoder.model(self.device, settings.model_cache_dir)
        self._cached: Encoded | None = None

    def answer(self, request: dict[str, Any]) -> dict[str, object]:
        image_id = _integer(request, "image_id")
        try:
            mode = ExploreMode(str(request.get("mode")))
        except ValueError as exc:
            raise ExploreError(f"unknown explore mode {request.get('mode')!r}") from exc

        started = time.perf_counter()
        cached = self._cached is not None and self._cached.image_id == image_id
        if not cached:
            self._cached = self._encode(image_id)
        encoded = self._cached
        assert encoded is not None
        encode_ms = (time.perf_counter() - started) * 1000.0

        computed = time.perf_counter()
        extra: dict[str, object] = {}
        if mode is ExploreMode.SIMILAR:
            positives = self._cells(encoded, request.get("points"))
            if not positives:
                raise ExploreError("similarity needs at least one positive point")
            negatives = self._cells(encoded, request.get("negatives"))
            grid: np.ndarray = similarity(encoded.features, positives, negatives)
            kind = MapKind.VALUES
        elif mode is ExploreMode.CLUSTERS:
            k = _integer(request, "k", DEFAULT_CLUSTERS)
            if not MIN_CLUSTERS <= k <= MAX_CLUSTERS:
                raise ExploreError(f"k must be between {MIN_CLUSTERS} and {MAX_CLUSTERS}")
            grid = cluster_grid(
                encoded.features, encoded.window, k, seed=_integer(request, "seed", 0)
            )
            kind = MapKind.LABELS
            extra["cells"] = [int(value) for value in grid.reshape(-1)]
            extra["clusters"] = int(grid.max())
        else:
            grid = pca_grid(encoded.features, encoded.window)
            kind = MapKind.RGB

        map_id = write_grid(self.scratch, kind, grid, encoded.transform, self.patch)
        rows, cols, width = encoded.features.shape
        return {
            "image_id": image_id,
            "mode": mode.value,
            "backbone": self.backbone.value,
            "device": self.device,
            "grid_rows": rows,
            "grid_cols": cols,
            "feature_dim": width,
            "patch_size": self.patch,
            "cached": cached,
            "encode_ms": 0.0 if cached else encode_ms,
            "compute_ms": (time.perf_counter() - computed) * 1000.0,
            "map_id": map_id,
            "map_kind": kind.value,
            "transform": encoded.transform.model_dump(mode="json"),
            **extra,
        }

    def _cells(self, encoded: Encoded, value: object) -> list[tuple[int, int]]:
        if value is None:
            return []
        if not isinstance(value, list) or len(value) > MAX_POINTS:
            raise ExploreError("malformed points")
        cells: list[tuple[int, int]] = []
        for point in value:
            if not isinstance(point, dict):
                raise ExploreError("malformed point")
            try:
                xy = (float(point["x"]), float(point["y"]))
            except (KeyError, TypeError, ValueError) as exc:
                raise ExploreError("malformed point") from exc
            cells.append(source_cell(encoded.transform, self.patch, encoded.window, xy))
        return cells

    def _encode(self, image_id: int) -> Encoded:
        with connection(self.settings.db_path) as conn:
            image = images_repo.get_image(conn, image_id)
        if image is None:
            raise ExploreError(f"no image with id {image_id}")
        source = decode.load(Path(image.path))
        transform = frame_transform(source.size, self.backbone)
        prepared = transform.prepare_image(source, resample=PILImage.Resampling.BILINEAR)
        self.scratch.mkdir(parents=True, exist_ok=True)
        frame = self.scratch / "frame.png"
        prepared.save(frame, format="PNG")

        preprocessing = PreprocessingConfig(width=self.side, height=self.side, color=ColorMode.RGB)
        record = ImageRecord(image_id=image_id, sample_id=image.sample_id, path=frame)
        features = self._forward(record, preprocessing)
        rows = cols = self.side // self.patch
        grid = np.asarray(features.numpy()[0], dtype=np.float32).reshape(rows, cols, -1)
        return Encoded(
            image_id=image_id,
            features=grid,
            transform=transform,
            window=covered_cells(transform, self.patch),
        )

    def _forward(self, record: ImageRecord, preprocessing: PreprocessingConfig) -> Any:
        try:
            return image_patch_features(
                self._model, [record], preprocessing, LAYERS.indices, self.device
            )
        except RuntimeError:
            # MPS is an optimisation, never an availability requirement.
            if self.device != "mps":
                raise
            self.device = "cpu"
            self._model = self._model.to("cpu")
            return image_patch_features(
                self._model, [record], preprocessing, LAYERS.indices, self.device
            )


def _preferred_device() -> str:
    import torch

    return "mps" if torch.backends.mps.is_available() else "cpu"


def _integer(request: dict[str, Any], key: str, default: int | None = None) -> int:
    value = request.get(key, default)
    if value is None:
        raise ExploreError(f"malformed {key}")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ExploreError(f"malformed {key}") from exc
