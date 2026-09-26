"""MobileSAM inference inside the resident worker, never the API process."""

from __future__ import annotations

import hashlib
import importlib
import uuid
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np

from anomaly_lab.annotation_bitmap import tight_bitmap_shape
from anomaly_lab.config import Settings
from anomaly_lab.db.connection import connection
from anomaly_lab.db.repositories import images as images_repo
from anomaly_lab.media import decode
from anomaly_lab.model_assets.catalog import get_spec
from anomaly_lab.model_assets.store import resolve_asset


class MobileSamError(RuntimeError):
    pass


class MobileSamSession:
    """One loaded checkpoint and at most one cached source-image embedding."""

    def __init__(self, settings: Settings, asset_key: str) -> None:
        spec = get_spec(asset_key)
        if spec is None:
            raise MobileSamError(f"unknown model asset {asset_key!r}")
        resolved = resolve_asset(settings, spec)
        if not resolved.ready:
            raise MobileSamError(f"{spec.title} is not ready: {resolved.reason}")
        self.settings = settings
        self.asset_path = resolved.path
        self.device = _preferred_device()
        self.model, self.predictor = _load_predictor(self.asset_path, self.device)
        self.image_id: int | None = None
        self.image_array: np.ndarray | None = None
        # The automatic-mask generator's own model, once it has had to leave the device.
        self.auto_model: Any | None = None
        self.auto_device = self.device

    def segment(self, request: dict[str, Any]) -> dict[str, object]:
        image_id = _integer(request, "image_id")
        if self.image_id != image_id:
            image = self._load_image(image_id)
            try:
                self.predictor.set_image(image)
            except RuntimeError:
                if self.device != "mps":
                    raise
                self._fall_back_to_cpu(image)
            self.image_id = image_id
            self.image_array = image

        point_coords, point_labels = _points(request.get("points"))
        box = _box(request.get("box"))
        if point_coords is None and box is None:
            raise MobileSamError("at least one point or a box is required")
        label_key = str(request.get("label_key", "defect"))
        operation = cast(Literal["add", "subtract"], str(request.get("operation", "add")))

        try:
            masks, scores, _ = self._predict(point_coords, point_labels, box)
        except RuntimeError:
            if self.device != "mps" or self.image_array is None:
                raise
            self._fall_back_to_cpu(self.image_array)
            masks, scores, _ = self._predict(point_coords, point_labels, box)
        candidates: list[dict[str, object]] = []
        seen: set[str] = set()
        for index in np.argsort(scores)[::-1].tolist():
            mask = np.asarray(masks[index], dtype=np.bool_)
            digest = hashlib.sha256(mask.tobytes()).hexdigest()
            if digest in seen:
                continue
            seen.add(digest)
            shape = tight_bitmap_shape(
                mask,
                shape_id=f"assist-{uuid.uuid4().hex}",
                label_key=label_key,
                operation=operation,
            )
            if shape is None:
                continue
            candidates.append(
                {
                    "shape": shape.model_dump(mode="json"),
                    "score": float(scores[index]),
                    "area": int(mask.sum()),
                }
            )
            if len(candidates) == 3:
                break
        return {"image_id": image_id, "device": self.device, "candidates": candidates}

    def region(self, request: dict[str, Any]) -> dict[str, object]:
        """Run the automatic-mask region extractor on one image, for the Prepare screen.

        The same selection `MobileSamRegionExtractor` applies in a build, over this
        session's already loaded model, so a live preview and a build agree. A region that
        cannot be found is an answer, not a protocol failure: it comes back as `error`
        rather than as an error frame, which would cost the resident its loaded model.
        """
        from anomaly_lab.regions.base import RegionExtractionError
        from anomaly_lab.regions.mobile_sam import (
            MobileSamRegionConfig,
            can_retry_on_cpu,
            select_region,
        )

        image_id = _integer(request, "image_id")
        try:
            config = MobileSamRegionConfig.model_validate(request.get("config") or {})
        except ValueError as exc:
            raise MobileSamError(f"malformed region config: {exc}") from exc
        image = self._load_image(image_id)
        try:
            records = self._generate(image, config)
        except (RuntimeError, TypeError) as exc:
            if self.auto_device != "mps" or not can_retry_on_cpu(exc):
                raise
            # The automatic generator emits float64 on the way, which MPS refuses; a build
            # falls back to CPU the same way. Only the generator moves: the prompt predictor
            # the annotation editor uses stays on the accelerator.
            self.auto_model = _load_predictor(self.asset_path, "cpu")[0]
            self.auto_device = "cpu"
            records = self._generate(image, config)
        height, width = image.shape[:2]
        try:
            extraction = select_region(records, height, width, config)
        except RegionExtractionError as exc:
            return {"image_id": image_id, "device": self.auto_device, "error": str(exc)}
        extraction.metadata["device"] = self.auto_device
        return {
            "image_id": image_id,
            "device": self.auto_device,
            "extraction": extraction.model_dump(mode="json"),
        }

    def _generate(self, image: np.ndarray, config: Any) -> list[dict[str, Any]]:
        package = importlib.import_module("mobile_sam")
        generator = package.SamAutomaticMaskGenerator(
            self.auto_model if self.auto_model is not None else self.model,
            points_per_side=config.points_per_side,
            points_per_batch=config.points_per_batch,
            pred_iou_thresh=config.predicted_iou_threshold,
            stability_score_thresh=config.stability_threshold,
            crop_n_layers=0,
            min_mask_region_area=0,
            output_mode="binary_mask",
        )
        records: list[dict[str, Any]] = generator.generate(image)
        return records

    def _predict(
        self,
        point_coords: np.ndarray | None,
        point_labels: np.ndarray | None,
        box: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        result: tuple[np.ndarray, np.ndarray, np.ndarray] = self.predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=box,
            multimask_output=True,
        )
        return result

    def _fall_back_to_cpu(self, image: np.ndarray) -> None:
        # MPS support is an optimisation, never an availability requirement. Rebuild
        # rather than move a half-run predictor: it may retain stale device tensors.
        self.device = "cpu"
        self.model, self.predictor = _load_predictor(self.asset_path, self.device)
        self.predictor.set_image(image)

    def _load_image(self, image_id: int) -> np.ndarray:
        with connection(self.settings.db_path) as conn:
            image = images_repo.get_image(conn, image_id)
        if image is None:
            raise MobileSamError(f"no image with id {image_id}")
        decoded = decode.load(Path(image.path))
        return np.asarray(decoded.convert("RGB"), dtype=np.uint8)


def _preferred_device() -> str:
    import torch

    return "mps" if torch.backends.mps.is_available() else "cpu"


def _load_predictor(checkpoint: Path, device: str) -> tuple[Any, Any]:
    try:
        package = importlib.import_module("mobile_sam")
    except ImportError as exc:
        raise MobileSamError(
            "MobileSAM runtime is not installed; install the backend's 'dl' extra"
        ) from exc
    registry: Any = package.sam_model_registry
    predictor_type: Any = package.SamPredictor
    model = registry["vit_t"](checkpoint=str(checkpoint))
    model.to(device=device)
    model.eval()
    return model, predictor_type(model)


def _integer(request: dict[str, Any], key: str) -> int:
    try:
        return int(request[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise MobileSamError(f"malformed {key}") from exc


def _points(value: object) -> tuple[np.ndarray | None, np.ndarray | None]:
    if not isinstance(value, list) or not value:
        return None, None
    coords: list[tuple[float, float]] = []
    labels: list[int] = []
    for point in value:
        if not isinstance(point, dict):
            raise MobileSamError("malformed point prompt")
        try:
            coords.append((float(point["x"]), float(point["y"])))
        except (KeyError, TypeError, ValueError) as exc:
            raise MobileSamError("malformed point prompt") from exc
        labels.append(1 if point.get("kind") == "positive" else 0)
    return np.asarray(coords, dtype=np.float32), np.asarray(labels, dtype=np.int32)


def _box(value: object) -> np.ndarray | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise MobileSamError("malformed box prompt")
    try:
        return np.asarray(
            [float(value[key]) for key in ("x0", "y0", "x1", "y1")],
            dtype=np.float32,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise MobileSamError("malformed box prompt") from exc
