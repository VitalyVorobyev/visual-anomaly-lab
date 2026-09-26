"""SAM 3 inside the resident worker: a phrase in, the instances it names out.

Explore's Text mode. The checkpoint is a catalogued model asset (`model_assets/catalog.py`),
resolved and verified by key in this process, so the request channel cannot choose what is
loaded — the same rule the MobileSAM resident keeps.

**The image is encoded once.** SAM 3 splits into a vision encoder, which is most of the cost,
and a text encoder plus decoder that are cheap. The last image's vision features are cached
the way `ExploreSession` caches a patch grid, so the first phrase on an image pays for the
encoder and every later phrase on it pays only for the text and the decoder.

**Masks go to the scratch store, not the reply.** An answer is up to `MAX_INSTANCES` masks at
the source image's size; it is written as one bit-packed instance map
(`store.MapKind.INSTANCES`) and the reply carries scores, boxes and areas only, so its size
does not grow with the image.

Torch and transformers are imported inside functions only: the API process imports this
module's constants, and must not import either.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from anomaly_lab.config import Settings
from anomaly_lab.db.connection import connection
from anomaly_lab.db.repositories import images as images_repo
from anomaly_lab.explore.store import MapKind, explore_dir, write_grid
from anomaly_lab.media import decode
from anomaly_lab.model_assets.catalog import get_spec
from anomaly_lab.model_assets.store import resolve_asset
from anomaly_lab.regions.transform import SpatialTransform

MAX_INSTANCES = 24
"""The most instances one answer keeps, best first; the rest are counted as dropped. More
than this on one image is a texture, not a set of objects a person would pick between."""

MAX_PHRASE_LENGTH = 80
DEFAULT_THRESHOLD = 0.5
"""SAM 3's own presence-times-detection score an instance must reach."""
MASK_THRESHOLD = 0.5


class TextExploreError(RuntimeError):
    """A request that cannot be answered, for a reason the reader can act on."""


@dataclass(frozen=True)
class Instance:
    score: float
    box: tuple[float, float, float, float]
    """`(x0, y0, x1, y1)` in source pixels."""
    area: int


def rank_instances(
    masks: np.ndarray, scores: np.ndarray, boxes: np.ndarray, *, limit: int = MAX_INSTANCES
) -> tuple[np.ndarray, list[Instance], int]:
    """Best first, empty masks out, at most `limit` kept; returns the masks, rows and drop count.

    Pure numpy, so the order and the bound are tested without SAM 3.
    """
    areas = masks.reshape(masks.shape[0], -1).sum(axis=1) if masks.shape[0] else np.zeros(0)
    order = [int(index) for index in np.argsort(-scores, kind="stable") if areas[index] > 0]
    kept = order[:limit]
    rows = [
        Instance(
            score=float(scores[index]),
            box=(
                float(boxes[index][0]),
                float(boxes[index][1]),
                float(boxes[index][2]),
                float(boxes[index][3]),
            ),
            area=int(areas[index]),
        )
        for index in kept
    ]
    height, width = masks.shape[1:]
    stack = (
        np.asarray(masks[kept], dtype=np.bool_)
        if kept
        else np.zeros((0, height, width), dtype=np.bool_)
    )
    return stack, rows, len(order) - len(kept)


def clean_phrase(value: object) -> str:
    phrase = " ".join(str(value or "").split())
    if not phrase:
        raise TextExploreError("a phrase is required")
    if len(phrase) > MAX_PHRASE_LENGTH:
        raise TextExploreError(f"a phrase is at most {MAX_PHRASE_LENGTH} characters")
    return phrase


@dataclass
class _Encoded:
    image_id: int
    size: tuple[int, int]
    """`(width, height)` of the source image."""
    vision: Any
    original_sizes: list[list[int]]


class TextSession:
    """One loaded SAM 3 and the vision features of the last image it encoded."""

    def __init__(self, settings: Settings, asset_key: str) -> None:
        spec = get_spec(asset_key)
        if spec is None:
            raise TextExploreError(f"unknown model asset {asset_key!r}")
        resolved = resolve_asset(settings, spec)
        if not resolved.ready:
            raise TextExploreError(f"{spec.title} is not ready: {resolved.reason}")
        self.settings = settings
        self.scratch = explore_dir(settings)
        self.device = _preferred_device()
        self._directory = resolved.path.parent
        self._processor, self._model = _load(self._directory, self.device)
        self._cached: _Encoded | None = None

    def answer(self, request: dict[str, Any]) -> dict[str, object]:
        image_id = _integer(request, "image_id")
        phrase = clean_phrase(request.get("phrase"))
        threshold = _unit(request, "threshold", DEFAULT_THRESHOLD)

        timings = {"encode": 0.0, "prompt": 0.0}

        def run() -> tuple[bool, tuple[np.ndarray, np.ndarray, np.ndarray]]:
            started = time.perf_counter()
            hit = self._cached is not None and self._cached.image_id == image_id
            if not hit:
                self._cached = None
                self._cached = self._encode(image_id)
            timings["encode"] = 0.0 if hit else (time.perf_counter() - started) * 1000.0
            prompted = time.perf_counter()
            assert self._cached is not None
            answer = self._prompt(self._cached, phrase, threshold)
            timings["prompt"] = (time.perf_counter() - prompted) * 1000.0
            return hit, answer

        cached, (masks, scores, boxes) = self._with_fallback(run)
        encoded = self._cached
        assert encoded is not None
        stack, instances, dropped = rank_instances(masks, scores, boxes)
        map_id: str | None = None
        if instances:
            width, height = encoded.size
            identity = SpatialTransform.resolve(
                source_size=(width, height), prepared_size=(width, height)
            )
            map_id = write_grid(self.scratch, MapKind.INSTANCES, stack, identity, 1)
        return {
            "image_id": image_id,
            "phrase": phrase,
            "threshold": threshold,
            "device": self.device,
            "cached": cached,
            "encode_ms": timings["encode"],
            "prompt_ms": timings["prompt"],
            "map_id": map_id,
            "instances": [
                {
                    "score": item.score,
                    "box": {
                        "x0": item.box[0],
                        "y0": item.box[1],
                        "x1": item.box[2],
                        "y1": item.box[3],
                    },
                    "area": item.area,
                }
                for item in instances
            ],
            "dropped": dropped,
        }

    def _with_fallback(self, body: Any) -> Any:
        try:
            return body()
        except RuntimeError:
            # MPS is an optimisation, never an availability requirement. The cached vision
            # features live on the device that failed, so they go with it and the image is
            # encoded again on the CPU.
            if self.device != "mps":
                raise
            self.device = "cpu"
            self._model = self._model.to("cpu")
            self._cached = None
            return body()

    def _encode(self, image_id: int) -> _Encoded:
        import torch

        with connection(self.settings.db_path) as conn:
            image = images_repo.get_image(conn, image_id)
        if image is None:
            raise TextExploreError(f"no image with id {image_id}")
        source = decode.load(Path(image.path)).convert("RGB")
        inputs = self._processor(images=source, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            vision = self._model.get_vision_features(pixel_values=inputs["pixel_values"])
        return _Encoded(
            image_id=image_id,
            size=source.size,
            vision=vision,
            original_sizes=inputs["original_sizes"].tolist(),
        )

    def _prompt(
        self, encoded: _Encoded, phrase: str, threshold: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        import torch

        text = self._processor(text=phrase, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            outputs = self._model(
                vision_embeds=encoded.vision,
                input_ids=text["input_ids"],
                attention_mask=text["attention_mask"],
            )
        result = self._processor.post_process_instance_segmentation(
            outputs,
            threshold=threshold,
            mask_threshold=MASK_THRESHOLD,
            target_sizes=encoded.original_sizes,
        )[0]
        width, height = encoded.size
        masks = np.asarray(result["masks"].detach().cpu().numpy(), dtype=np.bool_)
        if masks.shape[0] == 0:
            # SAM 3 reports "nothing" at its own mask resolution; the answer is at ours.
            masks = np.zeros((0, height, width), dtype=np.bool_)
        scores = np.asarray(result["scores"].detach().float().cpu().numpy(), dtype=np.float64)
        boxes = np.asarray(result["boxes"].detach().float().cpu().numpy(), dtype=np.float64)
        return masks, scores, boxes.reshape(-1, 4)


def _load(directory: Path, device: str) -> tuple[Any, Any]:
    try:
        from transformers import Sam3Model, Sam3Processor
    except ImportError as exc:
        raise TextExploreError(
            "SAM 3 needs transformers; install the backend's 'dl' extra"
        ) from exc
    processor = Sam3Processor.from_pretrained(directory, local_files_only=True)
    model: Any = Sam3Model.from_pretrained(directory, local_files_only=True)
    model = model.to(device).eval()
    model.requires_grad_(False)
    return processor, model


def _preferred_device() -> str:
    import torch

    return "mps" if torch.backends.mps.is_available() else "cpu"


def _integer(request: dict[str, Any], key: str) -> int:
    try:
        return int(request[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise TextExploreError(f"malformed {key}") from exc


def _unit(request: dict[str, Any], key: str, default: float) -> float:
    try:
        value = float(request.get(key, default))
    except (TypeError, ValueError) as exc:
        raise TextExploreError(f"malformed {key}") from exc
    if not 0.0 < value < 1.0:
        raise TextExploreError(f"{key} must lie strictly between 0 and 1")
    return value
