"""Deterministic materialisation of source-frame annotation documents.

A completed document becomes two PNGs from one rasterisation: a class-index mask, where each
pixel holds the index of the class drawn there last (0 is background), and the binary mask
every anomaly consumer reads, which is exactly `index > 0`. The class table that gives the
indices their meaning is returned with the pixel count of each class, and pinned on the
revision, so a class's presence is a database read rather than a decode (ADR-0040).

The same pass records object instances: every `add` shape belongs to the instance its
`instance_id` names, or to its own when unset, and an instance owns the pixels its shapes
set that nothing later cut or overdrew. They are written as a JSON file beside the masks.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import assert_never

import numpy as np
from PIL import Image

from anomaly_lab.annotation_bitmap import AnnotationBitmapError, decode_shape
from anomaly_lab.domain.annotations import (
    AnnotationDocument,
    BitmapShape,
    BoxShape,
    ClassTableEntry,
    PolygonShape,
)
from anomaly_lab.eval.pixel import connected_regions
from anomaly_lab.media.decode import sha256_of

# An imported binary mask speaks about this class; a `source_mask` base is drawn in it.
SOURCE_MASK_CLASS = "defect"
# The class-index PNG is 8-bit, and 0 is background.
MAX_CLASSES = 255


class AnnotationRenderError(ValueError):
    pass


@dataclass(frozen=True)
class Instance:
    """One object instance of a completed revision, in source-frame pixels."""

    instance_id: str
    label_key: str
    # Tight bounding box of its final pixels, `[x0, y0, x1, y1]` with x1/y1 exclusive.
    box: tuple[int, int, int, int]
    pixels: int

    def as_json(self) -> dict[str, object]:
        return {
            "instance_id": self.instance_id,
            "label_key": self.label_key,
            "box": list(self.box),
            "pixels": self.pixels,
        }


@dataclass(frozen=True)
class RenderedTruth:
    mask_sha256: str
    class_mask_sha256: str
    class_table: list[ClassTableEntry]
    instances_sha256: str | None = None
    instances: tuple[Instance, ...] = ()


def instances_destination(mask_destination: Path) -> Path:
    """The instances file sits beside the binary mask it was rendered with."""
    return mask_destination.with_name(f"{mask_destination.stem}.instances.json")


def render_truth(
    document: AnnotationDocument,
    mask_destination: Path,
    class_destination: Path,
    *,
    classes: Sequence[str],
    source_mask_path: Path | None,
    source_mask_sha256: str | None,
    instances_path: Path | None = None,
) -> RenderedTruth:
    """Atomically render one document into its binary and class-index masks.

    `classes` is the dataset's taxonomy in order at completion; class `classes[i]` is drawn
    with index `i + 1`. Every class is pinned, drawn or not, because a completed revision is
    a confirmed absence of every class that existed when it was completed. With
    `instances_path`, the document's instances are written there as well.
    """
    canvas, owner, keys = _rasterize(
        document,
        classes,
        source_mask_path=source_mask_path,
        source_mask_sha256=source_mask_sha256,
        track_instances=instances_path is not None,
    )
    index_of = {key: position + 1 for position, key in enumerate(classes)}
    counts = np.bincount(canvas.ravel(), minlength=len(classes) + 1)

    binary = np.where(canvas > 0, 255, 0).astype(np.uint8)
    instances: tuple[Instance, ...] = ()
    instances_sha256: str | None = None
    if instances_path is not None and owner is not None:
        instances = _instances(canvas, owner, keys, index_of)
        instances_sha256 = _write_json(
            {"instances": [instance.as_json() for instance in instances]}, instances_path
        )
    return RenderedTruth(
        mask_sha256=_write_png(binary, mask_destination),
        class_mask_sha256=_write_png(canvas, class_destination),
        class_table=[
            ClassTableEntry(key=key, index=index, pixels=int(counts[index]))
            for key, index in index_of.items()
        ],
        instances_sha256=instances_sha256,
        instances=instances,
    )


def _instances(
    canvas: np.ndarray,
    owner: np.ndarray,
    keys: Sequence[tuple[str, str]],
    index_of: dict[str, int],
) -> tuple[Instance, ...]:
    """Every instance left with pixels, in the order its first shape was drawn.

    `owner` holds, per pixel, the 1-based number of the instance that set it last (0: none).
    A pixel counts only while it still carries the instance's class. One pass over the
    owned pixels gives every count and bounding box, so the cost is linear in pixels.
    """
    if not keys:
        return ()
    class_of = np.array([0] + [index_of[label] for _, label in keys], dtype=np.int64)
    ys, xs = np.nonzero(owner)
    numbers = owner[ys, xs].astype(np.int64)
    kept = canvas[ys, xs] == class_of[numbers]
    ys, xs, numbers = ys[kept], xs[kept], numbers[kept]
    size = len(keys) + 1
    pixels = np.bincount(numbers, minlength=size)
    big = np.iinfo(np.int64).max
    x0 = np.full(size, big, dtype=np.int64)
    y0 = np.full(size, big, dtype=np.int64)
    x1 = np.full(size, -1, dtype=np.int64)
    y1 = np.full(size, -1, dtype=np.int64)
    np.minimum.at(x0, numbers, xs)
    np.minimum.at(y0, numbers, ys)
    np.maximum.at(x1, numbers, xs)
    np.maximum.at(y1, numbers, ys)
    return tuple(
        Instance(
            instance_id=key,
            label_key=label,
            box=(int(x0[number]), int(y0[number]), int(x1[number]) + 1, int(y1[number]) + 1),
            pixels=int(pixels[number]),
        )
        for number, (key, label) in enumerate(keys, start=1)
        if pixels[number] > 0
    )


def rasterize(
    document: AnnotationDocument,
    classes: Sequence[str],
    *,
    source_mask_path: Path | None,
    source_mask_sha256: str | None,
) -> np.ndarray:
    """The class-index canvas, in memory: class `classes[i]` is index `i + 1`, 0 background.

    Shapes are drawn in order: an `add` paints its class's index, a `subtract` clears.
    """
    canvas, _, _ = _rasterize(
        document,
        classes,
        source_mask_path=source_mask_path,
        source_mask_sha256=source_mask_sha256,
        track_instances=False,
    )
    return canvas


def document_instances(
    document: AnnotationDocument,
    classes: Sequence[str],
    *,
    source_mask_path: Path | None,
    source_mask_sha256: str | None,
) -> tuple[Instance, ...]:
    """Every object instance of a document, in memory, for detection truth (ADR-0039).

    The instances completion would write, followed by one instance per 8-connected component
    of the `source_mask` base that no drawn instance owns — the imported mask has no
    instances of its own, so its components stand for them, as they do for an image whose
    truth is an imported mask alone. Those are keyed `source-mask-<n>`, in scan order.
    """
    canvas, owner, keys = _rasterize(
        document,
        classes,
        source_mask_path=source_mask_path,
        source_mask_sha256=source_mask_sha256,
        track_instances=True,
    )
    index_of = {key: position + 1 for position, key in enumerate(classes)}
    drawn = _instances(canvas, owner, keys, index_of) if owner is not None else ()
    if document.base != "source_mask" or owner is None:
        return drawn
    unowned = (canvas == index_of[SOURCE_MASK_CLASS]) & (owner == 0)
    return drawn + component_instances(unowned, SOURCE_MASK_CLASS, prefix="source-mask")


def component_instances(mask: np.ndarray, label_key: str, *, prefix: str) -> tuple[Instance, ...]:
    """One instance per 8-connected component of a boolean mask, ordered by its first pixel.

    What an imported binary mask stands for when detection reads it: two touching objects
    become one instance, and one object broken in two becomes two.
    """
    width = mask.shape[1]
    regions = sorted(connected_regions(mask), key=lambda region: int(region.min()))
    found: list[Instance] = []
    for number, region in enumerate(regions, start=1):
        ys, xs = np.divmod(region, width)
        found.append(
            Instance(
                instance_id=f"{prefix}-{number}",
                label_key=label_key,
                box=(int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1),
                pixels=int(region.size),
            )
        )
    return tuple(found)


def _rasterize(
    document: AnnotationDocument,
    classes: Sequence[str],
    *,
    source_mask_path: Path | None,
    source_mask_sha256: str | None,
    track_instances: bool,
) -> tuple[np.ndarray, np.ndarray | None, list[tuple[str, str]]]:
    """The canvas, and with `track_instances` an int32 raster of which instance set each
    pixel last, drawn by the same calls so the two cannot disagree about coverage."""
    if len(classes) > MAX_CLASSES:
        raise AnnotationRenderError(f"a dataset can have at most {MAX_CLASSES} classes")
    index_of = {key: position + 1 for position, key in enumerate(classes)}
    size = (document.image_width, document.image_height)
    if document.base == "source_mask":
        if source_mask_path is None or source_mask_sha256 is None:
            raise AnnotationRenderError("source-mask base has no pinned source provenance")
        if not source_mask_path.is_file():
            raise AnnotationRenderError("the pinned source mask is no longer available")
        if sha256_of(source_mask_path) != source_mask_sha256:
            raise AnnotationRenderError(
                "the pinned source mask changed after the draft was created"
            )
        if SOURCE_MASK_CLASS not in index_of:
            raise AnnotationRenderError(f"the dataset has no {SOURCE_MASK_CLASS!r} class")
        base_index = index_of[SOURCE_MASK_CLASS]
        try:
            with Image.open(source_mask_path) as opened:
                if opened.size != size:
                    raise AnnotationRenderError(
                        "the source mask does not match the source image dimensions"
                    )
                canvas = opened.convert("L").point(lambda value: base_index if value > 0 else 0)
        except OSError as exc:
            raise AnnotationRenderError("the pinned source mask cannot be decoded") from exc
    else:
        canvas = Image.new("L", size, 0)

    canvas_pixels = np.array(canvas, dtype=np.uint8)
    owner = (
        np.zeros((document.image_height, document.image_width), dtype=np.int32)
        if track_instances
        else None
    )
    number_of: dict[str, int] = {}
    keys: list[tuple[str, str]] = []

    for shape in document.shapes:
        if shape.operation == "subtract":
            fill = 0
        elif shape.label_key in index_of:
            fill = index_of[shape.label_key]
        else:
            raise AnnotationRenderError(f"unknown class {shape.label_key!r}")
        owned = 0
        if owner is not None and shape.operation == "add":
            key = shape.instance_id or shape.id
            if key not in number_of:
                number_of[key] = len(keys) + 1
                keys.append((key, shape.label_key))
            owned = number_of[key]
        if isinstance(shape, BitmapShape):
            try:
                bitmap = decode_shape(shape)
            except AnnotationBitmapError as exc:
                raise AnnotationRenderError(str(exc)) from exc
            _paint_bitmap(canvas_pixels, shape, bitmap, fill)
            if owner is not None:
                _paint_bitmap(owner, shape, bitmap, owned)
        else:
            # Polygons and boxes both own the pixels whose centres they contain, so the two
            # rasters are painted from one coverage and cannot disagree about it.
            coverage = shape_coverage(shape, (document.image_width, document.image_height))
            if coverage is not None:
                paint(canvas_pixels, coverage, fill)
                if owner is not None:
                    paint(owner, coverage, owned)
    return canvas_pixels, owner, keys


@dataclass(frozen=True)
class Coverage:
    """The source pixels a vector shape owns: `mask` placed with its top-left at `(x, y)`."""

    x: int
    y: int
    mask: np.ndarray


def shape_coverage(shape: PolygonShape | BoxShape, size: tuple[int, int]) -> Coverage | None:
    """The pixels of a `(width, height)` frame whose centres a polygon or a box contains.

    `None` when it owns none — a degenerate or sub-pixel shape, or one outside the frame.
    """
    width, height = size
    if isinstance(shape, BoxShape):
        x0, y0, x1, y1 = shape.owned_pixels()
        x0, x1 = min(max(x0, 0), width), min(max(x1, 0), width)
        y0, y1 = min(max(y0, 0), height), min(max(y1, 0), height)
        if x1 <= x0 or y1 <= y0:
            return None
        return Coverage(x0, y0, np.ones((y1 - y0, x1 - x0), dtype=bool))
    if isinstance(shape, PolygonShape):
        return polygon_coverage([(point.x, point.y) for point in shape.points], size)
    assert_never(shape)  # pragma: no cover - the vector shapes are closed


def polygon_coverage(
    points: Sequence[tuple[float, float]], size: tuple[int, int]
) -> Coverage | None:
    """The pixels whose centres lie inside a polygon, by the even-odd rule.

    Pixel `(i, j)` covers `[i, i + 1) x [j, j + 1)`, and it is owned when its centre
    `(i + 0.5, j + 0.5)` is inside, which is `BoxShape.owned_pixels`' half-open rule: a
    centre on a left or top edge is in, one on a right or bottom edge is out, so the
    polygon of a box's four corners owns exactly the box's pixels. The test is the editor's
    readout (`pixelReadout.insidePolygon`) evaluated at every centre at once — an edge
    crosses row `j` when exactly one of its ends lies strictly below the centre line, at
    the x the readout computes by the same float operations, and a pixel is inside when
    an odd number of crossings lie strictly right of its centre. Holes and
    self-intersections follow the even-odd rule, as the readout and the canvas fill do;
    a polygon with no area owns nothing, and anything outside the frame is clipped.

    The cost is one entry per (edge, row it spans) plus the polygon's bounding box, never
    the frame.
    """
    width, height = size
    if len(points) < 3 or width <= 0 or height <= 0:
        return None
    coordinates = np.asarray(points, dtype=np.float64)
    if not np.isfinite(coordinates).all():
        raise AnnotationRenderError("a polygon point is not finite")
    # Edge k runs from vertex k (`a`) back to vertex k - 1 (`b`), as the readout walks it.
    ax, ay = coordinates[:, 0], coordinates[:, 1]
    bx, by = np.roll(ax, 1), np.roll(ay, 1)
    low = np.clip(np.minimum(ay, by), -1.0, height + 1.0)
    high = np.clip(np.maximum(ay, by), -1.0, height + 1.0)
    # A generous row range per edge, narrowed below by the exact comparison.
    first = np.clip(np.floor(low - 0.5).astype(np.int64), 0, height)
    stop = np.clip(np.ceil(high - 0.5).astype(np.int64) + 1, 0, height)
    spans = np.maximum(stop - first, 0)
    total = int(spans.sum())
    if total == 0:
        return None
    edge = np.repeat(np.arange(len(coordinates)), spans)
    offsets = np.arange(total) - np.repeat(np.cumsum(spans) - spans, spans)
    rows = np.repeat(first, spans) + offsets
    centre = rows + 0.5
    crosses = (ay[edge] > centre) != (by[edge] > centre)
    if not crosses.any():
        return None
    edge, rows, centre = edge[crosses], rows[crosses], centre[crosses]
    a_x, a_y, b_x, b_y = ax[edge], ay[edge], bx[edge], by[edge]
    crossing = ((b_x - a_x) * (centre - a_y)) / (b_y - a_y) + a_x
    # How many column centres lie strictly left of each crossing: exact, never rounded.
    cut = np.searchsorted(np.arange(width) + 0.5, crossing, side="left")
    top, bottom = int(rows.min()), int(rows.max()) + 1
    left, right = int(cut.min()), int(cut.max())
    if right <= left:
        return None
    span = right - left + 1
    tally = np.bincount(
        (rows - top) * span + (cut - left), minlength=(bottom - top) * span
    ).reshape(bottom - top, span)
    # Crossings strictly right of column i's centre are those whose cut exceeds i.
    right_of = tally.sum(axis=1, keepdims=True) - np.cumsum(tally, axis=1)
    mask = (right_of[:, :-1] % 2).astype(bool)
    if not mask.any():
        return None
    return Coverage(left, top, mask)


def paint(target: np.ndarray, coverage: Coverage, value: int) -> None:
    """Set the covered pixels of a `(height, width)` raster to `value`, in place."""
    height, width = coverage.mask.shape
    region = target[coverage.y : coverage.y + height, coverage.x : coverage.x + width]
    region[coverage.mask] = value


def _paint_bitmap(target: np.ndarray, shape: BitmapShape, bitmap: np.ndarray, value: int) -> None:
    region = target[shape.y : shape.y + shape.height, shape.x : shape.x + shape.width]
    region[bitmap] = value


def _write_json(payload: object, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp.json")
    try:
        temporary.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")))
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return sha256_of(destination)


def _write_png(pixels: np.ndarray, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp.png")
    try:
        Image.fromarray(pixels, mode="L").save(temporary, format="PNG", optimize=False)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return sha256_of(destination)
