"""Lossless PNG, LabelMe and COCO interchange for one source-frame annotation."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from PIL import Image, ImageDraw
from pydantic import BaseModel, ConfigDict, Field

from anomaly_lab.annotation_bitmap import (
    AnnotationBitmapError,
    decode_png,
    decode_png_base64,
    tight_bitmap_shape,
)
from anomaly_lab.domain.annotations import (
    AnnotationDocument,
    AnnotationPoint,
    AnnotationShape,
    BitmapShape,
    PolygonShape,
)


class AnnotationInterchangeError(ValueError):
    pass


class LabelMeShape(BaseModel):
    model_config = ConfigDict(extra="allow")

    label: str
    points: list[list[float]]
    group_id: int | None = None
    description: str = ""
    shape_type: str
    flags: dict[str, bool] = Field(default_factory=dict)
    mask: str | None = None


class LabelMeDocument(BaseModel):
    model_config = ConfigDict(extra="allow")

    version: str | None = None
    flags: dict[str, bool] = Field(default_factory=dict)
    shapes: list[LabelMeShape] = Field(default_factory=list)
    image_path: str = Field(alias="imagePath")
    image_data: str | None = Field(default=None, alias="imageData")
    image_height: int = Field(alias="imageHeight", gt=0)
    image_width: int = Field(alias="imageWidth", gt=0)


class CocoRle(BaseModel):
    size: tuple[int, int]
    counts: list[int] | str


class CocoImage(BaseModel):
    id: int
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    file_name: str


class CocoCategory(BaseModel):
    id: int
    name: str
    supercategory: str = "defect"


class CocoAnnotation(BaseModel):
    id: int
    image_id: int
    category_id: int
    segmentation: CocoRle | list[list[float]]
    area: float = Field(ge=0)
    bbox: tuple[float, float, float, float]
    iscrowd: Literal[0, 1] = 0


class CocoDocument(BaseModel):
    info: dict[str, Any] = Field(default_factory=dict)
    licenses: list[dict[str, Any]] = Field(default_factory=list)
    images: list[CocoImage]
    annotations: list[CocoAnnotation] = Field(default_factory=list)
    categories: list[CocoCategory]


def _label_key(value: str, known: dict[str, str]) -> str:
    direct = value.strip().lower()
    if direct in known:
        return direct
    by_name = {name.casefold(): key for key, name in known.items()}
    found = by_name.get(value.strip().casefold())
    if found is None:
        raise AnnotationInterchangeError(
            f"annotation label {value!r} is not in this dataset's taxonomy"
        )
    return found


def imported_document(
    current: AnnotationDocument,
    shapes: list[AnnotationShape],
    *,
    clear_label_key: str,
) -> AnnotationDocument:
    """Replace editable operations while preserving the immutable source-mask base."""
    replacement: list[AnnotationShape] = []
    if current.base == "source_mask":
        replacement.append(
            PolygonShape(
                id="import-clear-source-mask",
                label_key=clear_label_key,
                operation="subtract",
                points=[
                    AnnotationPoint(x=0, y=0),
                    AnnotationPoint(x=current.image_width, y=0),
                    AnnotationPoint(x=current.image_width, y=current.image_height),
                    AnnotationPoint(x=0, y=current.image_height),
                ],
            )
        )
    replacement.extend(shapes)
    return current.model_copy(update={"shapes": replacement})


def shapes_from_png(
    payload: bytes,
    *,
    size: tuple[int, int],
    label_key: str,
) -> list[AnnotationShape]:
    try:
        mask = decode_png(payload, expected_size=size)
    except AnnotationBitmapError as exc:
        raise AnnotationInterchangeError(str(exc)) from exc
    shape = tight_bitmap_shape(mask, shape_id="png-mask", label_key=label_key)
    return [] if shape is None else [shape]


def labelme_from_mask(
    mask: np.ndarray,
    *,
    image_path: str,
    label: str,
) -> LabelMeDocument:
    shape = tight_bitmap_shape(mask, shape_id="labelme-mask", label_key=label)
    shapes: list[LabelMeShape] = []
    if shape is not None:
        shapes.append(
            LabelMeShape(
                label=label,
                points=[
                    [float(shape.x), float(shape.y)],
                    [float(shape.x + shape.width - 1), float(shape.y + shape.height - 1)],
                ],
                shape_type="mask",
                mask=shape.png_base64,
            )
        )
    height, width = mask.shape
    return LabelMeDocument.model_validate(
        {
            "version": "7.0.2",
            "flags": {},
            "shapes": [item.model_dump() for item in shapes],
            "imagePath": image_path,
            "imageData": None,
            "imageHeight": int(height),
            "imageWidth": int(width),
        }
    )


def shapes_from_labelme(
    document: LabelMeDocument,
    *,
    size: tuple[int, int],
    known_labels: dict[str, str],
) -> list[AnnotationShape]:
    if (document.image_width, document.image_height) != size:
        raise AnnotationInterchangeError("LabelMe dimensions do not match the source image")
    shapes: list[AnnotationShape] = []
    for index, source in enumerate(document.shapes):
        label_key = _label_key(source.label, known_labels)
        if source.shape_type == "polygon":
            if len(source.points) < 3 or any(len(point) != 2 for point in source.points):
                raise AnnotationInterchangeError("a LabelMe polygon needs at least three points")
            shapes.append(
                PolygonShape(
                    id=f"labelme-polygon-{index + 1}",
                    label_key=label_key,
                    points=[AnnotationPoint(x=point[0], y=point[1]) for point in source.points],
                )
            )
            continue
        if source.shape_type == "mask":
            if source.mask is None or len(source.points) != 2:
                raise AnnotationInterchangeError(
                    "a LabelMe mask needs two bounding-box points and base64 PNG data"
                )
            (x0, y0), (x1, y1) = source.points
            integers = tuple(round(value) for value in (x0, y0, x1, y1))
            if any(
                abs(value - integer) > 1e-6
                for value, integer in zip((x0, y0, x1, y1), integers, strict=True)
            ):
                raise AnnotationInterchangeError("LabelMe mask bounds must be integer pixels")
            left, top, right, bottom = integers
            width, height = right - left + 1, bottom - top + 1
            try:
                mask = decode_png_base64(source.mask, expected_size=(width, height))
            except AnnotationBitmapError as exc:
                raise AnnotationInterchangeError(str(exc)) from exc
            shape = tight_bitmap_shape(
                mask,
                shape_id=f"labelme-mask-{index + 1}",
                label_key=label_key,
            )
            if shape is not None:
                shapes.append(shape.model_copy(update={"x": shape.x + left, "y": shape.y + top}))
            continue
        raise AnnotationInterchangeError(
            f"LabelMe shape_type {source.shape_type!r} is not supported; use polygon or mask"
        )
    return shapes


def encode_coco_rle(mask: np.ndarray) -> list[int]:
    flat = np.asarray(mask, dtype=bool).reshape(-1, order="F")
    counts: list[int] = []
    previous = False
    run = 0
    for value in flat:
        current = bool(value)
        if current == previous:
            run += 1
        else:
            counts.append(run)
            run = 1
            previous = current
    counts.append(run)
    return counts


def _decode_compressed_counts(value: str) -> list[int]:
    counts: list[int] = []
    position = 0
    while position < len(value):
        number = 0
        shift = 0
        more = True
        while more:
            if position >= len(value):
                raise AnnotationInterchangeError("COCO compressed RLE is truncated")
            code = ord(value[position]) - 48
            position += 1
            if code < 0 or code > 0x3F:
                raise AnnotationInterchangeError("COCO compressed RLE contains invalid data")
            number |= (code & 0x1F) << (5 * shift)
            more = bool(code & 0x20)
            if not more and code & 0x10:
                number |= -1 << (5 * (shift + 1))
            shift += 1
        if len(counts) > 2:
            number += counts[-2]
        if number < 0:
            raise AnnotationInterchangeError("COCO compressed RLE contains a negative run")
        counts.append(number)
    return counts


def decode_coco_rle(rle: CocoRle, *, size: tuple[int, int]) -> np.ndarray:
    width, height = size
    if rle.size != (height, width):
        raise AnnotationInterchangeError("COCO RLE dimensions do not match the source image")
    counts = rle.counts if isinstance(rle.counts, list) else _decode_compressed_counts(rle.counts)
    if any(count < 0 for count in counts) or sum(counts) != width * height:
        raise AnnotationInterchangeError("COCO RLE runs do not cover the source image exactly")
    flat = np.empty(width * height, dtype=bool)
    offset = 0
    value = False
    for count in counts:
        flat[offset : offset + count] = value
        offset += count
        value = not value
    return flat.reshape((height, width), order="F")


def coco_from_mask(
    mask: np.ndarray,
    *,
    image_path: str,
    label: str,
) -> CocoDocument:
    height, width = mask.shape
    ys, xs = np.nonzero(mask)
    annotations: list[CocoAnnotation] = []
    if xs.size:
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        annotations.append(
            CocoAnnotation(
                id=1,
                image_id=1,
                category_id=1,
                segmentation=CocoRle(size=(height, width), counts=encode_coco_rle(mask)),
                area=float(mask.sum()),
                bbox=(float(x0), float(y0), float(x1 - x0), float(y1 - y0)),
                iscrowd=1,
            )
        )
    return CocoDocument(
        info={"description": "visual-anomaly-lab annotation export"},
        images=[CocoImage(id=1, width=width, height=height, file_name=image_path)],
        annotations=annotations,
        categories=[CocoCategory(id=1, name=label)],
    )


def shapes_from_coco(
    document: CocoDocument,
    *,
    size: tuple[int, int],
    known_labels: dict[str, str],
) -> list[AnnotationShape]:
    matching = [image for image in document.images if (image.width, image.height) == size]
    if len(matching) != 1:
        raise AnnotationInterchangeError(
            "COCO import must contain exactly one image matching the source dimensions"
        )
    image = matching[0]
    categories = {category.id: category for category in document.categories}
    shapes: list[AnnotationShape] = []
    for annotation in document.annotations:
        if annotation.image_id != image.id:
            continue
        category = categories.get(annotation.category_id)
        if category is None:
            raise AnnotationInterchangeError(
                f"COCO annotation {annotation.id} references an unknown category"
            )
        label_key = _label_key(category.name, known_labels)
        if isinstance(annotation.segmentation, CocoRle):
            mask = decode_coco_rle(annotation.segmentation, size=size)
            shape = tight_bitmap_shape(
                mask,
                shape_id=f"coco-rle-{annotation.id}",
                label_key=label_key,
            )
            if shape is not None:
                shapes.append(shape)
            continue
        for polygon_index, coordinates in enumerate(annotation.segmentation):
            if len(coordinates) < 6 or len(coordinates) % 2:
                raise AnnotationInterchangeError(
                    f"COCO annotation {annotation.id} contains an invalid polygon"
                )
            points = [
                AnnotationPoint(x=coordinates[index], y=coordinates[index + 1])
                for index in range(0, len(coordinates), 2)
            ]
            shapes.append(
                PolygonShape(
                    id=f"coco-polygon-{annotation.id}-{polygon_index + 1}",
                    label_key=label_key,
                    points=points,
                )
            )
    return shapes


def render_shapes(shapes: list[AnnotationShape], *, size: tuple[int, int]) -> np.ndarray:
    """Test/interchange helper: rasterise additive imported shapes without a base."""
    canvas = Image.new("L", size, 0)
    draw = ImageDraw.Draw(canvas)
    for shape in shapes:
        if isinstance(shape, BitmapShape):
            bitmap = decode_png_base64(shape.png_base64, expected_size=(shape.width, shape.height))
            region = np.asarray(canvas)[
                shape.y : shape.y + shape.height,
                shape.x : shape.x + shape.width,
            ].copy()
            region[bitmap] = 255 if shape.operation == "add" else 0
            canvas.paste(Image.fromarray(region.astype(np.uint8), mode="L"), (shape.x, shape.y))
            draw = ImageDraw.Draw(canvas)
        else:
            draw.polygon(
                [(point.x, point.y) for point in shape.points],
                fill=255 if shape.operation == "add" else 0,
            )
    return np.asarray(canvas) > 0
