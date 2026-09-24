"""Deterministic materialisation of source-frame annotation documents.

A completed document becomes two PNGs from one rasterisation: a class-index mask, where each
pixel holds the index of the class drawn there last (0 is background), and the binary mask
every anomaly consumer reads, which is exactly `index > 0`. The class table that gives the
indices their meaning is returned with the pixel count of each class, and pinned on the
revision, so a class's presence is a database read rather than a decode (ADR-0040).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from anomaly_lab.annotation_bitmap import AnnotationBitmapError, decode_shape
from anomaly_lab.domain.annotations import AnnotationDocument, BitmapShape, ClassTableEntry
from anomaly_lab.media.decode import sha256_of

# An imported binary mask speaks about this class; a `source_mask` base is drawn in it.
SOURCE_MASK_CLASS = "defect"
# The class-index PNG is 8-bit, and 0 is background.
MAX_CLASSES = 255


class AnnotationRenderError(ValueError):
    pass


@dataclass(frozen=True)
class RenderedTruth:
    mask_sha256: str
    class_mask_sha256: str
    class_table: list[ClassTableEntry]


def render_truth(
    document: AnnotationDocument,
    mask_destination: Path,
    class_destination: Path,
    *,
    classes: Sequence[str],
    source_mask_path: Path | None,
    source_mask_sha256: str | None,
) -> RenderedTruth:
    """Atomically render one document into its binary and class-index masks.

    `classes` is the dataset's taxonomy in order at completion; class `classes[i]` is drawn
    with index `i + 1`. Every class is pinned, drawn or not, because a completed revision is
    a confirmed absence of every class that existed when it was completed.
    """
    canvas = rasterize(
        document,
        classes,
        source_mask_path=source_mask_path,
        source_mask_sha256=source_mask_sha256,
    )
    index_of = {key: position + 1 for position, key in enumerate(classes)}
    counts = np.bincount(canvas.ravel(), minlength=len(classes) + 1)

    binary = np.where(canvas > 0, 255, 0).astype(np.uint8)
    return RenderedTruth(
        mask_sha256=_write_png(binary, mask_destination),
        class_mask_sha256=_write_png(canvas, class_destination),
        class_table=[
            ClassTableEntry(key=key, index=index, pixels=int(counts[index]))
            for key, index in index_of.items()
        ],
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

    draw = ImageDraw.Draw(canvas)
    for shape in document.shapes:
        if shape.operation == "subtract":
            fill = 0
        elif shape.label_key in index_of:
            fill = index_of[shape.label_key]
        else:
            raise AnnotationRenderError(f"unknown class {shape.label_key!r}")
        if isinstance(shape, BitmapShape):
            try:
                bitmap = decode_shape(shape)
            except AnnotationBitmapError as exc:
                raise AnnotationRenderError(str(exc)) from exc
            region = np.asarray(canvas)[
                shape.y : shape.y + shape.height,
                shape.x : shape.x + shape.width,
            ].copy()
            region[bitmap] = fill
            canvas.paste(
                Image.fromarray(region.astype(np.uint8), mode="L"),
                (shape.x, shape.y),
            )
            draw = ImageDraw.Draw(canvas)
        else:
            draw.polygon([(point.x, point.y) for point in shape.points], fill=fill)
    return np.asarray(canvas, dtype=np.uint8)


def _write_png(pixels: np.ndarray, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp.png")
    try:
        Image.fromarray(pixels, mode="L").save(temporary, format="PNG", optimize=False)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return sha256_of(destination)
