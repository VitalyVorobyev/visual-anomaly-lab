"""Deterministic materialisation of source-frame annotation documents."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

from anomaly_lab.domain.annotations import AnnotationDocument
from anomaly_lab.media.decode import sha256_of


class AnnotationRenderError(ValueError):
    pass


def render_binary_mask(
    document: AnnotationDocument,
    destination: Path,
    *,
    source_mask_path: Path | None,
    source_mask_sha256: str | None,
) -> str:
    """Atomically render one document and return the materialised PNG digest."""
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
        try:
            with Image.open(source_mask_path) as opened:
                if opened.size != size:
                    raise AnnotationRenderError(
                        "the source mask does not match the source image dimensions"
                    )
                canvas = opened.convert("L").point(lambda value: 255 if value > 0 else 0)
        except OSError as exc:
            raise AnnotationRenderError("the pinned source mask cannot be decoded") from exc
    else:
        canvas = Image.new("L", size, 0)

    draw = ImageDraw.Draw(canvas)
    for shape in document.shapes:
        fill = 255 if shape.operation == "add" else 0
        draw.polygon([(point.x, point.y) for point in shape.points], fill=fill)

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp.png")
    try:
        canvas.save(temporary, format="PNG", optimize=False)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return sha256_of(destination)
