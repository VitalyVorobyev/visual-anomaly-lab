"""Lossless binary annotation layers, encoded as cropped PNGs."""

from __future__ import annotations

import base64
import binascii
import io
from typing import Literal

import numpy as np
from PIL import Image

from anomaly_lab.domain.annotations import BitmapShape

MAX_BITMAP_BYTES = 64 * 1024 * 1024


class AnnotationBitmapError(ValueError):
    pass


def decode_png_base64(value: str, *, expected_size: tuple[int, int]) -> np.ndarray:
    try:
        payload = base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise AnnotationBitmapError("bitmap data is not valid base64") from exc
    if len(payload) > MAX_BITMAP_BYTES:
        raise AnnotationBitmapError("bitmap PNG exceeds the 64 MiB annotation limit")
    return decode_png(payload, expected_size=expected_size)


def decode_png(payload: bytes, *, expected_size: tuple[int, int]) -> np.ndarray:
    if len(payload) > MAX_BITMAP_BYTES:
        raise AnnotationBitmapError("bitmap PNG exceeds the 64 MiB annotation limit")
    try:
        with Image.open(io.BytesIO(payload)) as opened:
            if opened.size != expected_size:
                raise AnnotationBitmapError(
                    f"bitmap dimensions {opened.size} do not match {expected_size}"
                )
            array = np.asarray(opened.convert("L")) > 0
    except (OSError, ValueError) as exc:
        if isinstance(exc, AnnotationBitmapError):
            raise
        raise AnnotationBitmapError("bitmap data is not a readable PNG") from exc
    return array


def encode_png(mask: np.ndarray) -> bytes:
    canvas = Image.fromarray(np.where(mask, 255, 0).astype(np.uint8), mode="L")
    output = io.BytesIO()
    canvas.save(output, format="PNG", optimize=False)
    return output.getvalue()


def encode_png_base64(mask: np.ndarray) -> str:
    return base64.b64encode(encode_png(mask)).decode("ascii")


def decode_shape(shape: BitmapShape) -> np.ndarray:
    return decode_png_base64(
        shape.png_base64,
        expected_size=(shape.width, shape.height),
    )


def tight_bitmap_shape(
    mask: np.ndarray,
    *,
    shape_id: str,
    label_key: str,
    operation: Literal["add", "subtract"] = "add",
) -> BitmapShape | None:
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    return BitmapShape(
        id=shape_id,
        label_key=label_key,
        operation=operation,
        x=x0,
        y=y0,
        width=x1 - x0,
        height=y1 - y0,
        png_base64=encode_png_base64(mask[y0:y1, x0:x1]),
    )
