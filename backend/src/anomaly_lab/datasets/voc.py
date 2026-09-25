"""Pascal VOC box annotations, read as boxes in the source frame's pixel-edge coordinates.

A VOC file lists every object of its image: a class name and a `bndbox` of `xmin`, `ymin`,
`xmax`, `ymax`. Those are **1-based pixel indices, inclusive** (the VOC devkit's convention,
and LabelImg's, which clamps a corner to 1), so the object covers pixels `xmin - 1` to
`xmax - 1` counted from 0. In the workbench's pixel-edge coordinates, where pixel `i` spans
`[i, i + 1)`, that is the box `[xmin - 1, xmax)`: its width is `xmax - xmin + 1`, and it owns
exactly the pixels the file names (`BoxShape.owned_pixels`).

A tool that wrote 0-based corners reads one pixel up and to the left of what it meant, and a
corner at 0 is clamped into the frame rather than refused; either is a pixel against boxes
tens of pixels wide. The size the file declares is checked against the image, because a
file paired with the wrong image is the error worth catching.
"""

from __future__ import annotations

import xml.etree.ElementTree as ElementTree
from dataclasses import dataclass
from pathlib import Path


class VocError(ValueError):
    """The file is not a VOC annotation this reader can use."""


@dataclass(frozen=True)
class VocObject:
    name: str
    box: tuple[float, float, float, float]
    """`(x0, y0, x1, y1)` in pixel-edge coordinates, `x1`/`y1` exclusive."""


@dataclass(frozen=True)
class VocAnnotation:
    size: tuple[int, int] | None
    """`(width, height)` as the file declares it, when it does."""
    objects: tuple[VocObject, ...]


def _number(element: ElementTree.Element, tag: str, path: Path) -> float:
    text = element.findtext(tag)
    if text is None or not text.strip():
        raise VocError(f"{path}: an object has no {tag}")
    try:
        return float(text)
    except ValueError as exc:
        raise VocError(f"{path}: {tag} {text!r} is not a number") from exc


def read_voc(path: Path) -> VocAnnotation:
    """Every object of one VOC file, in file order."""
    try:
        root = ElementTree.parse(path).getroot()
    except ElementTree.ParseError as exc:
        raise VocError(f"{path}: {exc}") from exc
    if root.tag != "annotation":
        raise VocError(f"{path}: the root element is <{root.tag}>, not <annotation>")

    size: tuple[int, int] | None = None
    declared = root.find("size")
    if declared is not None:
        width, height = _number(declared, "width", path), _number(declared, "height", path)
        if width > 0 and height > 0:
            size = (int(width), int(height))

    objects: list[VocObject] = []
    for element in root.findall("object"):
        name = (element.findtext("name") or "").strip()
        if not name:
            raise VocError(f"{path}: an object has no name")
        bndbox = element.find("bndbox")
        if bndbox is None:
            raise VocError(f"{path}: the {name} object has no bndbox")
        xmin, ymin = _number(bndbox, "xmin", path), _number(bndbox, "ymin", path)
        xmax, ymax = _number(bndbox, "xmax", path), _number(bndbox, "ymax", path)
        if xmax < xmin or ymax < ymin:
            raise VocError(f"{path}: the {name} box ({xmin}, {ymin}, {xmax}, {ymax}) is inverted")
        objects.append(VocObject(name, (xmin - 1, ymin - 1, xmax, ymax)))
    return VocAnnotation(size, tuple(objects))


def clamp_box(
    box: tuple[float, float, float, float], width: int, height: int
) -> tuple[float, float, float, float] | None:
    """`box` cut to the frame, or `None` when nothing of it is inside."""
    x0, y0 = max(0.0, box[0]), max(0.0, box[1])
    x1, y1 = min(float(width), box[2]), min(float(height), box[3])
    if x1 <= x0 or y1 <= y0:
        return None
    return (x0, y0, x1, y1)
