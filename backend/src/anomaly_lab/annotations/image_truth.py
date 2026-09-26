"""One image's truth as the sample view draws it: which classes, which boxes, which regions.

The presence rule is `annotations_repo.image_presence`, the one every class read shares: the
newest completed revision answers, else an imported ground-truth mask (the default class),
else a sample labelled normal (a confirmed absence), else nothing. This module only decides
how that answer is *drawn*:

- a revision's `add` boxes are drawn as rectangles with their class, never as a filled region —
  a box is an object's extent, and filling a PCB's six defect classes over the board hides
  the defects;
- every other region of a revision (bitmaps, polygons, a `source_mask` base) is rasterised
  without those boxes and drawn filled, in its class's colour;
- an imported mask is anomaly truth and is drawn as an outline, as every other anomaly
  screen draws it.

Truth is image-keyed below the editing unit (`docs/architecture/annotations.md`), so a sample
scope that materialised one revision per channel reads here exactly as an image scope does:
each channel's image answers for itself, and the channel count never appears.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image

from anomaly_lab.annotation_render import AnnotationRenderError, rasterize
from anomaly_lab.annotations.class_truth import ClassTruthError
from anomaly_lab.db.repositories import annotations as annotations_repo
from anomaly_lab.db.repositories import images as images_repo
from anomaly_lab.db.repositories.annotations import DEFAULT_LABEL_KEY
from anomaly_lab.domain.annotations import AnnotationDocument, BoxShape
from anomaly_lab.domain.entities import ClassPresence, Label
from anomaly_lab.media.decode import sha256_of
from anomaly_lab.models.preprocessing import load_mask

TruthSource = Literal["revision", "imported_mask", "normal", "none"]


@dataclass(frozen=True)
class TruthBox:
    label_key: str
    x: float
    y: float
    width: float
    height: float


@dataclass(frozen=True)
class ImageTruth:
    image_id: int
    dataset_id: int
    source: TruthSource
    revision_id: int | None
    classes: tuple[str, ...]
    """The classes present in this image, in the dataset's order."""
    boxes: tuple[TruthBox, ...]
    regions: bool
    """Whether there is a region layer to draw — anything but boxes."""
    outline: bool
    """Whether the region layer is anomaly truth, drawn as an outline rather than filled."""
    identity: str
    """What pins the answer, for a validator: a changed truth is a different value."""


@dataclass(frozen=True)
class RegionPlane:
    """The region layer as class indices: `labels == i + 1` is `classes[i]`, 0 background."""

    labels: np.ndarray
    classes: tuple[str, ...]
    outline: bool


def read_image_truth(conn: sqlite3.Connection, image_id: int) -> ImageTruth | None:
    """How one image's truth is drawn; `None` when there is no such image."""
    rows = annotations_repo.truth_rows_for_images(conn, [image_id])
    if not rows:
        return None
    dataset_id = images_repo.dataset_id_of(conn, image_id)
    row = rows[0]
    order = [label.key for label in annotations_repo.list_labels(conn, dataset_id)]
    created = annotations_repo.label_created_at(conn, dataset_id)
    present = tuple(
        key
        for key in order
        if annotations_repo.image_presence(row, key, created) is ClassPresence.PRESENT
    )

    if row["document"] is not None:
        document = json.loads(str(row["document"]))
        boxes = tuple(
            TruthBox(
                label_key=str(shape["label_key"]),
                x=float(shape["x"]),
                y=float(shape["y"]),
                width=float(shape["width"]),
                height=float(shape["height"]),
            )
            for shape in document.get("shapes", [])
            if shape.get("kind") == "box" and shape.get("operation", "add") == "add"
        )
        regions = document.get("base") == "source_mask" or any(
            shape.get("kind") != "box" and shape.get("operation", "add") == "add"
            for shape in document.get("shapes", [])
        )
        return ImageTruth(
            image_id=image_id,
            dataset_id=dataset_id,
            source="revision",
            revision_id=int(row["revision_id"]),
            classes=present,
            boxes=boxes,
            regions=regions,
            outline=False,
            identity=f"revision-{int(row['revision_id'])}",
        )
    if row["source_id"] is not None:
        return ImageTruth(
            image_id=image_id,
            dataset_id=dataset_id,
            source="imported_mask",
            revision_id=None,
            classes=present or (DEFAULT_LABEL_KEY,),
            boxes=(),
            regions=True,
            outline=True,
            identity=f"mask-{row['source_sha256'] or row['source_id']}",
        )
    normal = row["label"] == Label.NORMAL.value
    return ImageTruth(
        image_id=image_id,
        dataset_id=dataset_id,
        source="normal" if normal else "none",
        revision_id=None,
        classes=(),
        boxes=(),
        regions=False,
        outline=False,
        identity="normal" if normal else "none",
    )


def read_region_plane(conn: sqlite3.Connection, image_id: int) -> RegionPlane | None:
    """The region layer of an image's truth; `None` when it has none to draw.

    Raises `ClassTruthError` when the file a revision or an import pinned is missing or has
    changed since, as every other truth read does.
    """
    truth = read_image_truth(conn, image_id)
    if truth is None or not truth.regions:
        return None
    [row] = annotations_repo.truth_rows_for_images(conn, [image_id])
    size = (int(row["width"]), int(row["height"]))

    if truth.source == "imported_mask":
        path = Path(str(row["source_path"]))
        _verify(path, row["source_sha256"], image_id)
        region = load_mask(path, size=size)
        return RegionPlane(
            labels=region.astype(np.uint8), classes=(DEFAULT_LABEL_KEY,), outline=True
        )

    document = AnnotationDocument.model_validate_json(str(row["document"]))
    has_boxes = any(isinstance(shape, BoxShape) for shape in document.shapes)
    if row["class_table"] is not None:
        table = sorted(json.loads(str(row["class_table"])), key=lambda entry: int(entry["index"]))
        classes = tuple(str(entry["key"]) for entry in table)
        if not has_boxes:
            # The stored class mask is exactly this document's raster: read it, verified.
            path = Path(str(row["class_mask_path"]))
            _verify(path, row["class_mask_sha256"], image_id)
            with Image.open(path) as opened:
                index = np.asarray(opened.convert("L"))
            return RegionPlane(labels=_renumber(index, table), classes=classes, outline=False)
    else:
        classes = tuple(label.key for label in annotations_repo.list_labels(conn, truth.dataset_id))
    unboxed = document.model_copy(
        update={
            "shapes": [
                shape
                for shape in document.shapes
                if not (isinstance(shape, BoxShape) and shape.operation == "add")
            ]
        }
    )
    try:
        canvas = rasterize(
            unboxed,
            classes,
            source_mask_path=(
                Path(row["revision_source_path"]) if row["revision_source_path"] else None
            ),
            source_mask_sha256=row["revision_source_sha256"],
        )
    except AnnotationRenderError as exc:
        raise ClassTruthError(f"image {image_id}: {exc}") from exc
    return RegionPlane(labels=np.asarray(canvas, dtype=np.uint8), classes=classes, outline=False)


def _renumber(index: np.ndarray, table: list[dict[str, object]]) -> np.ndarray:
    """A stored class-index mask in table order: the `i`th entry becomes `i + 1`."""
    lookup = np.zeros(256, dtype=np.uint8)
    for position, entry in enumerate(table, start=1):
        lookup[int(str(entry["index"]))] = position
    return lookup[np.asarray(index, dtype=np.uint8)]


def _verify(path: Path, sha256: str | None, image_id: int) -> None:
    if not path.is_file():
        raise ClassTruthError(f"the truth of image {image_id} is unavailable")
    if sha256 is not None and sha256_of(path) != sha256:
        raise ClassTruthError(f"the truth of image {image_id} changed after it was pinned")
