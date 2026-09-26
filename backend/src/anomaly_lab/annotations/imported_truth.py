"""Class truth a dataset ships, entered as each image's first completed revision.

ADR-0039 decides what a class annotation is; ADR-0041 that it is not anomaly truth.

An imported binary mask answers for the default class alone, so a dataset annotated with
several classes -- as boxes, or as one mask per image of the class its directory names --
cannot enter as imported masks without losing its classes. It enters as what the editor would
have produced had someone drawn it: one `box` shape per object, or one `bitmap` shape of the
image's class, in a document on an empty base, completed through the ordinary draft lifecycle.
Completion pins the class table and writes the instances file, so the revision answers every
class of the dataset -- present where it has a region, absent where it has none -- and the class
reads (`annotations/class_truth.py`) read it like any drawn revision. It stays editable: a
person who fixes a region completes the next revision, and the imported one is kept.

Class truth is not anomaly truth (ADR-0041): entering it sets no sample label.

**Nothing is overwritten.** An image that already has a revision or an open draft keeps it and
is counted as skipped, so a second registration fills only what the first did not finish.

**A box keeps the pixels it names unless a smaller one overlaps it.** Completion gives a pixel
to the later shape, so boxes are drawn largest first, the rule `color_detector.paint_boxes`
trains on: where two overlap, the smaller keeps its pixels. An instance's box is the tight box
of what it finally owns, so an overlap that covers a whole edge of the larger box shrinks it;
every such box is counted (`reshaped`) rather than silently changed.

**A mask is foreground wherever it is non-zero**, the rule every imported mask is read by
(`models.preprocessing.load_mask`), so a mask entered as a class region covers exactly the
pixels it would have covered as an imported mask.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from anomaly_lab.annotation_bitmap import tight_bitmap_shape
from anomaly_lab.annotations.service import IMAGE_DRAFTS
from anomaly_lab.config import Settings
from anomaly_lab.db.connection import connection, transaction
from anomaly_lab.db.repositories import annotations as annotations_repo
from anomaly_lab.domain.annotations import (
    AnnotationDocument,
    AnnotationRevision,
    AnnotationShape,
    BoxShape,
)

# The editor's palette for a class added by hand (`CLASS_PALETTE` in
# `frontend/src/routes/dataset/ClassManager.tsx`), so an imported class looks like one a
# person added. Taken in order, skipping colours the dataset already uses.
CLASS_PALETTE = (
    "#e8590c",
    "#1c7ed6",
    "#2f9e44",
    "#ae3ec9",
    "#f59f00",
    "#0c8599",
    "#d6336c",
    "#5c940d",
)


@dataclass(frozen=True)
class ImportedClass:
    key: str
    name: str


@dataclass(frozen=True)
class ImportedBox:
    label_key: str
    box: tuple[float, float, float, float]
    """`(x0, y0, x1, y1)` in the source frame's pixel-edge coordinates, inside the frame."""


def ensure_classes(
    settings: Settings, dataset_id: int, classes: Sequence[ImportedClass]
) -> list[str]:
    """Add the classes the dataset lacks, after the ones it has; return the keys added.

    Must run before any revision is completed: a revision answers only for the classes that
    existed when it was completed.
    """
    added: list[str] = []
    with connection(settings.db_path) as conn, transaction(conn, immediate=True):
        annotations_repo.ensure_default_label(conn, dataset_id)
        existing = annotations_repo.list_labels(conn, dataset_id)
        known = {label.key for label in existing}
        used = {label.color.lower() for label in existing}
        position = max((label.position for label in existing), default=-1) + 1
        for entry in classes:
            if entry.key in known:
                continue
            colour = next(
                (colour for colour in CLASS_PALETTE if colour not in used),
                CLASS_PALETTE[position % len(CLASS_PALETTE)],
            )
            annotations_repo.create_label(
                conn, dataset_id, key=entry.key, name=entry.name, color=colour, position=position
            )
            known.add(entry.key)
            used.add(colour)
            added.append(entry.key)
            position += 1
    return added


def box_document(width: int, height: int, boxes: Sequence[ImportedBox]) -> AnnotationDocument:
    """One `box` shape per object, `import-<n>` in source order, drawn largest first."""
    shapes: list[AnnotationShape] = [
        BoxShape(
            id=f"import-{index + 1}",
            label_key=item.label_key,
            x=item.box[0],
            y=item.box[1],
            width=item.box[2] - item.box[0],
            height=item.box[3] - item.box[1],
        )
        for index, item in enumerate(boxes)
    ]
    shapes.sort(key=lambda shape: -_area(shape))
    return AnnotationDocument(image_width=width, image_height=height, shapes=shapes)


def _area(shape: AnnotationShape) -> float:
    assert isinstance(shape, BoxShape)
    return shape.width * shape.height


@dataclass(frozen=True)
class WrittenTruth:
    written: bool
    """False when the image already had a revision or an open draft, which it keeps."""
    reshaped: int = 0
    """Boxes whose instance box is not the box drawn: overlapped along a whole edge, or lost."""


def mask_document(width: int, height: int, label_key: str, mask: np.ndarray) -> AnnotationDocument:
    """One `bitmap` shape of `label_key`, cropped to the mask's extent; none for an empty mask."""
    if mask.shape != (height, width):
        raise ValueError(f"a {mask.shape[1]}x{mask.shape[0]} mask for a {width}x{height} image")
    shape = tight_bitmap_shape(mask, shape_id="import-1", label_key=label_key)
    return AnnotationDocument(
        image_width=width, image_height=height, shapes=[] if shape is None else [shape]
    )


def has_truth(settings: Settings, image_id: int) -> bool:
    """Whether the image already has a revision or an open draft, which an import keeps."""
    with connection(settings.db_path) as conn:
        return (
            annotations_repo.latest_revision(conn, image_id) is not None
            or annotations_repo.get_draft(conn, image_id) is not None
        )


def write_document_truth(
    settings: Settings, image_id: int, document: AnnotationDocument
) -> AnnotationRevision | None:
    """Complete `document` as the image's first revision, or return `None` and write nothing
    when the image already has a revision or an open draft, which it keeps."""
    if has_truth(settings, image_id):
        return None
    draft = IMAGE_DRAFTS.create(settings, image_id, document)
    return IMAGE_DRAFTS.complete(settings, image_id, IMAGE_DRAFTS.etag(draft))


def write_box_truth(
    settings: Settings, image_id: int, width: int, height: int, boxes: Sequence[ImportedBox]
) -> WrittenTruth:
    """Complete `boxes` as the image's first revision. An empty list asserts every class absent."""
    document = box_document(width, height, boxes)
    revision = write_document_truth(settings, image_id, document)
    if revision is None:
        return WrittenTruth(written=False)
    return WrittenTruth(written=True, reshaped=_reshaped(document, revision.instances_path))


def write_mask_truth(
    settings: Settings, image_id: int, width: int, height: int, label_key: str, mask: np.ndarray
) -> WrittenTruth:
    """Complete `mask` as the image's region of `label_key`, as its first revision.

    An empty mask asserts every class absent.
    """
    revision = write_document_truth(
        settings, image_id, mask_document(width, height, label_key, mask)
    )
    return WrittenTruth(written=revision is not None)


def _reshaped(document: AnnotationDocument, instances_path: str | None) -> int:
    if instances_path is None:  # pragma: no cover - completion always writes one now
        return 0
    stored = json.loads(Path(instances_path).read_text(encoding="utf-8"))["instances"]
    boxes = {str(entry["instance_id"]): [int(value) for value in entry["box"]] for entry in stored}
    reshaped = 0
    for shape in document.shapes:
        assert isinstance(shape, BoxShape)
        if boxes.get(shape.id) != list(shape.owned_pixels()):
            reshaped += 1
    return reshaped
