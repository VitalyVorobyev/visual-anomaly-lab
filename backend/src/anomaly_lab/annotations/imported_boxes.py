"""Box truth a dataset ships, entered as each image's first completed revision (ADR-0039).

An imported binary mask answers for the default class alone, so a dataset annotated as boxes
of several classes cannot enter as masks without losing either its classes or its boxes. It
enters as what the editor would have produced had someone drawn it: one `box` shape per object,
of the object's class, in a document on an empty base, completed through the ordinary draft
lifecycle. Completion pins the class table and writes the instances file, so the revision
answers every class of the dataset — present where it has a box, absent where it has none —
and `resolve_box_truth` reads it like any drawn revision. It stays editable: a person who fixes
a box completes the next revision, and the imported one is kept.

**Nothing is overwritten.** An image that already has a revision or an open draft keeps it and
is counted as skipped, so a second registration fills only what the first did not finish.

**A box keeps the pixels it names unless a smaller one overlaps it.** Completion gives a pixel
to the later shape, so boxes are drawn largest first, the rule `color_detector.paint_boxes`
trains on: where two overlap, the smaller keeps its pixels. An instance's box is the tight box
of what it finally owns, so an overlap that covers a whole edge of the larger box shrinks it;
every such box is counted (`reshaped`) rather than silently changed.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from anomaly_lab.annotations.service import IMAGE_DRAFTS
from anomaly_lab.config import Settings
from anomaly_lab.db.connection import connection, transaction
from anomaly_lab.db.repositories import annotations as annotations_repo
from anomaly_lab.domain.annotations import AnnotationDocument, AnnotationShape, BoxShape

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


def write_box_truth(
    settings: Settings, image_id: int, width: int, height: int, boxes: Sequence[ImportedBox]
) -> WrittenTruth:
    """Complete `boxes` as the image's first revision. An empty list asserts every class absent."""
    with connection(settings.db_path) as conn:
        if (
            annotations_repo.latest_revision(conn, image_id) is not None
            or annotations_repo.get_draft(conn, image_id) is not None
        ):
            return WrittenTruth(written=False)
    document = box_document(width, height, boxes)
    draft = IMAGE_DRAFTS.create(settings, image_id, document)
    revision = IMAGE_DRAFTS.complete(settings, image_id, IMAGE_DRAFTS.etag(draft))
    return WrittenTruth(written=True, reshaped=_reshaped(document, revision.instances_path))


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
