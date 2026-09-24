"""Per-image class truth, and its pixels in the source frame (ADR-0039, ADR-0040).

Two readings share one presence rule. A targeted task reads one class's region
(`resolve_class_truth`, `load_class_mask`); a supervised segmentation task reads every
pinned class at once as a label map (`resolve_label_truth`, `load_label_map`).

Presence is decided by `annotations_repo.image_presence`, the rule every class read shares.
This module adds the pixels: where a targeted task's references and queries read the
region of their class from. Each answer is loaded from what pinned it, and verified
against its digest before it is used:

- a revision with a class table reads its class-index mask at the class's index;
- an older revision re-rasterises its document in memory, which is what its class mask
  would have held;
- an imported ground-truth mask is the default class;
- an image of a sample labelled normal is an empty mask of the default class.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image

from anomaly_lab.annotation_render import AnnotationRenderError, rasterize
from anomaly_lab.db.repositories import annotations as annotations_repo
from anomaly_lab.db.repositories.annotations import DEFAULT_LABEL_KEY
from anomaly_lab.domain.annotations import AnnotationDocument
from anomaly_lab.domain.entities import ClassPresence
from anomaly_lab.media.decode import sha256_of
from anomaly_lab.models.base import IGNORE_INDEX
from anomaly_lab.models.preprocessing import load_mask


class ClassTruthError(ValueError):
    """Pinned truth is missing or changed since it was pinned."""


@dataclass(frozen=True)
class ClassTruth:
    """Where one image's region of one class comes from. Only answered images have one."""

    image_id: int
    label_key: str
    presence: Literal[ClassPresence.PRESENT, ClassPresence.ABSENT]
    kind: Literal["class_mask", "document", "source", "normal"]
    size: tuple[int, int]
    """`(width, height)` of the source image."""
    path: str | None = None
    sha256: str | None = None
    index: int | None = None
    document: str | None = None
    classes: tuple[str, ...] = ()
    source_mask_path: str | None = None
    source_mask_sha256: str | None = None

    @property
    def identity(self) -> str:
        """What pins this answer, for a ground-truth digest."""
        pinned = self.sha256 or self.source_mask_sha256 or "-"
        return f"{self.kind}:{self.presence.value}:{self.index or 0}:{pinned}"


def resolve_class_truth(
    conn: sqlite3.Connection, dataset_id: int, image_ids: Sequence[int], label_key: str
) -> dict[int, ClassTruth]:
    """The images whose truth answers for the class, and how to read each one's region.

    An unlabelled image is absent from the result, never an empty mask: a gap is not a
    negative.
    """
    created = annotations_repo.label_created_at(conn, dataset_id)
    classes = tuple(label.key for label in annotations_repo.list_labels(conn, dataset_id))
    found: dict[int, ClassTruth] = {}
    for row in annotations_repo.truth_rows_for_images(conn, image_ids):
        presence = annotations_repo.image_presence(row, label_key, created)
        if presence is ClassPresence.UNLABELED:
            continue
        image_id = int(row["image_id"])
        size = (int(row["width"]), int(row["height"]))
        if row["class_table"] is not None:
            index = next(
                int(entry["index"])
                for entry in json.loads(str(row["class_table"]))
                if entry["key"] == label_key
            )
            found[image_id] = ClassTruth(
                image_id,
                label_key,
                presence,
                "class_mask",
                size,
                path=str(row["class_mask_path"]),
                sha256=str(row["class_mask_sha256"]),
                index=index,
            )
        elif row["document"] is not None:
            found[image_id] = ClassTruth(
                image_id,
                label_key,
                presence,
                "document",
                size,
                document=str(row["document"]),
                classes=classes,
                source_mask_path=row["revision_source_path"],
                source_mask_sha256=row["revision_source_sha256"],
            )
        elif row["source_id"] is not None:
            found[image_id] = ClassTruth(
                image_id,
                label_key,
                presence,
                "source",
                size,
                path=str(row["source_path"]),
                sha256=row["source_sha256"],
            )
        else:
            found[image_id] = ClassTruth(image_id, label_key, presence, "normal", size)
    return found


def load_class_mask(truth: ClassTruth) -> np.ndarray:
    """The class's region as a boolean array in the source frame, `(height, width)`."""
    width, height = truth.size
    if truth.kind == "normal":
        return np.zeros((height, width), dtype=bool)
    if truth.kind == "document":
        document = AnnotationDocument.model_validate_json(truth.document or "{}")
        try:
            canvas = rasterize(
                document,
                truth.classes,
                source_mask_path=Path(truth.source_mask_path) if truth.source_mask_path else None,
                source_mask_sha256=truth.source_mask_sha256,
            )
        except AnnotationRenderError as exc:
            raise ClassTruthError(f"image {truth.image_id}: {exc}") from exc
        return np.asarray(canvas == truth.classes.index(truth.label_key) + 1)

    path = Path(truth.path or "")
    if not path.is_file():
        raise ClassTruthError(f"the {truth.kind} truth of image {truth.image_id} is unavailable")
    if truth.sha256 is not None and sha256_of(path) != truth.sha256:
        raise ClassTruthError(
            f"the {truth.kind} truth of image {truth.image_id} changed after it was pinned"
        )
    if truth.kind == "source":
        return load_mask(path, size=truth.size)
    return np.asarray(_read_index(path) == truth.index)


def _read_index(path: Path) -> np.ndarray:
    with Image.open(path) as opened:
        return np.asarray(opened.convert("L"))


# ------------------------------------------------------------------ every class at once

MAX_SEGMENTATION_CLASSES = IGNORE_INDEX - 1
"""Label maps are 8-bit, 0 is background and 255 is `IGNORE_INDEX`."""


@dataclass(frozen=True)
class LabelTruth:
    """Where one image's label map comes from. Only images answered for every pinned class
    have one."""

    image_id: int
    kind: Literal["class_mask", "document", "source", "normal"]
    size: tuple[int, int]
    """`(width, height)` of the source image."""
    stored: tuple[tuple[str, int], ...] = ()
    """For a class mask, which index each class key was stored under."""
    path: str | None = None
    sha256: str | None = None
    revision_id: int | None = None
    document: str | None = None
    classes: tuple[str, ...] = ()
    """For a document, the dataset's classes it is rasterised against."""
    source_mask_path: str | None = None
    source_mask_sha256: str | None = None

    @property
    def identity(self) -> str:
        """What pins this answer, for a ground-truth digest."""
        pinned = self.sha256 or self.source_mask_sha256 or "-"
        return f"{self.kind}:{self.revision_id or 0}:{pinned}"


def resolve_label_truth(
    conn: sqlite3.Connection, dataset_id: int, image_ids: Sequence[int], classes: Sequence[str]
) -> dict[int, LabelTruth]:
    """The images whose truth answers for every one of `classes`, and how to read each.

    One presence rule decides, the one a targeted task uses per class
    (`annotations_repo.image_presence`): a completed revision answers for every class that
    existed when it was completed; an imported mask, or a sample labelled normal, answers
    for the default class alone. An image that leaves any pinned class unanswered is
    unlabelled and absent from the result — a pixel of an unanswered class would otherwise
    read as background, which is a negative nobody asserted.
    """
    created = annotations_repo.label_created_at(conn, dataset_id)
    dataset_classes = tuple(label.key for label in annotations_repo.list_labels(conn, dataset_id))
    found: dict[int, LabelTruth] = {}
    for row in annotations_repo.truth_rows_for_images(conn, image_ids):
        if any(
            annotations_repo.image_presence(row, key, created) is ClassPresence.UNLABELED
            for key in classes
        ):
            continue
        image_id = int(row["image_id"])
        size = (int(row["width"]), int(row["height"]))
        revision_id = None if row["revision_id"] is None else int(row["revision_id"])
        if row["class_table"] is not None:
            found[image_id] = LabelTruth(
                image_id,
                "class_mask",
                size,
                stored=tuple(
                    (str(entry["key"]), int(entry["index"]))
                    for entry in json.loads(str(row["class_table"]))
                ),
                path=str(row["class_mask_path"]),
                sha256=str(row["class_mask_sha256"]),
                revision_id=revision_id,
            )
        elif row["document"] is not None:
            found[image_id] = LabelTruth(
                image_id,
                "document",
                size,
                revision_id=revision_id,
                document=str(row["document"]),
                classes=dataset_classes,
                source_mask_path=row["revision_source_path"],
                source_mask_sha256=row["revision_source_sha256"],
            )
        elif row["source_id"] is not None:
            found[image_id] = LabelTruth(
                image_id, "source", size, path=str(row["source_path"]), sha256=row["source_sha256"]
            )
        else:
            found[image_id] = LabelTruth(image_id, "normal", size)
    return found


def load_label_map(truth: LabelTruth, classes: Sequence[str]) -> np.ndarray:
    """The image's truth as a `uint8` label map in the source frame, `(height, width)`.

    `classes[i]` is `i + 1`, background is 0, and a pixel drawn in a class outside `classes`
    is `IGNORE_INDEX`. Each kind is verified against what pinned it, as `load_class_mask` is.
    """
    width, height = truth.size
    position = {key: index + 1 for index, key in enumerate(classes)}
    if truth.kind == "normal":
        return np.zeros((height, width), dtype=np.uint8)
    if truth.kind == "document":
        document = AnnotationDocument.model_validate_json(truth.document or "{}")
        try:
            canvas = rasterize(
                document,
                truth.classes,
                source_mask_path=Path(truth.source_mask_path) if truth.source_mask_path else None,
                source_mask_sha256=truth.source_mask_sha256,
            )
        except AnnotationRenderError as exc:
            raise ClassTruthError(f"image {truth.image_id}: {exc}") from exc
        stored = {key: index + 1 for index, key in enumerate(truth.classes)}
        return _relabel(np.asarray(canvas), stored, position)

    path = Path(truth.path or "")
    if not path.is_file():
        raise ClassTruthError(f"the {truth.kind} truth of image {truth.image_id} is unavailable")
    if truth.sha256 is not None and sha256_of(path) != truth.sha256:
        raise ClassTruthError(
            f"the {truth.kind} truth of image {truth.image_id} changed after it was pinned"
        )
    if truth.kind == "source":
        region = load_mask(path, size=truth.size)
        return np.where(region, position.get(DEFAULT_LABEL_KEY, IGNORE_INDEX), 0).astype(np.uint8)
    return _relabel(_read_index(path), dict(truth.stored), position)


def _relabel(
    stored: np.ndarray, stored_index: dict[str, int], position: dict[str, int]
) -> np.ndarray:
    """Renumber a stored class-index array into the run's pinned numbering."""
    table = np.full(256, IGNORE_INDEX, dtype=np.uint8)
    table[0] = 0
    for key, index in stored_index.items():
        table[index] = position.get(key, IGNORE_INDEX)
    return table[np.asarray(stored, dtype=np.uint8)]
