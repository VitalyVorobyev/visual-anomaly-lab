"""Annotation drafts and revisions: everything between the routes and the repository.

The routes in `api/routers/annotations.py` read headers, call one function here and set
response headers; every check, transaction and file write is in this module. Refusals are
`anomaly_lab.errors` categories, so nothing here knows it is behind HTTP.

**One draft lifecycle, two editing units (ADR-0036).** A dataset annotates either single
images or whole samples, and both units go through the same steps — read-or-seed, create
only if absent, save only at the version read, discard, complete. `DraftUnit` writes those
steps once; `ImageDrafts` and `SampleDrafts` supply what genuinely differs: how a unit is
found, what its seed is, which frame a document must match, and which rows hold it.
Completion differs in substance (one image and its imported-mask provenance, versus one
rendering fanned out to every image of a sample), so each unit completes on its own, from
the same locked prelude.
"""

from __future__ import annotations

import hashlib
import shutil
import sqlite3
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import ClassVar, Protocol
from uuid import uuid4

import numpy as np
from pydantic import BaseModel, Field

from anomaly_lab.annotation_bitmap import AnnotationBitmapError, decode_shape, encode_png
from anomaly_lab.annotation_interchange import (
    AnnotationInterchangeError,
    CocoDocument,
    LabelMeDocument,
    imported_document,
    shapes_from_coco,
    shapes_from_labelme,
    shapes_from_png,
)
from anomaly_lab.annotation_render import AnnotationRenderError, RenderedTruth, render_truth
from anomaly_lab.config import Settings
from anomaly_lab.db.connection import Transaction, connection, transaction
from anomaly_lab.db.repositories import annotations as annotations_repo
from anomaly_lab.db.repositories import datasets as datasets_repo
from anomaly_lab.db.repositories import images as images_repo
from anomaly_lab.db.repositories import masks as masks_repo
from anomaly_lab.db.repositories import samples as samples_repo
from anomaly_lab.domain.annotations import (
    AnnotationDocument,
    AnnotationDraft,
    AnnotationLabel,
    AnnotationLabelCreate,
    AnnotationLabelUpdate,
    AnnotationRevision,
    AnnotationSampleDraft,
    AnnotationShape,
    BitmapShape,
)
from anomaly_lab.domain.entities import AnnotationScope, ClassPresence, Image
from anomaly_lab.errors import (
    ConflictError,
    GoneError,
    InvalidInputError,
    NotFoundError,
    StaleVersionError,
)
from anomaly_lab.media.decode import UnreadableImageError, sha256_of
from anomaly_lab.models.preprocessing import load_mask
from anomaly_lab.schemas import API_MODEL_CONFIG

DRAFT_CHANGED = "the annotation draft changed elsewhere"
UNSAFE_DIRECTORY = "the app-owned annotation directory is unsafe"

ShapeBuilder = Callable[[tuple[int, int], dict[str, str]], list[AnnotationShape]]
"""Turns an imported file into shapes, given the draft's frame and the known label keys."""


# --- Read models ---------------------------------------------------------------------


class AnnotationDraftState(BaseModel):
    """What the editor opens on: a persisted draft, or the seed one would start from.

    `version` and `updated_at` are null exactly when `persisted` is false, and that null is the
    domain fact rather than a placeholder -- it is how the client knows its first save has to
    create the draft rather than update it. The write routes keep returning `AnnotationDraft`,
    where both are always present.
    """

    model_config = API_MODEL_CONFIG

    image_id: int
    persisted: bool
    document: AnnotationDocument
    version: int | None = None
    updated_at: str | None = None
    base_revision_id: int | None = None
    source_mask_id: int | None = None
    source_mask_path: str | None = None
    source_mask_sha256: str | None = None


class AnnotationSampleDraftState(BaseModel):
    """See `AnnotationDraftState`. Narrower, for the same reason `AnnotationSampleDraft` is."""

    model_config = API_MODEL_CONFIG

    sample_id: int
    persisted: bool
    document: AnnotationDocument
    version: int | None = None
    updated_at: str | None = None


class CopiedChannel(BaseModel):
    """What one target holds afterwards, so the caller can say it rather than guess it."""

    model_config = API_MODEL_CONFIG

    image_id: int
    version: int
    shape_count: int


class CopyRegionsResult(BaseModel):
    model_config = API_MODEL_CONFIG

    copied: int
    targets: list[CopiedChannel]


class OpenDraftUnit(BaseModel):
    """One unit standing between the dataset and a scope change, addressable."""

    model_config = API_MODEL_CONFIG

    sample_id: int
    sample_key: str
    image_id: int
    channel: str | None = None


class AnnotationScopeState(BaseModel):
    """A dataset's annotation scope, and everything that would stop it changing."""

    model_config = API_MODEL_CONFIG

    dataset_id: int
    scope: AnnotationScope
    samples: int
    multi_image_samples: int
    open_drafts: int = Field(
        description="Drafts open in the current scope. Any scope change requires zero."
    )
    open_draft_units: list[OpenDraftUnit] = Field(
        default_factory=list,
        description=(
            "Which units hold that work, so 'complete or discard them first' is reachable "
            "rather than merely true. Capped; `open_drafts` is the whole count."
        ),
    )
    can_use_sample_scope: bool
    blockers: list[str] = Field(
        default_factory=list,
        description="Every reason sample scope is unavailable, reported together.",
    )


# --- Shared checks -------------------------------------------------------------------


def _require_dataset(conn: sqlite3.Connection, dataset_id: int) -> None:
    if datasets_repo.get_dataset(conn, dataset_id) is None:
        raise NotFoundError(f"no dataset with id {dataset_id}")


def _annotation_scope(conn: sqlite3.Connection, dataset_id: int) -> AnnotationScope:
    dataset = datasets_repo.get_dataset(conn, dataset_id)
    if dataset is None:  # pragma: no cover - callers resolved the id from a row
        raise NotFoundError(f"no dataset with id {dataset_id}")
    return dataset.annotation_scope


def _image_dataset_id(conn: sqlite3.Connection, image_id: int) -> int:
    # Kept as a tiny SQL join here because an image's owning dataset is context for
    # taxonomy validation, not a new domain read model.
    row = conn.execute(
        """
        SELECT sample.dataset_id
          FROM image JOIN sample ON sample.id = image.sample_id
         WHERE image.id = ?
        """,
        (image_id,),
    ).fetchone()
    if row is None:
        raise NotFoundError(f"no image with id {image_id}")
    return int(row["dataset_id"])


def _sample_dataset_id(conn: sqlite3.Connection, sample_id: int) -> int:
    sample = samples_repo.get_sample(conn, sample_id)
    if sample is None:
        raise NotFoundError(f"no sample with id {sample_id}")
    return sample.dataset_id


def _validate_taxonomy(
    conn: sqlite3.Connection, dataset_id: int, document: AnnotationDocument
) -> None:
    """The checks that are about the document alone, not about which frame it belongs to.

    Split out because a sample-scoped document is validated against a *shared* frame that
    no single image owns, while these two checks are identical in both scopes.
    """
    known = {label.key for label in annotations_repo.list_labels(conn, dataset_id)}
    unknown = sorted({shape.label_key for shape in document.shapes} - known)
    if unknown:
        raise InvalidInputError(f"unknown annotation labels: {', '.join(unknown)}")
    for shape in document.shapes:
        if isinstance(shape, BitmapShape):
            try:
                decode_shape(shape)
            except AnnotationBitmapError as exc:
                raise InvalidInputError(str(exc)) from exc


def _shared_frame(images: list[Image]) -> tuple[int, int]:
    if not images:
        raise NotFoundError("this sample has no images to annotate")
    frames = {(image.width, image.height) for image in images}
    if len(frames) != 1:
        raise ConflictError("this sample's images do not share one source frame")
    return frames.pop()


def _document_sha256(document: AnnotationDocument) -> str:
    return hashlib.sha256(document.canonical_json().encode()).hexdigest()


def _revision_destination(settings: Settings, image_id: int, revision_no: int) -> Path:
    """Where a revision's mask is materialised, refused if the directory could redirect it."""
    destination = settings.annotation_image_dir(image_id) / f"revision-{revision_no}.png"
    if settings.annotations_dir.is_symlink() or destination.parent.is_symlink():
        raise ConflictError(UNSAFE_DIRECTORY)
    return destination


def _class_mask_destination(mask_destination: Path) -> Path:
    """The class-index mask sits beside the binary one it was rendered with."""
    return mask_destination.with_name(f"{mask_destination.stem}.classes.png")


def _render(
    conn: sqlite3.Connection,
    dataset_id: int,
    document: AnnotationDocument,
    destination: Path,
    *,
    source_mask_path: Path | None,
    source_mask_sha256: str | None,
) -> RenderedTruth:
    classes = [label.key for label in annotations_repo.list_labels(conn, dataset_id)]
    try:
        return render_truth(
            document,
            destination,
            _class_mask_destination(destination),
            classes=classes,
            source_mask_path=source_mask_path,
            source_mask_sha256=source_mask_sha256,
        )
    except AnnotationRenderError as exc:
        raise ConflictError(str(exc)) from exc


def _class_mask(destination: Path, rendered: RenderedTruth) -> annotations_repo.ClassMask:
    return annotations_repo.ClassMask(
        path=str(_class_mask_destination(destination)),
        sha256=rendered.class_mask_sha256,
        table=rendered.class_table,
    )


# --- The draft lifecycle, once ---------------------------------------------------------


class _Draft(Protocol):
    @property
    def version(self) -> int: ...

    @property
    def document(self) -> AnnotationDocument: ...


class _Seed(Protocol):
    @property
    def document(self) -> AnnotationDocument: ...


class DraftUnit[DraftT: _Draft, SeedT: _Seed](ABC):
    """The steps every editable unit goes through, written once.

    Every write takes `BEGIN IMMEDIATE` before it reads the version it checks, so a
    concurrent writer cannot slip between the check and the write.
    """

    scope: ClassVar[AnnotationScope]
    noun: ClassVar[str]
    wrong_scope: ClassVar[str]
    """The refusal a dataset edited in the *other* scope gets, with a pointer to it."""

    # What differs between the units.

    @abstractmethod
    def dataset_id(self, conn: sqlite3.Connection, unit_id: int) -> int: ...

    @abstractmethod
    def etag(self, draft: DraftT) -> str: ...

    @abstractmethod
    def get(self, conn: sqlite3.Connection, unit_id: int) -> DraftT | None: ...

    @abstractmethod
    def seed(self, conn: sqlite3.Connection, unit_id: int) -> SeedT:
        """What a draft for this unit *would* open on, without creating one."""

    @abstractmethod
    def persisted(self, draft: DraftT) -> SeedT: ...

    @abstractmethod
    def validate(
        self,
        conn: sqlite3.Connection,
        unit_id: int,
        dataset_id: int,
        document: AnnotationDocument,
        *,
        expected_base: str,
    ) -> None: ...

    @abstractmethod
    def insert(
        self, conn: sqlite3.Connection, unit_id: int, document: AnnotationDocument, seed: SeedT
    ) -> DraftT: ...

    @abstractmethod
    def update(
        self, conn: sqlite3.Connection, unit_id: int, version: int, document: AnnotationDocument
    ) -> DraftT | None: ...

    @abstractmethod
    def delete(self, conn: sqlite3.Connection, unit_id: int, version: int) -> None: ...

    # What is the same.

    def require_scope(self, conn: sqlite3.Connection, dataset_id: int) -> None:
        """Refuse a write through the scope this dataset does not edit in.

        A 409 with a pointer rather than a quiet redirect: two writers editing the same part
        through two different scopes would each hold a valid ETag for a different document
        and neither would detect the other.
        """
        if _annotation_scope(conn, dataset_id) is not self.scope:
            raise ConflictError(self.wrong_scope)

    def locked_current(
        self, conn: sqlite3.Connection, unit_id: int, expected: str | None
    ) -> tuple[int, DraftT]:
        """The unit's dataset and current draft, refused unless `expected` is its ETag.

        `expected=None` accepts whatever is current: the deliberate force-discard of a
        caller that has been shown the draft moved and chose to drop it anyway.
        """
        dataset_id = self.dataset_id(conn, unit_id)
        self.require_scope(conn, dataset_id)
        current = self.get(conn, unit_id)
        if current is None:
            raise NotFoundError(f"{self.noun} {unit_id} has no annotation draft")
        if expected is not None and expected != self.etag(current):
            raise StaleVersionError(DRAFT_CHANGED)
        return dataset_id, current

    def replace(
        self,
        conn: sqlite3.Connection,
        unit_id: int,
        dataset_id: int,
        current: DraftT,
        document: AnnotationDocument,
    ) -> DraftT:
        """Validate `document` against `current` and store it as the next version."""
        self.validate(conn, unit_id, dataset_id, document, expected_base=current.document.base)
        saved = self.update(conn, unit_id, current.version, document)
        if saved is None:  # pragma: no cover - BEGIN IMMEDIATE serialises writers
            raise StaleVersionError(DRAFT_CHANGED)
        return saved

    def read(self, settings: Settings, unit_id: int) -> tuple[SeedT, str | None]:
        """The persisted draft and its ETag, or the seed and `None`. Never a write."""
        with connection(settings.db_path) as conn:
            dataset_id = self.dataset_id(conn, unit_id)
            self.require_scope(conn, dataset_id)
            draft = self.get(conn, unit_id)
            if draft is None:
                return self.seed(conn, unit_id), None
        return self.persisted(draft), self.etag(draft)

    def create(self, settings: Settings, unit_id: int, document: AnnotationDocument) -> DraftT:
        """Create a draft from the first save, refusing if one already exists."""
        with connection(settings.db_path) as conn, transaction(conn, immediate=True):
            dataset_id = self.dataset_id(conn, unit_id)
            self.require_scope(conn, dataset_id)
            annotations_repo.ensure_default_label(conn, dataset_id)
            if self.get(conn, unit_id) is not None:
                raise StaleVersionError(DRAFT_CHANGED)
            # Resolved before anything is written, for its refusals as well as its content.
            seed = self.seed(conn, unit_id)
            self.validate(conn, unit_id, dataset_id, document, expected_base=seed.document.base)
            return self.insert(conn, unit_id, document, seed)

    def save(
        self, settings: Settings, unit_id: int, document: AnnotationDocument, expected: str
    ) -> DraftT:
        with connection(settings.db_path) as conn, transaction(conn, immediate=True):
            dataset_id, current = self.locked_current(conn, unit_id, expected)
            return self.replace(conn, unit_id, dataset_id, current, document)

    def discard(self, settings: Settings, unit_id: int, expected: str | None) -> None:
        with connection(settings.db_path) as conn, transaction(conn, immediate=True):
            _, current = self.locked_current(conn, unit_id, expected)
            self.delete(conn, unit_id, current.version)


# --- Image scope -----------------------------------------------------------------------


def _pin_source_mask(conn: sqlite3.Connection, seed: AnnotationDraftState) -> str | None:
    """Verify the imported mask on the way into a draft, and record its digest if it is new."""
    if seed.source_mask_id is None or seed.source_mask_path is None:
        return None
    if seed.base_revision_id is not None:
        # Provenance copied from a completed revision is already pinned and immutable.
        return seed.source_mask_sha256
    path = Path(seed.source_mask_path)
    if not path.is_file():
        raise ConflictError("the source mask is unavailable")
    actual = sha256_of(path)
    if seed.source_mask_sha256 is not None and seed.source_mask_sha256 != actual:
        raise ConflictError("the imported source mask changed after it entered the catalog")
    if seed.source_mask_sha256 is None:
        masks_repo.record_sha256(conn, seed.source_mask_id, actual)
    return actual


class ImageDrafts(DraftUnit[AnnotationDraft, AnnotationDraftState]):
    """A draft per image, which may open on an imported source mask."""

    scope = AnnotationScope.IMAGE
    noun = "image"
    wrong_scope = "this dataset annotates whole samples; use /api/samples/{sample_id}/annotations/…"

    def dataset_id(self, conn: sqlite3.Connection, unit_id: int) -> int:
        return _image_dataset_id(conn, unit_id)

    def etag(self, draft: AnnotationDraft) -> str:
        return f'"annotation-draft-{draft.image_id}-v{draft.version}"'

    def get(self, conn: sqlite3.Connection, unit_id: int) -> AnnotationDraft | None:
        return annotations_repo.get_draft(conn, unit_id)

    def seed(self, conn: sqlite3.Connection, unit_id: int) -> AnnotationDraftState:
        """Shared by the read and the create so the two can never disagree about a seed.

        Deliberately does not hash or record anything: a read must not write, and the
        imported mask's digest is verified where it is pinned (the create) and again where
        it is consumed (`render_truth` at completion), which are the two places that
        matter.
        """
        image = images_repo.get_image(conn, unit_id)
        if image is None:  # pragma: no cover - ownership join already found it
            raise NotFoundError(f"no image with id {unit_id}")

        latest = annotations_repo.latest_revision(conn, unit_id)
        if latest is not None:
            return AnnotationDraftState(
                image_id=unit_id,
                persisted=False,
                document=latest.document,
                base_revision_id=latest.id,
                source_mask_id=latest.source_mask_id,
                source_mask_path=latest.source_mask_path,
                source_mask_sha256=latest.source_mask_sha256,
            )

        source = masks_repo.get_mask_for_image(conn, unit_id)
        return AnnotationDraftState(
            image_id=unit_id,
            persisted=False,
            document=AnnotationDocument(
                image_width=image.width,
                image_height=image.height,
                base="source_mask" if source is not None else "empty",
            ),
            source_mask_id=source.id if source else None,
            source_mask_path=source.path if source else None,
            source_mask_sha256=source.sha256 if source else None,
        )

    def persisted(self, draft: AnnotationDraft) -> AnnotationDraftState:
        return AnnotationDraftState(persisted=True, **draft.model_dump(mode="python"))

    def validate(
        self,
        conn: sqlite3.Connection,
        unit_id: int,
        dataset_id: int,
        document: AnnotationDocument,
        *,
        expected_base: str,
    ) -> None:
        image = images_repo.get_image(conn, unit_id)
        if image is None:  # pragma: no cover - ownership join already found it
            raise NotFoundError(f"no image with id {unit_id}")
        if (document.image_width, document.image_height) != (image.width, image.height):
            raise InvalidInputError("annotation dimensions must match the source image exactly")
        if document.base != expected_base:
            raise InvalidInputError("a draft's base layer cannot be changed")
        _validate_taxonomy(conn, dataset_id, document)

    def insert(
        self,
        conn: sqlite3.Connection,
        unit_id: int,
        document: AnnotationDocument,
        seed: AnnotationDraftState,
    ) -> AnnotationDraft:
        return annotations_repo.create_draft(
            conn,
            unit_id,
            document,
            base_revision_id=seed.base_revision_id,
            source_mask_id=seed.source_mask_id,
            source_mask_path=seed.source_mask_path,
            source_mask_sha256=_pin_source_mask(conn, seed),
        )

    def update(
        self, conn: sqlite3.Connection, unit_id: int, version: int, document: AnnotationDocument
    ) -> AnnotationDraft | None:
        return annotations_repo.update_draft(conn, unit_id, version, document)

    def delete(self, conn: sqlite3.Connection, unit_id: int, version: int) -> None:
        annotations_repo.delete_draft(conn, unit_id, version)

    def complete(self, settings: Settings, image_id: int, expected: str) -> AnnotationRevision:
        """Materialise and freeze the current draft as the image's next revision."""
        with (
            connection(settings.db_path) as conn,
            transaction(conn, immediate=True) as tx,
        ):
            dataset_id, draft = self.locked_current(conn, image_id, expected)
            self.validate(
                conn, image_id, dataset_id, draft.document, expected_base=draft.document.base
            )
            latest = annotations_repo.latest_revision(conn, image_id)
            next_no = (latest.revision_no if latest else 0) + 1
            destination = _revision_destination(settings, image_id, next_no)
            # Registered before the render, so a render that fails part-way leaves nothing.
            tx.remove_on_rollback(destination)
            tx.remove_on_rollback(_class_mask_destination(destination))
            rendered = _render(
                conn,
                dataset_id,
                draft.document,
                destination,
                source_mask_path=Path(draft.source_mask_path) if draft.source_mask_path else None,
                source_mask_sha256=draft.source_mask_sha256,
            )
            return annotations_repo.insert_revision(
                conn,
                draft,
                document_sha256=_document_sha256(draft.document),
                mask_path=str(destination),
                mask_sha256=rendered.mask_sha256,
                class_mask=_class_mask(destination, rendered),
            )

    def import_file(
        self, settings: Settings, image_id: int, expected: str, build_shapes: ShapeBuilder
    ) -> AnnotationDraft:
        """Replace a draft's editable layers with shapes read from an interchange file."""
        with connection(settings.db_path) as conn, transaction(conn, immediate=True):
            dataset_id, current = self.locked_current(conn, image_id, expected)
            labels = annotations_repo.list_labels(conn, dataset_id)
            known = {label.key: label.name for label in labels}
            default_key = labels[0].key if labels else "defect"
            try:
                shapes = build_shapes(
                    (current.document.image_width, current.document.image_height), known
                )
                document = imported_document(current.document, shapes, clear_label_key=default_key)
            except AnnotationInterchangeError as exc:
                raise InvalidInputError(str(exc)) from exc
            return self.replace(conn, image_id, dataset_id, current, document)

    def copy_regions(
        self, settings: Settings, image_id: int, target_image_ids: Sequence[int], expected: str
    ) -> CopyRegionsResult:
        """Append this draft's regions to other channels of the same sample.

        Only the source is checked against a version: appending cannot lose what a target
        already holds. Equal dimensions are a refusal rather than a rescale, because an
        annotation never leaves its source frame (ADR-0032).
        """
        with connection(settings.db_path) as conn, transaction(conn, immediate=True):
            dataset_id, source = self.locked_current(conn, image_id, expected)
            if not source.document.shapes:
                raise InvalidInputError("this draft has no regions to copy")

            image = images_repo.get_image(conn, image_id)
            if image is None:  # pragma: no cover - the ownership join already found it
                raise NotFoundError(f"no image with id {image_id}")
            siblings = {
                sibling.id: sibling
                for sibling in images_repo.list_images_for_sample(conn, image.sample_id)
            }

            # De-duplicated up front: asking for the same channel twice is a client slip, and
            # honouring it literally would append the regions twice.
            wanted = list(dict.fromkeys(target_image_ids))
            copied: list[CopiedChannel] = []
            for target_id in wanted:
                if target_id == image_id:
                    raise InvalidInputError("a channel cannot be copied onto itself")
                sibling = siblings.get(target_id)
                if sibling is None:
                    raise ConflictError(f"image {target_id} is not another channel of this sample")
                if (sibling.width, sibling.height) != (image.width, image.height):
                    raise ConflictError(
                        f"image {target_id} is {sibling.width}x{sibling.height} and this one "
                        f"is {image.width}x{image.height}; an annotation never leaves its "
                        "source frame"
                    )

                current = self.get(conn, target_id)
                if current is None:
                    seed = self.seed(conn, target_id)
                    current = self.insert(conn, target_id, seed.document, seed)
                document = current.document.model_copy(
                    update={
                        "shapes": [
                            *current.document.shapes,
                            *_with_fresh_ids(source.document.shapes),
                        ]
                    }
                )
                saved = self.replace(conn, target_id, dataset_id, current, document)
                copied.append(
                    CopiedChannel(
                        image_id=target_id,
                        version=saved.version,
                        shape_count=len(saved.document.shapes),
                    )
                )
        return CopyRegionsResult(copied=len(source.document.shapes), targets=copied)


def _with_fresh_ids(shapes: Sequence[AnnotationShape]) -> list[AnnotationShape]:
    """The same geometry under new ids.

    A shape id is unique within *one* document, and each target already has its own. Copying
    ids verbatim would collide the second time a channel receives a copy -- and the document
    validator rejects duplicates, so the failure would be a 422 in the middle of a fan-out
    rather than anything a person could act on.
    """
    return [shape.model_copy(update={"id": uuid4().hex}) for shape in shapes]


# --- Sample scope (ADR-0036) ---------------------------------------------------------------
#
# One document, edited once, materialised onto every image of the sample. Truth below this
# boundary stays image-keyed: `resolve_ground_truth_masks`, pixel metrics, `has_mask`, the
# `MetricSet` digest and all three interchange formats never learn that scope exists.


class SampleDrafts(DraftUnit[AnnotationSampleDraft, AnnotationSampleDraftState]):
    """One shared draft per sample, in the frame all of its images share."""

    scope = AnnotationScope.SAMPLE
    noun = "sample"
    wrong_scope = (
        "this dataset annotates individual images; use /api/images/{image_id}/annotations/…"
    )

    def dataset_id(self, conn: sqlite3.Connection, unit_id: int) -> int:
        return _sample_dataset_id(conn, unit_id)

    def etag(self, draft: AnnotationSampleDraft) -> str:
        # A namespace of its own, not `annotation-draft-…`: an image id and a sample id are
        # both small integers and would collide into a token that validates against the wrong
        # document.
        return f'"annotation-sample-draft-{draft.sample_id}-v{draft.version}"'

    def get(self, conn: sqlite3.Connection, unit_id: int) -> AnnotationSampleDraft | None:
        return annotations_repo.get_sample_draft(conn, unit_id)

    def seed(self, conn: sqlite3.Connection, unit_id: int) -> AnnotationSampleDraftState:
        width, height = _shared_frame(images_repo.list_images_for_sample(conn, unit_id))
        document = AnnotationDocument(image_width=width, image_height=height)
        latest = annotations_repo.latest_revision_for_sample(conn, unit_id)
        if latest is not None:
            if latest.document.base != "empty":
                raise ConflictError(
                    "this sample's newest revision was opened on an imported source "
                    "mask, which belongs to one image and cannot be shared"
                )
            if (latest.document.image_width, latest.document.image_height) != (width, height):
                raise ConflictError("this sample's newest revision was drawn in a different frame")
            document = latest.document
        return AnnotationSampleDraftState(sample_id=unit_id, persisted=False, document=document)

    def persisted(self, draft: AnnotationSampleDraft) -> AnnotationSampleDraftState:
        return AnnotationSampleDraftState(persisted=True, **draft.model_dump(mode="python"))

    def validate(
        self,
        conn: sqlite3.Connection,
        unit_id: int,
        dataset_id: int,
        document: AnnotationDocument,
        *,
        expected_base: str,
    ) -> None:
        # `expected_base` is always "empty" here: a shared document has no base to inherit,
        # and the refusal says so rather than calling it a change.
        width, height = _shared_frame(images_repo.list_images_for_sample(conn, unit_id))
        if (document.image_width, document.image_height) != (width, height):
            raise InvalidInputError(
                "annotation dimensions must match the sample's shared frame exactly"
            )
        if document.base != "empty":
            raise InvalidInputError("a sample-scoped document has no base layer to inherit")
        _validate_taxonomy(conn, dataset_id, document)

    def insert(
        self,
        conn: sqlite3.Connection,
        unit_id: int,
        document: AnnotationDocument,
        seed: AnnotationSampleDraftState,
    ) -> AnnotationSampleDraft:
        return annotations_repo.create_sample_draft(conn, unit_id, document)

    def update(
        self, conn: sqlite3.Connection, unit_id: int, version: int, document: AnnotationDocument
    ) -> AnnotationSampleDraft | None:
        return annotations_repo.update_sample_draft(conn, unit_id, version, document)

    def delete(self, conn: sqlite3.Connection, unit_id: int, version: int) -> None:
        annotations_repo.delete_sample_draft(conn, unit_id, version)

    def complete(
        self, settings: Settings, sample_id: int, expected: str
    ) -> list[AnnotationRevision]:
        """Render once, then write the same bytes to every image's own revision path."""
        with (
            connection(settings.db_path) as conn,
            transaction(conn, immediate=True) as tx,
        ):
            dataset_id, draft = self.locked_current(conn, sample_id, expected)
            self.validate(conn, sample_id, dataset_id, draft.document, expected_base="empty")
            if settings.annotations_dir.is_symlink():
                raise ConflictError(UNSAFE_DIRECTORY)

            images = images_repo.list_images_for_sample(conn, sample_id)
            document_sha256 = _document_sha256(draft.document)
            revisions = [
                annotations_repo.insert_shared_revision(
                    conn,
                    image.id,
                    draft.document,
                    document_sha256=document_sha256,
                    mask_path=str(path),
                    mask_sha256=rendered.mask_sha256,
                    class_mask=_class_mask(path, rendered),
                )
                for image, path, rendered in _fan_out(
                    tx, settings, dataset_id, draft.document, images
                )
            ]
            annotations_repo.delete_sample_draft(conn, sample_id)
        return revisions


def _fan_out(
    tx: Transaction,
    settings: Settings,
    dataset_id: int,
    document: AnnotationDocument,
    images: list[Image],
) -> list[tuple[Image, Path, RenderedTruth]]:
    """One rendering, copied to each image's next revision path.

    The digest is therefore identical across the fan-out, which is what makes "these
    channels share one truth" checkable after the fact rather than merely intended. Each
    image keeps its own `revision_no`, so one that carries earlier image-scoped history
    continues counting from where it stopped. Every file is registered for removal if the
    transaction does not commit.
    """
    written: list[tuple[Image, Path, RenderedTruth]] = []
    first: tuple[Path, RenderedTruth] | None = None
    for image in images:
        next_no = annotations_repo.next_revision_no(tx.conn, image.id)
        destination = _revision_destination(settings, image.id, next_no)
        tx.remove_on_rollback(destination)
        tx.remove_on_rollback(_class_mask_destination(destination))
        if first is None:
            first = (
                destination,
                _render(
                    tx.conn,
                    dataset_id,
                    document,
                    destination,
                    source_mask_path=None,
                    source_mask_sha256=None,
                ),
            )
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(first[0], destination)
            shutil.copyfile(_class_mask_destination(first[0]), _class_mask_destination(destination))
        written.append((image, destination, first[1]))
    return written


IMAGE_DRAFTS = ImageDrafts()
SAMPLE_DRAFTS = SampleDrafts()


# --- Label taxonomy --------------------------------------------------------------------


class ClassCoverage(BaseModel):
    """How much truth one class has, in samples — what a targeted task can draw on (ADR-0040).

    `present` samples can be references or positive queries, `absent` ones are confirmed
    negatives, and `unlabeled` ones are excluded from every metric.
    """

    model_config = API_MODEL_CONFIG

    label_key: str
    present: int
    absent: int
    unlabeled: int


def class_coverage(settings: Settings, dataset_id: int) -> list[ClassCoverage]:
    with connection(settings.db_path) as conn:
        _require_dataset(conn, dataset_id)
        keys = [label.key for label in annotations_repo.list_labels(conn, dataset_id)]
        presence = annotations_repo.presence_by_class(conn, dataset_id, keys)
    return [
        ClassCoverage(
            label_key=key,
            present=sum(found is ClassPresence.PRESENT for found in presence[key].values()),
            absent=sum(found is ClassPresence.ABSENT for found in presence[key].values()),
            unlabeled=sum(found is ClassPresence.UNLABELED for found in presence[key].values()),
        )
        for key in keys
    ]


def list_labels(settings: Settings, dataset_id: int) -> list[AnnotationLabel]:
    with connection(settings.db_path) as conn:
        _require_dataset(conn, dataset_id)
        return annotations_repo.list_labels(conn, dataset_id)


def create_label(
    settings: Settings, dataset_id: int, body: AnnotationLabelCreate
) -> AnnotationLabel:
    with connection(settings.db_path) as conn:
        _require_dataset(conn, dataset_id)
        annotations_repo.ensure_default_label(conn, dataset_id)
        try:
            return annotations_repo.create_label(conn, dataset_id, **body.model_dump(mode="python"))
        except sqlite3.IntegrityError as exc:
            raise ConflictError(f"annotation label {body.key!r} exists") from exc


def update_label(
    settings: Settings, dataset_id: int, key: str, body: AnnotationLabelUpdate
) -> AnnotationLabel:
    with connection(settings.db_path) as conn:
        _require_dataset(conn, dataset_id)
        updated = annotations_repo.update_label(
            conn, dataset_id, key, **body.model_dump(mode="python")
        )
    if updated is None:
        raise NotFoundError(f"no annotation label {key!r}")
    return updated


# --- Revisions and masks ---------------------------------------------------------------


def list_revisions(settings: Settings, image_id: int) -> list[AnnotationRevision]:
    with connection(settings.db_path) as conn:
        _image_dataset_id(conn, image_id)
        return annotations_repo.list_revisions(conn, image_id)


def revision_mask(settings: Settings, image_id: int, revision_id: int) -> AnnotationRevision:
    """A completed revision whose materialised mask is where it should be, byte for byte."""
    with connection(settings.db_path) as conn:
        revision = annotations_repo.get_revision(conn, revision_id)
    if revision is None or revision.image_id != image_id:
        raise NotFoundError("no such annotation revision")
    path = Path(revision.mask_path)
    expected = settings.annotation_image_dir(image_id)
    if (
        settings.annotations_dir.is_symlink()
        or expected.is_symlink()
        or path.parent != expected
        or path.name != f"revision-{revision.revision_no}.png"
    ):
        raise ConflictError("the stored annotation path is unsafe")
    if not path.is_file() or sha256_of(path) != revision.mask_sha256:
        raise ConflictError("the materialised annotation mask is unavailable")
    return revision


def source_mask_png(settings: Settings, image_id: int) -> tuple[bytes, str]:
    """The imported mask a `source_mask` document pins, as a source-sized 0/255 PNG.

    Returned with the digest the bytes were verified against.
    """
    with connection(settings.db_path) as conn:
        _image_dataset_id(conn, image_id)
        source = masks_repo.get_mask_for_image(conn, image_id)
        image = images_repo.get_image(conn, image_id)
    if source is None or image is None:
        raise NotFoundError(f"image {image_id} has no imported mask")
    path = Path(source.path)
    if not path.is_file():
        raise ConflictError("the source mask is unavailable")
    actual = sha256_of(path)
    if source.sha256 is not None and source.sha256 != actual:
        raise ConflictError("the imported source mask changed after it entered the catalog")
    try:
        mask = load_mask(path, size=(image.width, image.height))
    except UnreadableImageError as exc:
        raise GoneError(str(exc)) from exc
    return encode_png(mask), actual


# --- Interchange -------------------------------------------------------------------------


def from_png(payload: bytes, label_key: str) -> ShapeBuilder:
    def build(size: tuple[int, int], known: dict[str, str]) -> list[AnnotationShape]:
        if label_key not in known:
            raise AnnotationInterchangeError(
                f"annotation label {label_key!r} is not in this dataset's taxonomy"
            )
        return shapes_from_png(payload, size=size, label_key=label_key)

    return build


def from_labelme(payload: LabelMeDocument) -> ShapeBuilder:
    return lambda size, known: shapes_from_labelme(payload, size=size, known_labels=known)


def from_coco(payload: CocoDocument) -> ShapeBuilder:
    return lambda size, known: shapes_from_coco(payload, size=size, known_labels=known)


def export_context(settings: Settings, image_id: int) -> tuple[np.ndarray, str, str]:
    """The current completed truth, the image's file name and the label to export it under."""
    with connection(settings.db_path) as conn:
        dataset_id = _image_dataset_id(conn, image_id)
        image = images_repo.get_image(conn, image_id)
        if image is None:  # pragma: no cover - ownership join already found it
            raise NotFoundError(f"no image with id {image_id}")
        labels = annotations_repo.list_labels(conn, dataset_id)
        label = next(
            (item for item in labels if item.key == "defect"), labels[0] if labels else None
        )
        mask = _resolved_mask(conn, image_id, size=(image.width, image.height))
    return mask, Path(image.path).name, label.key if label is not None else "defect"


def _resolved_mask(
    conn: sqlite3.Connection,
    image_id: int,
    *,
    size: tuple[int, int],
) -> np.ndarray:
    # A drifted truth file raises `GroundTruthDriftError`, which is a `ConflictError`.
    truth = annotations_repo.resolve_ground_truth_masks(conn, [image_id], verify_bytes=True).get(
        image_id
    )
    if truth is None:
        raise NotFoundError(f"image {image_id} has no completed annotation")
    try:
        mask = load_mask(Path(truth.path))
    except UnreadableImageError as exc:
        raise ConflictError(str(exc)) from exc
    if mask.shape != (size[1], size[0]):
        raise ConflictError(
            "the resolved annotation mask does not match the source image dimensions"
        )
    return mask


# --- Scope ---------------------------------------------------------------------------------


def _scope_state(conn: sqlite3.Connection, dataset_id: int) -> AnnotationScopeState:
    """Answer the whole question at once rather than failing on the first obstacle.

    An operator who fixes one blocker and is then told about a second has learned the
    system does not know its own mind. Every reason is collected before any is reported.
    """
    scope = _annotation_scope(conn, dataset_id)
    blockers: list[str] = []

    masks = masks_repo.count_masks_for_dataset(conn, dataset_id)
    if masks:
        blockers.append(
            f"{masks} imported source masks are pinned to individual images; a document "
            "shared across channels cannot carry one image's provenance"
        )

    mixed = annotations_repo.samples_with_mixed_dimensions(conn, dataset_id)
    if mixed:
        shown = ", ".join(mixed[:5])
        rest = f" and {len(mixed) - 5} more" if len(mixed) > 5 else ""
        blockers.append(
            f"{len(mixed)} samples mix image dimensions ({shown}{rest}); a shared document "
            "pins one source frame"
        )

    per_sample = scope is AnnotationScope.SAMPLE
    open_drafts = (
        annotations_repo.count_open_sample_drafts(conn, dataset_id)
        if per_sample
        else annotations_repo.count_open_image_drafts(conn, dataset_id)
    )
    units = (
        annotations_repo.list_open_sample_drafts(conn, dataset_id)
        if per_sample
        else annotations_repo.list_open_image_drafts(conn, dataset_id)
    )
    if open_drafts and not per_sample:
        # Not "unsaved": a draft row exists precisely because somebody saved one. What it is
        # missing is a completion, and saying otherwise sent an operator looking for an
        # unsaved editor they had already saved.
        blockers.append(
            f"{open_drafts} images hold annotation work that has not been completed; finish "
            "or discard it first, because sample scope would leave it unreachable"
        )

    return AnnotationScopeState(
        dataset_id=dataset_id,
        scope=scope,
        samples=samples_repo.count_samples(conn, dataset_id, samples_repo.SampleFilter()),
        multi_image_samples=annotations_repo.count_multi_image_samples(conn, dataset_id),
        open_drafts=open_drafts,
        open_draft_units=[OpenDraftUnit(**vars(unit)) for unit in units],
        can_use_sample_scope=not blockers,
        blockers=blockers,
    )


def read_scope(settings: Settings, dataset_id: int) -> AnnotationScopeState:
    with connection(settings.db_path) as conn:
        _require_dataset(conn, dataset_id)
        return _scope_state(conn, dataset_id)


def set_scope(settings: Settings, dataset_id: int, scope: AnnotationScope) -> AnnotationScopeState:
    """Move the editing surface. Completed revisions stay valid truth in either scope."""
    with connection(settings.db_path) as conn, transaction(conn, immediate=True):
        _require_dataset(conn, dataset_id)
        state = _scope_state(conn, dataset_id)
        if scope is not state.scope:
            if state.open_drafts:
                raise ConflictError(
                    f"{state.open_drafts} annotation drafts are open; complete or "
                    "discard them before changing scope"
                )
            if scope is AnnotationScope.SAMPLE and state.blockers:
                raise ConflictError("; ".join(state.blockers))
            datasets_repo.set_annotation_scope(conn, dataset_id, scope)
        return _scope_state(conn, dataset_id)
