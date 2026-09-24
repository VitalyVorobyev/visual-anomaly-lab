"""Source-frame annotation documents and their persisted lifecycle records."""

from __future__ import annotations

import json
from typing import Annotated, Literal, assert_never

from pydantic import BaseModel, Field, field_validator, model_validator

from anomaly_lab.schemas import API_MODEL_CONFIG


class AnnotationPoint(BaseModel):
    model_config = API_MODEL_CONFIG

    x: float
    y: float


ShapeId = Annotated[str, Field(min_length=1, max_length=128)]
InstanceId = Annotated[
    str | None,
    Field(
        min_length=1,
        max_length=128,
        # Unset is left out rather than written as null, so every document stored before
        # shapes had one keeps its canonical JSON, and therefore its digest.
        exclude_if=lambda value: value is None,
        description=(
            "Groups add shapes into one object instance at completion. Unset, a shape is its "
            "own instance, keyed by its id."
        ),
    ),
]


class PolygonShape(BaseModel):
    model_config = API_MODEL_CONFIG

    id: ShapeId
    label_key: str = Field(min_length=1, max_length=64, pattern=r"^[a-z][a-z0-9_-]*$")
    kind: Literal["polygon"] = "polygon"
    operation: Literal["add", "subtract"] = "add"
    instance_id: InstanceId = None
    points: list[AnnotationPoint] = Field(min_length=3)


class BoxShape(BaseModel):
    """An axis-aligned rectangle in the source frame; it rasterises as its four corners."""

    model_config = API_MODEL_CONFIG

    id: ShapeId
    label_key: str = Field(min_length=1, max_length=64, pattern=r"^[a-z][a-z0-9_-]*$")
    kind: Literal["box"] = "box"
    operation: Literal["add", "subtract"] = "add"
    instance_id: InstanceId = None
    x: float = Field(ge=0)
    y: float = Field(ge=0)
    width: float = Field(gt=0)
    height: float = Field(gt=0)

    def corners(self) -> list[tuple[float, float]]:
        """The polygon a box is: a box and this polygon cover exactly the same pixels."""
        x1, y1 = self.x + self.width, self.y + self.height
        return [(self.x, self.y), (x1, self.y), (x1, y1), (self.x, y1)]


class BitmapShape(BaseModel):
    """A cropped binary PNG layer positioned in the immutable source frame."""

    model_config = API_MODEL_CONFIG

    id: ShapeId
    label_key: str = Field(min_length=1, max_length=64, pattern=r"^[a-z][a-z0-9_-]*$")
    kind: Literal["bitmap"] = "bitmap"
    operation: Literal["add", "subtract"] = "add"
    instance_id: InstanceId = None
    x: int = Field(ge=0)
    y: int = Field(ge=0)
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    png_base64: str = Field(min_length=1, max_length=90_000_000)


AnnotationShape = Annotated[PolygonShape | BoxShape | BitmapShape, Field(discriminator="kind")]


class AnnotationDocument(BaseModel):
    """Editable vector truth in source-image pixel coordinates."""

    model_config = API_MODEL_CONFIG

    schema_version: Literal[1] = 1
    image_width: int = Field(gt=0)
    image_height: int = Field(gt=0)
    base: Literal["empty", "source_mask"] = "empty"
    shapes: list[AnnotationShape] = Field(default_factory=list)

    @model_validator(mode="after")
    def _valid_source_frame(self) -> AnnotationDocument:
        ids = [shape.id for shape in self.shapes]
        if len(ids) != len(set(ids)):
            raise ValueError("shape ids must be unique")
        instance_class: dict[str, str] = {}
        for shape in self.shapes:
            if shape.operation != "add":
                continue
            key = shape.instance_id or shape.id
            if instance_class.setdefault(key, shape.label_key) != shape.label_key:
                raise ValueError(f"instance {key!r} cannot hold shapes of two classes")
        for shape in self.shapes:
            if isinstance(shape, PolygonShape):
                for point in shape.points:
                    if not (0 <= point.x <= self.image_width and 0 <= point.y <= self.image_height):
                        raise ValueError("every point must lie inside the source-image frame")
            elif isinstance(shape, BoxShape):
                if (
                    shape.x + shape.width > self.image_width
                    or shape.y + shape.height > self.image_height
                ):
                    raise ValueError("every box must lie inside the source-image frame")
            elif isinstance(shape, BitmapShape):
                if (
                    shape.x + shape.width > self.image_width
                    or shape.y + shape.height > self.image_height
                ):
                    raise ValueError("every bitmap must lie inside the source-image frame")
            else:  # pragma: no cover - the discriminated union is closed
                assert_never(shape)
        return self

    def canonical_json(self) -> str:
        return json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))


class AnnotationLabel(BaseModel):
    model_config = API_MODEL_CONFIG

    id: int
    dataset_id: int
    key: str
    name: str
    color: str
    position: int
    created_at: str


class AnnotationLabelCreate(BaseModel):
    model_config = API_MODEL_CONFIG

    key: str = Field(min_length=1, max_length=64, pattern=r"^[a-z][a-z0-9_-]*$")
    name: str = Field(min_length=1, max_length=80)
    color: str = Field(pattern=r"^#[0-9a-fA-F]{6}$")
    position: int = Field(default=0, ge=0)


class AnnotationLabelUpdate(BaseModel):
    model_config = API_MODEL_CONFIG

    name: str = Field(min_length=1, max_length=80)
    color: str = Field(pattern=r"^#[0-9a-fA-F]{6}$")
    position: int = Field(ge=0)


class AnnotationDraft(BaseModel):
    model_config = API_MODEL_CONFIG

    image_id: int
    base_revision_id: int | None = None
    document: AnnotationDocument
    version: int
    source_mask_id: int | None = None
    source_mask_path: str | None = None
    source_mask_sha256: str | None = None
    updated_at: str


class AnnotationSampleDraft(BaseModel):
    """One editable document for every image of a sample.

    Deliberately narrower than `AnnotationDraft`: there is no source-mask provenance,
    because that is pinned per image and a document about to be written onto N images
    cannot carry one image's. A sample-scoped document is therefore always `base="empty"`,
    which the routes enforce rather than assume.
    """

    model_config = API_MODEL_CONFIG

    sample_id: int
    document: AnnotationDocument
    version: int
    updated_at: str


class ClassTableEntry(BaseModel):
    """One class as a completed revision pinned it: its index in the class mask, and its area."""

    model_config = API_MODEL_CONFIG

    key: str
    index: int = Field(ge=1, le=255)
    pixels: int = Field(ge=0)


class AnnotationRevision(BaseModel):
    model_config = API_MODEL_CONFIG

    id: int
    image_id: int
    revision_no: int
    document: AnnotationDocument
    document_sha256: str
    mask_path: str
    mask_sha256: str
    source_mask_id: int | None = None
    source_mask_path: str | None = None
    source_mask_sha256: str | None = None
    class_mask_path: str | None = Field(
        default=None,
        description="The class-index PNG: each pixel is a class-table index, 0 is background.",
    )
    class_mask_sha256: str | None = None
    class_table: list[ClassTableEntry] | None = Field(
        default=None,
        description=(
            "Every class the dataset had at completion, with its index and pixel count. Null "
            "for a revision completed before class masks were written."
        ),
    )
    instances_path: str | None = Field(
        default=None,
        description=(
            "The instances JSON: every object instance with its class, bounding box and "
            "pixel count. Null for a revision completed before instances were recorded."
        ),
    )
    instances_sha256: str | None = None
    completed_at: str

    @field_validator("class_table", mode="before")
    @classmethod
    def _decode_class_table(cls, value: object) -> object:
        return json.loads(value) if isinstance(value, str) else value
