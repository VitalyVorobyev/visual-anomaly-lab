"""Source-frame annotation documents and their persisted lifecycle records."""

from __future__ import annotations

import json
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from anomaly_lab.schemas import API_MODEL_CONFIG


class AnnotationPoint(BaseModel):
    model_config = API_MODEL_CONFIG

    x: float
    y: float


class PolygonShape(BaseModel):
    model_config = API_MODEL_CONFIG

    id: str = Field(min_length=1, max_length=128)
    label_key: str = Field(min_length=1, max_length=64, pattern=r"^[a-z][a-z0-9_-]*$")
    kind: Literal["polygon"] = "polygon"
    operation: Literal["add", "subtract"] = "add"
    points: list[AnnotationPoint] = Field(min_length=3)


class AnnotationDocument(BaseModel):
    """Editable vector truth in source-image pixel coordinates."""

    model_config = API_MODEL_CONFIG

    schema_version: Literal[1] = 1
    image_width: int = Field(gt=0)
    image_height: int = Field(gt=0)
    base: Literal["empty", "source_mask"] = "empty"
    shapes: list[PolygonShape] = Field(default_factory=list)

    @model_validator(mode="after")
    def _valid_source_frame(self) -> AnnotationDocument:
        ids = [shape.id for shape in self.shapes]
        if len(ids) != len(set(ids)):
            raise ValueError("shape ids must be unique")
        for shape in self.shapes:
            for point in shape.points:
                if not (0 <= point.x <= self.image_width and 0 <= point.y <= self.image_height):
                    raise ValueError("every point must lie inside the source-image frame")
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
    completed_at: str
