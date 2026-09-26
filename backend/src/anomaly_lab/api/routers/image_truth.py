"""An image's truth for the sample view: what it holds, and its regions as an overlay.

The read is `annotations/image_truth.py`; this is the HTTP edge. Classes carry the dataset's
own name and colour — the taxonomy the editor paints with — so the legend and the overlay
agree with the editor without the client knowing a palette.
"""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, HTTPException, Request, Response
from pydantic import BaseModel, Field

from anomaly_lab.annotations.class_truth import ClassTruthError
from anomaly_lab.annotations.image_truth import read_image_truth, read_region_plane
from anomaly_lab.config import Settings
from anomaly_lab.db.connection import connection
from anomaly_lab.db.repositories import annotations as annotations_repo
from anomaly_lab.domain.annotations import AnnotationLabel
from anomaly_lab.media.overlay import render_class_regions
from anomaly_lab.schemas import API_MODEL_CONFIG

router = APIRouter(prefix="/api/images", tags=["images"])

# The colour a class the dataset no longer lists is drawn in: neutral, and visible on a
# photograph. A class is never deleted, so this is a guard rather than a path.
UNLISTED_RGB = (139, 148, 155)


class TruthClass(BaseModel):
    model_config = API_MODEL_CONFIG

    key: str
    name: str
    color: str = Field(description="The class's colour in the dataset's taxonomy, `#rrggbb`.")


class TruthBox(BaseModel):
    model_config = API_MODEL_CONFIG

    label_key: str
    x: float
    y: float
    width: float
    height: float


class ImageTruthResponse(BaseModel):
    model_config = API_MODEL_CONFIG

    image_id: int
    source: Literal["revision", "imported_mask", "normal", "none"] = Field(
        description=(
            "What answers for this image: its newest completed revision, an imported "
            "ground-truth mask, its sample's normal label, or nothing yet."
        )
    )
    revision_id: int | None = None
    classes: list[TruthClass] = Field(description="The classes present, in the dataset's order.")
    boxes: list[TruthBox] = Field(description="Drawn boxes, source-frame pixel edges.")
    regions_url: str | None = Field(
        default=None,
        description="The region overlay, when the truth has regions other than boxes.",
    )
    outline: bool = Field(
        default=False, description="The regions are anomaly truth, drawn as an outline."
    )


def _settings(request: Request) -> Settings:
    settings: Settings = request.app.state.settings
    return settings


@router.get("/{image_id}/truth", summary="An image's truth, as the sample view draws it")
def get_image_truth(request: Request, image_id: int) -> ImageTruthResponse:
    settings = _settings(request)
    with connection(settings.db_path) as conn:
        truth = read_image_truth(conn, image_id)
        if truth is None:
            raise HTTPException(status_code=404, detail=f"no image with id {image_id}")
        labels = {
            label.key: label for label in annotations_repo.list_labels(conn, truth.dataset_id)
        }
    return ImageTruthResponse(
        image_id=image_id,
        source=truth.source,
        revision_id=truth.revision_id,
        classes=[_class(labels, key) for key in truth.classes],
        boxes=[TruthBox.model_validate(box, from_attributes=True) for box in truth.boxes],
        regions_url=(
            f"/api/images/{image_id}/truth/regions.png?v={truth.identity}"
            if truth.regions
            else None
        ),
        outline=truth.outline,
    )


@router.get(
    "/{image_id}/truth/regions.png",
    summary="An image's truth regions in its classes' colours, at the source size",
    response_class=Response,
    responses={200: {"content": {"image/png": {}}}},
)
def get_image_truth_regions(request: Request, image_id: int) -> Response:
    """Every region but a drawn box, filled in its class's colour — or, for an imported
    anomaly mask, its outline. Boxes are drawn by the client, with their class tags."""
    settings = _settings(request)
    with connection(settings.db_path) as conn:
        truth = read_image_truth(conn, image_id)
        if truth is None:
            raise HTTPException(status_code=404, detail=f"no image with id {image_id}")
        try:
            plane = read_region_plane(conn, image_id)
        except ClassTruthError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        labels = {
            label.key: label for label in annotations_repo.list_labels(conn, truth.dataset_id)
        }
    if plane is None:
        raise HTTPException(status_code=404, detail=f"image {image_id} has no truth regions")
    colours = [_rgb(labels.get(key)) for key in plane.classes] or [UNLISTED_RGB]
    payload = render_class_regions(plane.labels, colours, outline_only=plane.outline)
    # Revalidated rather than immutable: recolouring a class changes the picture, and the
    # `v` in the URL only names the truth.
    return Response(
        content=payload,
        media_type="image/png",
        headers={"Cache-Control": "no-cache"},
    )


def _class(labels: dict[str, AnnotationLabel], key: str) -> TruthClass:
    label = labels.get(key)
    if label is None:
        return TruthClass(key=key, name=key, color="#{:02x}{:02x}{:02x}".format(*UNLISTED_RGB))
    return TruthClass(key=key, name=label.name, color=label.color)


def _rgb(label: AnnotationLabel | None) -> tuple[int, int, int]:
    if label is None:
        return UNLISTED_RGB
    digits = label.color.lstrip("#")
    return (int(digits[0:2], 16), int(digits[2:4], 16), int(digits[4:6], 16))
