"""An image's truth as the sample view draws it: classes, boxes, and a region overlay."""

from __future__ import annotations

import io

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from anomaly_lab.annotation_bitmap import tight_bitmap_shape
from anomaly_lab.annotations.imported_truth import (
    ImportedClass,
    ensure_classes,
    write_document_truth,
)
from anomaly_lab.config import Settings
from anomaly_lab.db.connection import connection
from anomaly_lab.db.repositories import annotations as annotations_repo
from anomaly_lab.domain.annotations import AnnotationDocument, BoxShape

from .conftest import FIXTURE_SIZE, Fixture


def _rgba(payload: bytes) -> np.ndarray:
    with Image.open(io.BytesIO(payload)) as image:
        assert image.mode == "RGBA"
        return np.asarray(image)


def _rgb(colour: str) -> list[int]:
    digits = colour.lstrip("#")
    return [int(digits[index : index + 2], 16) for index in (0, 2, 4)]


def test_an_imported_mask_is_anomaly_truth_drawn_as_an_outline(
    client: TestClient, seeded: Fixture
) -> None:
    image_id = seeded.defect_image_ids[0]
    truth = client.get(f"/api/images/{image_id}/truth").json()
    assert truth["source"] == "imported_mask"
    assert [entry["key"] for entry in truth["classes"]] == ["defect"]
    assert truth["outline"] is True and truth["boxes"] == []

    regions = client.get(truth["regions_url"])
    assert regions.status_code == 200
    pixels = _rgba(regions.content)
    assert pixels.shape == (FIXTURE_SIZE, FIXTURE_SIZE, 4)
    # The fixture's mask is rows and columns 5..9: its border is drawn and its middle is not.
    assert pixels[5, 5, 3] > 0 and pixels[7, 5, 3] > 0
    assert pixels[7, 7, 3] == 0
    assert pixels[0, 0, 3] == 0
    assert pixels[5, 5, :3].tolist() == _rgb(truth["classes"][0]["color"])


def test_a_normal_sample_is_a_confirmed_absence_with_nothing_to_draw(
    client: TestClient, seeded: Fixture
) -> None:
    image_id = seeded.normal_image_ids[0]
    truth = client.get(f"/api/images/{image_id}/truth").json()
    assert (truth["source"], truth["classes"], truth["regions_url"]) == ("normal", [], None)
    assert client.get(f"/api/images/{image_id}/truth/regions.png").status_code == 404


def test_a_revision_draws_its_boxes_as_boxes_and_its_regions_filled(
    client: TestClient, settings: Settings, seeded: Fixture
) -> None:
    ensure_classes(
        settings,
        seeded.dataset_id,
        [ImportedClass(key="scratch", name="Scratch"), ImportedClass(key="hole", name="Hole")],
    )
    mask = np.zeros((FIXTURE_SIZE, FIXTURE_SIZE), dtype=bool)
    mask[7:15, 0:8] = True
    bitmap = tight_bitmap_shape(mask, shape_id="hole-1", label_key="hole")
    assert bitmap is not None
    box = BoxShape(id="scratch-1", label_key="scratch", x=8.0, y=1.0, width=6.0, height=5.0)
    image_id = seeded.normal_image_ids[1]
    revision = write_document_truth(
        settings,
        image_id,
        AnnotationDocument(
            image_width=FIXTURE_SIZE, image_height=FIXTURE_SIZE, shapes=[box, bitmap]
        ),
    )
    assert revision is not None

    truth = client.get(f"/api/images/{image_id}/truth").json()
    assert truth["source"] == "revision" and truth["revision_id"] == revision.id
    # In the dataset's order; `defect` comes first there, and is absent here.
    assert [entry["key"] for entry in truth["classes"]] == ["scratch", "hole"]
    assert [entry["name"] for entry in truth["classes"]] == ["Scratch", "Hole"]
    assert truth["boxes"] == [
        {"label_key": "scratch", "x": 8.0, "y": 1.0, "width": 6.0, "height": 5.0}
    ]
    assert truth["outline"] is False

    pixels = _rgba(client.get(truth["regions_url"]).content)
    hole = _rgb(truth["classes"][1]["color"])
    # The bitmap is filled in its class's colour, border and middle alike...
    assert pixels[11, 4, :3].tolist() == hole and pixels[11, 4, 3] > 0
    assert pixels[7, 4, :3].tolist() == hole and pixels[7, 4, 3] > pixels[11, 4, 3]
    # ...and the box is not part of the raster: the client draws it, with its tag.
    assert pixels[1:6, 8:14, 3].max() == 0


def test_a_recoloured_class_is_drawn_in_its_new_colour(
    client: TestClient, settings: Settings, seeded: Fixture
) -> None:
    image_id = seeded.defect_image_ids[1]
    with connection(settings.db_path) as conn:
        annotations_repo.ensure_default_label(conn, seeded.dataset_id)
        annotations_repo.update_label(
            conn, seeded.dataset_id, "defect", name="Defect", color="#123456", position=0
        )
    truth = client.get(f"/api/images/{image_id}/truth").json()
    assert truth["classes"][0]["color"] == "#123456"
    pixels = _rgba(client.get(truth["regions_url"]).content)
    assert pixels[5, 5, :3].tolist() == [0x12, 0x34, 0x56]


def test_an_unknown_image_has_no_truth(client: TestClient) -> None:
    assert client.get("/api/images/999999/truth").status_code == 404
    assert client.get("/api/images/999999/truth/regions.png").status_code == 404
