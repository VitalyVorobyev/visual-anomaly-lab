"""Lossless annotation interchange, using only generated binary masks."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from anomaly_lab.annotation_bitmap import encode_png
from anomaly_lab.annotation_interchange import (
    CocoAnnotation,
    CocoCategory,
    CocoDocument,
    CocoImage,
    CocoRle,
    LabelMeDocument,
    coco_from_mask,
    decode_coco_rle,
    encode_coco_rle,
    labelme_from_mask,
    render_shapes,
    shapes_from_coco,
    shapes_from_labelme,
    shapes_from_png,
)
from anomaly_lab.config import Settings
from anomaly_lab.db.connection import connection
from anomaly_lab.db.repositories import masks as masks_repo

from .conftest import FIXTURE_SIZE, Fixture


def _mask() -> np.ndarray:
    mask = np.zeros((11, 13), dtype=bool)
    mask[1:9, 2:10] = True
    mask[3:7, 4:8] = False
    mask[9:11, 11:13] = True
    return mask


def _compressed(counts: list[int]) -> str:
    encoded: list[str] = []
    for index, count in enumerate(counts):
        value = count - counts[index - 2] if index > 2 else count
        more = True
        while more:
            code = value & 0x1F
            value >>= 5
            more = value != (-1 if code & 0x10 else 0)
            if more:
                code |= 0x20
            encoded.append(chr(code + 48))
    return "".join(encoded)


def _array(payload: bytes) -> np.ndarray:
    with Image.open(io.BytesIO(payload)) as opened:
        return np.asarray(opened.convert("L")) > 0


def test_png_and_labelme_mask_shapes_preserve_holes_and_disconnected_regions() -> None:
    expected = _mask()
    size = (expected.shape[1], expected.shape[0])

    png_shapes = shapes_from_png(encode_png(expected), size=size, label_key="defect")
    assert np.array_equal(render_shapes(png_shapes, size=size), expected)

    exported = labelme_from_mask(expected, image_path="part.png", label="defect")
    assert exported.shapes[0].shape_type == "mask"
    imported = shapes_from_labelme(
        exported,
        size=size,
        known_labels={"defect": "Defect"},
    )
    assert np.array_equal(render_shapes(imported, size=size), expected)


def test_coco_uncompressed_and_compressed_rle_use_the_standard_column_major_order() -> None:
    expected = _mask()
    size = (expected.shape[1], expected.shape[0])
    counts = encode_coco_rle(expected)

    assert np.array_equal(
        decode_coco_rle(CocoRle(size=expected.shape, counts=counts), size=size), expected
    )
    assert np.array_equal(
        decode_coco_rle(
            CocoRle(size=expected.shape, counts=_compressed(counts)),
            size=size,
        ),
        expected,
    )

    exported = coco_from_mask(expected, image_path="part.png", label="defect")
    imported = shapes_from_coco(
        exported,
        size=size,
        known_labels={"defect": "Defect"},
    )
    assert np.array_equal(render_shapes(imported, size=size), expected)


def test_coco_and_labelme_polygons_remain_editable_vectors() -> None:
    size = (16, 16)
    labelme = LabelMeDocument.model_validate(
        {
            "version": "7.0.2",
            "flags": {},
            "shapes": [
                {
                    "label": "Scratch",
                    "points": [[2, 2], [10, 2], [2, 10]],
                    "shape_type": "polygon",
                    "flags": {},
                }
            ],
            "imagePath": "part.png",
            "imageData": None,
            "imageHeight": 16,
            "imageWidth": 16,
        }
    )
    imported_labelme = shapes_from_labelme(
        labelme,
        size=size,
        known_labels={"scratch": "Scratch"},
    )
    assert imported_labelme[0].kind == "polygon"
    assert imported_labelme[0].label_key == "scratch"

    coco = CocoDocument(
        images=[CocoImage(id=9, width=16, height=16, file_name="part.png")],
        categories=[CocoCategory(id=4, name="Scratch")],
        annotations=[
            CocoAnnotation(
                id=3,
                image_id=9,
                category_id=4,
                segmentation=[[2, 2, 10, 2, 2, 10]],
                area=32,
                bbox=(2, 2, 8, 8),
            )
        ],
    )
    imported_coco = shapes_from_coco(
        coco,
        size=size,
        known_labels={"scratch": "Scratch"},
    )
    assert imported_coco[0].kind == "polygon"
    assert imported_coco[0].label_key == "scratch"


def _open(client: TestClient, image_id: int) -> str:
    seed = client.get(f"/api/images/{image_id}/annotations/draft")
    assert seed.status_code == 200, seed.text
    opened = client.post(
        f"/api/images/{image_id}/annotations/draft",
        json=seed.json()["document"],
        headers={"If-None-Match": "*"},
    )
    assert opened.status_code == 201, opened.text
    return opened.headers["etag"]


def _complete(client: TestClient, image_id: int, etag: str) -> None:
    completed = client.post(
        f"/api/images/{image_id}/annotations/complete",
        headers={"If-Match": etag},
    )
    assert completed.status_code == 200, completed.text


def test_api_round_trips_all_three_formats_without_rewriting_the_source(
    client: TestClient,
    settings: Settings,
    seeded: Fixture,
) -> None:
    image_id = seeded.defect_image_ids[0]
    with connection(settings.db_path) as conn:
        source_row = masks_repo.get_mask_for_image(conn, image_id)
    assert source_row is not None
    source_path = Path(source_row.path)
    source_bytes = source_path.read_bytes()
    source = client.get(f"/api/images/{image_id}/annotations/export/png")
    assert source.status_code == 200
    expected = _array(source.content)

    labelme = client.get(f"/api/images/{image_id}/annotations/export/labelme")
    assert labelme.status_code == 200
    assert labelme.json()["shapes"][0]["shape_type"] == "mask"
    etag = _open(client, image_id)
    imported = client.put(
        f"/api/images/{image_id}/annotations/draft/import/labelme",
        json=labelme.json(),
        headers={"If-Match": etag},
    )
    assert imported.status_code == 200, imported.text
    _complete(client, image_id, imported.headers["etag"])
    assert np.array_equal(
        _array(client.get(f"/api/images/{image_id}/annotations/export/png").content),
        expected,
    )
    assert source_path.read_bytes() == source_bytes

    coco = client.get(f"/api/images/{image_id}/annotations/export/coco")
    assert coco.status_code == 200
    assert isinstance(coco.json()["annotations"][0]["segmentation"]["counts"], list)
    etag = _open(client, image_id)
    imported = client.put(
        f"/api/images/{image_id}/annotations/draft/import/coco",
        json=coco.json(),
        headers={"If-Match": etag},
    )
    assert imported.status_code == 200, imported.text
    _complete(client, image_id, imported.headers["etag"])
    assert np.array_equal(
        _array(client.get(f"/api/images/{image_id}/annotations/export/png").content),
        expected,
    )

    etag = _open(client, image_id)
    missing = client.put(
        f"/api/images/{image_id}/annotations/draft/import/png",
        content=source.content,
        headers={"Content-Type": "image/png"},
    )
    assert missing.status_code == 428
    imported = client.put(
        f"/api/images/{image_id}/annotations/draft/import/png",
        content=source.content,
        headers={"Content-Type": "image/png", "If-Match": etag},
    )
    assert imported.status_code == 200, imported.text
    _complete(client, image_id, imported.headers["etag"])
    assert np.array_equal(
        _array(client.get(f"/api/images/{image_id}/annotations/export/png").content),
        expected,
    )
    assert source_path.read_bytes() == source_bytes


def test_import_refuses_unknown_taxonomy_and_wrong_dimensions(
    client: TestClient,
    seeded: Fixture,
) -> None:
    image_id = seeded.normal_image_ids[0]
    etag = _open(client, image_id)
    payload: dict[str, Any] = {
        "version": "7.0.2",
        "flags": {},
        "shapes": [
            {
                "label": "not-in-taxonomy",
                "points": [[1, 1], [3, 1], [1, 3]],
                "shape_type": "polygon",
                "flags": {},
            }
        ],
        "imagePath": "part.png",
        "imageData": None,
        "imageHeight": FIXTURE_SIZE,
        "imageWidth": FIXTURE_SIZE,
    }
    unknown = client.put(
        f"/api/images/{image_id}/annotations/draft/import/labelme",
        json=payload,
        headers={"If-Match": etag},
    )
    assert unknown.status_code == 422
    assert "taxonomy" in unknown.json()["detail"]

    payload["shapes"] = []
    payload["imageWidth"] = FIXTURE_SIZE + 1
    wrong_size = client.put(
        f"/api/images/{image_id}/annotations/draft/import/labelme",
        json=payload,
        headers={"If-Match": etag},
    )
    assert wrong_size.status_code == 422
    assert "dimensions" in wrong_size.json()["detail"]
