"""Boxes, instance ids and the instances file a completion writes (ADR-0039).

The shape union grew without a schema version: every stored v1 document reads unchanged
and keeps its digest. Every fixture is a synthetic document or a generated PNG.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image
from pydantic import ValidationError

from anomaly_lab.annotation_render import render_truth
from anomaly_lab.config import Settings
from anomaly_lab.db.migrate import apply_migrations_to, discover_migrations
from anomaly_lab.domain.annotations import AnnotationDocument
from anomaly_lab.media.decode import sha256_of

from .conftest import Fixture, multishot_dataset

# A document as a stored v1 revision holds it: no boxes and no instance ids. Its digest was
# computed before either existed, and must never move.
V1_DOCUMENT: dict[str, Any] = {
    "schema_version": 1,
    "image_width": 16,
    "image_height": 12,
    "base": "empty",
    "shapes": [
        {
            "id": "p1",
            "label_key": "defect",
            "kind": "polygon",
            "operation": "add",
            "points": [{"x": 1, "y": 1}, {"x": 8, "y": 1}, {"x": 8, "y": 6.5}],
        },
        {
            "id": "b1",
            "label_key": "defect",
            "kind": "bitmap",
            "operation": "subtract",
            "x": 2,
            "y": 2,
            "width": 1,
            "height": 1,
            "png_base64": (
                "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAAAAAA6fptVAAAACklEQVR4nGP4DwABAQEAWjEbTgAAAABJRU5ErkJggg=="
            ),
        },
    ],
}
V1_SHA256 = "933364cd5ddcd030556b4927b79210008d818584a111e63cf328c6cdcedcaf32"


def _box(
    shape_id: str,
    label_key: str,
    rect: tuple[float, float, float, float],
    *,
    op: str = "add",
    instance_id: str | None = None,
) -> dict[str, Any]:
    x, y, width, height = rect
    shape: dict[str, Any] = {
        "id": shape_id,
        "label_key": label_key,
        "kind": "box",
        "operation": op,
        "x": x,
        "y": y,
        "width": width,
        "height": height,
    }
    if instance_id is not None:
        shape["instance_id"] = instance_id
    return shape


def _document(*shapes: dict[str, Any], size: tuple[int, int] = (16, 16)) -> AnnotationDocument:
    return AnnotationDocument.model_validate(
        {"image_width": size[0], "image_height": size[1], "shapes": list(shapes)}
    )


def _render(tmp_path: Path, name: str, document: AnnotationDocument) -> tuple[np.ndarray, Any]:
    rendered = render_truth(
        document,
        tmp_path / f"{name}.png",
        tmp_path / f"{name}.classes.png",
        classes=["defect", "scratch"],
        source_mask_path=None,
        source_mask_sha256=None,
        instances_path=tmp_path / f"{name}.instances.json",
    )
    with Image.open(tmp_path / f"{name}.classes.png") as opened:
        return np.asarray(opened), rendered


def test_a_stored_v1_document_reads_unchanged_and_keeps_its_digest() -> None:
    document = AnnotationDocument.model_validate(V1_DOCUMENT)
    assert document.schema_version == 1
    canonical = document.canonical_json()
    assert "instance_id" not in canonical
    assert hashlib.sha256(canonical.encode()).hexdigest() == V1_SHA256
    # An unset instance id is not written at the API either, so a round trip is exact.
    assert "instance_id" not in document.model_dump(mode="json")["shapes"][0]


def test_a_box_covers_exactly_the_pixels_of_its_four_corner_polygon(tmp_path: Path) -> None:
    rect = (1.5, 2.25, 9.5, 6.1)
    x, y, width, height = rect
    polygon = {
        "id": "p",
        "label_key": "scratch",
        "kind": "polygon",
        "points": [
            {"x": x, "y": y},
            {"x": x + width, "y": y},
            {"x": x + width, "y": y + height},
            {"x": x, "y": y + height},
        ],
    }
    as_box, _ = _render(tmp_path, "box", _document(_box("b", "scratch", rect)))
    as_polygon, _ = _render(tmp_path, "polygon", _document(polygon))
    assert as_box.any()
    assert np.array_equal(as_box, as_polygon)


def test_a_box_must_lie_inside_the_frame() -> None:
    _document(_box("edge", "defect", (12, 12, 4, 4)))  # touching the far edge is inside
    with pytest.raises(ValidationError, match="every box must lie inside"):
        _document(_box("out", "defect", (12, 12, 5, 4)))
    with pytest.raises(ValidationError):
        _document(_box("flat", "defect", (1, 1, 0, 4)))


def test_one_instance_cannot_hold_two_classes() -> None:
    with pytest.raises(ValidationError, match="instance 'obj' cannot hold shapes of two"):
        _document(
            _box("a", "defect", (0, 0, 2, 2), instance_id="obj"),
            _box("b", "scratch", (4, 4, 2, 2), instance_id="obj"),
        )


def test_instances_are_grouped_cut_overdrawn_and_dropped_when_empty(tmp_path: Path) -> None:
    """A box `(x, y, w, h)` fills columns x..x+w and rows y..y+h inclusive, as its polygon does.

    - `a` and `b` are one instance, `obj`: 16 + 16 pixels.
    - `c` (scratch, its own instance) overdraws `obj` on columns and rows 2..3: 4 pixels
      move from `obj` to `c`, and `obj`'s box still reaches (0, 0) and (11, 3).
    - `d` is cut away entirely by `e`, and is therefore not an instance at all.
    """
    document = _document(
        _box("a", "defect", (0, 0, 3, 3), instance_id="obj"),
        _box("b", "defect", (8, 0, 3, 3), instance_id="obj"),
        _box("c", "scratch", (2, 2, 3, 3)),
        _box("d", "defect", (12, 12, 2, 2)),
        _box("e", "defect", (11, 11, 4, 4), op="subtract"),
    )
    _, rendered = _render(tmp_path, "revision-1", document)

    path = tmp_path / "revision-1.instances.json"
    assert rendered.instances_sha256 == sha256_of(path)
    assert json.loads(path.read_text()) == {
        "instances": [
            {"instance_id": "obj", "label_key": "defect", "box": [0, 0, 12, 4], "pixels": 28},
            {"instance_id": "c", "label_key": "scratch", "box": [2, 2, 6, 6], "pixels": 16},
        ]
    }


def test_instances_of_polygons_and_bitmaps_count_their_final_pixels(tmp_path: Path) -> None:
    """The class table and the instances describe one raster, so their counts agree."""
    document = _document(
        {
            "id": "tri",
            "label_key": "defect",
            "kind": "polygon",
            "points": [{"x": 1, "y": 1}, {"x": 10, "y": 1}, {"x": 1, "y": 10}],
        },
        _box("sq", "scratch", (9, 9, 4, 4)),
    )
    classes, rendered = _render(tmp_path, "mixed", document)
    by_id = {instance.instance_id: instance for instance in rendered.instances}
    assert by_id["tri"].pixels == int((classes == 1).sum())
    assert by_id["sq"].pixels == int((classes == 2).sum())
    assert by_id["sq"].box == (9, 9, 14, 14)


def _complete_image(client: TestClient, image_id: int, shapes: list[Any]) -> dict[str, Any]:
    seed = client.get(f"/api/images/{image_id}/annotations/draft").json()["document"]
    created = client.post(
        f"/api/images/{image_id}/annotations/draft",
        json={**seed, "shapes": shapes},
        headers={"If-None-Match": "*"},
    )
    assert created.status_code == 201, created.text
    completed = client.post(
        f"/api/images/{image_id}/annotations/complete",
        headers={"If-Match": created.headers["etag"]},
    )
    assert completed.status_code == 200, completed.text
    revision: dict[str, Any] = completed.json()
    return revision


def test_an_image_completion_writes_and_pins_its_instances(
    client: TestClient, seeded: Fixture
) -> None:
    image_id = seeded.normal_image_ids[0]
    revision = _complete_image(client, image_id, [_box("a", "defect", (1, 1, 3, 3))])
    path = Path(revision["instances_path"])
    assert path.name == "revision-1.instances.json"
    assert sha256_of(path) == revision["instances_sha256"]
    assert json.loads(path.read_text())["instances"] == [
        {"instance_id": "a", "label_key": "defect", "box": [1, 1, 5, 5], "pixels": 16}
    ]
    # The document round-trips with its box, and the box is truth like any other region.
    assert revision["document"]["shapes"][0]["kind"] == "box"
    assert revision["class_table"][0]["pixels"] == 16


def test_a_sample_completion_fans_out_the_instances_file(
    client: TestClient, settings: Settings, tmp_path: Path
) -> None:
    dataset_id, sample_ids, _ = multishot_dataset(settings, tmp_path / "src")
    client.put(f"/api/datasets/{dataset_id}/annotation-scope", json={"scope": "sample"})
    sample_id = sample_ids[0]
    seed = client.get(f"/api/samples/{sample_id}/annotations/draft").json()["document"]
    created = client.post(
        f"/api/samples/{sample_id}/annotations/draft",
        json={**seed, "shapes": [_box("a", "defect", (1, 1, 4, 4), instance_id="part")]},
        headers={"If-None-Match": "*"},
    )
    assert created.status_code == 201, created.text
    completed = client.post(
        f"/api/samples/{sample_id}/annotations/complete",
        headers={"If-Match": created.headers["etag"]},
    )
    assert completed.status_code == 200, completed.text
    revisions = completed.json()
    assert len({revision["instances_sha256"] for revision in revisions}) == 1
    assert len({revision["instances_path"] for revision in revisions}) == len(revisions)
    for revision in revisions:
        path = Path(revision["instances_path"])
        assert sha256_of(path) == revision["instances_sha256"]
        assert json.loads(path.read_text())["instances"][0]["instance_id"] == "part"


def test_migration_024_leaves_earlier_revisions_without_instances(tmp_path: Path) -> None:
    conn = sqlite3.connect(tmp_path / "catalog.db")
    conn.row_factory = sqlite3.Row
    try:
        for migration in discover_migrations():
            if migration.number > 23:
                break
            conn.executescript(
                f"BEGIN;\n{migration.sql}\nPRAGMA user_version = {migration.number};\nCOMMIT;"
            )
        conn.execute("INSERT INTO dataset (name, root_path) VALUES ('d', '/d')")
        conn.execute("INSERT INTO sample (dataset_id, group_key, external_id) VALUES (1, 'g', '1')")
        conn.execute(
            "INSERT INTO image (sample_id, path, width, height, bit_depth, file_size, sha256) "
            "VALUES (1, '/d/1.png', 16, 12, 8, 64, 'h')"
        )
        conn.execute(
            "INSERT INTO annotation_revision (image_id, revision_no, document, document_sha256, "
            "mask_path, mask_sha256) VALUES (1, 1, ?, ?, '/m.png', 'm')",
            (json.dumps(V1_DOCUMENT), V1_SHA256),
        )
        conn.commit()

        assert apply_migrations_to(conn) >= 24

        row = conn.execute("SELECT * FROM annotation_revision").fetchone()
        assert row["instances_path"] is None
        assert row["instances_sha256"] is None
    finally:
        conn.close()
