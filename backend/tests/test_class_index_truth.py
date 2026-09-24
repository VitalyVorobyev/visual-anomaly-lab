"""Completion writes a class-index mask beside the binary one and pins its class table
(ADR-0040). The binary mask is unchanged, and a class's presence reads the table."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from anomaly_lab.annotation_render import AnnotationRenderError, render_truth
from anomaly_lab.config import Settings
from anomaly_lab.db.connection import connection
from anomaly_lab.db.repositories import annotations as annotations_repo
from anomaly_lab.domain.annotations import AnnotationDocument
from anomaly_lab.domain.entities import ClassPresence
from anomaly_lab.media.decode import sha256_of

from .conftest import Fixture, multishot_dataset


def _polygon(shape_id: str, label_key: str, box: tuple[int, int, int, int], op: str = "add") -> Any:
    x0, y0, x1, y1 = box
    return {
        "id": shape_id,
        "label_key": label_key,
        "kind": "polygon",
        "operation": op,
        "points": [{"x": x0, "y": y0}, {"x": x1, "y": y0}, {"x": x1, "y": y1}, {"x": x0, "y": y1}],
    }


def _document(*shapes: Any, base: str = "empty") -> AnnotationDocument:
    return AnnotationDocument.model_validate(
        {"image_width": 16, "image_height": 16, "base": base, "shapes": list(shapes)}
    )


def _read(path: Path) -> np.ndarray:
    with Image.open(path) as opened:
        return np.asarray(opened)


def test_later_shapes_win_and_a_cut_clears_every_class(tmp_path: Path) -> None:
    document = _document(
        _polygon("a", "defect", (0, 0, 8, 8)),
        _polygon("b", "scratch", (4, 4, 12, 12)),
        _polygon("c", "defect", (10, 10, 14, 14), op="subtract"),
    )
    rendered = render_truth(
        document,
        tmp_path / "revision-1.png",
        tmp_path / "revision-1.classes.png",
        classes=["defect", "scratch", "dent"],
        source_mask_path=None,
        source_mask_sha256=None,
    )
    index = _read(tmp_path / "revision-1.classes.png")
    binary = _read(tmp_path / "revision-1.png")

    assert index[1, 1] == 1  # defect
    assert index[6, 6] == 2  # scratch drawn over defect
    assert index[11, 11] == 0  # cut away
    assert np.array_equal(binary, np.where(index > 0, 255, 0))
    table = {entry.key: (entry.index, entry.pixels) for entry in rendered.class_table}
    assert table["dent"] == (3, 0)  # pinned although not drawn: a confirmed absence
    assert table["defect"][1] == int((index == 1).sum()) > 0
    assert table["scratch"][1] == int((index == 2).sum()) > 0
    assert rendered.class_mask_sha256 == sha256_of(tmp_path / "revision-1.classes.png")


def test_the_binary_mask_matches_a_plain_binary_render(tmp_path: Path) -> None:
    """Anomaly consumers read the binary mask; which class drew a pixel must not change it."""
    shapes = [
        _polygon("a", "defect", (1, 1, 9, 5)),
        _polygon("b", "scratch", (3, 3, 13, 13)),
        _polygon("c", "defect", (5, 5, 7, 7), op="subtract"),
    ]
    two = render_truth(
        _document(*shapes),
        tmp_path / "two.png",
        tmp_path / "two.classes.png",
        classes=["defect", "scratch"],
        source_mask_path=None,
        source_mask_sha256=None,
    )
    one_class = [{**shape, "label_key": "defect"} for shape in shapes]
    one = render_truth(
        _document(*one_class),
        tmp_path / "one.png",
        tmp_path / "one.classes.png",
        classes=["defect", "scratch"],
        source_mask_path=None,
        source_mask_sha256=None,
    )
    assert two.mask_sha256 == one.mask_sha256


def test_a_source_mask_base_is_drawn_in_the_default_class(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    pixels = np.zeros((16, 16), dtype=np.uint8)
    pixels[2:6, 2:6] = 255
    Image.fromarray(pixels).save(source)
    rendered = render_truth(
        _document(base="source_mask"),
        tmp_path / "r.png",
        tmp_path / "r.classes.png",
        classes=["scratch", "defect"],
        source_mask_path=source,
        source_mask_sha256=sha256_of(source),
    )
    index = _read(tmp_path / "r.classes.png")
    assert set(np.unique(index)) == {0, 2}
    assert {entry.key: entry.pixels for entry in rendered.class_table} == {
        "scratch": 0,
        "defect": 16,
    }


def test_a_shape_of_an_unknown_class_is_refused(tmp_path: Path) -> None:
    with pytest.raises(AnnotationRenderError, match="unknown class 'scratch'"):
        render_truth(
            _document(_polygon("a", "scratch", (0, 0, 4, 4))),
            tmp_path / "r.png",
            tmp_path / "r.classes.png",
            classes=["defect"],
            source_mask_path=None,
            source_mask_sha256=None,
        )


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


def test_completion_pins_the_table_and_presence_reads_it(
    client: TestClient, settings: Settings, seeded: Fixture
) -> None:
    created = client.post(
        f"/api/datasets/{seeded.dataset_id}/annotation-labels",
        json={"key": "scratch", "name": "Scratch", "color": "#00aa00", "position": 1},
    )
    assert created.status_code == 200, created.text
    drawn, cut = seeded.normal_image_ids[:2]

    revision = _complete_image(client, drawn, [_polygon("a", "scratch", (2, 2, 10, 10))])
    assert [entry["key"] for entry in revision["class_table"]] == ["defect", "scratch"]
    class_path = Path(revision["class_mask_path"])
    assert class_path.name == "revision-1.classes.png"
    assert sha256_of(class_path) == revision["class_mask_sha256"]

    # A region cut away entirely is drawn in the document and absent from the pixels.
    _complete_image(
        client,
        cut,
        [
            _polygon("a", "scratch", (2, 2, 6, 6)),
            _polygon("b", "scratch", (0, 0, 15, 15), op="subtract"),
        ],
    )
    with connection(settings.db_path) as conn:
        found = annotations_repo.class_presence(conn, seeded.dataset_id, "scratch")
        sample_of = {
            image_id: int(
                conn.execute("SELECT sample_id FROM image WHERE id = ?", (image_id,)).fetchone()[0]
            )
            for image_id in (drawn, cut)
        }
    assert found[sample_of[drawn]] is ClassPresence.PRESENT
    assert found[sample_of[cut]] is ClassPresence.ABSENT


def test_a_class_created_after_completion_is_unlabelled_there(
    client: TestClient, settings: Settings, seeded: Fixture
) -> None:
    image_id = seeded.normal_image_ids[0]
    _complete_image(client, image_id, [])
    client.post(
        f"/api/datasets/{seeded.dataset_id}/annotation-labels",
        json={"key": "scratch", "name": "Scratch", "color": "#00aa00", "position": 1},
    )
    with connection(settings.db_path) as conn:
        found = annotations_repo.presence_by_class(conn, seeded.dataset_id, ["defect", "scratch"])
        sample = int(
            conn.execute("SELECT sample_id FROM image WHERE id = ?", (image_id,)).fetchone()[0]
        )
    assert found["defect"][sample] is ClassPresence.ABSENT
    assert found["scratch"][sample] is ClassPresence.UNLABELED


def test_a_sample_completion_fans_out_the_class_mask_too(
    client: TestClient, settings: Settings, tmp_path: Path
) -> None:
    dataset_id, sample_ids, _ = multishot_dataset(settings, tmp_path / "src")
    client.put(f"/api/datasets/{dataset_id}/annotation-scope", json={"scope": "sample"})
    sample_id = sample_ids[0]
    seed = client.get(f"/api/samples/{sample_id}/annotations/draft").json()["document"]
    created = client.post(
        f"/api/samples/{sample_id}/annotations/draft",
        json={**seed, "shapes": [_polygon("a", "defect", (1, 1, 5, 5))]},
        headers={"If-None-Match": "*"},
    )
    completed = client.post(
        f"/api/samples/{sample_id}/annotations/complete",
        headers={"If-Match": created.headers["etag"]},
    )
    assert completed.status_code == 200, completed.text
    revisions = completed.json()
    assert len({revision["class_mask_sha256"] for revision in revisions}) == 1
    assert len({revision["class_mask_path"] for revision in revisions}) == len(revisions)
    for revision in revisions:
        assert sha256_of(Path(revision["class_mask_path"])) == revision["class_mask_sha256"]
        assert revision["class_table"][0]["pixels"] > 0
