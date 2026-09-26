"""Box truth a dataset ships: VOC files read as pixel-edge boxes, entered as revisions."""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path

import pytest

from anomaly_lab.annotations.class_truth import load_boxes, resolve_box_truth
from anomaly_lab.annotations.imported_truth import (
    CLASS_PALETTE,
    ImportedBox,
    ImportedClass,
    ensure_classes,
    write_box_truth,
)
from anomaly_lab.config import Settings
from anomaly_lab.datasets.voc import VocError, clamp_box, read_voc
from anomaly_lab.db.repositories import annotations as annotations_repo
from anomaly_lab.db.repositories import datasets as datasets_repo
from anomaly_lab.db.repositories import images as images_repo
from anomaly_lab.db.repositories import samples as samples_repo
from anomaly_lab.domain.entities import Label


def _voc(
    path: Path, body: str, size: str = "<size><width>20</width><height>10</height></size>"
) -> Path:
    path.write_text(f"<annotation>{size}{body}</annotation>", encoding="utf-8")
    return path


def _object(name: str, x0: str, y0: str, x1: str, y1: str) -> str:
    return (
        f"<object><name>{name}</name><bndbox><xmin>{x0}</xmin><ymin>{y0}</ymin>"
        f"<xmax>{x1}</xmax><ymax>{y1}</ymax></bndbox></object>"
    )


def test_voc_corners_are_one_based_and_inclusive(tmp_path: Path) -> None:
    """(1, 1)-(1, 1) is the single top-left pixel: the pixel-edge box (0, 0)-(1, 1)."""
    path = _voc(
        tmp_path / "a.xml",
        _object("spur", "1", "1", "1", "1") + _object(" short ", "4.0", "2", "9", "5"),
    )
    annotation = read_voc(path)
    assert annotation.size == (20, 10)
    assert [(item.name, item.box) for item in annotation.objects] == [
        ("spur", (0.0, 0.0, 1.0, 1.0)),
        ("short", (3.0, 1.0, 9.0, 5.0)),
    ]


def test_a_voc_file_without_a_size_or_objects_is_still_an_answer(tmp_path: Path) -> None:
    annotation = read_voc(_voc(tmp_path / "a.xml", "", size=""))
    assert (annotation.size, annotation.objects) == (None, ())


@pytest.mark.parametrize(
    ("body", "message"),
    [
        (_object("spur", "5", "1", "4", "2"), "inverted"),
        (_object("", "1", "1", "2", "2"), "no name"),
        ("<object><name>spur</name></object>", "no bndbox"),
        (_object("spur", "a", "1", "2", "2"), "not a number"),
    ],
)
def test_a_malformed_voc_object_is_refused(tmp_path: Path, body: str, message: str) -> None:
    with pytest.raises(VocError, match=message):
        read_voc(_voc(tmp_path / "a.xml", body))


def test_a_file_that_is_not_voc_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "a.xml"
    path.write_text("<coco/>", encoding="utf-8")
    with pytest.raises(VocError, match="not <annotation>"):
        read_voc(path)
    path.write_text("<annotation>", encoding="utf-8")
    with pytest.raises(VocError):
        read_voc(path)


def test_a_box_is_clamped_to_the_frame_or_dropped_outside_it() -> None:
    assert clamp_box((-1.0, 0.0, 5.0, 12.0), 20, 10) == (0.0, 0.0, 5.0, 10.0)
    assert clamp_box((20.0, 0.0, 25.0, 5.0), 20, 10) is None


def _image(conn: sqlite3.Connection, width: int = 20, height: int = 10) -> tuple[int, int]:
    dataset = datasets_repo.create_dataset(conn, name="boxes", root_path="/boxes")
    sample, _ = samples_repo.upsert_sample(
        conn, dataset.id, group_key="g", external_id="1", label=Label.DEFECT, notes=None
    )
    image, _ = images_repo.upsert_image(
        conn,
        sample.id,
        channel_id=None,
        path="/boxes/1.png",
        width=width,
        height=height,
        bit_depth=8,
        file_size=1,
        sha256="0" * 64,
    )
    return dataset.id, image.id


CLASSES = (ImportedClass("short", "Short"), ImportedClass("spur", "Spur"))


def test_classes_are_added_after_the_default_in_the_editors_palette(
    settings: Settings, migrated_db: sqlite3.Connection
) -> None:
    dataset_id, _ = _image(migrated_db)
    assert ensure_classes(settings, dataset_id, CLASSES) == ["short", "spur"]
    assert ensure_classes(settings, dataset_id, CLASSES) == []
    labels = annotations_repo.list_labels(migrated_db, dataset_id)
    assert [(label.key, label.position) for label in labels] == [
        ("defect", 0),
        ("short", 1),
        ("spur", 2),
    ]
    assert [label.color for label in labels[1:]] == list(CLASS_PALETTE[:2])


def test_the_import_palette_is_the_editors() -> None:
    """One palette, written twice: an imported class must look like one added by hand."""
    manager = (
        Path(__file__).resolve().parents[2] / "frontend/src/routes/dataset/ClassManager.tsx"
    ).read_text(encoding="utf-8")
    block = manager.split("export const CLASS_PALETTE = [", 1)[1].split("]", 1)[0]
    assert tuple(re.findall(r'"(#[0-9a-f]{6})"', block)) == CLASS_PALETTE


def test_overlapping_boxes_keep_the_smaller_one_whole(
    settings: Settings, migrated_db: sqlite3.Connection
) -> None:
    """Drawn largest first: the smaller box keeps its pixels, and the larger keeps its box
    unless the smaller covers a whole edge of it, which is counted."""
    dataset_id, image_id = _image(migrated_db)
    ensure_classes(settings, dataset_id, CLASSES)
    small = ImportedBox("spur", (2.0, 2.0, 4.0, 4.0))
    corner = ImportedBox("short", (0.0, 0.0, 10.0, 8.0))
    written = write_box_truth(settings, image_id, 20, 10, [small, corner])
    assert (written.written, written.reshaped) == (True, 0)

    labels = [label.key for label in annotations_repo.list_labels(migrated_db, dataset_id)]
    truth = resolve_box_truth(migrated_db, dataset_id, [image_id], labels)
    assert sorted((box.label_key, box.box) for box in load_boxes(truth[image_id])) == [
        ("short", (0.0, 0.0, 10.0, 8.0)),
        ("spur", (2.0, 2.0, 4.0, 4.0)),
    ]


def test_an_edge_covering_overlap_is_counted_as_reshaped(
    settings: Settings, migrated_db: sqlite3.Connection
) -> None:
    dataset_id, image_id = _image(migrated_db)
    ensure_classes(settings, dataset_id, CLASSES)
    edge = ImportedBox("spur", (0.0, 0.0, 2.0, 8.0))
    wide = ImportedBox("short", (0.0, 0.0, 10.0, 6.0))
    written = write_box_truth(settings, image_id, 20, 10, [edge, wide])
    assert (written.written, written.reshaped) == (True, 1)

    labels = [label.key for label in annotations_repo.list_labels(migrated_db, dataset_id)]
    truth = resolve_box_truth(migrated_db, dataset_id, [image_id], labels)
    assert sorted((box.label_key, box.box) for box in load_boxes(truth[image_id])) == [
        ("short", (2.0, 0.0, 10.0, 6.0)),
        ("spur", (0.0, 0.0, 2.0, 8.0)),
    ]


def test_an_image_with_no_boxes_answers_every_class_absent(
    settings: Settings, migrated_db: sqlite3.Connection
) -> None:
    dataset_id, image_id = _image(migrated_db)
    ensure_classes(settings, dataset_id, CLASSES)
    assert write_box_truth(settings, image_id, 20, 10, []).written
    assert not write_box_truth(settings, image_id, 20, 10, []).written

    labels = [label.key for label in annotations_repo.list_labels(migrated_db, dataset_id)]
    truth = resolve_box_truth(migrated_db, dataset_id, [image_id], labels)
    assert load_boxes(truth[image_id]) == []
