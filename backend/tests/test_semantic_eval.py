"""Supervised semantic segmentation's evaluator and truth (ADR-0039): a confusion matrix in
constant memory, read into hand-computed numbers; every kind of truth read as a label map
in the run's pinned numbering; and label maps that are never interpolated."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from anomaly_lab.annotations.class_truth import (
    ClassTruthError,
    LabelTruth,
    load_label_map,
    resolve_label_truth,
)
from anomaly_lab.config import Settings
from anomaly_lab.db.connection import connection
from anomaly_lab.eval.semantic import ConfusionAccumulator, SemanticEvalError
from anomaly_lab.media.decode import sha256_of
from anomaly_lab.models.base import IGNORE_INDEX, Device, InferContext, NullReporter
from anomaly_lab.models.diagnostics import DiagnosticWriter
from anomaly_lab.models.preprocessing import PreprocessingConfig
from anomaly_lab.regions.transform import PixelBounds, SpatialTransform

from .conftest import FIXTURE_SIZE, Fixture

CLASSES = ("scratch", "stain")


def _two_classes_and_background() -> tuple[np.ndarray, np.ndarray]:
    """A 4x4 image: two scratch pixels, four stain pixels, ten background.

    truth      predicted
    1 1 0 0    1 0 0 0
    0 0 0 0    0 0 0 1
    2 2 0 0    2 2 2 0
    2 2 0 0    2 0 0 0
    """
    truth = np.array([[1, 1, 0, 0], [0, 0, 0, 0], [2, 2, 0, 0], [2, 2, 0, 0]], dtype=np.uint8)
    predicted = np.array([[1, 0, 0, 0], [0, 0, 0, 1], [2, 2, 2, 0], [2, 0, 0, 0]], dtype=np.uint8)
    return truth, predicted


def test_the_confusion_matrix_reads_into_hand_computed_metrics() -> None:
    truth, predicted = _two_classes_and_background()
    accumulator = ConfusionAccumulator(CLASSES)
    accumulator.add(truth, predicted, inference_ms=2.0)
    metrics = accumulator.metrics()

    # Rows are truth, columns predicted, background first.
    assert metrics["confusion"] == {
        "classes": ["background", "scratch", "stain"],
        "counts": [[8, 1, 1], [1, 1, 0], [1, 0, 3]],
    }
    # scratch: 1 hit, union 2 + 2 - 1 = 3. stain: 3 hits, union 4 + 4 - 3 = 5.
    # background: 8 hits, union 10 + 10 - 8 = 12.
    assert metrics["per_class_iou"] == {"scratch": pytest.approx(1 / 3), "stain": 0.6}
    assert metrics["background_iou"] == pytest.approx(8 / 12)
    assert metrics["mean_iou"] == pytest.approx((1 / 3 + 0.6) / 2)
    assert metrics["pixel_accuracy"] == pytest.approx(12 / 16)
    assert metrics["per_class_accuracy"] == {"scratch": 0.5, "stain": 0.75}
    assert metrics["mean_class_accuracy"] == pytest.approx(0.625)
    assert metrics["frequency_weighted_iou"] == pytest.approx(
        (10 * 8 / 12 + 2 * (1 / 3) + 4 * 0.6) / 16
    )
    assert metrics["images"] == {"labelled": 1, "unlabeled": 0, "without_prediction": 0}


def test_two_images_pool_into_the_matrix_one_would_give() -> None:
    truth, predicted = _two_classes_and_background()
    pooled = ConfusionAccumulator(CLASSES)
    pooled.add(np.vstack([truth, truth]), np.vstack([predicted, predicted]), inference_ms=1.0)
    split = ConfusionAccumulator(CLASSES)
    split.add(truth, predicted, inference_ms=1.0)
    split.add(truth, predicted, inference_ms=1.0)
    assert np.array_equal(pooled.counts, split.counts)
    assert split.metrics()["mean_iou"] == pooled.metrics()["mean_iou"]


def test_a_class_nobody_drew_or_predicted_has_no_iou_rather_than_zero() -> None:
    truth = np.array([[0, 1], [0, 1]], dtype=np.uint8)
    accumulator = ConfusionAccumulator(CLASSES)
    accumulator.add(truth, truth, inference_ms=0.0)
    metrics = accumulator.metrics()
    assert metrics["per_class_iou"] == {"scratch": 1.0, "stain": None}
    assert metrics["per_class_accuracy"] == {"scratch": 1.0, "stain": None}
    assert metrics["mean_iou"] == 1.0

    predicted_only = ConfusionAccumulator(CLASSES)
    predicted_only.add(truth, np.array([[2, 1], [0, 1]], dtype=np.uint8), inference_ms=0.0)
    # Predicted but never true: a union with no hit is a real zero, and no accuracy.
    assert predicted_only.metrics()["per_class_iou"]["stain"] == 0.0
    assert predicted_only.metrics()["per_class_accuracy"]["stain"] is None

    empty = ConfusionAccumulator(CLASSES).metrics()
    assert empty["mean_iou"] is None
    assert empty["pixel_accuracy"] is None
    assert empty["frequency_weighted_iou"] is None


def test_ignored_pixels_are_counted_and_nowhere_else() -> None:
    truth = np.array([[1, IGNORE_INDEX], [0, IGNORE_INDEX]], dtype=np.uint8)
    accumulator = ConfusionAccumulator(CLASSES)
    accumulator.add(truth, np.array([[1, 2], [0, 2]], dtype=np.uint8), inference_ms=0.0)
    metrics = accumulator.metrics()
    assert metrics["ignored_pixels"] == 2
    assert int(accumulator.counts.sum()) == 2
    assert metrics["pixel_accuracy"] == 1.0


def test_a_label_the_run_does_not_have_is_refused_by_name() -> None:
    accumulator = ConfusionAccumulator(CLASSES)
    with pytest.raises(SemanticEvalError, match="index 3; the run has 2 classes"):
        accumulator.add(
            np.zeros((2, 2), dtype=np.uint8), np.full((2, 2), 3, dtype=np.uint8), inference_ms=0.0
        )
    with pytest.raises(SemanticEvalError, match="shape"):
        accumulator.add(
            np.zeros((2, 2), dtype=np.uint8), np.zeros((3, 2), dtype=np.uint8), inference_ms=0.0
        )


def test_a_stored_class_mask_is_renumbered_into_the_run_s_classes(tmp_path: Path) -> None:
    stored = np.array([[0, 1], [2, 3]], dtype=np.uint8)
    path = tmp_path / "classes.png"
    Image.fromarray(stored, mode="L").save(path)
    truth = LabelTruth(
        1,
        "class_mask",
        (2, 2),
        # Stored under the taxonomy at completion; the run pins another order and one fewer.
        stored=(("defect", 1), ("stain", 2), ("scratch", 3)),
        path=str(path),
        sha256=sha256_of(path),
    )
    labels = load_label_map(truth, ("scratch", "stain"))
    assert labels.tolist() == [[0, IGNORE_INDEX], [2, 1]]

    path.write_bytes(b"changed")
    with pytest.raises(ClassTruthError, match="changed after it was pinned"):
        load_label_map(truth, ("scratch", "stain"))


def _complete(client: TestClient, image_id: int, shapes: list[dict[str, Any]]) -> None:
    seed = client.get(f"/api/images/{image_id}/annotations/draft").json()["document"]
    draft = client.post(
        f"/api/images/{image_id}/annotations/draft",
        json={**seed, "shapes": shapes},
        headers={"If-None-Match": "*"},
    )
    assert draft.status_code == 201, draft.text
    done = client.post(
        f"/api/images/{image_id}/annotations/complete",
        headers={"If-Match": draft.headers["etag"]},
    )
    assert done.status_code == 200, done.text


def _square(key: str, low: int, high: int) -> dict[str, Any]:
    return {
        "id": f"{key}-{low}",
        "label_key": key,
        "kind": "polygon",
        "operation": "add",
        "points": [
            {"x": low, "y": low},
            {"x": high, "y": low},
            {"x": high, "y": high},
            {"x": low, "y": high},
        ],
    }


def test_an_image_is_labelled_only_when_its_truth_answers_every_pinned_class(
    client: TestClient, settings: Settings, seeded: Fixture
) -> None:
    created = client.post(
        f"/api/datasets/{seeded.dataset_id}/annotation-labels",
        json={"key": "stain", "name": "Stain", "color": "#22aa22"},
    )
    assert created.status_code == 200, created.text
    completed, normal, _ = seeded.normal_image_ids
    source = seeded.defect_image_ids[0]
    _complete(client, completed, [_square("defect", 1, 5), _square("stain", 9, 14)])

    with connection(settings.db_path) as conn:
        both = resolve_label_truth(
            conn, seeded.dataset_id, [completed, normal, source], ("defect", "stain")
        )
        defect_only = resolve_label_truth(
            conn, seeded.dataset_id, [completed, normal, source], ("defect",)
        )
    # An imported mask and a normal label answer for the default class alone.
    assert set(both) == {completed}
    assert {image_id: truth.kind for image_id, truth in defect_only.items()} == {
        completed: "class_mask",
        normal: "normal",
        source: "source",
    }

    labels = load_label_map(both[completed], ("defect", "stain"))
    assert labels.shape == (FIXTURE_SIZE, FIXTURE_SIZE)
    assert labels[3, 3] == 1 and labels[11, 11] == 2 and labels[7, 7] == 0
    # Read against a run that pinned only `defect`, the stain is nobody's truth.
    narrow = load_label_map(defect_only[completed], ("defect",))
    assert narrow[3, 3] == 1 and narrow[11, 11] == IGNORE_INDEX
    assert set(np.unique(load_label_map(defect_only[source], ("defect",)))) == {0, 1}
    assert not load_label_map(defect_only[normal], ("defect",)).any()


def _letterbox() -> SpatialTransform:
    """A 6x4 source, cropped to its left 4x4, scaled to 2x2 and padded to 4x2."""
    return SpatialTransform.resolve(
        source_size=(6, 4),
        prepared_size=(4, 2),
        region=PixelBounds(left=0, top=0, right=4, bottom=4),
        padding_fraction=0.0,
    )


def test_label_maps_cross_the_region_transform_nearest_and_filled() -> None:
    transform = _letterbox()
    source = np.array(
        [[1, 1, 2, 2, 5, 5], [1, 1, 2, 2, 5, 5], [0, 0, 3, 3, 5, 5], [0, 0, 3, 3, 5, 5]],
        dtype=np.uint8,
    )
    prepared = transform.prepare_labels(source, fill=IGNORE_INDEX)
    assert prepared.shape == (2, 4)
    # Every value is one the source had or the fill — nothing between two classes.
    assert set(np.unique(prepared)) <= {0, 1, 2, 3, IGNORE_INDEX}
    assert (prepared == IGNORE_INDEX).sum() == 4

    back = transform.project_labels(prepared, fill=0)
    assert back.shape == (4, 6)
    assert np.array_equal(back[:, :4], source[:, :4])
    assert not back[:, 4:].any()


def _infer_context(tmp_path: Path) -> InferContext:
    return InferContext(
        artifact_dir=tmp_path,
        cache_dir=tmp_path,
        preprocessing=PreprocessingConfig(width=8, height=8),
        device=Device.CPU,
        reporter=NullReporter(),
        diagnostics=DiagnosticWriter(tmp_path / "diagnostics", enabled=False),
        label_projector=lambda _, values: _letterbox().project_labels(values, fill=0),
    )


def test_a_label_map_is_stored_in_the_source_frame_and_checked(tmp_path: Path) -> None:
    ctx = _infer_context(tmp_path)
    path = ctx.write_label_map(7, np.array([[0, 1, 2, 0], [0, 2, 1, 0]]), classes=2)
    with Image.open(path) as opened:
        stored = np.asarray(opened)
    assert path.name == "7.labels.png"
    assert stored.shape == (4, 6) and set(np.unique(stored)) == {0, 1, 2}

    with pytest.raises(ValueError, match=r"outside 0\.\.2"):
        ctx.write_label_map(7, np.full((2, 4), 3), classes=2)
    with pytest.raises(ValueError, match="class indices"):
        ctx.write_label_map(7, np.zeros((2, 4), dtype=np.float32), classes=2)
