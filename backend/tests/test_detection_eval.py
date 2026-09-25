"""Object detection's evaluator and truth (ADR-0039): COCO's matching and AP read into
hand-computed numbers, `None` where nothing can be measured, every kind of truth read as
boxes, and boxes that cross the region transform in both directions."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient

from anomaly_lab.annotations.class_truth import (
    BoxTruth,
    ClassTruthError,
    load_boxes,
    resolve_box_truth,
)
from anomaly_lab.config import Settings
from anomaly_lab.db.connection import connection
from anomaly_lab.eval.detection import (
    IOU_THRESHOLDS,
    DetectionAccumulator,
    DetectionEvalError,
    average_precision,
    box_iou,
    match,
)
from anomaly_lab.media.decode import sha256_of
from anomaly_lab.models.base import (
    MAX_INSTANCES_PER_IMAGE,
    Box,
    Device,
    InferContext,
    NullReporter,
    PredictedInstance,
    TargetBox,
)
from anomaly_lab.models.diagnostics import DiagnosticWriter
from anomaly_lab.models.preprocessing import PreprocessingConfig
from anomaly_lab.regions.transform import PixelBounds, SpatialTransform

from .conftest import Fixture

CLASSES = ("scratch", "stain")


def _found(key: str, box: Box, confidence: float) -> PredictedInstance:
    return PredictedInstance(key, box, confidence)


def test_iou_is_overlap_over_union_in_pixel_edges() -> None:
    left = np.array([[0, 0, 10, 10]], dtype=float)
    right = np.array([[0, 0, 10, 10], [5, 0, 15, 10], [20, 20, 30, 30]], dtype=float)
    assert box_iou(left, right).tolist() == [[1.0, pytest.approx(50 / 150), 0.0]]
    assert box_iou(left, np.zeros((0, 4))).shape == (1, 0)


def test_two_boxes_one_match_gives_the_hand_computed_ap() -> None:
    """Two truth boxes; one exact detection at 0.9, one false positive at 0.8.

    Ranked: TP then FP, so recall is 0.5, 0.5 and precision 1, 0.5; the envelope is 1 up to
    recall 0.5 and nothing reaches beyond it. Of COCO's 101 recall points, 0.00 to 0.50 — 51
    of them — read precision 1 and the rest 0, at every IoU threshold, since the match is exact.
    """
    accumulator = DetectionAccumulator(("scratch",))
    accumulator.add(
        [TargetBox("scratch", (0, 0, 10, 10)), TargetBox("scratch", (20, 20, 30, 30))],
        [_found("scratch", (0, 0, 10, 10), 0.9), _found("scratch", (40, 40, 45, 45), 0.8)],
        inference_ms=3.0,
    )
    metrics = accumulator.metrics()
    assert metrics["ap"] == pytest.approx(51 / 101)
    assert metrics["ap50"] == pytest.approx(51 / 101)
    assert metrics["ap75"] == pytest.approx(51 / 101)
    assert metrics["recall"] == 0.5 and metrics["recall50"] == 0.5
    assert metrics["per_class_ap"] == {"scratch": pytest.approx(51 / 101)}
    assert metrics["truth_instances"] == {"scratch": 2}
    assert metrics["predicted_instances"] == {"scratch": 2}
    assert metrics["images"] == {"labelled": 1, "unlabeled": 0, "without_prediction": 0}
    assert metrics["iou_thresholds"] == list(IOU_THRESHOLDS)


def test_a_loose_box_counts_below_its_iou_and_not_above() -> None:
    """IoU 0.68 matches at 0.50, 0.55, 0.60 and 0.65 — four of ten thresholds, AP 1 at each."""
    accumulator = DetectionAccumulator(("scratch",))
    accumulator.add(
        [TargetBox("scratch", (0, 0, 10, 10))],
        [_found("scratch", (0, 0, 10, 6.8), 0.9)],
        inference_ms=0.0,
    )
    metrics = accumulator.metrics()
    assert metrics["ap"] == pytest.approx(0.4)
    assert metrics["ap50"] == 1.0
    assert metrics["ap75"] == 0.0
    assert metrics["recall"] == pytest.approx(0.4)


def test_the_more_confident_detection_takes_the_truth_and_a_duplicate_is_false() -> None:
    truth = np.array([[0, 0, 10, 10]], dtype=float)
    ranked = np.array([[0, 0, 10, 10], [0, 0, 10, 9]], dtype=float)
    matched = match(truth, ranked)
    assert matched[0].all()
    assert not matched[1].any()
    # A better fit ranked second waits its turn: COCO is greedy by confidence, not an
    # assignment, so it matches only where the first one's IoU of 0.9 falls short.
    reversed_ranking = match(truth, ranked[::-1])
    assert reversed_ranking[0].tolist() == [True] * 9 + [False]
    assert reversed_ranking[1].tolist() == [False] * 9 + [True]


def test_ties_break_in_order_and_the_envelope_fills_dips() -> None:
    # TP, FP, TP over two truth boxes: precision 1, 1/2, 2/3; the envelope lifts 1/2 to 2/3.
    confidence = np.array([0.9, 0.8, 0.7])
    matched = np.array([True, False, True])
    expected = (51 * 1.0 + 50 * (2 / 3)) / 101
    assert average_precision(confidence, matched, 2) == pytest.approx(expected)
    assert average_precision(np.array([]), np.array([], dtype=bool), 2) == 0.0
    assert average_precision(confidence, matched, 0) is None


def test_a_class_with_no_truth_has_no_ap_rather_than_zero() -> None:
    accumulator = DetectionAccumulator(CLASSES)
    accumulator.add(
        [TargetBox("scratch", (0, 0, 10, 10))],
        # A stain nobody drew: a false positive, but no stain recall exists to measure.
        [_found("scratch", (0, 0, 10, 10), 0.9), _found("stain", (20, 20, 30, 30), 0.95)],
        inference_ms=0.0,
    )
    metrics = accumulator.metrics()
    assert metrics["per_class_ap"] == {"scratch": 1.0, "stain": None}
    assert metrics["per_class_recall"] == {"scratch": 1.0, "stain": None}
    assert metrics["ap"] == 1.0

    missed = DetectionAccumulator(CLASSES)
    missed.add([TargetBox("stain", (0, 0, 4, 4))], [], inference_ms=0.0)
    # Truth and no detection: measured, and zero.
    assert missed.metrics()["per_class_ap"] == {"scratch": None, "stain": 0.0}

    empty = DetectionAccumulator(CLASSES).metrics()
    assert empty["ap"] is None and empty["ap50"] is None and empty["recall"] is None


def test_truth_of_a_class_the_run_does_not_pin_is_counted_and_nowhere_else() -> None:
    accumulator = DetectionAccumulator(("scratch",))
    accumulator.add(
        [TargetBox("scratch", (0, 0, 4, 4)), TargetBox("dent", (8, 8, 12, 12))],
        [_found("scratch", (0, 0, 4, 4), 0.5)],
        inference_ms=0.0,
    )
    metrics = accumulator.metrics()
    assert metrics["ignored_instances"] == 1
    assert metrics["truth_instances"] == {"scratch": 1}
    assert metrics["ap"] == 1.0


def test_a_detection_the_run_does_not_pin_or_too_many_are_refused() -> None:
    accumulator = DetectionAccumulator(("scratch",))
    with pytest.raises(DetectionEvalError, match="'dent', which the run does not pin"):
        accumulator.add([], [_found("dent", (0, 0, 1, 1), 0.5)], inference_ms=0.0)
    flood = [_found("scratch", (0, 0, 1, 1), 0.5)] * (MAX_INSTANCES_PER_IMAGE + 1)
    with pytest.raises(DetectionEvalError, match="at most 100"):
        accumulator.add([], flood, inference_ms=0.0)


# ------------------------------------------------------------------- the write seam


def _letterbox() -> SpatialTransform:
    """A 6x4 source, cropped to its left 4x4, scaled to 2x2 and padded to 4x2."""
    return SpatialTransform.resolve(
        source_size=(6, 4),
        prepared_size=(4, 2),
        region=PixelBounds(left=0, top=0, right=4, bottom=4),
        padding_fraction=0.0,
    )


def test_a_box_crosses_the_region_transform_both_ways() -> None:
    transform = _letterbox()
    assert (transform.pad_left, transform.scale_x) == (1, 0.5)
    prepared = transform.prepare_box((0, 0, 2, 4))
    assert prepared == (1.0, 0.0, 2.0, 2.0)
    assert transform.project_box((1.0, 0.0, 2.0, 2.0)) == (0.0, 0.0, 2.0, 4.0)
    # Outside the crop, or wholly in the padding: nothing is left of it.
    assert transform.prepare_box((4, 0, 6, 4)) is None
    assert transform.project_box((0, 0, 1, 2)) is None
    # Partly outside: clipped, never stretched.
    assert transform.prepare_box((3, 0, 6, 4)) == (2.5, 0.0, 3.0, 2.0)


def _infer_context(tmp_path: Path) -> InferContext:
    return InferContext(
        artifact_dir=tmp_path,
        cache_dir=tmp_path,
        preprocessing=PreprocessingConfig(width=8, height=8),
        device=Device.CPU,
        reporter=NullReporter(),
        diagnostics=DiagnosticWriter(tmp_path / "diagnostics", enabled=False),
        box_projector=lambda _, box: _letterbox().project_box(box),
    )


def test_detections_are_stored_in_the_source_frame_and_checked(tmp_path: Path) -> None:
    ctx = _infer_context(tmp_path)
    path = ctx.write_instances(
        7,
        [
            _found("scratch", (1, 0, 2, 2), 0.4),
            _found("stain", (2, 0, 3, 1), 0.9),
            _found("stain", (0, 0, 1, 2), 0.8),  # wholly in the padding: dropped
        ],
        classes=CLASSES,
    )
    assert path.name == "7.instances.json"
    stored = json.loads(path.read_text())["instances"]
    assert stored == [
        {"label_key": "stain", "box": [2.0, 0.0, 4.0, 2.0], "confidence": 0.9},
        {"label_key": "scratch", "box": [0.0, 0.0, 2.0, 4.0], "confidence": 0.4},
    ]

    with pytest.raises(ValueError, match="'dent', which the run does not pin"):
        ctx.write_instances(7, [_found("dent", (1, 0, 2, 2), 0.5)], classes=CLASSES)
    with pytest.raises(ValueError, match="no area"):
        ctx.write_instances(7, [_found("stain", (2, 0, 2, 2), 0.5)], classes=CLASSES)
    with pytest.raises(ValueError, match="confidence nan"):
        ctx.write_instances(7, [_found("stain", (1, 0, 2, 2), float("nan"))], classes=CLASSES)
    flood = [_found("stain", (1, 0, 2, 2), 0.5)] * (MAX_INSTANCES_PER_IMAGE + 1)
    with pytest.raises(ValueError, match="keep the most confident"):
        ctx.write_instances(7, flood, classes=CLASSES)

    # Found nothing is an answer, and is written as one.
    empty = ctx.write_instances(8, [], classes=CLASSES)
    assert json.loads(empty.read_text()) == {"instances": []}


# ----------------------------------------------------------------------- box truth


def _complete(client: TestClient, image_id: int, document: dict[str, Any]) -> None:
    draft = client.post(
        f"/api/images/{image_id}/annotations/draft",
        json=document,
        headers={"If-None-Match": "*"},
    )
    assert draft.status_code == 201, draft.text
    done = client.post(
        f"/api/images/{image_id}/annotations/complete",
        headers={"If-Match": draft.headers["etag"]},
    )
    assert done.status_code == 200, done.text


def _box(shape_id: str, key: str, x: int, y: int, size: int, **extra: Any) -> dict[str, Any]:
    return {
        "id": shape_id,
        "label_key": key,
        "kind": "box",
        "x": x,
        "y": y,
        "width": size,
        "height": size,
        **extra,
    }


def test_every_kind_of_truth_reads_as_boxes(
    client: TestClient, settings: Settings, seeded: Fixture
) -> None:
    created = client.post(
        f"/api/datasets/{seeded.dataset_id}/annotation-labels",
        json={"key": "stain", "name": "Stain", "color": "#22aa22"},
    )
    assert created.status_code == 200, created.text
    drawn, normal, _ = seeded.normal_image_ids
    source, over_source = seeded.defect_image_ids[:2]

    seed = client.get(f"/api/images/{drawn}/annotations/draft").json()["document"]
    # Two shapes of one instance box as one; a stain is its own.
    _complete(
        client,
        drawn,
        {
            **seed,
            "shapes": [
                _box("a", "defect", 1, 1, 3, instance_id="part"),
                _box("b", "defect", 4, 1, 3, instance_id="part"),
                _box("c", "stain", 10, 10, 4),
            ],
        },
    )
    # Drawn over the imported mask: the base belongs to no drawn instance, so it is boxed by
    # its component beside the stain drawn on top of it.
    base = client.get(f"/api/images/{over_source}/annotations/draft").json()["document"]
    assert base["base"] == "source_mask"
    _complete(client, over_source, {**base, "shapes": [_box("s", "stain", 12, 12, 3)]})

    with connection(settings.db_path) as conn:
        both = resolve_box_truth(
            conn, seeded.dataset_id, [drawn, normal, source, over_source], ("defect", "stain")
        )
        defect_only = resolve_box_truth(
            conn, seeded.dataset_id, [drawn, normal, source, over_source], ("defect",)
        )
    # The rule segmentation labels by: an imported mask and a normal label answer for the
    # default class alone.
    assert set(both) == {drawn, over_source}
    assert {image_id: truth.kind for image_id, truth in defect_only.items()} == {
        drawn: "instances",
        normal: "normal",
        source: "source",
        over_source: "document",
    }

    # An instance's box is the tight box of the pixels it owns, and a drawn box owns the
    # pixels it covers, so two 3-pixel boxes at x = 1 and x = 4 box as columns 1 to 6.
    assert load_boxes(both[drawn]) == [
        TargetBox("defect", (1.0, 1.0, 7.0, 4.0)),
        TargetBox("stain", (10.0, 10.0, 14.0, 14.0)),
    ]
    assert load_boxes(defect_only[normal]) == []
    assert load_boxes(defect_only[source]) == [TargetBox("defect", (5.0, 5.0, 10.0, 10.0))]
    assert load_boxes(both[over_source]) == [
        TargetBox("stain", (12.0, 12.0, 15.0, 15.0)),
        TargetBox("defect", (5.0, 5.0, 10.0, 10.0)),
    ]


def test_a_revision_keeps_the_boxes_it_wrote(
    client: TestClient, settings: Settings, seeded: Fixture
) -> None:
    # A revision completed while a box's outline was rasterised inclusively pinned an
    # instances file one pixel larger than drawn. Completed revisions are immutable: it is read
    # as written, against its own digest, and only a new completion uses the pixel-edge rule.
    image_id = seeded.normal_image_ids[0]
    seed = client.get(f"/api/images/{image_id}/annotations/draft").json()["document"]
    _complete(client, image_id, {**seed, "shapes": [_box("a", "defect", 1, 1, 3)]})
    with connection(settings.db_path) as conn:
        truth = resolve_box_truth(conn, seeded.dataset_id, [image_id], ("defect",))[image_id]
        assert truth.kind == "instances"
        assert load_boxes(truth) == [TargetBox("defect", (1.0, 1.0, 4.0, 4.0))]

    # What that earlier completion pinned instead. A revision row is immutable, so the old
    # file stands beside this one with its own digest, as an earlier revision's would.
    earlier = Path(truth.path or "").with_name("revision-0.instances.json")
    old = {"instance_id": "a", "label_key": "defect", "box": [1, 1, 5, 5], "pixels": 16}
    earlier.write_text(json.dumps({"instances": [old]}))
    pinned = replace(truth, path=str(earlier), sha256=sha256_of(earlier))
    assert load_boxes(pinned) == [TargetBox("defect", (1.0, 1.0, 5.0, 5.0))]


def test_an_instances_file_that_changed_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "revision-1.instances.json"
    path.write_text(json.dumps({"instances": [{"label_key": "a", "box": [0, 0, 2, 2]}]}))
    truth = BoxTruth(1, "instances", (4, 4), path=str(path), sha256=sha256_of(path))
    assert load_boxes(truth) == [TargetBox("a", (0.0, 0.0, 2.0, 2.0))]
    path.write_text(json.dumps({"instances": []}))
    with pytest.raises(ClassTruthError, match="changed after it was pinned"):
        load_boxes(truth)
