"""`color_detector`, the torch-free floor for object detection — and, through it, the whole
detection slice in the torch-free job: pin classes, train on boxed truth, write boxes, read
COCO's AP."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from anomaly_lab.config import Settings
from anomaly_lab.db.connection import connection
from anomaly_lab.domain.entities import JobKind
from anomaly_lab.models.base import (
    Device,
    ImageRecord,
    InferContext,
    NullReporter,
    TargetBox,
    TrainContext,
)
from anomaly_lab.models.color_detector import (
    ColorDetectorConfig,
    ColorDetectorModel,
    paint_boxes,
)
from anomaly_lab.models.diagnostics import DiagnosticWriter
from anomaly_lab.models.pixel_reference import PixelReferenceModel
from anomaly_lab.models.preprocessing import PreprocessingConfig

from .conftest import Fixture, create_experiment, run_handler

SIZE = 16
RED, GREEN = (200, 30, 30), (30, 190, 40)
CLASSES = ("rust", "moss")


class _Boxes:
    classes = CLASSES

    def __init__(self, boxes: dict[int, list[TargetBox]]) -> None:
        self._boxes = boxes

    def boxes(self, image_id: int) -> list[TargetBox]:
        return self._boxes[image_id]


Squares = tuple[tuple[int, int] | None, tuple[int, int] | None]


def _records(
    tmp_path: Path, squares: Sequence[Squares], *, first_id: int = 1
) -> tuple[list[ImageRecord], dict[int, list[TargetBox]]]:
    """Blue images with a red 4x4 square (`rust`) and a green one (`moss`), and their boxes."""
    records: list[ImageRecord] = []
    boxes: dict[int, list[TargetBox]] = {}
    for image_id, pair in enumerate(squares, start=first_id):
        rng = np.random.default_rng(image_id)
        pixels = np.zeros((SIZE, SIZE, 3), dtype=np.float64)
        pixels[..., 2] = 180 + rng.normal(0, 4, (SIZE, SIZE))
        boxes[image_id] = []
        for key, corner, colour in zip(CLASSES, pair, (RED, GREEN), strict=True):
            if corner is not None:
                y, x = corner
                pixels[y : y + 4, x : x + 4] = colour
                boxes[image_id].append(TargetBox(key, (x, y, x + 4, y + 4)))
        path = tmp_path / f"{image_id}.png"
        Image.fromarray(np.clip(pixels, 0, 255).astype(np.uint8)).save(path)
        records.append(ImageRecord(image_id=image_id, sample_id=image_id, path=path))
    return records, boxes


def _contexts(tmp_path: Path, targets: _Boxes | None) -> tuple[TrainContext, InferContext]:
    common: dict[str, Any] = {
        "artifact_dir": tmp_path,
        "cache_dir": tmp_path,
        "preprocessing": PreprocessingConfig(width=SIZE, height=SIZE),
        "device": Device.CPU,
        "reporter": NullReporter(),
        "diagnostics": DiagnosticWriter(tmp_path / "diagnostics", enabled=False),
    }
    return TrainContext(**common, box_targets=targets), InferContext(**common)


def _stored(ctx: InferContext, image_id: int) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = json.loads(ctx.instances_path(image_id).read_text())["instances"]
    return found


def test_box_interiors_paint_by_pixel_centre_and_the_smaller_box_wins() -> None:
    painted = paint_boxes(
        [TargetBox("rust", (0, 0, 6, 6)), TargetBox("moss", (2.4, 2.6, 4, 4))],
        CLASSES,
        8,
        8,
    )
    assert (painted[:6, :6] == 1).sum() == 34 and (painted == 2).sum() == 2
    # A centre at 2.5 is inside a box from 2.4, one at 2.5 is not inside a box from 2.6.
    assert painted[3, 2] == 2 and painted[2, 3] == 1
    assert not painted[6:, :].any() and not painted[:, 6:].any()


def test_it_boxes_each_class_and_scores_its_most_confident_find(tmp_path: Path) -> None:
    training, boxes = _records(tmp_path, [((2, 2), (9, 9)), ((8, 1), (1, 10))])
    queries, truth = _records(tmp_path, [((10, 3), (3, 10)), (None, None)], first_id=11)

    train_ctx, infer_ctx = _contexts(tmp_path, _Boxes(boxes))
    model = ColorDetectorModel(ColorDetectorConfig(smoothing_sigma=0.0))
    model.fit(training, train_ctx)
    both, neither = model.predict(queries, infer_ctx)

    found = _stored(infer_ctx, 11)
    assert {(entry["label_key"], tuple(entry["box"])) for entry in found} == {
        (target.label_key, target.box) for target in truth[11]
    }
    assert both.instances == infer_ctx.instances_path(11)
    assert both.score == found[0]["confidence"] > 0.9
    assert _stored(infer_ctx, 12) == []
    assert neither.score == 0.0


def test_the_fit_is_deterministic_and_survives_a_round_trip(tmp_path: Path) -> None:
    training, boxes = _records(tmp_path, [((2, 2), (9, 9)), ((8, 1), (1, 10))])
    train_ctx, infer_ctx = _contexts(tmp_path, _Boxes(boxes))

    first = ColorDetectorModel(ColorDetectorConfig())
    first.fit(training, train_ctx)
    second = ColorDetectorModel(ColorDetectorConfig())
    second.fit(training, train_ctx)
    first.save(tmp_path)
    loaded = ColorDetectorModel(ColorDetectorConfig())
    loaded.load(tmp_path)

    # No seed to vary: pixels are sampled with `evenly_spaced`, so the same boxes always give
    # the same models, and a loaded one the same detections.
    stored = []
    for model in (first, second, loaded):
        model.predict(training[:1], infer_ctx)
        stored.append(_stored(infer_ctx, 1))
    assert stored[0] == stored[1] == stored[2]


def test_small_components_and_detections_past_the_cap_are_dropped(tmp_path: Path) -> None:
    training, boxes = _records(tmp_path, [((2, 2), (9, 9)), ((8, 1), (1, 10))])
    train_ctx, infer_ctx = _contexts(tmp_path, _Boxes(boxes))
    model = ColorDetectorModel(ColorDetectorConfig(smoothing_sigma=0.0, max_detections=1))
    model.fit(training, train_ctx)
    model.predict(training[:1], infer_ctx)
    assert len(_stored(infer_ctx, 1)) == 1

    strict = ColorDetectorModel(ColorDetectorConfig(smoothing_sigma=0.0, min_area=17))
    strict.fit(training, train_ctx)
    strict.predict(training[:1], infer_ctx)
    assert _stored(infer_ctx, 1) == []


def test_it_refuses_to_fit_without_boxes(tmp_path: Path) -> None:
    training, _ = _records(tmp_path, [((2, 2), None)])
    no_targets, _ = _contexts(tmp_path, None)
    with pytest.raises(RuntimeError, match="needs box targets"):
        ColorDetectorModel(ColorDetectorConfig()).fit(training, no_targets)


# ---------------------------------------------------------------- the slice, end to end


def _sample_of(settings: Settings, image_id: int) -> int:
    with connection(settings.db_path) as conn:
        return int(
            conn.execute("SELECT sample_id FROM image WHERE id = ?", (image_id,)).fetchone()[0]
        )


def _supervised_split(client: TestClient, settings: Settings, seeded: Fixture) -> int:
    """Two defects and a normal train; everything else tests."""
    train = [*seeded.defect_image_ids[:2], seeded.normal_image_ids[0]]
    split = client.post(
        "/api/splits",
        json={
            "dataset_id": seeded.dataset_id,
            "name": "annotated train",
            "seed": 0,
            "params": {
                "strategy": "manual",
                "sample_ids": [_sample_of(settings, image_id) for image_id in train],
            },
        },
    )
    assert split.status_code == 200, split.text
    return int(split.json()["id"])


def test_the_whole_detection_slice_runs_without_torch(
    client: TestClient, settings: Settings, seeded: Fixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    anomaly = create_experiment(client, seeded)  # builds the region profile the run pins
    seen: dict[str, Any] = {}
    original_fit = ColorDetectorModel.fit

    def spy(self: ColorDetectorModel, train: Sequence[ImageRecord], ctx: TrainContext) -> None:
        seen["detection"] = (ctx.targets, ctx.label_targets, ctx.box_targets)
        original_fit(self, train, ctx)

    monkeypatch.setattr(ColorDetectorModel, "fit", spy)
    created = create_experiment(
        client,
        seeded,
        name="colour boxes",
        split_id=_supervised_split(client, settings, seeded),
        model_type="color_detector",
        task="object_detection",
        config={},
    )
    assert created["classes"] == ["defect"]
    assert created["target_label"] is None

    trained = run_handler(settings, JobKind.TRAIN, {"experiment_id": created["id"]})
    assert trained["train_images"] == 3
    targets, label_targets, box_targets = seen["detection"]
    assert targets is None and label_targets is None
    # The imported masks are boxed by their components: one 5x5 square per defect image.
    assert box_targets.classes == ("defect",)
    assert box_targets.boxes(seeded.defect_image_ids[0]) == [
        TargetBox("defect", (5.0, 5.0, 10.0, 10.0))
    ]
    assert box_targets.boxes(seeded.normal_image_ids[0]) == []

    scored = run_handler(settings, JobKind.INFER, {"experiment_id": created["id"]})
    metrics = scored["metrics"]["test"]
    with connection(settings.db_path) as conn:
        total = int(conn.execute("SELECT COUNT(*) FROM image").fetchone()[0]) - 3
    assert metrics["images"] == {"labelled": total, "unlabeled": 0, "without_prediction": 0}
    assert metrics["classes"] == ["defect"]
    defects = len(seeded.defect_image_ids) - 2
    assert metrics["truth_instances"] == {"defect": defects}
    assert metrics["ap50"] > 0.5
    assert metrics["ap"] == metrics["per_class_ap"]["defect"]
    assert metrics["recall50"] > 0.5

    detail = client.get(f"/api/experiments/{created['id']}").json()
    assert detail["headline_metric"] == "ap"
    assert detail["headline_value"] == metrics["ap"]
    assert detail["metrics"][0]["ground_truth_stale"] is False
    ranked = client.get(f"/api/experiments/{created['id']}/results", params={"subset": "test"})
    assert ranked.status_code == 200 and len(ranked.json()["samples"]) == total

    # The screens that belong to other tasks say so rather than guessing.
    image_id = seeded.defect_image_ids[-1]
    labels = client.get(f"/api/experiments/{created['id']}/images/{image_id}/labels")
    assert labels.status_code == 409
    outcomes = client.get(f"/api/experiments/{created['id']}/segmentation-outcomes")
    assert outcomes.status_code == 409

    # An anomaly fit is still given no ground truth of any kind.
    original = PixelReferenceModel.fit

    def fit(self: PixelReferenceModel, train: Sequence[ImageRecord], ctx: TrainContext) -> None:
        seen["anomaly"] = (ctx.targets, ctx.label_targets, ctx.box_targets)
        original(self, train, ctx)

    monkeypatch.setattr(PixelReferenceModel, "fit", fit)
    run_handler(settings, JobKind.TRAIN, {"experiment_id": anomaly["id"]})
    assert seen["anomaly"] == (None, None, None)


def test_creation_refuses_a_method_the_task_does_not_take(
    client: TestClient, settings: Settings, seeded: Fixture
) -> None:
    create_experiment(client, seeded)
    body: dict[str, Any] = {
        "name": "refused",
        "dataset_id": seeded.dataset_id,
        "split_id": _supervised_split(client, settings, seeded),
        "region_profile_id": seeded.region_profile_id,
        "model_type": "color_classifier",
        "task": "object_detection",
        "config": {},
    }
    refused = client.post("/api/experiments", json=body)
    assert refused.status_code == 422
    assert "does not support the task 'object_detection'" in refused.text

    as_segmentation = client.post(
        "/api/experiments",
        json={**body, "model_type": "color_detector", "task": "semantic_segmentation"},
    )
    assert as_segmentation.status_code == 422
    assert "does not support the task 'semantic_segmentation'" in as_segmentation.text
