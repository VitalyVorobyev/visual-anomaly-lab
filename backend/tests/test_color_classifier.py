"""`color_classifier`, the torch-free floor for supervised segmentation — and, through it,
the whole supervised slice in the torch-free job: pin classes, train on annotated images,
write label maps, read a confusion matrix."""

from __future__ import annotations

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
from anomaly_lab.experiments.context import ExperimentJobError
from anomaly_lab.models.base import (
    IGNORE_INDEX,
    Device,
    ImageRecord,
    InferContext,
    NullReporter,
    TrainContext,
)
from anomaly_lab.models.color_classifier import ColorClassifierConfig, ColorClassifierModel
from anomaly_lab.models.diagnostics import DiagnosticWriter
from anomaly_lab.models.pixel_reference import PixelReferenceModel
from anomaly_lab.models.preprocessing import PreprocessingConfig

from .conftest import Fixture, create_experiment, run_handler


def decode_plane(payload: bytes) -> np.ndarray:
    """A one-channel value plane (`media/values.py`) as a `(height, width)` array."""
    width, height = (int(value) for value in np.frombuffer(payload[4:12], dtype="<u4"))
    return np.frombuffer(payload[24:], dtype="<f4").reshape(height, width)


SIZE = 16
RED, GREEN = (200, 30, 30), (30, 190, 40)


class _Labels:
    classes = ("rust", "moss")

    def __init__(self, labels: dict[int, np.ndarray]) -> None:
        self._labels = labels

    def labels(self, image_id: int) -> np.ndarray:
        return self._labels[image_id]


def _image(path: Path, red: tuple[int, int] | None, green: tuple[int, int] | None) -> np.ndarray:
    """Blue with a little noise, a red 4x4 square (`rust`) and a green one (`moss`)."""
    rng = np.random.default_rng(len(str(path)))
    pixels = np.zeros((SIZE, SIZE, 3), dtype=np.float64)
    pixels[..., 2] = 180 + rng.normal(0, 4, (SIZE, SIZE))
    labels = np.zeros((SIZE, SIZE), dtype=np.uint8)
    for index, (corner, colour) in enumerate(((red, RED), (green, GREEN)), start=1):
        if corner is not None:
            y, x = corner
            pixels[y : y + 4, x : x + 4] = colour
            labels[y : y + 4, x : x + 4] = index
    Image.fromarray(np.clip(pixels, 0, 255).astype(np.uint8)).save(path)
    return labels


Squares = tuple[tuple[int, int] | None, tuple[int, int] | None]


def _records(
    tmp_path: Path, squares: Sequence[Squares], *, first_id: int = 1
) -> tuple[list[ImageRecord], dict[int, np.ndarray]]:
    records: list[ImageRecord] = []
    labels: dict[int, np.ndarray] = {}
    for image_id, (red, green) in enumerate(squares, start=first_id):
        path = tmp_path / f"{image_id}.png"
        labels[image_id] = _image(path, red, green)
        records.append(ImageRecord(image_id=image_id, sample_id=image_id, path=path))
    return records, labels


def _contexts(tmp_path: Path, targets: _Labels | None) -> tuple[TrainContext, InferContext]:
    common: dict[str, Any] = {
        "artifact_dir": tmp_path,
        "cache_dir": tmp_path,
        "preprocessing": PreprocessingConfig(width=SIZE, height=SIZE),
        "device": Device.CPU,
        "reporter": NullReporter(),
        "diagnostics": DiagnosticWriter(tmp_path / "diagnostics", enabled=False),
    }
    return TrainContext(**common, label_targets=targets), InferContext(**common)


def _stored(ctx: InferContext, image_id: int) -> np.ndarray:
    with Image.open(ctx.label_map_path(image_id)) as opened:
        return np.asarray(opened)


def test_it_labels_each_class_by_colour_and_scores_the_share_it_found(tmp_path: Path) -> None:
    training, labels = _records(tmp_path, [((2, 2), (9, 9)), ((8, 1), (1, 10))])
    queries, truth = _records(tmp_path, [((10, 3), (3, 10)), (None, None)], first_id=11)

    train_ctx, infer_ctx = _contexts(tmp_path, _Labels(labels))
    model = ColorClassifierModel(ColorClassifierConfig(smoothing_sigma=0.0))
    model.fit(training, train_ctx)
    both, neither = model.predict(queries, infer_ctx)

    assert np.array_equal(_stored(infer_ctx, 11), truth[11])
    assert not _stored(infer_ctx, 12).any()
    assert both.label_map == infer_ctx.label_map_path(11)
    assert both.score == pytest.approx(32 / SIZE**2)
    assert neither.score == 0.0
    probability = np.load(infer_ctx.map_path(11))
    assert probability[truth[11] > 0].min() > 0.9 > 0.1 > probability[truth[11] == 0].max()


def test_the_fit_is_deterministic_and_survives_a_round_trip(tmp_path: Path) -> None:
    training, labels = _records(tmp_path, [((2, 2), (9, 9)), ((8, 1), (1, 10))])
    train_ctx, infer_ctx = _contexts(tmp_path, _Labels(labels))

    first = ColorClassifierModel(ColorClassifierConfig())
    first.fit(training, train_ctx)
    second = ColorClassifierModel(ColorClassifierConfig())
    second.fit(training, train_ctx)
    first.save(tmp_path)
    loaded = ColorClassifierModel(ColorClassifierConfig())
    loaded.load(tmp_path)

    # There is no seed to vary: pixels are sampled with `evenly_spaced`, so the same
    # training images always give the same models, and a loaded one the same maps.
    maps = []
    for model in (first, second, loaded):
        model.predict(training[:1], infer_ctx)
        maps.append(_stored(infer_ctx, 1).copy())
    assert np.array_equal(maps[0], maps[1]) and np.array_equal(maps[0], maps[2])


def test_the_pixel_cap_samples_evenly_and_says_so(tmp_path: Path) -> None:
    training, labels = _records(tmp_path, [((2, 2), (9, 9)), ((8, 1), (1, 10))])
    logged: list[str] = []

    class Recorder(NullReporter):
        def log(self, message: str, level: str = "info") -> None:
            logged.append(message)

    train_ctx, _ = _contexts(tmp_path, _Labels(labels))
    train_ctx.reporter = Recorder()
    ColorClassifierModel(ColorClassifierConfig(max_pixels_per_class=100)).fit(training, train_ctx)
    assert any("'background' on 100 of 448 training pixels" in line for line in logged)
    assert any("'rust' on all 32 training pixels" in line for line in logged)


def test_ignored_pixels_are_fitted_on_by_nobody(tmp_path: Path) -> None:
    training, labels = _records(tmp_path, [((2, 2), None)])
    # Mark every red pixel ignored, and a class left with no pixel is never predicted.
    labels[1][labels[1] == 1] = IGNORE_INDEX
    labels[1][0, 0] = 2
    logged: list[str] = []

    class Recorder(NullReporter):
        def log(self, message: str, level: str = "info") -> None:
            logged.append(message)

    train_ctx, infer_ctx = _contexts(tmp_path, _Labels(labels))
    train_ctx.reporter = Recorder()
    model = ColorClassifierModel(ColorClassifierConfig(smoothing_sigma=0.0))
    model.fit(training, train_ctx)
    assert any("no training pixel of 'rust'" in line for line in logged)
    model.predict(training, infer_ctx)
    assert 1 not in np.unique(_stored(infer_ctx, 1))


def test_it_refuses_to_fit_without_labels_or_without_any_class(tmp_path: Path) -> None:
    training, labels = _records(tmp_path, [(None, None)])
    no_targets, _ = _contexts(tmp_path, None)
    with pytest.raises(RuntimeError, match="needs label targets"):
        ColorClassifierModel(ColorClassifierConfig()).fit(training, no_targets)
    blank, _ = _contexts(tmp_path, _Labels(labels))
    with pytest.raises(RuntimeError, match="no pixel of any class of rust, moss"):
        ColorClassifierModel(ColorClassifierConfig()).fit(training, blank)


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


def test_the_whole_supervised_slice_runs_without_torch(
    client: TestClient, settings: Settings, seeded: Fixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    anomaly = create_experiment(client, seeded)  # builds the region profile the run pins
    created = create_experiment(
        client,
        seeded,
        name="colour classes",
        split_id=_supervised_split(client, settings, seeded),
        model_type="color_classifier",
        task="semantic_segmentation",
        config={},
    )
    assert created["classes"] == ["defect"]
    assert created["target_label"] is None

    trained = run_handler(settings, JobKind.TRAIN, {"experiment_id": created["id"]})
    assert trained["train_images"] == 3
    scored = run_handler(settings, JobKind.INFER, {"experiment_id": created["id"]})
    metrics = scored["metrics"]["test"]
    with connection(settings.db_path) as conn:
        total = int(conn.execute("SELECT COUNT(*) FROM image").fetchone()[0]) - 3
    assert metrics["images"] == {"labelled": total, "unlabeled": 0, "without_prediction": 0}
    assert metrics["classes"] == ["defect"]
    assert metrics["per_class_iou"]["defect"] > 0.5
    assert metrics["mean_iou"] == metrics["per_class_iou"]["defect"]
    assert metrics["pixel_accuracy"] > 0.9

    detail = client.get(f"/api/experiments/{created['id']}").json()
    assert detail["headline_metric"] == "mean_iou"
    assert detail["headline_value"] == metrics["mean_iou"]
    assert detail["classes"] == ["defect"]
    assert detail["metrics"][0]["ground_truth_stale"] is False
    ranked = client.get(f"/api/experiments/{created['id']}/results", params={"subset": "test"})
    assert ranked.status_code == 200 and len(ranked.json()["samples"]) == total

    # Each sample's verdict, from its own label maps: nothing thresholded, every one labelled.
    outcomes = client.get(
        f"/api/experiments/{created['id']}/segmentation-outcomes", params={"subset": "test"}
    )
    assert outcomes.status_code == 200, outcomes.text
    body = outcomes.json()
    assert body["threshold_rule"] == "the method's label map, as written"
    verdicts = body["samples"]
    assert len(verdicts) == len(ranked.json()["samples"])
    allowed = {"hit", "low_iou", "miss", "false_class", "false_presence", "correct_absence"}
    assert {verdict["outcome"] for verdict in verdicts} <= allowed
    assert any(verdict["outcome"] == "hit" for verdict in verdicts)
    assert all(
        verdict["iou"] is None
        for verdict in verdicts
        if verdict["outcome"] in ("correct_absence", "false_presence")
    )

    # The label maps as value planes, predicted and true, in the run's numbering.
    image_id = seeded.defect_image_ids[-1]
    base = f"/api/experiments/{created['id']}/images/{image_id}/labels"
    predicted = decode_plane(client.get(base).content)
    truth = decode_plane(client.get(base, params={"truth": "true"}).content)
    assert predicted.shape == truth.shape
    assert set(np.unique(predicted)) <= {0.0, 1.0}
    assert set(np.unique(truth)) == {0.0, 1.0}
    unscored = seeded.defect_image_ids[0]  # in train, so never scored
    missing = client.get(f"/api/experiments/{created['id']}/images/{unscored}/labels")
    assert missing.status_code == 404
    refused = client.get(f"/api/experiments/{anomaly['id']}/images/{image_id}/labels")
    assert refused.status_code == 409

    # An anomaly fit is still given no ground truth of either kind.
    seen: dict[str, Any] = {}
    original = PixelReferenceModel.fit

    def fit(self: PixelReferenceModel, train: Sequence[ImageRecord], ctx: TrainContext) -> None:
        seen["targets"], seen["label_targets"] = ctx.targets, ctx.label_targets
        original(self, train, ctx)

    monkeypatch.setattr(PixelReferenceModel, "fit", fit)
    run_handler(settings, JobKind.TRAIN, {"experiment_id": anomaly["id"]})
    assert seen == {"targets": None, "label_targets": None}


def test_a_class_added_before_creation_leaves_imported_truth_unlabelled(
    client: TestClient, settings: Settings, seeded: Fixture
) -> None:
    create_experiment(client, seeded)
    client.post(
        f"/api/datasets/{seeded.dataset_id}/annotation-labels",
        json={"key": "stain", "name": "Stain", "color": "#22aa22", "position": 1},
    )
    created = create_experiment(
        client,
        seeded,
        name="two classes",
        split_id=_supervised_split(client, settings, seeded),
        model_type="color_classifier",
        task="semantic_segmentation",
        config={},
    )
    # Pinned in taxonomy order, and an imported mask answers for `defect` alone.
    assert created["classes"] == ["defect", "stain"]
    with pytest.raises(ExperimentJobError, match="ground truth for every one of defect, stain"):
        run_handler(settings, JobKind.TRAIN, {"experiment_id": created["id"]})


def test_creation_refuses_a_method_or_a_target_the_task_does_not_take(
    client: TestClient, settings: Settings, seeded: Fixture
) -> None:
    create_experiment(client, seeded)
    body: dict[str, Any] = {
        "name": "refused",
        "dataset_id": seeded.dataset_id,
        "split_id": _supervised_split(client, settings, seeded),
        "region_profile_id": seeded.region_profile_id,
        "model_type": "pixel_reference",
        "task": "semantic_segmentation",
        "config": {},
    }
    refused = client.post("/api/experiments", json=body)
    assert refused.status_code == 422
    assert "does not support the task 'semantic_segmentation'" in refused.text

    as_anomaly = client.post(
        "/api/experiments", json={**body, "model_type": "color_classifier", "task": "anomaly"}
    )
    assert as_anomaly.status_code == 422
    assert "does not support the task 'anomaly'" in as_anomaly.text

    targeted = client.post(
        "/api/experiments",
        json={**body, "model_type": "color_classifier", "target_label": "defect"},
    )
    assert targeted.status_code == 422
    assert "does not take a target class" in targeted.text
