"""`dino_linear_det` against a seeded random ViT: boxed images in, boxes out that COCO's AP
reads as found, the seed in both directions, a round trip that detects identically, a cancelled
fit that leaves nothing behind, and a fit refused without boxes. Accuracy on real data needs
pretrained weights and belongs to the public detection gate, not here."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("timm")

from PIL import Image

from anomaly_lab.eval.detection import DetectionAccumulator, read_instances
from anomaly_lab.map_files import read_map
from anomaly_lab.models.base import (
    Device,
    ImageRecord,
    InferContext,
    ModelCancelledError,
    NullReporter,
    PredictedInstance,
    ProgressReporter,
    TargetBox,
    TrainContext,
)
from anomaly_lab.models.diagnostics import DiagnosticWriter
from anomaly_lab.models.dino_backbone import DinoBackbone
from anomaly_lab.models.dino_linear_det import DinoLinearDetConfig, DinoLinearDetModel
from anomaly_lab.models.preprocessing import PreprocessingConfig

SIZE = 64
RED, GREEN = (220, 40, 40), (40, 200, 60)
CLASSES = ("rust", "moss")


class _Boxes:
    classes = CLASSES

    def __init__(self, boxes: dict[int, list[TargetBox]]) -> None:
        self._boxes = boxes

    def boxes(self, image_id: int) -> list[TargetBox]:
        return self._boxes[image_id]


def _records(
    folder: Path, count: int, first_id: int = 1
) -> tuple[list[ImageRecord], dict[int, list[TargetBox]]]:
    """Dark noise, a red square (`rust`) and a striped green square (`moss`), placed by image.

    Each square is 32 pixels on a side, two patches of the 16-pixel encoder, and they sit in
    opposite corners — so every object is its own component."""
    folder.mkdir(parents=True, exist_ok=True)
    records: list[ImageRecord] = []
    boxes: dict[int, list[TargetBox]] = {}
    for image_id in range(first_id, first_id + count):
        rng = np.random.default_rng(image_id)
        pixels = rng.integers(0, 60, (SIZE, SIZE, 3)).astype(np.uint8)
        top = 32 * (image_id % 2)
        pixels[top : top + 32, 0:32] = RED
        pixels[32 - top : 64 - top, 32:64] = GREEN
        pixels[32 - top : 64 - top : 4, 32:64] = (230, 230, 230)
        boxes[image_id] = [
            TargetBox("rust", (0.0, float(top), 32.0, float(top + 32))),
            TargetBox("moss", (32.0, float(32 - top), 64.0, float(64 - top))),
        ]
        path = folder / f"{image_id}.png"
        Image.fromarray(pixels).save(path)
        records.append(ImageRecord(image_id=image_id, sample_id=image_id, path=path))
    return records, boxes


def _contexts(
    artifact_dir: Path,
    boxes: dict[int, list[TargetBox]] | None,
    reporter: ProgressReporter | None = None,
) -> tuple[TrainContext, InferContext]:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    common: dict[str, Any] = {
        "artifact_dir": artifact_dir,
        "cache_dir": artifact_dir.parent / "cache",
        "preprocessing": PreprocessingConfig(width=SIZE, height=SIZE),
        "device": Device.CPU,
        "reporter": reporter or NullReporter(),
        "diagnostics": DiagnosticWriter(artifact_dir / "diagnostics", enabled=False),
    }
    targets = None if boxes is None else _Boxes(boxes)
    return TrainContext(**common, box_targets=targets), InferContext(**common)


def _model(seed: int = 0, **overrides: Any) -> DinoLinearDetModel:
    settings: dict[str, Any] = {
        "backbone": DinoBackbone.DINOV3_VIT_S16,
        "pretrained_backbone": False,
        "allow_downloads": False,
        "epochs": 30,
        "batch_size": 256,
        "learning_rate": 0.01,
        "seed": seed,
    }
    settings.update(overrides)
    return DinoLinearDetModel(DinoLinearDetConfig(**settings))


def _detect(
    model: DinoLinearDetModel, artifact_dir: Path, images: list[ImageRecord]
) -> list[list[PredictedInstance]]:
    _, infer_ctx = _contexts(artifact_dir, None)
    model.predict(images, infer_ctx)
    stored = [read_instances(infer_ctx.instances_path(record.image_id)) for record in images]
    return [found or [] for found in stored]


def test_boxed_images_in_boxes_that_coco_reads_as_found_out(tmp_path: Path) -> None:
    training, boxes = _records(tmp_path / "images", 4)
    queries, truth = _records(tmp_path / "images", 4, first_id=11)
    train_ctx, infer_ctx = _contexts(tmp_path / "run", boxes)
    model = _model()
    model.fit(training, train_ctx)
    predictions = model.predict(queries, infer_ctx)

    accumulator = DetectionAccumulator(CLASSES)
    for prediction, record in zip(predictions, queries, strict=True):
        found = read_instances(infer_ctx.instances_path(record.image_id))
        assert found is not None
        assert prediction.instances == infer_ctx.instances_path(record.image_id)
        assert prediction.score == (found[0].confidence if found else 0.0)
        assert all(0.0 < item.confidence <= 1.0 for item in found)
        foreground = read_map(infer_ctx.map_path(record.image_id))
        assert foreground.shape == (SIZE, SIZE)
        assert foreground.min() >= 0.0 and foreground.max() <= 1.0
        accumulator.add(truth[record.image_id], found, inference_ms=prediction.inference_ms)
    metrics = accumulator.metrics()
    assert metrics["ap50"] > 0.8, metrics
    assert metrics["recall50"] > 0.8, metrics
    assert all(value > 0.8 for value in metrics["per_class_ap50"].values()), metrics


def test_the_pixel_plan_is_logged_before_encoding(tmp_path: Path) -> None:
    training, boxes = _records(tmp_path / "images", 2)
    events: list[str] = []

    class Recorder(NullReporter):
        def log(self, message: str, level: str = "info") -> None:
            events.append(message)

        def progress(self, fraction: float, message: str | None = None) -> None:
            events.append(f"progress {message}")

    train_ctx, _ = _contexts(tmp_path / "run", boxes, Recorder())
    _model(epochs=2, pixels_per_image=64).fit(training, train_ctx)
    plan = next(index for index, line in enumerate(events) if line.startswith("pixel plan"))
    encoded = next(index for index, line in enumerate(events) if "encoded 1/2" in line)
    assert plan < encoded
    assert "at most 64 labelled pixels from each" in events[plan]


def test_one_seed_is_one_answer_and_another_seed_is_another(tmp_path: Path) -> None:
    training, boxes = _records(tmp_path / "images", 3)
    queries, _ = _records(tmp_path / "images", 2, first_id=11)
    fitted: list[tuple[np.ndarray, list[list[PredictedInstance]]]] = []
    for name, seed in (("a", 0), ("b", 0), ("c", 1)):
        train_ctx, _ = _contexts(tmp_path / name, boxes)
        model = _model(seed, epochs=3)
        model.fit(training, train_ctx)
        state = model._head._state
        weights = np.concatenate([state["weights"].ravel(), state["held_out_bias"]])
        fitted.append((weights, _detect(model, tmp_path / name, queries)))
    (first, first_found), (again, again_found), (other, _) = fitted
    assert np.array_equal(first, again)
    assert first_found == again_found
    assert not np.array_equal(first, other)


def test_a_saved_model_detects_identically_after_loading(tmp_path: Path) -> None:
    training, boxes = _records(tmp_path / "images", 3)
    queries, _ = _records(tmp_path / "images", 2, first_id=11)
    train_ctx, _ = _contexts(tmp_path / "run", boxes)
    model = _model(epochs=3)
    model.fit(training, train_ctx)
    before = _detect(model, tmp_path / "run", queries)
    model.save(tmp_path / "run")
    assert sorted(path.name for path in (tmp_path / "run").glob("dino_linear_*")) == [
        "dino_linear_det.json",
        "dino_linear_seg.json",
        "dino_linear_seg.npz",
    ]

    restored = _model(epochs=3)
    restored.load(tmp_path / "run")
    assert _detect(restored, tmp_path / "loaded", queries) == before
    with pytest.raises(RuntimeError, match="fitted with dinov3_vit_s16 \\(last\\)"):
        _model(layers="last_two").load(tmp_path / "run")


def test_a_cancelled_fit_leaves_no_artefact(tmp_path: Path) -> None:
    training, boxes = _records(tmp_path / "images", 2)

    class CancelDuringTraining(NullReporter):
        def __init__(self) -> None:
            self.training = False

        def progress(self, fraction: float, message: str | None = None) -> None:
            self.training = self.training or (message or "").startswith("epoch")

        def should_cancel(self) -> bool:
            return self.training

    run = tmp_path / "run"
    train_ctx, _ = _contexts(run, boxes, CancelDuringTraining())
    model = _model(epochs=5)
    with pytest.raises(ModelCancelledError):
        model.fit(training, train_ctx)
    with pytest.raises(RuntimeError, match="never fitted"):
        model.save(run)
    assert not any(run.glob("dino_linear_*"))
    with pytest.raises(RuntimeError, match="before it was fitted"):
        model.predict(training, _contexts(run, None)[1])


def test_it_refuses_to_fit_without_boxes(tmp_path: Path) -> None:
    training, _ = _records(tmp_path / "images", 1)
    train_ctx, _ = _contexts(tmp_path / "run", None)
    with pytest.raises(RuntimeError, match="dino_linear_det detects annotated classes"):
        _model().fit(training, train_ctx)
