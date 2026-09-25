"""`dino_linear_seg` against a seeded random ViT: labelled images in, a label map out that finds
the classes, the seed in both directions, a round trip that segments identically, and a
cancelled fit that leaves nothing behind. Accuracy on real data needs pretrained weights and
belongs to the public gate, not here."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("timm")

from PIL import Image

from anomaly_lab.map_files import read_map
from anomaly_lab.models.base import (
    IGNORE_INDEX,
    Device,
    ImageRecord,
    InferContext,
    ModelCancelledError,
    NullReporter,
    ProgressReporter,
    TrainContext,
)
from anomaly_lab.models.diagnostics import DiagnosticWriter
from anomaly_lab.models.dino_backbone import DinoBackbone
from anomaly_lab.models.dino_linear_seg import DinoLinearSegConfig, DinoLinearSegModel
from anomaly_lab.models.preprocessing import PreprocessingConfig

SIZE = 64
RED, GREEN = (220, 40, 40), (40, 200, 60)


class _Labels:
    classes = ("rust", "moss")

    def __init__(self, labels: dict[int, np.ndarray]) -> None:
        self._labels = labels

    def labels(self, image_id: int) -> np.ndarray:
        return self._labels[image_id]


def _records(
    folder: Path, count: int, first_id: int = 1
) -> tuple[list[ImageRecord], dict[int, np.ndarray]]:
    """Dark noise, a red block (`rust`) and a striped green block (`moss`), placed by image."""
    folder.mkdir(parents=True, exist_ok=True)
    records: list[ImageRecord] = []
    labels: dict[int, np.ndarray] = {}
    for image_id in range(first_id, first_id + count):
        rng = np.random.default_rng(image_id)
        pixels = rng.integers(0, 60, (SIZE, SIZE, 3)).astype(np.uint8)
        truth = np.zeros((SIZE, SIZE), dtype=np.uint8)
        shift = 16 * (image_id % 2)
        pixels[shift : shift + 32, 0:32] = RED
        truth[shift : shift + 32, 0:32] = 1
        pixels[32 - shift : 64 - shift, 32:64] = GREEN
        pixels[32 - shift : 64 - shift : 4, 32:64] = (230, 230, 230)
        truth[32 - shift : 64 - shift, 32:64] = 2
        path = folder / f"{image_id}.png"
        Image.fromarray(pixels).save(path)
        records.append(ImageRecord(image_id=image_id, sample_id=image_id, path=path))
        labels[image_id] = truth
    return records, labels


def _contexts(
    artifact_dir: Path,
    labels: dict[int, np.ndarray] | None,
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
    targets = None if labels is None else _Labels(labels)
    return TrainContext(**common, label_targets=targets), InferContext(**common)


def _model(seed: int = 0, **overrides: Any) -> DinoLinearSegModel:
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
    return DinoLinearSegModel(DinoLinearSegConfig(**settings))


def _stored(ctx: InferContext, image_id: int) -> np.ndarray:
    with Image.open(ctx.label_map_path(image_id)) as opened:
        return np.asarray(opened).copy()


def _segment(
    model: DinoLinearSegModel, artifact_dir: Path, images: list[ImageRecord]
) -> list[np.ndarray]:
    _, infer_ctx = _contexts(artifact_dir, None)
    model.predict(images, infer_ctx)
    return [_stored(infer_ctx, record.image_id) for record in images]


@pytest.mark.parametrize("overrides", [{}, {"refine": "guided", "class_balancing": "none"}])
def test_labelled_images_in_a_label_map_that_finds_the_classes_out(
    tmp_path: Path, overrides: dict[str, Any]
) -> None:
    training, labels = _records(tmp_path / "images", 4)
    queries, truth = _records(tmp_path / "images", 2, first_id=11)
    train_ctx, infer_ctx = _contexts(tmp_path / "run", labels)
    model = _model(**overrides)
    model.fit(training, train_ctx)
    predictions = model.predict(queries, infer_ctx)

    for prediction, record in zip(predictions, queries, strict=True):
        predicted = _stored(infer_ctx, record.image_id)
        assert predicted.shape == (SIZE, SIZE)
        assert set(np.unique(predicted)) <= {0, 1, 2}
        accuracy = float(np.mean(predicted == truth[record.image_id]))
        assert accuracy > 0.8, accuracy
        assert prediction.label_map == infer_ctx.label_map_path(record.image_id)
        assert prediction.score == pytest.approx(float(np.mean(predicted > 0)))
        foreground = read_map(infer_ctx.map_path(record.image_id))
        assert foreground.shape == (SIZE, SIZE)
        assert foreground.min() >= 0.0 and foreground.max() <= 1.0


def test_the_plan_is_logged_before_encoding_and_ignored_pixels_are_never_learned(
    tmp_path: Path,
) -> None:
    training, labels = _records(tmp_path / "images", 2)
    for truth in labels.values():
        truth[truth == 2] = IGNORE_INDEX
    events: list[str] = []

    class Recorder(NullReporter):
        def log(self, message: str, level: str = "info") -> None:
            events.append(message)

        def progress(self, fraction: float, message: str | None = None) -> None:
            events.append(f"progress {message}")

    train_ctx, _ = _contexts(tmp_path / "run", labels, Recorder())
    model = _model(epochs=2, pixels_per_image=64)
    model.fit(training, train_ctx)
    plan = next(index for index, line in enumerate(events) if line.startswith("pixel plan"))
    encoded = next(index for index, line in enumerate(events) if "encoded 1/2" in line)
    assert plan < encoded
    assert "at most 64 labelled pixels from each" in events[plan]
    assert any("no training pixel of 'moss'" in line for line in events)
    assert "pixel sampling: per_class" in events
    # What each class got is logged against what the chosen images held.
    trained = next(line for line in events if line.startswith("training the head on"))
    assert "background" in trained and " of " in trained

    queries, _ = _records(tmp_path / "images", 1, first_id=11)
    (predicted,) = _segment(model, tmp_path / "run", queries)
    assert 2 not in np.unique(predicted)


@pytest.mark.parametrize("overrides", [{}, {"pixel_sampling": "raster", "logit_bias": "none"}])
def test_one_seed_is_one_answer_and_another_seed_is_another(
    tmp_path: Path, overrides: dict[str, Any]
) -> None:
    training, labels = _records(tmp_path / "images", 2)
    queries, _ = _records(tmp_path / "images", 2, first_id=11)
    fitted: list[tuple[np.ndarray, list[np.ndarray]]] = []
    for name, seed in (("a", 0), ("b", 0), ("c", 1)):
        train_ctx, _ = _contexts(tmp_path / name, labels)
        model = _model(seed, epochs=3, **overrides)
        model.fit(training, train_ctx)
        state = np.concatenate(
            [model._state["weights"].ravel(), model._state.get("held_out_bias", np.zeros(0))]
        )
        fitted.append((state, _segment(model, tmp_path / name, queries)))
    (first, first_maps), (again, again_maps), (other, _) = fitted
    assert np.array_equal(first, again)
    assert all(np.array_equal(a, b) for a, b in zip(first_maps, again_maps, strict=True))
    assert not np.array_equal(first, other)


def test_restoring_the_prior_reads_the_same_head_and_gives_up_only_foreground(
    tmp_path: Path,
) -> None:
    """The bias is applied at prediction: one head, several readings. Both foreground classes
    cover a quarter of every frame and background half, so under `inverse_frequency` each is
    shifted by the same `log(1/2)` and the corrected map can only hand pixels to background."""
    training, labels = _records(tmp_path / "images", 4)
    queries, truth = _records(tmp_path / "images", 2, first_id=11)
    readings: dict[str, tuple[np.ndarray, list[np.ndarray]]] = {}
    for choice in ("none", "training_prior", "held_out_iou"):
        train_ctx, _ = _contexts(tmp_path / choice, labels)
        model = _model(epochs=3, logit_bias=choice)
        model.fit(training, train_ctx)
        shift = model._state["prior_shift"]
        assert shift[0] == 0.0
        np.testing.assert_allclose(shift[1:], np.log(0.5), rtol=1e-4)
        assert ("held_out_bias" in model._state) is (choice == "held_out_iou")
        readings[choice] = (
            model._state["weights"].copy(),
            _segment(model, tmp_path / choice, queries),
        )
    plain, plain_maps = readings["none"]
    corrected, corrected_maps = readings["training_prior"]
    fitted, fitted_maps = readings["held_out_iou"]
    assert np.array_equal(plain, corrected) and np.array_equal(plain, fitted)
    for before, after in zip(plain_maps, corrected_maps, strict=True):
        assert np.all((after > 0) <= (before > 0))
        assert np.all(after[after > 0] == before[after > 0])
    # Fitted for IoU on held-out folds, the classes are still found.
    for predicted, record in zip(fitted_maps, queries, strict=True):
        assert float(np.mean(predicted == truth[record.image_id])) > 0.8


def test_a_saved_model_segments_identically_after_loading(tmp_path: Path) -> None:
    training, labels = _records(tmp_path / "images", 2)
    queries, _ = _records(tmp_path / "images", 2, first_id=11)
    train_ctx, _ = _contexts(tmp_path / "run", labels)
    model = _model(epochs=3, logit_bias="none")
    model.fit(training, train_ctx)
    before = _segment(model, tmp_path / "run", queries)
    model.save(tmp_path / "run")
    assert sorted(path.name for path in (tmp_path / "run").glob("dino_linear_seg*")) == [
        "dino_linear_seg.json",
        "dino_linear_seg.npz",
    ]

    restored = _model(epochs=3, logit_bias="none")
    restored.load(tmp_path / "run")
    after = _segment(restored, tmp_path / "loaded", queries)
    assert all(np.array_equal(a, b) for a, b in zip(before, after, strict=True))

    with pytest.raises(RuntimeError, match="fitted with dinov3_vit_s16 \\(last\\)"):
        _model(layers="last_two").load(tmp_path / "run")

    # The prior is saved with the head, so the corrected reading survives the round trip too.
    corrected = _model(epochs=3, logit_bias="training_prior")
    corrected.load(tmp_path / "run")
    first = _segment(corrected, tmp_path / "corrected-a", queries)
    again = _model(epochs=3, logit_bias="training_prior")
    again.load(tmp_path / "run")
    assert all(
        np.array_equal(a, b)
        for a, b in zip(first, _segment(again, tmp_path / "corrected-b", queries), strict=True)
    )
    # The held-out constants are fitted only when asked for, so this head cannot be read so.
    folded = _model(epochs=3, logit_bias="held_out_iou")
    folded.load(tmp_path / "run")
    with pytest.raises(RuntimeError, match="not fitted with logit_bias 'held_out_iou'"):
        _segment(folded, tmp_path / "corrected-c", queries)


def test_a_head_fitted_for_held_out_iou_segments_identically_after_loading(
    tmp_path: Path,
) -> None:
    training, labels = _records(tmp_path / "images", 4)
    queries, _ = _records(tmp_path / "images", 2, first_id=11)
    train_ctx, _ = _contexts(tmp_path / "run", labels)
    model = _model(epochs=3, logit_bias="held_out_iou")
    model.fit(training, train_ctx)
    before = _segment(model, tmp_path / "run", queries)
    model.save(tmp_path / "run")
    restored = _model(epochs=3, logit_bias="held_out_iou")
    restored.load(tmp_path / "run")
    np.testing.assert_array_equal(restored._state["held_out_bias"], model._state["held_out_bias"])
    after = _segment(restored, tmp_path / "loaded", queries)
    assert all(np.array_equal(a, b) for a, b in zip(before, after, strict=True))


def test_a_cancelled_fit_leaves_no_artefact(tmp_path: Path) -> None:
    training, labels = _records(tmp_path / "images", 2)

    class CancelDuringTraining(NullReporter):
        def __init__(self) -> None:
            self.training = False

        def progress(self, fraction: float, message: str | None = None) -> None:
            self.training = self.training or (message or "").startswith("epoch")

        def should_cancel(self) -> bool:
            return self.training

    run = tmp_path / "run"
    train_ctx, _ = _contexts(run, labels, CancelDuringTraining())
    model = _model(epochs=5)
    with pytest.raises(ModelCancelledError):
        model.fit(training, train_ctx)
    assert not any(run.glob("dino_linear_seg*"))
    with pytest.raises(RuntimeError, match="never fitted"):
        model.save(run)
    assert not any(run.glob("dino_linear_seg*"))


def test_it_refuses_to_fit_without_labels(tmp_path: Path) -> None:
    training, _ = _records(tmp_path / "images", 1)
    train_ctx, _ = _contexts(tmp_path / "run", None)
    with pytest.raises(RuntimeError, match="needs label targets"):
        _model().fit(training, train_ctx)
