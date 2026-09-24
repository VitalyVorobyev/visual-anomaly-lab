"""`proto_seg` against a seeded random ViT: references in, a map out under each axis, the
debias rank it resolved, the seed in both directions, and a round trip that scores
identically. Accuracy needs pretrained
weights and belongs to the public gate, not here."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("timm")

from PIL import Image

from anomaly_lab.models.base import (
    Device,
    ImageRecord,
    InferContext,
    NullReporter,
    TrainContext,
)
from anomaly_lab.models.calibration import Calibration
from anomaly_lab.models.diagnostics import DiagnosticWriter
from anomaly_lab.models.dino_backbone import DinoBackbone
from anomaly_lab.models.preprocessing import PreprocessingConfig
from anomaly_lab.models.proto_seg import Adaptation, ProtoSegConfig, ProtoSegModel

SIZE = 112


class _Targets:
    label_key = "stain"

    def __init__(self, masks: dict[int, np.ndarray]) -> None:
        self._masks = masks

    def mask(self, image_id: int) -> np.ndarray:
        return self._masks[image_id]


def _records(
    tmp_path: Path, count: int, first_id: int = 1
) -> tuple[list[ImageRecord], dict[int, np.ndarray]]:
    records: list[ImageRecord] = []
    masks: dict[int, np.ndarray] = {}
    for image_id in range(first_id, first_id + count):
        rng = np.random.default_rng(image_id)
        pixels = rng.integers(0, 80, (SIZE, SIZE, 3)).astype(np.uint8)
        mask = np.zeros((SIZE, SIZE), dtype=bool)
        top = 16 + 8 * (image_id % 4)
        mask[top : top + 48, top : top + 48] = True
        pixels[mask] = (220, 40, 40)
        path = tmp_path / f"{image_id}.png"
        Image.fromarray(pixels).save(path)
        records.append(ImageRecord(image_id=image_id, sample_id=image_id, path=path))
        masks[image_id] = mask
    return records, masks


def _contexts(tmp_path: Path, targets: _Targets | None) -> tuple[TrainContext, InferContext]:
    common: dict[str, Any] = {
        "artifact_dir": tmp_path,
        "cache_dir": tmp_path / "cache",
        "preprocessing": PreprocessingConfig(width=SIZE, height=SIZE),
        "device": Device.CPU,
        "reporter": NullReporter(),
        "diagnostics": DiagnosticWriter(tmp_path / "diagnostics", enabled=False),
    }
    return TrainContext(**common, targets=targets), InferContext(**common)


def _model(seed: int = 0, **overrides: Any) -> ProtoSegModel:
    return ProtoSegModel(
        ProtoSegConfig(
            backbone=DinoBackbone.DINOV3_VIT_S16,
            pretrained_backbone=False,
            allow_downloads=False,
            seed=seed,
            **overrides,
        )
    )


def _run(
    model: ProtoSegModel,
    tmp_path: Path,
    queries: Sequence[ImageRecord],
    masks: dict[int, np.ndarray],
    references: Sequence[ImageRecord],
) -> list[float]:
    train_ctx, infer_ctx = _contexts(tmp_path, _Targets(masks))
    model.fit(references, train_ctx)
    return [prediction.score for prediction in model.predict(queries, infer_ctx)]


@pytest.mark.parametrize(
    "overrides",
    [
        {},
        {"adaptation": Adaptation.LINEAR_ADAPT, "refine": "bilinear"},
        {"positional_debias": False, "clusters_per_class": 0},
    ],
)
def test_references_in_a_probability_map_out(tmp_path: Path, overrides: dict[str, Any]) -> None:
    references, masks = _records(tmp_path, 2)
    queries, _ = _records(tmp_path, 1, first_id=10)
    train_ctx, infer_ctx = _contexts(tmp_path, _Targets(masks))
    model = _model(**overrides)
    model.fit(references, train_ctx)
    (prediction,) = model.predict(queries, infer_ctx)

    probability = np.load(infer_ctx.map_path(10))
    assert probability.shape == (SIZE, SIZE)
    assert probability.min() >= 0.0 and probability.max() <= 1.0
    assert 0.0 <= prediction.score <= 1.0


def test_the_resolved_debias_rank_is_logged(tmp_path: Path) -> None:
    references, masks = _records(tmp_path, 1)
    logged: list[str] = []

    class Recorder(NullReporter):
        def log(self, message: str, level: str = "info") -> None:
            logged.append(message)

    train_ctx, _ = _contexts(tmp_path, _Targets(masks))
    train_ctx.reporter = Recorder()
    _model().fit(references, train_ctx)
    # A 7x7 grid of a 768-wide (two-layer) ViT-S: at most half of 49 directions.
    assert any("removing 24 of 768 directions" in line for line in logged), logged


def test_one_seed_is_one_answer_and_another_seed_is_another(tmp_path: Path) -> None:
    references, masks = _records(tmp_path, 2)
    queries, _ = _records(tmp_path, 2, first_id=10)
    first = _run(_model(0), tmp_path / "a", queries, masks, references)
    again = _run(_model(0), tmp_path / "b", queries, masks, references)
    other = _run(_model(1), tmp_path / "c", queries, masks, references)
    assert first == again
    assert first != other


def test_a_saved_model_scores_identically_after_loading(tmp_path: Path) -> None:
    references, masks = _records(tmp_path, 2)
    queries, _ = _records(tmp_path, 1, first_id=10)
    model = _model()
    before = _run(model, tmp_path, queries, masks, references)
    model.save(tmp_path)

    restored = _model()
    restored.load(tmp_path)
    _, infer_ctx = _contexts(tmp_path, None)
    after = [prediction.score for prediction in restored.predict(queries, infer_ctx)]
    assert after == before


def test_it_refuses_to_fit_without_masks(tmp_path: Path) -> None:
    references, _ = _records(tmp_path, 1)
    train_ctx, _ = _contexts(tmp_path, None)
    with pytest.raises(RuntimeError, match="needs references with masks"):
        _model().fit(references, train_ctx)


class _Recorder(NullReporter):
    def __init__(self) -> None:
        self.lines: list[str] = []

    def log(self, message: str, level: str = "info") -> None:
        self.lines.append(message)


@pytest.mark.parametrize("adaptation", list(Adaptation))
def test_leave_one_out_rescales_the_same_bank(tmp_path: Path, adaptation: Adaptation) -> None:
    references, masks = _records(tmp_path, 3)
    queries, _ = _records(tmp_path, 1, first_id=10)
    plain_dir, calibrated_dir = tmp_path / "plain", tmp_path / "calibrated"
    plain_dir.mkdir()
    calibrated_dir.mkdir()
    train_ctx, plain_ctx = _contexts(plain_dir, _Targets(masks))
    _, calibrated_ctx = _contexts(calibrated_dir, _Targets(masks))
    recorder = _Recorder()
    train_ctx.reporter = recorder

    plain = _model(adaptation=adaptation)
    plain.fit(references, train_ctx)
    calibrated = _model(adaptation=adaptation, calibration=Calibration.LEAVE_ONE_OUT)
    calibrated.fit(references, train_ctx)
    assert any("leave-one-out over 3 references" in line for line in recorder.lines), recorder.lines
    scale = calibrated._calibration
    assert not scale.is_identity

    (before,) = plain.predict(queries, plain_ctx)
    (after,) = calibrated.predict(queries, calibrated_ctx)
    np.testing.assert_allclose(
        np.load(calibrated_ctx.map_path(10)),
        scale.apply(np.load(plain_ctx.map_path(10))),
        rtol=1e-5,
        atol=1e-6,
    )
    assert after.score == pytest.approx(scale.apply_score(before.score))

    calibrated.save(calibrated_dir)
    restored = _model(adaptation=adaptation, calibration=Calibration.LEAVE_ONE_OUT)
    restored.load(calibrated_dir)
    assert [p.score for p in restored.predict(queries, calibrated_ctx)] == [after.score]


def test_calibration_is_seeded_in_both_directions(tmp_path: Path) -> None:
    references, masks = _records(tmp_path, 3)
    queries, _ = _records(tmp_path, 2, first_id=10)
    loo = {"calibration": Calibration.LEAVE_ONE_OUT}
    first = _run(_model(0, **loo), tmp_path / "a", queries, masks, references)
    again = _run(_model(0, **loo), tmp_path / "b", queries, masks, references)
    other = _run(_model(1, **loo), tmp_path / "c", queries, masks, references)
    assert first == again
    assert first != other


def test_one_reference_is_left_unscaled_and_says_so(tmp_path: Path) -> None:
    references, masks = _records(tmp_path, 1)
    queries, _ = _records(tmp_path, 1, first_id=10)
    recorder = _Recorder()
    train_ctx, infer_ctx = _contexts(tmp_path, _Targets(masks))
    train_ctx.reporter = recorder
    calibrated = _model(calibration=Calibration.LEAVE_ONE_OUT)
    calibrated.fit(references, train_ctx)
    assert any("cannot be left out" in line for line in recorder.lines), recorder.lines
    plain = _run(_model(), tmp_path / "plain", queries, masks, references)
    assert [p.score for p in calibrated.predict(queries, infer_ctx)] == plain
