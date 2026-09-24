"""`color_prototype`, the torch-free few-shot floor — and, through it, the whole few-shot
slice in the torch-free job: split, create, train on references, score, evaluate."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from anomaly_lab.config import Settings
from anomaly_lab.domain.entities import JobKind
from anomaly_lab.models.base import (
    Device,
    ImageRecord,
    InferContext,
    NullReporter,
    TrainContext,
)
from anomaly_lab.models.color_prototype import (
    ColorPrototypeConfig,
    ColorPrototypeModel,
    srgb_to_lab,
)
from anomaly_lab.models.diagnostics import DiagnosticWriter
from anomaly_lab.models.preprocessing import PreprocessingConfig

from .conftest import Fixture, create_experiment, run_handler

SIZE = 16


class _Targets:
    label_key = "stain"

    def __init__(self, masks: dict[int, np.ndarray]) -> None:
        self._masks = masks

    def mask(self, image_id: int) -> np.ndarray:
        return self._masks[image_id]


def _image(path: Path, square: tuple[int, int] | None) -> np.ndarray:
    """Blue with a little noise, and a red 5x5 square where asked."""
    rng = np.random.default_rng(len(str(path)))
    pixels = np.zeros((SIZE, SIZE, 3), dtype=np.float64)
    pixels[..., 2] = 180 + rng.normal(0, 4, (SIZE, SIZE))
    mask = np.zeros((SIZE, SIZE), dtype=bool)
    if square is not None:
        y, x = square
        pixels[y : y + 5, x : x + 5] = (200, 30, 30)
        mask[y : y + 5, x : x + 5] = True
    Image.fromarray(np.clip(pixels, 0, 255).astype(np.uint8)).save(path)
    return mask


def _contexts(tmp_path: Path, targets: _Targets | None) -> tuple[TrainContext, InferContext]:
    common: dict[str, Any] = {
        "artifact_dir": tmp_path,
        "cache_dir": tmp_path,
        "preprocessing": PreprocessingConfig(width=SIZE, height=SIZE),
        "device": Device.CPU,
        "reporter": NullReporter(),
        "diagnostics": DiagnosticWriter(tmp_path / "diagnostics", enabled=False),
    }
    return TrainContext(**common, targets=targets), InferContext(**common)


def _records(
    tmp_path: Path, squares: Sequence[tuple[int, int] | None], *, first_id: int = 1
) -> tuple[list[ImageRecord], dict[int, np.ndarray]]:
    records: list[ImageRecord] = []
    masks: dict[int, np.ndarray] = {}
    for image_id, square in enumerate(squares, start=first_id):
        path = tmp_path / f"{image_id}.png"
        masks[image_id] = _image(path, square)
        records.append(ImageRecord(image_id=image_id, sample_id=image_id, path=path))
    return records, masks


def _map(ctx: InferContext, image_id: int) -> np.ndarray:
    return np.asarray(np.load(ctx.map_path(image_id)), dtype=np.float32)


def test_it_finds_the_class_by_colour_and_scores_its_presence(tmp_path: Path) -> None:
    references, masks = _records(tmp_path, [(2, 2), (8, 9)])
    queries, query_masks = _records(tmp_path, [(10, 3), None], first_id=11)

    train_ctx, infer_ctx = _contexts(tmp_path, _Targets(masks))
    model = ColorPrototypeModel(ColorPrototypeConfig(smoothing_sigma=0.0))
    model.fit(references, train_ctx)
    present, absent = model.predict(queries, infer_ctx)

    probability = _map(infer_ctx, 11)
    region = query_masks[11]
    assert probability[region].min() > 0.9
    assert probability[~region].max() < 0.1
    assert present.score > 0.9 > 0.1 > absent.score
    assert _map(infer_ctx, 12).max() < 0.1


def test_the_fit_is_deterministic_and_survives_a_round_trip(tmp_path: Path) -> None:
    references, masks = _records(tmp_path, [(2, 2), (8, 9)])
    train_ctx, infer_ctx = _contexts(tmp_path, _Targets(masks))

    first = ColorPrototypeModel(ColorPrototypeConfig())
    first.fit(references, train_ctx)
    second = ColorPrototypeModel(ColorPrototypeConfig())
    second.fit(references, train_ctx)
    first.save(tmp_path)
    loaded = ColorPrototypeModel(ColorPrototypeConfig())
    loaded.load(tmp_path)

    # There is no seed to vary: pixels are sampled with `evenly_spaced`, so the same
    # references always give the same models.
    scores = [model.predict(references, infer_ctx)[0].score for model in (first, second, loaded)]
    assert scores[0] == scores[1] == scores[2]


def test_the_pixel_cap_samples_evenly_and_says_so(tmp_path: Path) -> None:
    references, masks = _records(tmp_path, [(2, 2)])
    logged: list[str] = []

    class Recorder(NullReporter):
        def log(self, message: str, level: str = "info") -> None:
            logged.append(message)

    train_ctx, _ = _contexts(tmp_path, _Targets(masks))
    train_ctx.reporter = Recorder()
    ColorPrototypeModel(ColorPrototypeConfig(max_pixels_per_class=100)).fit(references, train_ctx)
    assert any("100 of 231 reference pixels" in line for line in logged)


def test_it_refuses_to_fit_without_masks_or_without_the_class(tmp_path: Path) -> None:
    references, masks = _records(tmp_path, [None])
    no_targets, _ = _contexts(tmp_path, None)
    with pytest.raises(RuntimeError, match="needs references with masks"):
        ColorPrototypeModel(ColorPrototypeConfig()).fit(references, no_targets)
    blank, _ = _contexts(tmp_path, _Targets(masks))
    with pytest.raises(RuntimeError, match="no pixel of the class for 'stain'"):
        ColorPrototypeModel(ColorPrototypeConfig()).fit(references, blank)


def test_lab_puts_white_at_lightness_100() -> None:
    lab = srgb_to_lab(np.array([[1.0, 1.0, 1.0], [0.0, 0.0, 0.0]]))
    assert lab[0] == pytest.approx([100.0, 0.0, 0.0], abs=0.1)
    assert lab[1] == pytest.approx([0.0, 0.0, 0.0], abs=0.1)


def test_the_whole_few_shot_slice_runs_without_torch(
    client: TestClient, settings: Settings, seeded: Fixture
) -> None:
    create_experiment(client, seeded)  # builds the region profile the run pins
    split = client.post(
        "/api/splits",
        json={
            "dataset_id": seeded.dataset_id,
            "name": "one shot",
            "seed": 0,
            "params": {"strategy": "few_shot", "label_key": "defect", "shots": 1},
        },
    )
    assert split.status_code == 200, split.text
    created = create_experiment(
        client,
        seeded,
        name="colour floor",
        split_id=split.json()["id"],
        model_type="color_prototype",
        task="few_shot_segmentation",
        target_label="defect",
        config={},
    )
    assert created["target_label"] == "defect"

    trained = run_handler(settings, JobKind.TRAIN, {"experiment_id": created["id"]})
    assert trained["train_images"] == 1
    scored = run_handler(settings, JobKind.INFER, {"experiment_id": created["id"]})
    metrics = scored["metrics"]["test"]
    assert metrics["images"]["present"] == 2
    assert metrics["images"]["absent"] == 11
    assert metrics["foreground_iou"] > 0.5
    assert metrics["image_presence_roc_auc"] == 1.0

    detail = client.get(f"/api/experiments/{created['id']}").json()
    assert detail["headline_metric"] == "foreground_iou"
    assert detail["headline_value"] == metrics["foreground_iou"]
    outcomes = client.get(
        f"/api/experiments/{created['id']}/segmentation-outcomes", params={"subset": "test"}
    )
    assert outcomes.status_code == 200, outcomes.text
    tally = [verdict["outcome"] for verdict in outcomes.json()["samples"]]
    assert len(tally) == 13 and "unlabeled" not in tally

    anomaly_id = client.get("/api/experiments").json()[-1]["id"]
    refused = client.get(f"/api/experiments/{anomaly_id}/segmentation-outcomes")
    assert refused.status_code == 409

    # Compare reads anomaly runs at thresholds; a few-shot run is refused by name.
    compared = client.get("/api/compare", params={"ids": [anomaly_id, created["id"]]})
    assert compared.status_code == 422
    assert "/api/compare/few-shot" in compared.text


def test_few_shot_runs_of_one_class_compare_across_reference_draws(
    client: TestClient, settings: Settings, seeded: Fixture
) -> None:
    anomaly = create_experiment(client, seeded)
    runs = []
    for shots, seed in ((1, 0), (2, 5)):
        split = client.post(
            "/api/splits",
            json={
                "dataset_id": seeded.dataset_id,
                "name": f"{shots} shots seed {seed}",
                "seed": seed,
                "params": {"strategy": "few_shot", "label_key": "defect", "shots": shots},
            },
        ).json()
        run = create_experiment(
            client,
            seeded,
            name=f"colour {shots}",
            split_id=split["id"],
            model_type="color_prototype",
            task="few_shot_segmentation",
            target_label="defect",
            config={},
        )
        run_handler(settings, JobKind.TRAIN, {"experiment_id": run["id"]})
        run_handler(settings, JobKind.INFER, {"experiment_id": run["id"]})
        runs.append(run["id"])

    report = client.get("/api/compare/few-shot", params={"ids": runs})
    assert report.status_code == 200, report.text
    body = report.json()
    assert body["target_label"] == "defect"
    assert [run["references"] for run in body["runs"]] == [1, 2]
    assert [run["seed"] for run in body["runs"]] == [0, 5]
    assert all(run["metrics"]["foreground_iou"] is not None for run in body["runs"])
    # Queries both runs scored: everything but the union of their references.
    assert 14 - 3 <= len(body["samples"]) <= 14 - 2
    assert all(len(sample["outcomes"]) == 2 for sample in body["samples"])

    mixed = client.get("/api/compare/few-shot", params={"ids": [runs[0], anomaly["id"]]})
    assert mixed.status_code == 422
    assert "not a few-shot one" in mixed.text
