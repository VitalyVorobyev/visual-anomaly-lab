"""Object detection's result screens (ADR-0039): the confidence cut each subset resolves by
one printed rule (ADR-0028), each image's boxes matched at it, each sample's verdict, and the
routes that serve them — hand-computed cases first, then the torch-free slice end to end."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient

from anomaly_lab.config import Settings
from anomaly_lab.domain.entities import JobKind
from anomaly_lab.eval.detection import (
    CUT_RULE,
    UNRESOLVED_RULE,
    DetectionAccumulator,
    _outcome,
    match_image,
    resolve_cut,
)
from anomaly_lab.models.base import PredictedInstance, TargetBox

from .conftest import Fixture, create_experiment, run_handler
from .test_color_detector import _sample_of, _supervised_split


def test_the_cut_maximises_f1_at_iou_half() -> None:
    """TP, FP, TP, FP over two truth boxes: F1 is 2/3, 1/2, 4/5, 2/3 as each is kept, so the
    cut is the third confidence, with precision 2/3 and recall 1."""
    cut = resolve_cut(np.array([0.9, 0.8, 0.7, 0.6]), np.array([True, False, True, False]), truth=2)
    assert cut.value == 0.7
    assert cut.f1 == pytest.approx(0.8)
    assert cut.precision == pytest.approx(2 / 3)
    assert cut.recall == 1.0


def test_tied_confidences_are_kept_or_dropped_together() -> None:
    # Keeping only the matched one of the two at 0.5 would give F1 1.0, but no cut can.
    cut = resolve_cut(np.array([0.5, 0.9, 0.5]), np.array([True, True, False]), truth=2)
    assert cut.value == 0.5
    assert cut.f1 == pytest.approx(0.8)


def test_no_truth_or_no_detection_resolves_no_cut() -> None:
    assert resolve_cut(np.array([0.9]), np.array([False]), truth=0).value is None
    assert resolve_cut(np.array([]), np.array([], dtype=bool), truth=3).value is None


def test_the_evaluator_stores_the_cut_with_its_rule() -> None:
    accumulator = DetectionAccumulator(("scratch", "stain"))
    accumulator.add(
        [TargetBox("scratch", (0, 0, 10, 10)), TargetBox("stain", (20, 20, 30, 30))],
        [
            PredictedInstance("scratch", (0, 0, 10, 10), 0.9),
            PredictedInstance("stain", (50, 50, 60, 60), 0.8),
            PredictedInstance("stain", (20, 20, 30, 30), 0.7),
            PredictedInstance("scratch", (70, 70, 80, 80), 0.6),
        ],
        inference_ms=0.0,
    )
    metrics = accumulator.metrics()
    # Pooled across both classes: the same sequence as the first test.
    assert metrics["confidence_cut"] == 0.7
    assert metrics["cut_rule"] == CUT_RULE
    assert metrics["cut_iou"] == 0.5
    assert metrics["f1_at_cut"] == pytest.approx(0.8)
    assert metrics["recall_at_cut"] == 1.0

    empty = DetectionAccumulator(("scratch",)).metrics()
    assert empty["confidence_cut"] is None and empty["cut_rule"] == UNRESOLVED_RULE


def test_an_image_is_matched_at_the_cut_class_by_class() -> None:
    truth = [
        TargetBox("scratch", (0, 0, 10, 10)),
        TargetBox("scratch", (20, 20, 30, 30)),
        TargetBox("dent", (40, 40, 50, 50)),  # a class the run does not pin
    ]
    predicted = [
        PredictedInstance("scratch", (0, 0, 10, 10), 0.9),
        PredictedInstance("scratch", (60, 60, 70, 70), 0.8),
        PredictedInstance("scratch", (20, 20, 30, 30), 0.3),
        PredictedInstance("stain", (40, 40, 50, 50), 0.95),  # right place, wrong class
    ]
    classes = ("scratch", "stain")

    at_cut = match_image(truth, predicted, classes, 0.5)
    assert at_cut.kept == [True, True, False, True]
    assert at_cut.matched == [True, False, False, False]
    assert at_cut.found == [True, False, False]
    assert (at_cut.truth, at_cut.true_positives, at_cut.false_positives, at_cut.missed) == (
        2,
        1,
        2,
        1,
    )

    # Without a cut every detection counts, and the low one finds the second scratch.
    uncut = match_image(truth, predicted, classes, None)
    assert uncut.found == [True, True, False]
    assert (uncut.missed, uncut.false_positives) == (0, 2)


@pytest.mark.parametrize(
    ("truth", "missed", "false_positives", "outcome"),
    [
        (0, 0, 0, "correct_absence"),
        (0, 0, 2, "false_presence"),
        (2, 0, 0, "hit"),
        (2, 1, 0, "miss"),
        (2, 0, 1, "false_presence"),
        (2, 1, 1, "mixed"),
    ],
)
def test_a_sample_outcome_reads_its_three_counts(
    truth: int, missed: int, false_positives: int, outcome: str
) -> None:
    assert _outcome(truth, missed, false_positives) == outcome


# ---------------------------------------------------------------- the routes, end to end


def _scored_detection(
    client: TestClient,
    settings: Settings,
    seeded: Fixture,
    *,
    split_id: int | None = None,
    name: str = "colour boxes",
) -> dict[str, Any]:
    created = create_experiment(
        client,
        seeded,
        name=name,
        split_id=split_id if split_id is not None else _supervised_split(client, settings, seeded),
        model_type="color_detector",
        task="object_detection",
        config={},
    )
    run_handler(settings, JobKind.TRAIN, {"experiment_id": created["id"]})
    run_handler(settings, JobKind.INFER, {"experiment_id": created["id"]})
    return created


def test_the_routes_serve_boxes_verdicts_and_a_tile_at_one_cut(
    client: TestClient, settings: Settings, seeded: Fixture
) -> None:
    anomaly = create_experiment(client, seeded)
    created = _scored_detection(client, settings, seeded)
    base = f"/api/experiments/{created['id']}"
    detail = client.get(base).json()
    test_metrics = next(entry for entry in detail["metrics"] if entry["subset"] == "test")
    cut = test_metrics["metrics"]["confidence_cut"]
    assert cut is not None

    image_id = seeded.defect_image_ids[-1]
    boxes = client.get(f"{base}/images/{image_id}/boxes")
    assert boxes.status_code == 200, boxes.text
    body = boxes.json()
    assert body["subset"] == "test" and body["confidence_cut"] == cut
    assert body["threshold_rule"].startswith("confidence ≥") and CUT_RULE in body["threshold_rule"]
    assert body["classes"] == ["defect"] and body["iou_threshold"] == 0.5
    assert body["truth"] == [
        {"label_key": "defect", "class_index": 0, "box": [5.0, 5.0, 10.0, 10.0], "found": True}
    ]
    predictions = body["predictions"]
    assert predictions and len(predictions) <= 100
    confidences = [entry["confidence"] for entry in predictions]
    assert confidences == sorted(confidences, reverse=True)
    assert all(entry["kept"] == (entry["confidence"] >= cut) for entry in predictions)
    assert any(entry["matched"] for entry in predictions)

    # Unscored (a training image), another task, and an unknown run say so.
    assert client.get(f"{base}/images/{seeded.defect_image_ids[0]}/boxes").status_code == 404
    refused = client.get(f"/api/experiments/{anomaly['id']}/images/{image_id}/boxes")
    assert refused.status_code == 409

    outcomes = client.get(f"{base}/detection-outcomes", params={"subset": "test"})
    assert outcomes.status_code == 200, outcomes.text
    report = outcomes.json()
    assert report["confidence_cut"] == cut and report["iou_threshold"] == 0.5
    assert report["threshold_rule"] == body["threshold_rule"]
    ranked = client.get(f"{base}/results", params={"subset": "test"}).json()["samples"]
    assert len(report["samples"]) == len(ranked)
    allowed = {"hit", "miss", "false_presence", "mixed", "correct_absence", "unlabeled"}
    assert {entry["outcome"] for entry in report["samples"]} <= allowed
    this = next(
        entry for entry in report["samples"] if entry["sample_id"] == _sample_of(settings, image_id)
    )
    assert this["matched"] == 1 and this["missed"] == 0
    scores = [entry["score"] for entry in report["samples"]]
    assert scores == sorted(scores, reverse=True)
    assert client.get(f"/api/experiments/{anomaly['id']}/detection-outcomes").status_code == 409

    tile = client.get(
        f"{base}/images/{image_id}/box-map", params={"colours": "34d399,f87171,fbbf24"}
    )
    assert tile.status_code == 200, tile.text
    assert tile.headers["content-type"].startswith("image/svg+xml")
    svg = tile.text
    assert svg.startswith("<svg") and 'stroke-dasharray="4 3"' in svg
    assert 'stroke="rgb(52,211,153)"' in svg  # the found truth and its match, in `match`
    kept = sum(1 for entry in predictions if entry["kept"])
    assert svg.count("<rect") == 1 + kept
    again = client.get(
        f"{base}/images/{image_id}/box-map",
        params={"colours": "34d399,f87171,fbbf24"},
        headers={"If-None-Match": tile.headers["etag"]},
    )
    assert again.status_code == 304
    only_truth = client.get(
        f"{base}/images/{image_id}/box-map",
        params={"colours": "34d399,f87171,fbbf24", "predictions": "false"},
    )
    assert only_truth.text.count("<rect") == 1
    bad = client.get(f"{base}/images/{image_id}/box-map", params={"colours": "34d399"})
    assert bad.status_code == 422


def test_a_run_whose_detections_were_removed_still_draws_its_truth(
    client: TestClient, settings: Settings, seeded: Fixture
) -> None:
    created = _scored_detection(client, settings, seeded)
    image_id = seeded.defect_image_ids[-1]
    stored = Path(created["artifact_dir"]) / "maps" / f"{image_id}.instances.json"
    assert json.loads(stored.read_text())["instances"]
    stored.unlink()
    body = client.get(f"/api/experiments/{created['id']}/images/{image_id}/boxes").json()
    assert body["predictions"] is None
    assert [entry["found"] for entry in body["truth"]] == [False]


def test_compare_puts_detection_runs_side_by_side_without_their_cuts(
    client: TestClient, settings: Settings, seeded: Fixture
) -> None:
    anomaly = create_experiment(client, seeded)
    split_id = _supervised_split(client, settings, seeded)
    first = _scored_detection(client, settings, seeded, split_id=split_id, name="boxes a")
    second = _scored_detection(client, settings, seeded, split_id=split_id, name="boxes b")

    compared = client.get("/api/compare/detection", params={"ids": [first["id"], second["id"]]})
    assert compared.status_code == 200, compared.text
    report = compared.json()
    assert report["subset"] == "test" and report["classes"] == ["defect"]
    assert [run["name"] for run in report["runs"]] == ["boxes a", "boxes b"]
    for run in report["runs"]:
        assert run["scored"] is True
        assert run["metrics"]["ap"] is not None and "per_class_ap" in run["metrics"]
        # Each run's cut is its own and never crosses a column (ADR-0028).
        assert not {"confidence_cut", "f1_at_cut", "cut_rule"} & set(run["metrics"])

    mixed = client.get("/api/compare/detection", params={"ids": [first["id"], anomaly["id"]]})
    assert mixed.status_code == 422 and "not a detection one" in mixed.text
    other = client.post(
        "/api/splits",
        json={
            "dataset_id": seeded.dataset_id,
            "name": "other train",
            "seed": 0,
            "params": {
                "strategy": "manual",
                "sample_ids": [_sample_of(settings, seeded.defect_image_ids[0])],
            },
        },
    )
    assert other.status_code == 200, other.text
    elsewhere = _scored_detection(
        client, settings, seeded, split_id=int(other.json()["id"]), name="boxes c"
    )
    other_split = client.get(
        "/api/compare/detection", params={"ids": [first["id"], elsewhere["id"]]}
    )
    assert other_split.status_code == 422
    refused = client.get("/api/compare", params={"ids": [first["id"], second["id"]]})
    assert refused.status_code == 422 and "/api/compare/detection" in refused.text
