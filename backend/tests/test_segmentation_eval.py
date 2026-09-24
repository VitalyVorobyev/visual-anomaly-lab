"""The few-shot segmentation evaluator (ADR-0040): pooled pixel counts with hand-computed
answers, `None` for what cannot be computed, unlabelled images excluded and counted, a
method's own mask read before its map, and a digest that moves with the truth."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from anomaly_lab.config import Settings
from anomaly_lab.db.connection import connection
from anomaly_lab.db.repositories import experiments as experiments_repo
from anomaly_lab.db.repositories import results as results_repo
from anomaly_lab.domain.entities import ImageResult, Subset, Task
from anomaly_lab.eval.evaluators import evaluator_for
from anomaly_lab.eval.segmentation import (
    BOUNDARY_TOLERANCE_PX,
    THRESHOLD_RULE,
    SegmentationAccumulator,
    boundary,
    sample_outcomes,
)
from anomaly_lab.models.preprocessing import load_mask

from .conftest import FIXTURE_SIZE, Fixture, create_few_shot_experiment


def _square(x0: int, y0: int, size: int, frame: int = 8) -> np.ndarray:
    mask = np.zeros((frame, frame), dtype=bool)
    mask[y0 : y0 + size, x0 : x0 + size] = True
    return mask


def test_pooled_counts_against_a_hand_computed_answer() -> None:
    accumulator = SegmentationAccumulator()
    # 16 true pixels; the prediction is shifted one column: 12 hit, 4 extra, 4 missed.
    accumulator.add(_square(2, 2, 4), _square(3, 2, 4), score=0.9, inference_ms=10.0)
    # A confirmed absence with 4 predicted pixels: false positives, and one flagged image.
    accumulator.add(_square(0, 0, 0), _square(0, 0, 2), score=0.2, inference_ms=30.0)
    metrics = accumulator.metrics()

    assert metrics["foreground_iou"] == pytest.approx(12 / 24)
    assert metrics["foreground_dice"] == pytest.approx(24 / 36)
    assert metrics["image_present_recall"] == 1.0
    assert metrics["image_absent_false_positive_rate"] == 1.0
    assert metrics["image_presence_roc_auc"] == 1.0
    # 16 of 64 pixels is not a small region, so there is nothing to report.
    assert metrics["image_small_region_recall"] is None
    assert metrics["images"] == {"present": 1, "absent": 1, "unlabeled": 0, "without_prediction": 0}
    assert metrics["threshold_rule"] == THRESHOLD_RULE
    assert metrics["boundary_tolerance_px"] == BOUNDARY_TOLERANCE_PX
    assert metrics["timing"]["mean_ms"] == 20.0


def test_what_cannot_be_computed_is_none_not_zero() -> None:
    accumulator = SegmentationAccumulator()
    accumulator.add(_square(2, 2, 4), _square(2, 2, 4), score=0.5, inference_ms=1.0)
    metrics = accumulator.metrics()
    assert metrics["image_absent_false_positive_rate"] is None
    assert metrics["image_presence_roc_auc"] is None  # one class only
    assert metrics["foreground_iou"] == 1.0
    assert metrics["boundary_f1"] == 1.0

    empty = SegmentationAccumulator().metrics()
    assert empty["foreground_iou"] is None
    assert empty["boundary_f1"] is None


def test_a_boundary_is_the_ring_and_a_near_miss_still_matches() -> None:
    ring = boundary(_square(2, 2, 4))
    assert ring.sum() == 12 and not ring[3:5, 3:5].any()

    accumulator = SegmentationAccumulator()
    # A one-pixel shift is inside the tolerance, so every boundary pixel still matches.
    accumulator.add(_square(2, 2, 4, 16), _square(3, 2, 4, 16), score=1.0, inference_ms=1.0)
    assert accumulator.metrics()["boundary_f1"] == 1.0

    far = SegmentationAccumulator()
    far.add(_square(0, 0, 3, 16), _square(10, 10, 3, 16), score=1.0, inference_ms=1.0)
    assert far.metrics()["boundary_f1"] is None  # no boundary pixel matched either way


def test_small_region_recall_counts_only_small_regions() -> None:
    accumulator = SegmentationAccumulator()
    tiny = _square(5, 5, 1, 16)  # 1 of 256 pixels
    accumulator.add(tiny, np.zeros_like(tiny), score=0.1, inference_ms=1.0)
    accumulator.add(_square(0, 0, 8, 16), _square(0, 0, 8, 16), score=0.9, inference_ms=1.0)
    metrics = accumulator.metrics()
    assert metrics["image_small_region_recall"] == 0.0
    assert metrics["image_present_recall"] == 0.5


def _sample_of(settings: Settings, image_id: int) -> int:
    with connection(settings.db_path) as conn:
        return int(
            conn.execute("SELECT sample_id FROM image WHERE id = ?", (image_id,)).fetchone()[0]
        )


def _truth(settings: Settings, image_id: int) -> np.ndarray:
    with connection(settings.db_path) as conn:
        path = conn.execute(
            "SELECT path FROM mask WHERE image_id = ? AND kind = 'ground_truth'", (image_id,)
        ).fetchone()[0]
    return load_mask(Path(path), size=(FIXTURE_SIZE, FIXTURE_SIZE))


def test_the_evaluator_reads_stored_predictions_against_class_truth(
    client: TestClient, settings: Settings, seeded: Fixture
) -> None:
    reference, *queries = seeded.defect_image_ids
    normals = seeded.normal_image_ids
    run_id = create_few_shot_experiment(client, settings, seeded, [_sample_of(settings, reference)])
    with connection(settings.db_path) as conn:
        experiment = experiments_repo.get_experiment(conn, run_id)
        assert experiment is not None
        # One query normal loses its label, so its truth no longer answers for `defect`.
        conn.execute(
            "UPDATE sample SET label = 'unlabeled' WHERE id = ?",
            (_sample_of(settings, normals[2]),),
        )
    maps = Path(experiment.artifact_dir) / "maps"
    maps.mkdir(parents=True, exist_ok=True)

    rows: list[ImageResult] = []
    true_pixels = 0
    for image_id in queries:
        truth = _truth(settings, image_id)
        true_pixels += int(truth.sum())
        np.save(maps / f"{image_id}.npy", truth.astype(np.float32))
        rows.append(
            ImageResult(
                experiment_id=run_id,
                image_id=image_id,
                score=0.9,
                map_path=str(maps / f"{image_id}.npy"),
                inference_ms=5.0,
            )
        )
    # A normal whose map says nothing, and one whose written mask flags four pixels — the
    # mask is read, not the empty map beside it.
    for image_id in normals[:2]:
        np.save(maps / f"{image_id}.npy", np.zeros((FIXTURE_SIZE, FIXTURE_SIZE), np.float32))
        rows.append(
            ImageResult(
                experiment_id=run_id,
                image_id=image_id,
                score=0.1,
                map_path=str(maps / f"{image_id}.npy"),
                inference_ms=5.0,
            )
        )
    flagged = np.zeros((FIXTURE_SIZE, FIXTURE_SIZE), dtype=np.uint8)
    flagged[0:2, 0:2] = 255
    Image.fromarray(flagged).save(maps / f"{normals[1]}.mask.png")
    rows.append(
        ImageResult(
            experiment_id=run_id, image_id=normals[2], score=0.5, map_path=None, inference_ms=5.0
        )
    )

    evaluator = evaluator_for(Task.FEW_SHOT_SEGMENTATION)
    with connection(settings.db_path) as conn:
        results_repo.replace_image_results(conn, run_id, rows)
        metrics = evaluator.evaluate_and_store(conn, experiment)[Subset.TEST]
        stored = results_repo.list_metric_sets(conn, run_id)
        fresh = evaluator.current_digest(conn, experiment, Subset.TEST)

    assert metrics["images"] == {
        "present": len(queries),
        "absent": 2,
        "unlabeled": 1,
        "without_prediction": 0,
    }
    assert metrics["foreground_iou"] == pytest.approx(true_pixels / (true_pixels + 4))
    assert metrics["image_present_recall"] == 1.0
    assert metrics["image_absent_false_positive_rate"] == 0.5
    assert metrics["image_presence_roc_auc"] == 1.0
    assert stored[0].ground_truth_digest == fresh

    with connection(settings.db_path) as conn:
        outcomes = {
            verdict.sample_id: verdict
            for verdict in sample_outcomes(conn, experiment, Subset.TEST).samples
        }
    for image_id in queries:
        found = outcomes[_sample_of(settings, image_id)]
        assert (found.outcome, found.iou) == ("hit", 1.0)
    assert outcomes[_sample_of(settings, normals[0])].outcome == "correct_absence"
    assert outcomes[_sample_of(settings, normals[1])].outcome == "false_presence"
    assert outcomes[_sample_of(settings, normals[2])].outcome == "unlabeled"

    # Relabelling the query normal gives it an answer, and the stored metrics go stale.
    with connection(settings.db_path) as conn:
        conn.execute(
            "UPDATE sample SET label = 'normal' WHERE id = ?", (_sample_of(settings, normals[2]),)
        )
        assert evaluator.current_digest(conn, experiment, Subset.TEST) != fresh
