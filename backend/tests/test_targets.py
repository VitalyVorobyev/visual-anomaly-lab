"""Ground truth reaches `fit` only as `TrainContext.targets`, and only for a targeted task
(ADR-0039, ADR-0040). Also: each kind of class truth loads to the right pixels, and a
predicted mask is stored in the source frame."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from anomaly_lab.annotations.class_truth import (
    ClassTruthError,
    load_class_mask,
    resolve_class_truth,
)
from anomaly_lab.config import Settings
from anomaly_lab.db.connection import connection
from anomaly_lab.db.repositories import experiments as experiments_repo
from anomaly_lab.db.repositories import splits as splits_repo
from anomaly_lab.domain.entities import JobKind, Subset, Task
from anomaly_lab.eval.evaluators import evaluator_for
from anomaly_lab.experiments.context import ExperimentJobError
from anomaly_lab.models.base import (
    Device,
    ImageRecord,
    InferContext,
    NullReporter,
    TrainContext,
)
from anomaly_lab.models.diagnostics import DiagnosticWriter
from anomaly_lab.models.pixel_reference import PixelReferenceModel
from anomaly_lab.models.preprocessing import PreprocessingConfig, load_mask

from .conftest import FIXTURE_SIZE, Fixture, create_experiment, run_handler


def _sample_of(settings: Settings, image_id: int) -> int:
    with connection(settings.db_path) as conn:
        return int(
            conn.execute("SELECT sample_id FROM image WHERE id = ?", (image_id,)).fetchone()[0]
        )


def _capture_fit(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    seen: dict[str, Any] = {}
    original = PixelReferenceModel.fit

    def fit(self: PixelReferenceModel, train: Sequence[ImageRecord], ctx: TrainContext) -> None:
        seen["records"] = [record.image_id for record in train]
        seen["targets"] = ctx.targets
        if ctx.targets is not None:
            seen["label_key"] = ctx.targets.label_key
            seen["masks"] = {record.image_id: ctx.targets.mask(record.image_id) for record in train}
        original(self, train, ctx)

    monkeypatch.setattr(PixelReferenceModel, "fit", fit)
    return seen


def test_an_anomaly_fit_is_given_no_targets(
    client: TestClient, settings: Settings, seeded: Fixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen = _capture_fit(monkeypatch)
    experiment = create_experiment(client, seeded)
    run_handler(settings, JobKind.TRAIN, {"experiment_id": experiment["id"]})
    assert seen["targets"] is None


def _few_shot_run(
    client: TestClient, settings: Settings, seeded: Fixture, references: list[int]
) -> int:
    """A few-shot experiment row on a manual split, written past the create guard.

    No method declares the task yet, so `create_experiment` refuses it. The train handler is
    what is under test, and it reads the row as stored; the anomaly run is only there to
    build the region profile the row pins.
    """
    anomaly = create_experiment(client, seeded)
    with connection(settings.db_path) as conn:
        base = experiments_repo.get_experiment(conn, anomaly["id"])
        assert base is not None
        split = splits_repo.create_split(
            conn,
            seeded.dataset_id,
            name="references",
            strategy="manual",
            seed=0,
            params={"strategy": "manual", "sample_ids": references},
            assignments={
                sample: Subset.TRAIN if sample in references else Subset.TEST
                for sample in splits_repo.list_sample_ids(conn, seeded.split_id)
            },
        )
        row = experiments_repo.create_experiment(
            conn,
            name="few-shot",
            dataset_id=seeded.dataset_id,
            split_id=split.id,
            region_profile_id=base.region_profile_id,
            region_manifest_sha256=base.region_manifest_sha256,
            model_type="pixel_reference",
            task=Task.FEW_SHOT_SEGMENTATION.value,
            target_label="defect",
            model_config=base.model_config_,
            preprocessing_config=base.preprocessing_config,
            eval_config={},
            artifact_dir="",
        )
    return row.id


def test_a_few_shot_fit_gets_every_answered_reference_and_its_mask(
    client: TestClient, settings: Settings, seeded: Fixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen = _capture_fit(monkeypatch)
    defect, normal = seeded.defect_image_ids[0], seeded.normal_image_ids[0]
    run_id = _few_shot_run(
        client, settings, seeded, [_sample_of(settings, defect), _sample_of(settings, normal)]
    )
    result = run_handler(settings, JobKind.TRAIN, {"experiment_id": run_id})

    # A reference labelled normal is a confirmed absence, so it is fitted on as a negative.
    assert sorted(seen["records"]) == sorted([defect, normal])
    assert result["val_images"] == 0
    assert seen["label_key"] == "defect"
    with connection(settings.db_path) as conn:
        source = conn.execute(
            "SELECT path FROM mask WHERE image_id = ? AND kind = 'ground_truth'", (defect,)
        ).fetchone()[0]
    # The region profile is full-frame at the source size, so the prepared mask is the source.
    expected = load_mask(Path(source), size=(FIXTURE_SIZE, FIXTURE_SIZE))
    assert seen["masks"][defect].shape == (FIXTURE_SIZE, FIXTURE_SIZE)
    assert np.array_equal(seen["masks"][defect], expected)
    assert expected.any()
    assert not seen["masks"][normal].any()


def test_a_reference_without_truth_for_the_class_is_left_out_by_name(
    client: TestClient, settings: Settings, seeded: Fixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen = _capture_fit(monkeypatch)
    defect = seeded.defect_image_ids[0]
    with connection(settings.db_path) as conn:
        conn.execute(
            "UPDATE sample SET label = 'unlabeled' WHERE id = ?",
            (_sample_of(settings, seeded.normal_image_ids[0]),),
        )
    run_id = _few_shot_run(
        client,
        settings,
        seeded,
        [_sample_of(settings, defect), _sample_of(settings, seeded.normal_image_ids[0])],
    )
    result = run_handler(settings, JobKind.TRAIN, {"experiment_id": run_id})
    assert seen["records"] == [defect]
    assert result["excluded_images"] == 1


def test_a_few_shot_run_with_no_answered_reference_is_refused(
    client: TestClient, settings: Settings, seeded: Fixture
) -> None:
    unlabelled = seeded.normal_image_ids[0]
    with connection(settings.db_path) as conn:
        conn.execute(
            "UPDATE sample SET label = 'unlabeled' WHERE id = ?",
            (_sample_of(settings, unlabelled),),
        )
    run_id = _few_shot_run(client, settings, seeded, [_sample_of(settings, unlabelled)])
    with pytest.raises(ExperimentJobError, match=r"no reference .* has ground truth for 'defect'"):
        run_handler(settings, JobKind.TRAIN, {"experiment_id": run_id})


def _complete_legacy(settings: Settings, image_id: int, shapes: list[dict[str, Any]]) -> None:
    """A revision as written before migration 023: no class mask, no table."""
    document = {
        "schema_version": 1,
        "image_width": FIXTURE_SIZE,
        "image_height": FIXTURE_SIZE,
        "base": "empty",
        "shapes": shapes,
    }
    with connection(settings.db_path) as conn:
        conn.execute(
            "INSERT INTO annotation_revision (image_id, revision_no, document, document_sha256, "
            "mask_path, mask_sha256) VALUES (?, 1, ?, 'doc', '/mask.png', 'mask')",
            (image_id, json.dumps(document)),
        )


def test_every_kind_of_truth_loads_the_region_of_its_class(
    client: TestClient, settings: Settings, seeded: Fixture
) -> None:
    square = {
        "id": "a",
        "label_key": "defect",
        "kind": "polygon",
        "operation": "add",
        "points": [{"x": 2, "y": 2}, {"x": 8, "y": 2}, {"x": 8, "y": 8}, {"x": 2, "y": 8}],
    }
    legacy, completed, normal = seeded.normal_image_ids
    source = seeded.defect_image_ids[0]
    _complete_legacy(settings, legacy, [square])

    seed = client.get(f"/api/images/{completed}/annotations/draft").json()["document"]
    draft = client.post(
        f"/api/images/{completed}/annotations/draft",
        json={**seed, "shapes": [square]},
        headers={"If-None-Match": "*"},
    )
    done = client.post(
        f"/api/images/{completed}/annotations/complete",
        headers={"If-Match": draft.headers["etag"]},
    )
    assert done.status_code == 200, done.text

    with connection(settings.db_path) as conn:
        truths = resolve_class_truth(
            conn, seeded.dataset_id, [legacy, completed, normal, source], "defect"
        )
    assert {image_id: truth.kind for image_id, truth in truths.items()} == {
        legacy: "document",
        completed: "class_mask",
        normal: "normal",
        source: "source",
    }
    drawn = load_class_mask(truths[completed])
    assert np.array_equal(load_class_mask(truths[legacy]), drawn)
    assert drawn[4, 4] and not drawn[12, 12]
    assert not load_class_mask(truths[normal]).any()
    assert load_class_mask(truths[source]).any()

    Path(truths[completed].path or "").write_bytes(b"changed")
    with pytest.raises(ClassTruthError, match="changed after it was pinned"):
        load_class_mask(truths[completed])


def test_a_predicted_mask_is_stored_in_the_source_frame(tmp_path: Path) -> None:
    ctx = InferContext(
        artifact_dir=tmp_path,
        cache_dir=tmp_path,
        preprocessing=PreprocessingConfig(width=8, height=8),
        device=Device.CPU,
        reporter=NullReporter(),
        diagnostics=DiagnosticWriter(tmp_path / "diagnostics", enabled=False),
        mask_projector=lambda _image_id, values: np.kron(values, np.ones((2, 2), dtype=bool)),
    )
    prepared = np.zeros((1, 4, 4), dtype=bool)
    prepared[0, 1, 1] = True
    path = ctx.write_mask(7, prepared)

    assert path == tmp_path / "maps" / "7.mask.png"
    with Image.open(path) as opened:
        stored = np.asarray(opened)
    assert stored.shape == (8, 8)
    assert set(np.unique(stored)) == {0, 255}
    assert stored[2:4, 2:4].all() and stored.sum() == 4 * 255

    with pytest.raises(ValueError, match="must be 2-D"):
        ctx.write_mask(8, np.zeros((2, 4, 4), dtype=bool))


def test_the_infer_log_names_the_task_s_headline() -> None:
    assert evaluator_for(Task.ANOMALY).headline == "sample_roc_auc"
