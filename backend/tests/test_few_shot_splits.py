"""A few-shot task's references are a split, and a class's truth includes its absence
(ADR-0040): presence per sample, the `manual` and `few_shot` strategies, the coverage read,
and the target class an experiment is frozen with."""

from __future__ import annotations

import json
import sqlite3
from typing import Any

import pytest
from fastapi.testclient import TestClient

from anomaly_lab.config import Settings
from anomaly_lab.datasets.splitting import (
    SplitParams,
    SplitPlanError,
    SplitStrategy,
    plan_few_shot_split,
    plan_manual_split,
)
from anomaly_lab.db.connection import connection
from anomaly_lab.db.repositories import annotations as annotations_repo
from anomaly_lab.db.repositories import splits as splits_repo
from anomaly_lab.domain.entities import ClassPresence, Split, Subset, Task
from anomaly_lab.errors import InvalidInputError
from anomaly_lab.experiments.service import validate_target
from tests.conftest import TEST_DEFECTS, TEST_NORMALS, TRAIN_NORMALS, Fixture, create_experiment

SAMPLES = TRAIN_NORMALS + TEST_NORMALS + TEST_DEFECTS


def _complete(conn: sqlite3.Connection, image_id: int, shapes: list[dict[str, Any]]) -> None:
    """A completed revision with these shapes. Presence reads only the document."""
    document = {
        "schema_version": 1,
        "image_width": 16,
        "image_height": 16,
        "base": "empty",
        "shapes": shapes,
    }
    number = conn.execute(
        "SELECT COALESCE(MAX(revision_no), 0) + 1 FROM annotation_revision WHERE image_id = ?",
        (image_id,),
    ).fetchone()[0]
    conn.execute(
        "INSERT INTO annotation_revision (image_id, revision_no, document, document_sha256, "
        "mask_path, mask_sha256) VALUES (?, ?, ?, 'doc', '/mask.png', 'mask')",
        (image_id, number, json.dumps(document)),
    )


def _polygon(label_key: str, operation: str = "add") -> dict[str, Any]:
    points = [{"x": 1, "y": 1}, {"x": 6, "y": 1}, {"x": 6, "y": 6}]
    return {
        "id": f"{label_key}-{operation}",
        "label_key": label_key,
        "kind": "polygon",
        "operation": operation,
        "points": points,
    }


def _sample_of(conn: sqlite3.Connection, image_id: int) -> int:
    return int(conn.execute("SELECT sample_id FROM image WHERE id = ?", (image_id,)).fetchone()[0])


def _add_class(conn: sqlite3.Connection, dataset_id: int, key: str) -> None:
    annotations_repo.create_label(
        conn, dataset_id, key=key, name=key.title(), color="#00aa00", position=1
    )


# --- presence --------------------------------------------------------------------------


def test_imported_truth_answers_for_the_default_class_only(
    settings: Settings, seeded: Fixture
) -> None:
    with connection(settings.db_path) as conn:
        _add_class(conn, seeded.dataset_id, "scratch")
        found = annotations_repo.presence_by_class(conn, seeded.dataset_id, ["defect", "scratch"])

        defects = {_sample_of(conn, image_id) for image_id in seeded.defect_image_ids}
    assert {s for s, p in found["defect"].items() if p is ClassPresence.PRESENT} == defects
    # A sample labelled normal is a confirmed absence of the class its label speaks about...
    assert sum(p is ClassPresence.ABSENT for p in found["defect"].values()) == (
        TRAIN_NORMALS + TEST_NORMALS
    )
    # ...and says nothing about any other class.
    assert set(found["scratch"].values()) == {ClassPresence.UNLABELED}


def test_a_completed_revision_decides_presence_and_absence(
    settings: Settings, seeded: Fixture
) -> None:
    with connection(settings.db_path) as conn:
        _add_class(conn, seeded.dataset_id, "scratch")
        drawn, blank, subtracted = seeded.normal_image_ids
        _complete(conn, drawn, [_polygon("scratch")])
        _complete(conn, blank, [])
        _complete(conn, subtracted, [_polygon("scratch", "subtract")])
        # A defect whose newest revision draws another class no longer shows `defect`.
        _complete(conn, seeded.defect_image_ids[0], [_polygon("scratch")])
        found = annotations_repo.presence_by_class(conn, seeded.dataset_id, ["defect", "scratch"])
        sample = {image_id: _sample_of(conn, image_id) for image_id in (drawn, blank, subtracted)}
        redrawn = _sample_of(conn, seeded.defect_image_ids[0])

    assert found["scratch"][sample[drawn]] is ClassPresence.PRESENT
    assert found["scratch"][sample[blank]] is ClassPresence.ABSENT
    assert found["scratch"][sample[subtracted]] is ClassPresence.ABSENT
    assert found["scratch"][redrawn] is ClassPresence.PRESENT
    assert found["defect"][redrawn] is ClassPresence.ABSENT


def test_a_revision_says_nothing_about_a_class_created_after_it(
    settings: Settings, seeded: Fixture
) -> None:
    with connection(settings.db_path) as conn:
        image_id = seeded.normal_image_ids[0]
        _complete(conn, image_id, [])
        _add_class(conn, seeded.dataset_id, "scratch")
        # Timestamps have millisecond resolution; make "later" unambiguous.
        conn.execute(
            "UPDATE annotation_label SET created_at = '2999-01-01T00:00:00.000Z' WHERE key = ?",
            ("scratch",),
        )
        found = annotations_repo.presence_by_class(conn, seeded.dataset_id, ["defect", "scratch"])
        sample = _sample_of(conn, image_id)
    assert found["defect"][sample] is ClassPresence.ABSENT
    assert found["scratch"][sample] is ClassPresence.UNLABELED


def test_a_sample_is_absent_only_when_every_image_is(catalog: Any, migrated_db: Any) -> None:
    # `group-a/1` is a normal sample with two images; annotating one of them for another class
    # leaves the other unlabelled for it, so the sample is too.
    _add_class(migrated_db, catalog.dataset_id, "scratch")
    first_image = catalog.image_ids[0]
    _complete(migrated_db, first_image, [])
    found = annotations_repo.class_presence(migrated_db, catalog.dataset_id, "scratch")
    assert found[catalog.sample_ids["group-a/1"]] is ClassPresence.UNLABELED

    _complete(migrated_db, catalog.image_ids[1], [])
    found = annotations_repo.class_presence(migrated_db, catalog.dataset_id, "scratch")
    assert found[catalog.sample_ids["group-a/1"]] is ClassPresence.ABSENT


# --- strategies ------------------------------------------------------------------------


def test_split_params_refuse_what_a_strategy_cannot_build() -> None:
    with pytest.raises(ValueError, match="at least one reference"):
        SplitParams(strategy=SplitStrategy.MANUAL)
    with pytest.raises(ValueError, match="twice"):
        SplitParams(strategy=SplitStrategy.MANUAL, sample_ids=[3, 3])
    with pytest.raises(ValueError, match="label_key"):
        SplitParams(strategy=SplitStrategy.FEW_SHOT, shots=2)
    with pytest.raises(ValueError, match="number of shots"):
        SplitParams(strategy=SplitStrategy.FEW_SHOT, label_key="defect")
    with pytest.raises(ValueError):
        SplitParams(strategy=SplitStrategy.FEW_SHOT, label_key="defect", shots=0)


def test_manual_references_train_and_everything_else_tests(
    settings: Settings, seeded: Fixture
) -> None:
    with connection(settings.db_path) as conn:
        references = [_sample_of(conn, image_id) for image_id in seeded.defect_image_ids[:2]]
        plan = plan_manual_split(conn, seeded.dataset_id, references)
        with pytest.raises(SplitPlanError, match="not in dataset"):
            plan_manual_split(conn, seeded.dataset_id, [*references, 10_000])

    assert len(plan) == SAMPLES
    assert sorted(s for s, subset in plan.items() if subset is Subset.TRAIN) == sorted(references)
    assert Subset.VAL not in plan.values()


def test_a_few_shot_draw_is_seeded_in_both_directions(settings: Settings, seeded: Fixture) -> None:
    with connection(settings.db_path) as conn:
        _add_class(conn, seeded.dataset_id, "scratch")
        for image_id in seeded.normal_image_ids + seeded.defect_image_ids:
            _complete(conn, image_id, [_polygon("scratch")])

        def draw(seed: int) -> set[int]:
            plan = plan_few_shot_split(
                conn, seeded.dataset_id, seed=seed, label_key="scratch", shots=2
            )
            assert len(plan) == SAMPLES
            return {s for s, subset in plan.items() if subset is Subset.TRAIN}

        eligible = {_sample_of(conn, i) for i in seeded.normal_image_ids + seeded.defect_image_ids}
        first = draw(0)
        assert first == draw(0)
        assert any(draw(seed) != first for seed in range(1, 6))
        assert first <= eligible and len(first) == 2

        with pytest.raises(SplitPlanError, match="only 6 samples show it"):
            plan_few_shot_split(conn, seeded.dataset_id, seed=0, label_key="scratch", shots=7)


def test_the_api_builds_both_strategies_and_refuses_an_unknown_class(
    client: TestClient, settings: Settings, seeded: Fixture
) -> None:
    drawn = client.post(
        "/api/splits",
        json={
            "dataset_id": seeded.dataset_id,
            "name": "two shots",
            "seed": 3,
            "params": {"strategy": "few_shot", "label_key": "defect", "shots": 2},
        },
    )
    assert drawn.status_code == 200, drawn.text
    composition = {row["subset"]: row["total"] for row in drawn.json()["composition"]}
    assert composition == {"train": 2, "val": 0, "test": SAMPLES - 2}

    unknown = client.post(
        "/api/splits",
        json={
            "dataset_id": seeded.dataset_id,
            "name": "nothing",
            "params": {"strategy": "few_shot", "label_key": "nothing", "shots": 1},
        },
    )
    assert unknown.status_code == 409
    assert "no annotation class 'nothing'" in unknown.json()["detail"]

    with connection(settings.db_path) as conn:
        reference = _sample_of(conn, seeded.normal_image_ids[0])
    manual = client.post(
        "/api/splits",
        json={
            "dataset_id": seeded.dataset_id,
            "name": "hand-picked",
            "params": {"strategy": "manual", "sample_ids": [reference]},
        },
    )
    assert manual.status_code == 200, manual.text
    assert manual.json()["params"]["sample_ids"] == [reference]


def test_coverage_counts_samples_per_class(client: TestClient, seeded: Fixture) -> None:
    response = client.get(f"/api/datasets/{seeded.dataset_id}/annotation-labels/coverage")
    assert response.status_code == 200, response.text
    assert response.json() == [
        {
            "label_key": "defect",
            "present": TEST_DEFECTS,
            "absent": TRAIN_NORMALS + TEST_NORMALS,
            "unlabeled": 0,
        }
    ]


# --- the target class ------------------------------------------------------------------


def _split(conn: sqlite3.Connection, seeded: Fixture, params: SplitParams, plan: Any) -> Split:
    return splits_repo.create_split(
        conn,
        seeded.dataset_id,
        name=f"split-{len(splits_repo.list_splits(conn, seeded.dataset_id))}",
        strategy=params.strategy.value,
        seed=0,
        params=params.model_dump(mode="json"),
        assignments=plan,
    )


def test_a_target_class_belongs_to_a_targeted_task_only(
    settings: Settings, seeded: Fixture
) -> None:
    with connection(settings.db_path) as conn:
        official = splits_repo.get_split(conn, seeded.split_id)
        assert official is not None
        validate_target(conn, Task.ANOMALY, None, official)
        with pytest.raises(InvalidInputError, match="does not take a target class"):
            validate_target(conn, Task.ANOMALY, "defect", official)
        with pytest.raises(InvalidInputError, match="needs a target class"):
            validate_target(conn, Task.FEW_SHOT_SEGMENTATION, None, official)
        with pytest.raises(InvalidInputError, match="no annotation class 'scratch'"):
            validate_target(conn, Task.FEW_SHOT_SEGMENTATION, "scratch", official)


def test_every_reference_must_show_the_target_class(settings: Settings, seeded: Fixture) -> None:
    with connection(settings.db_path) as conn:
        _add_class(conn, seeded.dataset_id, "scratch")
        defects = [_sample_of(conn, image_id) for image_id in seeded.defect_image_ids]
        normal = _sample_of(conn, seeded.normal_image_ids[0])

        good = SplitParams(strategy=SplitStrategy.MANUAL, sample_ids=defects[:2])
        validate_target(
            conn,
            Task.FEW_SHOT_SEGMENTATION,
            "defect",
            _split(conn, seeded, good, plan_manual_split(conn, seeded.dataset_id, defects[:2])),
        )

        mixed = SplitParams(strategy=SplitStrategy.MANUAL, sample_ids=[defects[0], normal])
        mixed_split = _split(
            conn, seeded, mixed, plan_manual_split(conn, seeded.dataset_id, [defects[0], normal])
        )
        with pytest.raises(InvalidInputError, match=rf"references \[{normal}\]"):
            validate_target(conn, Task.FEW_SHOT_SEGMENTATION, "defect", mixed_split)

        drawn = SplitParams(strategy=SplitStrategy.FEW_SHOT, label_key="defect", shots=1)
        drawn_split = _split(
            conn,
            seeded,
            drawn,
            plan_few_shot_split(conn, seeded.dataset_id, seed=0, label_key="defect", shots=1),
        )
        with pytest.raises(InvalidInputError, match="draws references of 'defect'"):
            validate_target(conn, Task.FEW_SHOT_SEGMENTATION, "scratch", drawn_split)


def test_an_anomaly_run_is_created_without_a_target_and_reads_back_null(
    client: TestClient, seeded: Fixture
) -> None:
    created = create_experiment(client, seeded)
    assert created["task"] == "anomaly"
    assert created["target_label"] is None

    refused = client.post(
        "/api/experiments",
        json={
            "name": "targeted",
            "dataset_id": seeded.dataset_id,
            "split_id": seeded.split_id,
            "region_profile_id": seeded.region_profile_id,
            "model_type": "pixel_reference",
            "target_label": "defect",
        },
    )
    assert refused.status_code == 422
    assert "does not take a target class" in refused.text


# --- reading a class ---------------------------------------------------------------------


def test_samples_can_be_listed_by_their_presence_of_a_class(
    client: TestClient, seeded: Fixture
) -> None:
    url = f"/api/datasets/{seeded.dataset_id}/samples"
    present = client.get(url, params={"class_key": "defect", "presence": "present"})
    absent = client.get(url, params={"class_key": "defect", "presence": "absent"})
    assert present.json()["total"] == TEST_DEFECTS
    assert absent.json()["total"] == TRAIN_NORMALS + TEST_NORMALS
    assert client.get(url, params={"class_key": "defect"}).status_code == 422


def test_one_class_s_region_is_outlined_from_its_own_truth(
    client: TestClient, seeded: Fixture
) -> None:
    defect = client.get(
        f"/api/images/{seeded.defect_image_ids[0]}/mask", params={"class_key": "defect"}
    )
    assert defect.status_code == 200
    assert defect.headers["content-type"] == "image/png"
    # A normal image is a confirmed absence: an empty outline, not a missing one.
    normal = client.get(
        f"/api/images/{seeded.normal_image_ids[0]}/mask", params={"class_key": "defect"}
    )
    assert normal.status_code == 200
    missing = client.get(
        f"/api/images/{seeded.normal_image_ids[0]}/mask", params={"class_key": "scratch"}
    )
    assert missing.status_code == 404
    prepared = client.get(
        f"/api/images/{seeded.defect_image_ids[0]}/mask",
        params={"class_key": "defect", "frame": "prepared", "experiment_id": 1},
    )
    assert prepared.status_code == 422
