"""A supervised task's split (ADR-0039): annotated samples drawn into train and test,
stratified by the set of classes each one shows, under a seed."""

from __future__ import annotations

import sqlite3
from typing import Any

import pytest
from fastapi.testclient import TestClient

from anomaly_lab.annotations.class_truth import resolve_box_truth
from anomaly_lab.config import Settings
from anomaly_lab.datasets.splitting import (
    SplitParams,
    SplitStrategy,
    draw_class_stratified,
)
from anomaly_lab.db.connection import connection
from anomaly_lab.db.repositories import annotations as annotations_repo
from anomaly_lab.domain.entities import Subset
from tests.conftest import Fixture
from tests.test_few_shot_splits import _add_class, _complete, _polygon, _sample_of


def _signatures(**strata: int) -> dict[int, frozenset[str]]:
    """`empty=10, a=5, a_b=5` → ten samples showing nothing, five `{a}`, five `{a, b}`."""
    shown: dict[int, frozenset[str]] = {}
    for name, count in strata.items():
        classes = frozenset() if name == "empty" else frozenset(name.split("_"))
        for _ in range(count):
            shown[len(shown) + 1] = classes
    return shown


def _train(plan: dict[int, Subset]) -> set[int]:
    return {sample_id for sample_id, subset in plan.items() if subset is Subset.TRAIN}


def _in(shown: dict[int, frozenset[str]], plan: dict[int, Subset], key: str, subset: Subset) -> int:
    return sum(1 for s, found in shown.items() if key in found and plan[s] is subset)


# --- the draw ----------------------------------------------------------------------------


def test_each_class_signature_trains_in_proportion() -> None:
    shown = _signatures(empty=10, a=5, a_b=5)
    plan = draw_class_stratified(shown, seed=0, train_fraction=0.6)

    assert set(plan) == set(shown)
    assert set(plan.values()) == {Subset.TRAIN, Subset.TEST}
    train = _train(plan)
    assert len(train) == 12
    by_signature = {
        signature: sum(1 for s in train if shown[s] == signature)
        for signature in {frozenset(), frozenset({"a"}), frozenset({"a", "b"})}
    }
    assert by_signature == {frozenset(): 6, frozenset({"a"}): 3, frozenset({"a", "b"}): 3}


def test_leftover_samples_keep_the_total_exact() -> None:
    # 3 * 0.5 floors to 1 per stratum; the total asks for round(9 * 0.5) = 4, so exactly one
    # stratum takes a second sample, and none takes more.
    shown = _signatures(empty=3, a=3, b=3)
    plan = draw_class_stratified(shown, seed=5, train_fraction=0.5)
    train = _train(plan)
    assert len(train) == 4
    per_stratum = sorted(
        sum(1 for s in train if shown[s] == signature)
        for signature in {frozenset(), frozenset({"a"}), frozenset({"b"})}
    )
    assert per_stratum == [1, 1, 2]


def test_the_draw_is_seeded_in_both_directions() -> None:
    shown = _signatures(empty=12, a=6, a_b=6, b=6)
    first = draw_class_stratified(shown, seed=0, train_fraction=0.7)
    assert first == draw_class_stratified(shown, seed=0, train_fraction=0.7)
    assert any(
        draw_class_stratified(shown, seed=seed, train_fraction=0.7) != first for seed in range(1, 6)
    )


def test_a_class_seen_twice_trains_even_when_its_strata_round_to_nothing() -> None:
    # Ten plain samples, then `a` on two samples of two different one-sample strata. 12 * 0.2
    # is 2, and both go to the plain stratum; the draw alone would leave `a` untrained.
    shown = _signatures(empty=10, a_b=1, a=1)
    for seed in range(8):
        plan = draw_class_stratified(shown, seed=seed, train_fraction=0.2)
        assert _in(shown, plan, "a", Subset.TRAIN) == 1
        assert _in(shown, plan, "a", Subset.TEST) == 1


def test_a_class_seen_twice_is_tested_too_where_that_strands_nothing() -> None:
    # 5 * 0.8 is 4 and the larger remainder is `a`'s: both `a` samples train. One can move to
    # test, because `a` still trains on the other and neither carries another class.
    shown = _signatures(empty=3, a=2)
    plan = draw_class_stratified(shown, seed=0, train_fraction=0.8)
    assert _in(shown, plan, "a", Subset.TRAIN) == 1
    assert _in(shown, plan, "a", Subset.TEST) == 1


def test_a_class_stays_untested_rather_than_strand_another_class() -> None:
    # `x` is on two samples, and each is the only sample of another class (`y`, `z`). Both
    # train; moving either to test would leave `y` or `z` with nothing to learn from, so `x`
    # has no test sample and its test metrics will be None.
    shown = _signatures(empty=3, x_y=1, x_z=1)
    plan = draw_class_stratified(shown, seed=0, train_fraction=0.9)
    assert _in(shown, plan, "x", Subset.TEST) == 0
    assert _in(shown, plan, "y", Subset.TRAIN) == 1
    assert _in(shown, plan, "z", Subset.TRAIN) == 1


def test_params_refuse_training_on_samples_without_truth() -> None:
    with pytest.raises(ValueError, match="no ground truth"):
        SplitParams(strategy=SplitStrategy.CLASS_STRATIFIED, unlabeled_subset=Subset.TRAIN)
    with pytest.raises(ValueError):
        SplitParams(strategy=SplitStrategy.CLASS_STRATIFIED, train_fraction=1.0)
    # The anomaly fractions do not apply, so their sum is not checked.
    SplitParams(
        strategy=SplitStrategy.CLASS_STRATIFIED,
        train_normal_fraction=0.9,
        val_normal_fraction=0.9,
    )


# --- which samples are annotated ------------------------------------------------------------


def test_a_sample_answers_only_when_every_image_does(catalog: Any, migrated_db: Any) -> None:
    # `group-a/1` has two images; one completed annotation leaves the other unanswered for
    # the new class, so the sample cannot train a run that segments it.
    _add_class(migrated_db, catalog.dataset_id, "scratch")
    sample = catalog.sample_ids["group-a/1"]
    _complete(migrated_db, catalog.image_ids[0], [_polygon("scratch")])
    shown = annotations_repo.classes_shown_by_sample(
        migrated_db, catalog.dataset_id, ["defect", "scratch"]
    )
    assert shown[sample] is None

    _complete(migrated_db, catalog.image_ids[1], [])
    shown = annotations_repo.classes_shown_by_sample(
        migrated_db, catalog.dataset_id, ["defect", "scratch"]
    )
    assert shown[sample] == frozenset({"scratch"})


# --- through the API ---------------------------------------------------------------------


def _image_ids(conn: sqlite3.Connection, dataset_id: int) -> list[int]:
    return [
        int(row[0])
        for row in conn.execute(
            "SELECT image.id FROM image JOIN sample ON sample.id = image.sample_id "
            "WHERE sample.dataset_id = ? ORDER BY image.id",
            (dataset_id,),
        )
    ]


def _draw(client: TestClient, seeded: Fixture, name: str, **params: Any) -> Any:
    return client.post(
        "/api/splits",
        json={
            "dataset_id": seeded.dataset_id,
            "name": name,
            "seed": params.pop("seed", 0),
            "params": {"strategy": "class_stratified", **params},
        },
    )


def test_the_api_draws_annotated_samples_and_parks_the_rest(
    client: TestClient, settings: Settings, seeded: Fixture
) -> None:
    with connection(settings.db_path) as conn:
        _add_class(conn, seeded.dataset_id, "scratch")
        images = _image_ids(conn, seeded.dataset_id)
        # Thirteen of fourteen samples answered for both classes, five of them showing a
        # scratch; the last defect is never annotated after `scratch` was created.
        annotated = images[:-1]
        for index, image_id in enumerate(annotated):
            _complete(conn, image_id, [_polygon("scratch")] if index % 3 == 0 else [])
        unannotated = _sample_of(conn, images[-1])
        scratched = {_sample_of(conn, i) for n, i in enumerate(annotated) if n % 3 == 0}

    response = _draw(client, seeded, "supervised", train_fraction=0.6)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["strategy"] == "class_stratified"
    assert body["params"]["classes"] == ["defect", "scratch"]
    composition = {row["subset"]: row["total"] for row in body["composition"]}
    # round(13 * 0.6) = 8 of the annotated samples train; the rest test beside the one
    # sample with no truth.
    assert composition == {"train": 8, "val": 0, "test": 6}

    with connection(settings.db_path) as conn:
        rows = conn.execute(
            "SELECT sample_id, subset FROM split_assignment WHERE split_id = ?", (body["id"],)
        ).fetchall()
    plan = {int(row["sample_id"]): Subset(row["subset"]) for row in rows}
    assert plan[unannotated] is Subset.TEST
    assert any(plan[s] is Subset.TRAIN for s in scratched)
    assert any(plan[s] is Subset.TEST for s in scratched)

    again = _draw(client, seeded, "supervised again", train_fraction=0.6)
    with connection(settings.db_path) as conn:
        rows = conn.execute(
            "SELECT sample_id, subset FROM split_assignment WHERE split_id = ?",
            (again.json()["id"],),
        ).fetchall()
    assert {int(row["sample_id"]): Subset(row["subset"]) for row in rows} == plan

    left_out = _draw(client, seeded, "annotated only", unlabeled_subset=None)
    assert left_out.status_code == 200, left_out.text
    assert sum(row["total"] for row in left_out.json()["composition"]) == 13


def test_the_api_refuses_a_dataset_with_too_little_truth(
    client: TestClient, settings: Settings, seeded: Fixture
) -> None:
    with connection(settings.db_path) as conn:
        _add_class(conn, seeded.dataset_id, "scratch")
    refused = _draw(client, seeded, "nothing yet")
    assert refused.status_code == 409
    assert "0 of 14 samples have ground truth" in refused.json()["detail"]

    parked = _draw(client, seeded, "train on nothing", unlabeled_subset="train")
    assert parked.status_code == 422


def _box(label_key: str) -> dict[str, Any]:
    return {
        "id": f"{label_key}-box",
        "label_key": label_key,
        "kind": "box",
        "operation": "add",
        "x": 2,
        "y": 2,
        "width": 3,
        "height": 4,
    }


def test_a_detection_run_reads_boxes_on_every_sample_the_draw_placed(
    client: TestClient, settings: Settings, seeded: Fixture
) -> None:
    # Detection truth answers by the presence rule the draw stratifies by, so a split drawn
    # over boxed truth trains and tests only on images a detection run can read as boxes,
    # and leaves out every image it cannot.
    with connection(settings.db_path) as conn:
        _add_class(conn, seeded.dataset_id, "scratch")
        images = _image_ids(conn, seeded.dataset_id)
        boxed = images[:-1]
        for index, image_id in enumerate(boxed):
            _complete(conn, image_id, [_box("scratch")] if index % 3 == 0 else [])

    response = _draw(client, seeded, "detection", train_fraction=0.6, unlabeled_subset=None)
    assert response.status_code == 200, response.text
    body = response.json()
    classes = body["params"]["classes"]
    assert classes == ["defect", "scratch"]

    with connection(settings.db_path) as conn:
        rows = conn.execute(
            "SELECT image.id AS image_id, split_assignment.subset AS subset "
            "FROM split_assignment JOIN image ON image.sample_id = split_assignment.sample_id "
            "WHERE split_assignment.split_id = ?",
            (body["id"],),
        ).fetchall()
        by_subset: dict[Subset, set[int]] = {}
        for row in rows:
            by_subset.setdefault(Subset(row["subset"]), set()).add(int(row["image_id"]))
        drawn = by_subset[Subset.TRAIN] | by_subset[Subset.TEST]
        truth = resolve_box_truth(conn, seeded.dataset_id, sorted(drawn), classes)
        left_out = resolve_box_truth(conn, seeded.dataset_id, [images[-1]], classes)

    assert set(by_subset) == {Subset.TRAIN, Subset.TEST}
    assert drawn == set(boxed)
    assert set(truth) == drawn
    assert {entry.kind for entry in truth.values()} == {"document"}
    # The one image never annotated after `scratch` existed is neither drawn nor boxed.
    assert left_out == {}
