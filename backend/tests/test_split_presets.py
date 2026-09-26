"""Split presets, dry runs, derived names and deletion.

A split cannot be edited, so what it will contain is shown before it exists: a preset is a
request that works on this dataset, with the composition Create would produce, and the custom
form's preview is the same dry run. Neither writes anything. An unnamed split is named after
its params and the first seed they have not used. A split no experiment ran on can be deleted;
one an experiment ran on cannot, and nothing cascades.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from anomaly_lab.config import Settings
from anomaly_lab.db.connection import connection
from anomaly_lab.db.repositories import splits as splits_repo
from tests.conftest import TEST_DEFECTS, TEST_NORMALS, TRAIN_NORMALS, Fixture, create_experiment
from tests.test_truth_is_task_scoped import CLASSES, IMAGES_PER_CLASS, _one_dataset, _tree


def _presets(client: TestClient, dataset_id: int) -> dict[str, dict[str, Any]]:
    response = client.get(f"/api/datasets/{dataset_id}/split-presets")
    assert response.status_code == 200, response.text
    return {preset["key"]: preset for preset in response.json()}


def _by_subset(composition: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {row["subset"]: row for row in composition}


def _split_count(settings: Settings, dataset_id: int) -> int:
    with connection(settings.db_path) as conn:
        return len(splits_repo.list_splits(conn, dataset_id))


# --- presets -------------------------------------------------------------------------------


def test_an_anomaly_dataset_is_offered_the_standard_draw_only(
    client: TestClient, settings: Settings, seeded: Fixture
) -> None:
    """No manifest, so no published partition; no class truth, so no class presets."""
    before = _split_count(settings, seeded.dataset_id)

    presets = _presets(client, seeded.dataset_id)

    assert list(presets) == ["standard"]
    standard = presets["standard"]
    assert standard["label"] == "Standard · 60/20/20, normals only"
    assert standard["tasks"] == ["anomaly"]
    assert standard["params"]["strategy"] == "normal_only_train"
    assert standard["name"] == "Standard · 60/20/20, normals only · seed 0"
    composition = _by_subset(standard["composition"])
    assert composition["train"]["defect"] == 0
    assert composition["train"]["normal"] > 0
    total = sum(row["total"] for row in standard["composition"])
    assert total == TRAIN_NORMALS + TEST_NORMALS + TEST_DEFECTS
    # A dry run writes nothing.
    assert _split_count(settings, seeded.dataset_id) == before


def test_a_preset_creates_exactly_what_its_dry_run_showed(
    client: TestClient, seeded: Fixture
) -> None:
    preset = _presets(client, seeded.dataset_id)["standard"]

    created = client.post(
        "/api/splits", json={"dataset_id": seeded.dataset_id, "params": preset["params"]}
    )

    assert created.status_code == 200, created.text
    split = created.json()
    assert split["name"] == preset["name"]
    assert split["seed"] == preset["seed"]
    assert split["composition"] == preset["composition"]
    assert split["tasks"] == ["anomaly"]


def test_pressing_create_twice_draws_the_next_seed(client: TestClient, seeded: Fixture) -> None:
    """An untouched name never collides: a repeated request is a new, differently named draw."""
    body = {"dataset_id": seeded.dataset_id, "params": {"strategy": "normal_only_train"}}

    first = client.post("/api/splits", json=body).json()
    second = client.post("/api/splits", json=body).json()

    assert (first["seed"], second["seed"]) == (0, 1)
    assert first["name"].endswith("· seed 0")
    assert second["name"].endswith("· seed 1")
    assert _presets(client, seeded.dataset_id)["standard"]["seed"] == 2


def test_a_derived_name_takes_a_suffix_when_it_is_taken(
    client: TestClient, seeded: Fixture
) -> None:
    body = {"dataset_id": seeded.dataset_id, "seed": 3, "params": {"strategy": "normal_only_train"}}

    first = client.post("/api/splits", json=body).json()
    second = client.post("/api/splits", json=body).json()

    assert first["name"] == "Standard · 60/20/20, normals only · seed 3"
    assert second["name"] == "Standard · 60/20/20, normals only · seed 3 (2)"


def test_a_published_partition_is_offered_only_with_a_manifest_that_has_one(
    client: TestClient, settings: Settings, seeded: Fixture
) -> None:
    # The fixture's dataset holds an `imported` split but was never committed from a
    # manifest, so there is nothing to adopt and the preset must not be offered.
    assert "published" not in _presets(client, seeded.dataset_id)


def test_a_class_dataset_is_offered_few_shot_and_by_class_presets(
    client: TestClient, settings: Settings, tmp_path: Path
) -> None:
    dataset_id = _one_dataset(settings, _tree(tmp_path / "tree"))

    presets = _presets(client, dataset_id)

    # No anomaly verdicts, so no anomaly preset; four samples per class, so no 5-shot.
    assert "standard" not in presets
    assert "few_shot_5" not in presets
    one_shot = presets["few_shot_1"]
    assert one_shot["tasks"] == ["few_shot_segmentation"]
    assert [entry["key"] for entry in one_shot["classes"]] == list(CLASSES)
    assert all(entry["samples"] == IMAGES_PER_CLASS for entry in one_shot["classes"])
    assert one_shot["params"]["label_key"] == CLASSES[0]
    composition = _by_subset(one_shot["composition"])
    assert composition["train"]["total"] == 1
    train_classes = {entry["key"]: entry["samples"] for entry in composition["train"]["classes"]}
    assert train_classes[CLASSES[0]] == 1
    test_classes = {entry["key"]: entry["samples"] for entry in composition["test"]["classes"]}
    assert test_classes[CLASSES[0]] == IMAGES_PER_CLASS - 1
    assert one_shot["name"] == f"1-shot · {CLASSES[0]} · seed 0"

    by_class = presets["by_class"]
    assert by_class["tasks"] == ["semantic_segmentation", "object_detection"]
    by_class_composition = _by_subset(by_class["composition"])
    assert by_class_composition["train"]["total"] > 0
    assert by_class_composition["test"]["total"] > 0


# --- preview -------------------------------------------------------------------------------


def test_a_preview_is_the_dry_run_of_custom_params(
    client: TestClient, settings: Settings, seeded: Fixture
) -> None:
    response = client.post(
        f"/api/datasets/{seeded.dataset_id}/splits/preview",
        json={"seed": 4, "params": {"train_normal_fraction": 0.5, "val_normal_fraction": 0.1}},
    )

    assert response.status_code == 200, response.text
    preview = response.json()
    assert preview["error"] is None
    assert preview["name"] == "50/10/40, normals only · seed 4"
    normals = TRAIN_NORMALS + TEST_NORMALS
    assert _by_subset(preview["composition"])["train"]["normal"] == round(normals * 0.5)
    assert _split_count(settings, seeded.dataset_id) == 1


def test_a_preview_that_cannot_be_drawn_says_why(client: TestClient, seeded: Fixture) -> None:
    response = client.post(
        f"/api/datasets/{seeded.dataset_id}/splits/preview",
        json={"params": {"strategy": "few_shot", "label_key": "nothing", "shots": 1}},
    )

    assert response.status_code == 200
    preview = response.json()
    assert "no annotation class 'nothing'" in preview["error"]
    assert preview["composition"] == []


# --- deletion ------------------------------------------------------------------------------


def test_a_split_no_experiment_ran_on_is_deleted_with_its_assignments(
    client: TestClient, settings: Settings, seeded: Fixture
) -> None:
    split = client.post("/api/splits", json={"dataset_id": seeded.dataset_id}).json()

    preview = client.get(f"/api/splits/{split['id']}/deletion-preview").json()
    assert preview["can_delete"] is True
    assert preview["experiments"] == []
    assert preview["assignments"] == sum(row["total"] for row in split["composition"])

    deleted = client.delete(f"/api/splits/{split['id']}")

    assert deleted.status_code == 200, deleted.text
    assert deleted.json() == {"deleted": True, "assignments_removed": preview["assignments"]}
    assert client.get(f"/api/splits/{split['id']}").status_code == 404
    with connection(settings.db_path) as conn:
        assert splits_repo.assignments(conn, split["id"]) == {}


def test_a_split_an_experiment_ran_on_is_refused_and_names_it(
    client: TestClient, seeded: Fixture
) -> None:
    """RESTRICT, not cascade: a run's numbers only mean something against its split."""
    experiment = create_experiment(client, seeded, name="held")

    preview = client.get(f"/api/splits/{seeded.split_id}/deletion-preview").json()
    assert preview["can_delete"] is False
    assert preview["experiments"] == [{"experiment_id": experiment["id"], "name": "held"}]
    assert "held" in preview["blocker"]

    refused = client.delete(f"/api/splits/{seeded.split_id}")

    assert refused.status_code == 409
    assert "held" in refused.json()["detail"]
    assert client.get(f"/api/splits/{seeded.split_id}").status_code == 200
    listed = client.get("/api/splits", params={"dataset_id": seeded.dataset_id}).json()
    assert listed[0]["experiments"] == [{"experiment_id": experiment["id"], "name": "held"}]


def test_deleting_a_split_that_does_not_exist_is_a_404(client: TestClient) -> None:
    assert client.delete("/api/splits/999").status_code == 404
    assert client.get("/api/splits/999/deletion-preview").status_code == 404
