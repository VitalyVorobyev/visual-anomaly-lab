"""End to end for a published dataset: table → catalog → masks → the published split.

The three claims under test are the ones that make a benchmark number trustworthy:
ground truth reaches the catalog, `verify` covers it afterwards, and the partition a
split is built on is the source's own rather than one we drew.
"""

from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from anomaly_lab.db.connection import connection
from anomaly_lab.db.repositories import images as images_repo
from anomaly_lab.db.repositories import masks as masks_repo
from anomaly_lab.db.repositories import samples as samples_repo
from anomaly_lab.domain.entities import Label, MaskKind
from tests.conftest import write_image

TERMINAL_WAIT_SECONDS = 60.0

TRAIN_NORMAL = 4
TEST_NORMAL = 2
TEST_ANOMALY = 3


@pytest.fixture
def published(tmp_path: Path) -> Path:
    """A tree with a split table and masks, in the shape a benchmark publishes."""
    root = tmp_path / "source"
    rows: list[list[str]] = []

    for index in range(TRAIN_NORMAL):
        image = f"widget/Images/Normal/t{index}.png"
        write_image(root / image)
        rows.append(["widget", "train", "normal", image, ""])

    for index in range(TEST_NORMAL):
        image = f"widget/Images/Normal/v{index}.png"
        write_image(root / image)
        rows.append(["widget", "test", "normal", image, ""])

    for index in range(TEST_ANOMALY):
        image = f"widget/Images/Anomaly/{index}.png"
        mask = f"widget/Masks/Anomaly/{index}.png"
        write_image(root / image)
        write_image(root / mask, mode="L")
        rows.append(["widget", "test", "anomaly", image, mask])

    table = root / "split_csv" / "1cls.csv"
    table.parent.mkdir(parents=True, exist_ok=True)
    with table.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["object", "split", "label", "image", "mask"])
        writer.writerows(rows)
    return root


def _await_job(client: TestClient, job_id: int) -> dict[str, Any]:
    deadline = time.monotonic() + TERMINAL_WAIT_SECONDS
    while time.monotonic() < deadline:
        payload: dict[str, Any] = client.get(f"/api/jobs/{job_id}").json()
        if payload["status"] in {"succeeded", "failed", "cancelled"}:
            return payload
        time.sleep(0.05)
    msg = f"job {job_id} never finished"
    raise AssertionError(msg)


def _import(client: TestClient, root: Path, **options: Any) -> dict[str, Any]:
    """Scan and commit in one step, the way the import screen does it."""
    started = client.post(
        "/api/import/scan",
        json={
            "root_path": str(root),
            "dataset_name": "widget",
            "adapter": "csv_table",
            "options": {"csv_path": "split_csv/1cls.csv", **options},
        },
    )
    assert started.status_code == 200, started.text
    job = _await_job(client, started.json()["id"])
    assert job["status"] == "succeeded", job

    manifest = client.get(f"/api/import/manifests/{job['result']['manifest_id']}")
    assert manifest.status_code == 200, manifest.text

    committed = client.post("/api/import/commit", json={"manifest": manifest.json()})
    assert committed.status_code == 200, committed.text
    return dict(committed.json())


# -- masks reach the catalog -----------------------------------------------------------


def test_masks_are_committed_alongside_their_images(
    client: TestClient, published: Path, settings: Any
) -> None:
    result = _import(client, published)

    assert result["masks"] == TEST_ANOMALY
    with connection(settings.db_path) as conn:
        stored = masks_repo.list_masks_for_dataset(conn, result["dataset_id"])

    assert len(stored) == TEST_ANOMALY
    assert all(mask.kind is MaskKind.GROUND_TRUTH for mask in stored)
    assert all(Path(mask.path).is_file() for mask in stored)


def test_a_mask_belongs_to_the_image_it_annotates(
    client: TestClient, published: Path, settings: Any
) -> None:
    """A mask on the wrong image is worse than none: it would score as a silent miss."""
    result = _import(client, published)

    with connection(settings.db_path) as conn:
        images = images_repo.list_images_for_dataset(conn, result["dataset_id"])
        by_image = masks_repo.list_masks_for_images(conn, [image.id for image in images])

    for image in images:
        found = by_image[image.id]
        if "Anomaly" in image.path:
            assert len(found) == 1
            assert Path(found[0].path).stem == Path(image.path).stem
        else:
            assert found == [], "a normal image has nothing to annotate"


def test_re_importing_repoints_masks_rather_than_duplicating_them(
    client: TestClient, published: Path, settings: Any
) -> None:
    _import(client, published)
    second = _import(client, published)

    with connection(settings.db_path) as conn:
        stored = masks_repo.count_masks_for_dataset(conn, second["dataset_id"])

    assert stored == TEST_ANOMALY


def test_a_dataset_without_masks_gets_none(client: TestClient, tmp_path: Path) -> None:
    root = tmp_path / "plain"
    for index in range(2):
        write_image(root / "good" / f"{index}.png")

    started = client.post(
        "/api/import/scan",
        json={
            "root_path": str(root),
            "dataset_name": "plain",
            "adapter": "folder_classes",
            "options": {"normal_dirs": ["good"]},
        },
    )
    job = _await_job(client, started.json()["id"])
    manifest = client.get(f"/api/import/manifests/{job['result']['manifest_id']}").json()
    result = client.post("/api/import/commit", json={"manifest": manifest}).json()

    assert result["masks"] == 0
    assert result["samples_created"] == 2


# -- verify covers them ----------------------------------------------------------------


def test_verify_walks_masks_and_reports_a_missing_one(client: TestClient, published: Path) -> None:
    result = _import(client, published)

    started = client.post("/api/import/verify", json={"dataset_id": result["dataset_id"]})
    job = _await_job(client, started.json()["id"])

    assert job["status"] == "succeeded"
    assert job["result"]["masks_checked"] == TEST_ANOMALY
    assert job["result"]["masks_verified"] == TEST_ANOMALY
    assert job["result"]["masks_missing_count"] == 0

    (published / "widget" / "Masks" / "Anomaly" / "0.png").unlink()
    again = _await_job(
        client,
        client.post("/api/import/verify", json={"dataset_id": result["dataset_id"]}).json()["id"],
    )

    assert again["result"]["masks_missing_count"] == 1
    assert again["result"]["masks_verified"] == TEST_ANOMALY - 1
    # The images themselves are untouched, so the two reports stay separable.
    assert again["result"]["missing_count"] == 0


# -- the published split -----------------------------------------------------------------


def _create_split(client: TestClient, dataset_id: int, **body: Any) -> Any:
    return client.post(
        "/api/splits",
        json={"dataset_id": dataset_id, "name": "official", **body},
    )


def test_an_imported_split_reproduces_the_published_partition(
    client: TestClient, published: Path
) -> None:
    result = _import(client, published)

    response = _create_split(client, result["dataset_id"], params={"strategy": "imported"})
    assert response.status_code == 200, response.text
    split = response.json()

    composition = {row["subset"]: row for row in split["composition"]}
    assert composition["train"]["total"] == TRAIN_NORMAL
    assert composition["train"]["defect"] == 0
    assert composition["test"]["total"] == TEST_NORMAL + TEST_ANOMALY
    assert composition["test"]["defect"] == TEST_ANOMALY


def test_an_empty_validation_subset_is_ordinary_not_a_failure(
    client: TestClient, published: Path
) -> None:
    """The official one-class protocols have no val subset at all (ADR-0011 amended)."""
    result = _import(client, published)

    split = _create_split(client, result["dataset_id"], params={"strategy": "imported"}).json()

    subsets = {row["subset"] for row in split["composition"]}
    assert (
        "val" not in subsets
        or next(row["total"] for row in split["composition"] if row["subset"] == "val") == 0
    )


def test_the_split_records_which_import_asserted_it(client: TestClient, published: Path) -> None:
    """A seed reproduces nothing here; the manifest is what the partition came from."""
    result = _import(client, published)

    split = _create_split(client, result["dataset_id"], params={"strategy": "imported"}).json()

    assert split["strategy"] == "imported"
    assert split["params"]["manifest_id"] == result["manifest_id"]


def test_importing_a_split_a_dataset_never_published_is_refused(
    client: TestClient, tmp_path: Path
) -> None:
    root = tmp_path / "unsplit"
    for index in range(2):
        write_image(root / "good" / f"{index}.png")
    started = client.post(
        "/api/import/scan",
        json={
            "root_path": str(root),
            "dataset_name": "unsplit",
            "adapter": "folder_classes",
            "options": {"normal_dirs": ["good"]},
        },
    )
    job = _await_job(client, started.json()["id"])
    manifest = client.get(f"/api/import/manifests/{job['result']['manifest_id']}").json()
    result = client.post("/api/import/commit", json={"manifest": manifest}).json()

    response = _create_split(client, result["dataset_id"], params={"strategy": "imported"})

    assert response.status_code == 409
    assert "without split information" in response.json()["detail"]


def test_a_seeded_split_still_works_on_an_imported_dataset(
    client: TestClient, published: Path
) -> None:
    """Adopting a partition is a choice, not a lock-in."""
    result = _import(client, published)

    response = _create_split(
        client, result["dataset_id"], name="ours", seed=7, params={"strategy": "normal_only_train"}
    )

    assert response.status_code == 200, response.text
    assert response.json()["strategy"] == "normal_only_train"


# -- the defect type survives ------------------------------------------------------------


def test_a_defect_type_read_at_import_reaches_the_sample(
    client: TestClient, tmp_path: Path, settings: Any
) -> None:
    root = tmp_path / "typed"
    for index in range(2):
        write_image(root / "Nick" / f"{index}.png")
    write_image(root / "Scratch" / "0.png")

    started = client.post(
        "/api/import/scan",
        json={
            "root_path": str(root),
            "dataset_name": "typed",
            "adapter": "folder_classes",
            "options": {"defect_dirs": ["Nick", "Scratch"]},
        },
    )
    job = _await_job(client, started.json()["id"])
    manifest = client.get(f"/api/import/manifests/{job['result']['manifest_id']}").json()
    result = client.post("/api/import/commit", json={"manifest": manifest}).json()

    with connection(settings.db_path) as conn:
        stored = samples_repo.list_samples(conn, result["dataset_id"], limit=100, offset=0)

    assert {sample.notes for sample in stored} == {"Nick", "Scratch"}
    assert all(sample.label is Label.DEFECT for sample in stored)
