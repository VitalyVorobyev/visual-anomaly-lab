"""Local public benchmark discovery and one-action registration."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient
from PIL import Image

from anomaly_lab.api.app import create_app
from anomaly_lab.config import Settings
from anomaly_lab.datasets.reference_packs import FSS_PANEL
from anomaly_lab.datasets.splitting import plan_few_shot_split
from anomaly_lab.db.connection import connection
from anomaly_lab.db.repositories import annotations as annotations_repo
from anomaly_lab.db.repositories import datasets as datasets_repo
from anomaly_lab.domain.entities import ClassPresence, Subset


def _write_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (4, 4), (128, 128, 128)).save(path)


def _wait(client: TestClient, job_id: int) -> dict[str, Any]:
    for _ in range(200):
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] in {"succeeded", "failed", "cancelled"}:
            return dict(job)
        time.sleep(0.02)
    raise AssertionError("reference registration did not finish")


def test_absent_packs_are_instructional_and_gkn_registers_in_one_action(
    tmp_path: Path,
) -> None:
    references = tmp_path / "references"
    gkn = references / "GKN Blade Surface Defect Dataset" / "Data_GKN"
    _write_image(gkn / "Good" / "good.png")
    _write_image(gkn / "Nick" / "nick.png")
    _write_image(gkn / "Scratch" / "scratch.png")
    settings = Settings(data_dir=tmp_path / "data", reference_datasets_dir=references)

    with TestClient(create_app(settings)) as client:
        catalog = client.get("/api/reference-packs").json()
        assert [(pack["key"], pack["status"]) for pack in catalog["packs"]] == [
            ("visa", "absent"),
            ("gkn", "available"),
            ("fss1000", "absent"),
        ]
        assert catalog["pending_datasets"] == 1
        assert catalog["packs"][0]["install_url"].startswith("https://")

        started = client.post("/api/reference-packs/register", json={"pack_keys": ["gkn"]})
        assert started.status_code == 200, started.text
        job = _wait(client, started.json()["id"])
        assert job["status"] == "succeeded", job
        assert job["result"]["registered"] == 1

        datasets = client.get("/api/datasets").json()
        assert [(dataset["name"], dataset["samples"]) for dataset in datasets] == [
            ("GKN Blade Surface Defect", 3)
        ]
        assert datasets[0]["label_counts"] == {
            "normal": 1,
            "defect": 2,
            "unlabeled": 0,
        }

        after = client.get("/api/reference-packs").json()
        assert after["packs"][1]["status"] == "registered"
        assert after["pending_datasets"] == 0
        assert (
            client.post("/api/reference-packs/register", json={"pack_keys": ["gkn"]}).status_code
            == 409
        )


def test_a_registered_dataset_inherits_its_pack_collection_and_blurb(tmp_path: Path) -> None:
    """Membership is derived, so it holds for datasets registered before the column existed."""
    references = tmp_path / "references"
    gkn = references / "GKN Blade Surface Defect Dataset" / "Data_GKN"
    _write_image(gkn / "Good" / "good.png")
    _write_image(gkn / "Nick" / "nick.png")
    _write_image(gkn / "Scratch" / "scratch.png")
    settings = Settings(data_dir=tmp_path / "data", reference_datasets_dir=references)

    with TestClient(create_app(settings)) as client:
        started = client.post("/api/reference-packs/register", json={"pack_keys": ["gkn"]})
        _wait(client, started.json()["id"])

        registered = client.get("/api/datasets").json()[0]
        # Nothing was written at registration; both values come from the pack spec.
        assert registered["notes"] is None
        assert registered["collection"] == "GKN"
        assert "turbine blades" in registered["description"]

        dataset_id = registered["id"]
        moved = client.patch(
            f"/api/datasets/{dataset_id}",
            json={"collection": "Blades", "notes": "My working copy."},
        ).json()
        assert (moved["collection"], moved["description"]) == ("Blades", "My working copy.")

        # Clearing the override falls back to the pack rather than to nothing.
        restored = client.patch(
            f"/api/datasets/{dataset_id}", json={"collection": None, "notes": None}
        ).json()
        assert restored["collection"] == "GKN"
        assert "turbine blades" in restored["description"]


def test_a_dataset_that_merely_shares_a_name_inherits_nothing(tmp_path: Path) -> None:
    """The matcher guards this; a user's own `candle` must not be filed under VisA."""
    settings = Settings(data_dir=tmp_path / "data", reference_datasets_dir=tmp_path / "references")
    with TestClient(create_app(settings)) as client:
        with connection(settings.db_path) as conn:
            datasets_repo.create_dataset(
                conn, name="candle", root_path=str(tmp_path / "mine"), adapter="csv_table"
            )

        mine = client.get("/api/datasets").json()[0]
        assert mine["collection"] is None
        assert mine["description"] is None


def test_an_incomplete_pack_is_not_offered_for_registration(tmp_path: Path) -> None:
    references = tmp_path / "references"
    (references / "VisA_20220922").mkdir(parents=True)
    settings = Settings(data_dir=tmp_path / "data", reference_datasets_dir=references)
    with TestClient(create_app(settings)) as client:
        visa = client.get("/api/reference-packs").json()["packs"][0]
        assert visa["status"] == "incomplete"
        assert visa["missing"]
        response = client.post("/api/reference-packs/register", json={"pack_keys": ["visa"]})
        assert response.status_code == 409


def _write_fss_pair(directory: Path, index: int, shade: int) -> None:
    """One FSS-1000-shaped pair: `<n>.jpg` and the `<n>.png` mask of the object in it."""
    directory.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), (shade, index * 40, 90)).save(directory / f"{index}.jpg")
    mask = Image.new("L", (8, 8), 0)
    mask.paste(255, (2, 2, 6, 6))
    mask.save(directory / f"{index}.png")


def test_fss1000_registers_one_few_shot_dataset_per_panel_class(tmp_path: Path) -> None:
    """A target's own masks are its truth; the rest of the panel are confirmed absences."""
    references = tmp_path / "references"
    classes = references / "FSS-1000" / "fewshot_data"
    for position, name in enumerate(FSS_PANEL):
        for index in (1, 2):
            _write_fss_pair(classes / name, index, shade=position * 12)
    # Outside the panel, and a stray `.jpeg` beside a paired `.jpg`: neither is imported.
    _write_fss_pair(classes / "not_in_panel", 1, shade=250)
    Image.new("RGB", (8, 8), (1, 2, 3)).save(classes / FSS_PANEL[0] / "1.jpeg")
    settings = Settings(data_dir=tmp_path / "data", reference_datasets_dir=references)

    with TestClient(create_app(settings)) as client:
        fss = client.get("/api/reference-packs").json()["packs"][2]
        assert (fss["key"], fss["status"], len(fss["datasets"])) == ("fss1000", "available", 20)

        started = client.post("/api/reference-packs/register", json={"pack_keys": ["fss1000"]})
        job = _wait(client, started.json()["id"])
        assert job["status"] == "succeeded", job
        assert job["result"]["registered"] == 20

        datasets = {dataset["name"]: dataset for dataset in client.get("/api/datasets").json()}
        target = datasets[FSS_PANEL[0]]
        assert target["samples"] == 2 * 20
        assert target["label_counts"] == {"normal": 38, "defect": 2, "unlabeled": 0}
        assert target["collection"] == "FSS-1000"

    with connection(settings.db_path) as conn:
        presence = annotations_repo.class_presence(conn, target["id"], "defect")
        assert sum(found is ClassPresence.PRESENT for found in presence.values()) == 2
        assert sum(found is ClassPresence.ABSENT for found in presence.values()) == 38
        split = plan_few_shot_split(conn, target["id"], seed=0, label_key="defect", shots=1)
    references_drawn = [sample for sample, subset in split.items() if subset is Subset.TRAIN]
    assert len(references_drawn) == 1
    assert presence[references_drawn[0]] is ClassPresence.PRESENT
