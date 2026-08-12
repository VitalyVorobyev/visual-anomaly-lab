"""Local public benchmark discovery and one-action registration."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient
from PIL import Image

from anomaly_lab.api.app import create_app
from anomaly_lab.config import Settings


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
