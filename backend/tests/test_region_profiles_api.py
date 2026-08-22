"""Region profile catalogue and immutable-revision API."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from anomaly_lab.config import Settings
from anomaly_lab.db.connection import connection
from anomaly_lab.domain.entities import JobKind
from anomaly_lab.jobs.context import JobContext
from anomaly_lab.regions.preparation import run_region_prepare_job
from tests.conftest import Fixture, SeededCatalog, create_experiment, write_image


def test_extractor_catalogue_exposes_schema_without_loading_a_checkpoint(
    client: TestClient,
) -> None:
    response = client.get("/api/region-extractors")

    assert response.status_code == 200
    items = response.json()
    assert [item["key"] for item in items] == [
        "identity",
        "foreground_threshold",
        "mobile_sam",
    ]
    threshold = next(item for item in items if item["key"] == "foreground_threshold")
    assert threshold["config_schema"]["properties"]["max_analysis_side"]["maximum"] == 2048


def test_profile_revisions_validate_schema_and_append(
    client: TestClient, catalog: SeededCatalog
) -> None:
    dataset_id = catalog.dataset_id
    body = {
        "name": "Dominant object",
        "extractor_type": "foreground_threshold",
        "extractor_config": {"min_contrast": 18},
        "prepared_width": 256,
        "prepared_height": 192,
        "padding_fraction": 0.05,
        "seed": 91,
    }
    first = client.post(f"/api/datasets/{dataset_id}/region-profiles", json=body)
    second = client.post(
        f"/api/datasets/{dataset_id}/region-profiles",
        json={**body, "extractor_config": {"min_contrast": 24}},
    )

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json()["revision_no"] == 1
    assert second.json()["revision_no"] == 2
    assert first.json()["failure_policy"] == "fail"
    listed = client.get(f"/api/datasets/{dataset_id}/region-profiles")
    assert [item["revision_no"] for item in listed.json()] == [2, 1]
    fetched = client.get(f"/api/region-profiles/{first.json()['id']}")
    assert fetched.json() == first.json()


def test_profile_creation_rejects_unknown_or_invalid_extractor_config(
    client: TestClient, catalog: SeededCatalog
) -> None:
    dataset_id = catalog.dataset_id
    base = {
        "name": "Broken",
        "prepared_width": 256,
        "prepared_height": 256,
        "seed": 17,
    }

    unknown = client.post(
        f"/api/datasets/{dataset_id}/region-profiles",
        json={**base, "extractor_type": "magic", "extractor_config": {}},
    )
    invalid = client.post(
        f"/api/datasets/{dataset_id}/region-profiles",
        json={
            **base,
            "extractor_type": "mobile_sam",
            "extractor_config": {"points_per_side": 200},
        },
    )

    assert unknown.status_code == 422
    assert invalid.status_code == 422


def test_preview_and_build_routes_expose_a_persisted_visual_audit(
    client: TestClient,
    catalog: SeededCatalog,
    settings: Settings,
    tmp_path: Path,
) -> None:
    with connection(settings.db_path) as conn:
        for image_id in catalog.image_ids:
            path = write_image(tmp_path / "sources" / f"{image_id}.png", size=(16, 12))
            conn.execute(
                "UPDATE image SET path = ?, width = 16, height = 12 WHERE id = ?",
                (str(path), image_id),
            )

    created = client.post(
        f"/api/datasets/{catalog.dataset_id}/region-profiles",
        json={
            "name": "full frame",
            "extractor_type": "identity",
            "extractor_config": {},
            "prepared_width": 20,
            "prepared_height": 18,
            "padding_fraction": 0.0,
            "resample": "bilinear",
            "seed": 17,
        },
    )
    profile_id = int(created.json()["id"])

    preview = client.post(f"/api/region-profiles/{profile_id}/preview")
    preview_job = _wait_for_job(client, int(preview.json()["id"]))
    assert preview_job["status"] == "succeeded"
    assert preview_job["result"]["sampled"] == len(catalog.image_ids)
    assert client.get(f"/api/region-profiles/{profile_id}/build").status_code == 404

    build = client.post(f"/api/region-profiles/{profile_id}/build")
    build_job = _wait_for_job(client, int(build.json()["id"]))
    assert build_job["status"] == "succeeded"

    report = client.get(f"/api/region-profiles/{profile_id}/build")
    assert report.status_code == 200
    assert report.json()["succeeded"] == len(catalog.image_ids)
    assert len(report.json()["preview_entries"]) == len(catalog.image_ids)
    prepared = client.get(f"/api/region-profiles/{profile_id}/prepared/{catalog.image_ids[0]}")
    assert prepared.status_code == 200
    assert prepared.headers["content-type"] == "image/png"


def test_deleting_a_profile_removes_its_row_and_its_prepared_pixels(
    client: TestClient, settings: Settings, seeded: Fixture
) -> None:
    """A preset the operator no longer wants, and the disk it was holding."""
    profile_id = seeded.region_profile_id
    run_region_prepare_job(
        JobContext(
            job_id=61,
            kind=JobKind.REGION_PREPARE,
            params={"dataset_id": seeded.dataset_id, "profile_id": profile_id, "mode": "build"},
            settings=settings,
        )
    )
    prepared = settings.region_profile_dir(profile_id)
    assert prepared.exists()

    preview = client.get(f"/api/region-profiles/{profile_id}/deletion-preview")
    assert preview.status_code == 200
    assert preview.json()["can_delete"] is True
    assert preview.json()["generated_files"] > 0

    removed = client.delete(f"/api/region-profiles/{profile_id}")

    assert removed.status_code == 200
    assert removed.json()["deleted"] is True
    assert removed.json()["prepared_removed"] is True
    assert removed.json()["freed_files"] > 0
    assert not prepared.exists()
    assert client.get(f"/api/region-profiles/{profile_id}").status_code == 404


def test_a_profile_an_experiment_pins_is_refused_by_name(
    client: TestClient, seeded: Fixture
) -> None:
    """`ON DELETE RESTRICT` would raise an `IntegrityError` naming a constraint.

    That is not an answer anyone can act on, so the experiments are counted first and the
    refusal names them — both on the preview, which greys the button, and on the route,
    which is what a second client racing the first one meets.
    """
    experiment = create_experiment(client, seeded)

    preview = client.get(f"/api/region-profiles/{seeded.region_profile_id}/deletion-preview")
    assert preview.status_code == 200
    assert preview.json()["can_delete"] is False
    assert preview.json()["experiments"] == [
        {"experiment_id": experiment["id"], "name": experiment["name"]}
    ]
    assert "still use this input" in preview.json()["blocker"]

    refused = client.delete(f"/api/region-profiles/{seeded.region_profile_id}")

    assert refused.status_code == 409
    assert "delete them first" in refused.json()["detail"]
    assert client.get(f"/api/region-profiles/{seeded.region_profile_id}").status_code == 200


def test_a_profile_with_preparation_work_in_flight_is_refused(
    client: TestClient, settings: Settings, seeded: Fixture
) -> None:
    """The same guard `_enqueue_preparation` uses to refuse a second concurrent build."""
    profile_id = seeded.region_profile_id
    with connection(settings.db_path) as conn:
        conn.execute(
            """
            INSERT INTO job (kind, status, params)
            VALUES ('region_prepare', 'running', ?)
            """,
            (json.dumps({"dataset_id": seeded.dataset_id, "profile_id": profile_id}),),
        )
        conn.commit()

    preview = client.get(f"/api/region-profiles/{profile_id}/deletion-preview")
    assert preview.json()["can_delete"] is False
    assert len(preview.json()["active_jobs"]) == 1

    refused = client.delete(f"/api/region-profiles/{profile_id}")

    assert refused.status_code == 409
    assert "cancel or wait" in refused.json()["detail"]


def test_deletion_never_follows_a_symlinked_storage_root(
    client: TestClient, settings: Settings, seeded: Fixture, tmp_path: Path
) -> None:
    """Neither preview nor cleanup follows a symlink out of app-owned storage."""
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    root = settings.region_profiles_dir
    if root.exists():
        root.rmdir()
    root.symlink_to(elsewhere, target_is_directory=True)

    preview = client.get(f"/api/region-profiles/{seeded.region_profile_id}/deletion-preview")
    assert preview.json()["storage_location_safe"] is False
    assert preview.json()["can_delete"] is False

    refused = client.delete(f"/api/region-profiles/{seeded.region_profile_id}")

    assert refused.status_code == 409
    assert elsewhere.exists()


def _wait_for_job(client: TestClient, job_id: int) -> dict[str, Any]:
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        payload: dict[str, Any] = client.get(f"/api/jobs/{job_id}").json()
        if payload["status"] in {"succeeded", "failed", "cancelled"}:
            return payload
        time.sleep(0.02)
    raise AssertionError(f"job {job_id} did not finish")
