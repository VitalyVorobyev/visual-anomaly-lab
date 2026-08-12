"""Region profile catalogue and immutable-revision API."""

from __future__ import annotations

from fastapi.testclient import TestClient

from tests.conftest import SeededCatalog


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
