"""The reference studio's live preview (ADR-0040), through the real resident process.

Torch-free: `color_prototype` fits in milliseconds, so the whole round trip — resolve,
spawn, fit on the references, segment one image, render the map — runs in the default job.
"""

from __future__ import annotations

from typing import Any

from fastapi.testclient import TestClient

from anomaly_lab.config import Settings
from anomaly_lab.db.connection import connection

from .conftest import Fixture, create_experiment


def _sample_of(settings: Settings, image_id: int) -> int:
    with connection(settings.db_path) as conn:
        return int(
            conn.execute("SELECT sample_id FROM image WHERE id = ?", (image_id,)).fetchone()[0]
        )


def _preview(client: TestClient, seeded: Fixture, **body: Any) -> Any:
    payload = {"class_key": "defect", "method": "color_prototype", **body}
    return client.post(f"/api/datasets/{seeded.dataset_id}/studio/preview", json=payload)


def test_references_in_one_image_segmented_out_and_warm_the_second_time(
    client: TestClient, settings: Settings, seeded: Fixture
) -> None:
    profile = create_experiment(client, seeded)["region_profile_id"]
    reference = _sample_of(settings, seeded.defect_image_ids[0])
    query = seeded.defect_image_ids[1]

    first = _preview(client, seeded, profile_id=profile, references=[reference], image_id=query)
    assert first.status_code == 200, first.text
    body = first.json()
    assert body["warm"] is False
    assert 0.0 <= body["score"] <= 1.0
    assert body["foreground_share"] > 0.0
    rendered = client.get(body["map_url"])
    assert rendered.status_code == 200
    assert rendered.headers["content-type"] == "image/png"

    second = _preview(
        client,
        seeded,
        profile_id=profile,
        references=[reference],
        image_id=seeded.normal_image_ids[0],
    )
    assert second.json()["warm"] is True
    assert second.json()["generation"] == body["generation"]

    # New references are a new fit, never the old one served warm.
    other = _preview(
        client,
        seeded,
        profile_id=profile,
        references=[reference, _sample_of(settings, seeded.defect_image_ids[2])],
        image_id=query,
    )
    assert other.json()["warm"] is False
    assert other.json()["generation"] != body["generation"]


def test_a_preview_that_cannot_be_fitted_is_refused_by_name(
    client: TestClient, settings: Settings, seeded: Fixture
) -> None:
    profile = create_experiment(client, seeded)["region_profile_id"]
    reference = _sample_of(settings, seeded.defect_image_ids[0])
    anomaly = _preview(
        client,
        seeded,
        method="pixel_reference",
        profile_id=profile,
        references=[reference],
        image_id=seeded.defect_image_ids[1],
    )
    assert anomaly.status_code == 422
    assert "does not segment a class" in anomaly.text

    unlabelled = _preview(
        client,
        seeded,
        class_key="scratch",
        profile_id=profile,
        references=[reference],
        image_id=seeded.defect_image_ids[1],
    )
    assert unlabelled.status_code == 422

    assert client.get("/api/studio/previews/nothing/1.png").status_code == 404
