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


# ------------------------------------------------------------------ from preview to truth


def _region(client: TestClient, image_id: int, **body: Any) -> Any:
    return client.post(
        f"/api/images/{image_id}/studio/region", json={"class_key": "defect", **body}
    )


def test_accepting_a_preview_makes_its_region_the_class_s_truth(
    client: TestClient, settings: Settings, seeded: Fixture
) -> None:
    from anomaly_lab.db.repositories import annotations as annotations_repo

    profile = create_experiment(client, seeded)["region_profile_id"]
    reference = _sample_of(settings, seeded.defect_image_ids[0])
    query = seeded.defect_image_ids[1]
    preview = _preview(client, seeded, profile_id=profile, references=[reference], image_id=query)
    generation = preview.json()["generation"]

    accepted = _region(client, query, action="accept", generation=generation)
    assert accepted.status_code == 200, accepted.text
    assert accepted.json()["completed"] is True
    revision = client.get(f"/api/images/{query}/annotations/revisions").json()[-1]
    table = {entry["key"]: entry["pixels"] for entry in revision["class_table"]}
    assert table["defect"] > 0

    # Confirming it absent removes the region and completes a revision that says so.
    absent = _region(client, query, action="absent")
    assert absent.status_code == 200, absent.text
    with connection(settings.db_path) as conn:
        found = annotations_repo.class_presence(conn, seeded.dataset_id, "defect")
    assert found[_sample_of(settings, query)].value == "absent"


def test_fixing_opens_a_draft_and_an_open_draft_is_never_completed_behind_it(
    client: TestClient, settings: Settings, seeded: Fixture
) -> None:
    profile = create_experiment(client, seeded)["region_profile_id"]
    reference = _sample_of(settings, seeded.defect_image_ids[0])
    query = seeded.defect_image_ids[2]
    generation = _preview(
        client, seeded, profile_id=profile, references=[reference], image_id=query
    ).json()["generation"]

    fixed = _region(client, query, action="fix", generation=generation)
    assert fixed.status_code == 200, fixed.text
    assert fixed.json()["completed"] is False
    draft = client.get(f"/api/images/{query}/annotations/draft").json()
    assert draft["persisted"] is True
    assert any(shape["id"].startswith("studio-") for shape in draft["document"]["shapes"])

    refused = _region(client, query, action="accept", generation=generation)
    assert refused.status_code == 409
    assert "open annotation draft" in refused.text
    assert _region(client, query, action="accept").status_code == 422


def test_other_classes_keep_their_shapes(
    client: TestClient, settings: Settings, seeded: Fixture
) -> None:
    client.post(
        f"/api/datasets/{seeded.dataset_id}/annotation-labels",
        json={"key": "scratch", "name": "Scratch", "color": "#00aa00", "position": 1},
    )
    image_id = seeded.normal_image_ids[0]
    seed = client.get(f"/api/images/{image_id}/annotations/draft").json()["document"]
    scratch = {
        "id": "scratch-1",
        "label_key": "scratch",
        "kind": "polygon",
        "operation": "add",
        "points": [{"x": 1, "y": 1}, {"x": 6, "y": 1}, {"x": 6, "y": 6}],
    }
    created = client.post(
        f"/api/images/{image_id}/annotations/draft",
        json={**seed, "shapes": [scratch]},
        headers={"If-None-Match": "*"},
    )
    client.post(
        f"/api/images/{image_id}/annotations/complete",
        headers={"If-Match": created.headers["etag"]},
    )

    absent = _region(client, image_id, action="absent")
    assert absent.status_code == 200, absent.text
    revision = client.get(f"/api/images/{image_id}/annotations/revisions").json()[-1]
    assert revision["document"]["shapes"][0] == scratch
    table = {entry["key"]: entry["pixels"] for entry in revision["class_table"]}
    assert table["scratch"] > 0 and table["defect"] == 0
