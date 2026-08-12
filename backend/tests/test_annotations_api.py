"""Versioned annotation API, using only generated PNG fixtures."""

from __future__ import annotations

import io
import sqlite3
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from anomaly_lab.config import Settings
from anomaly_lab.datasets.verify import run_verify_job
from anomaly_lab.db.connection import connection
from anomaly_lab.db.repositories import masks as masks_repo
from anomaly_lab.domain.entities import JobKind
from anomaly_lab.jobs.context import JobContext
from anomaly_lab.media.decode import sha256_of

from .conftest import Fixture


def _open(client: TestClient, image_id: int) -> tuple[dict[str, object], str]:
    response = client.post(f"/api/images/{image_id}/annotations/draft")
    assert response.status_code == 200, response.text
    assert response.headers["etag"].startswith('"annotation-draft-')
    return response.json(), response.headers["etag"]


def _triangle(document: dict[str, object]) -> dict[str, object]:
    return {
        **document,
        "shapes": [
            {
                "id": "defect-1",
                "label_key": "defect",
                "kind": "polygon",
                "operation": "add",
                "points": [{"x": 2, "y": 2}, {"x": 12, "y": 2}, {"x": 2, "y": 12}],
            }
        ],
    }


def test_dataset_gets_a_stable_default_defect_taxonomy(client: TestClient, seeded: Fixture) -> None:
    first = client.get(f"/api/datasets/{seeded.dataset_id}/annotation-labels")
    second = client.get(f"/api/datasets/{seeded.dataset_id}/annotation-labels")

    assert first.status_code == 200
    assert first.json() == second.json()
    assert [(item["key"], item["name"]) for item in first.json()] == [("defect", "Defect")]

    created = client.post(
        f"/api/datasets/{seeded.dataset_id}/annotation-labels",
        json={"key": "scratch", "name": "Scratch", "color": "#4F46E5", "position": 1},
    )
    assert created.status_code == 200
    assert created.json()["color"] == "#4f46e5"
    duplicate = client.post(
        f"/api/datasets/{seeded.dataset_id}/annotation-labels",
        json={"key": "scratch", "name": "Duplicate", "color": "#000000"},
    )
    assert duplicate.status_code == 409

    updated = client.put(
        f"/api/datasets/{seeded.dataset_id}/annotation-labels/scratch",
        json={"name": "Surface scratch", "color": "#123456", "position": 2},
    )
    assert updated.status_code == 200
    assert updated.json()["key"] == "scratch"
    assert updated.json()["name"] == "Surface scratch"


def test_draft_save_is_etag_guarded_and_completion_materialises_a_binary_png(
    client: TestClient, settings: Settings, seeded: Fixture
) -> None:
    image_id = seeded.normal_image_ids[0]
    opened, original_etag = _open(client, image_id)
    document = _triangle(opened["document"])  # type: ignore[arg-type]

    missing = client.put(f"/api/images/{image_id}/annotations/draft", json=document)
    assert missing.status_code == 428

    saved = client.put(
        f"/api/images/{image_id}/annotations/draft",
        json=document,
        headers={"If-Match": original_etag},
    )
    assert saved.status_code == 200, saved.text
    saved_etag = saved.headers["etag"]
    assert saved_etag != original_etag
    assert saved.json()["version"] == 2

    stale = client.put(
        f"/api/images/{image_id}/annotations/draft",
        json=document,
        headers={"If-Match": original_etag},
    )
    assert stale.status_code == 412

    completed = client.post(
        f"/api/images/{image_id}/annotations/complete",
        headers={"If-Match": saved_etag},
    )
    assert completed.status_code == 200, completed.text
    revision = completed.json()
    assert revision["revision_no"] == 1
    assert len(revision["document_sha256"]) == 64
    assert len(revision["mask_sha256"]) == 64
    assert client.get(f"/api/images/{image_id}/annotations/draft").status_code == 404

    mask = client.get(f"/api/images/{image_id}/annotations/revisions/{revision['id']}/mask")
    assert mask.status_code == 200
    with Image.open(io.BytesIO(mask.content)) as opened_mask:
        values = np.asarray(opened_mask)
    assert set(np.unique(values)) <= {0, 255}
    assert values[3, 3] == 255
    assert values[-1, -1] == 0
    assert Path(revision["mask_path"]).is_relative_to(settings.annotations_dir)


def test_source_mask_is_pinned_but_never_rewritten(
    client: TestClient, settings: Settings, seeded: Fixture
) -> None:
    image_id = seeded.defect_image_ids[0]
    with connection(settings.db_path) as conn:
        source = masks_repo.get_mask_for_image(conn, image_id)
    assert source is not None
    source_path = Path(source.path)
    before = source_path.read_bytes()

    draft, etag = _open(client, image_id)
    assert draft["document"]["base"] == "source_mask"  # type: ignore[index]
    assert draft["source_mask_sha256"] == sha256_of(source_path)

    completed = client.post(
        f"/api/images/{image_id}/annotations/complete", headers={"If-Match": etag}
    )
    assert completed.status_code == 200, completed.text
    assert source_path.read_bytes() == before

    generated = Path(completed.json()["mask_path"])
    assert generated != source_path
    with Image.open(source_path) as expected, Image.open(generated) as actual:
        assert np.array_equal(np.asarray(expected) > 0, np.asarray(actual) > 0)

    with connection(settings.db_path) as conn:
        catalogued = masks_repo.get_mask_for_image(conn, image_id)
    assert catalogued is not None
    assert catalogued.sha256 == sha256_of(source_path)

    report = run_verify_job(
        JobContext(
            job_id=991,
            kind=JobKind.VERIFY,
            params={"dataset_id": seeded.dataset_id},
            settings=settings,
        )
    )
    assert report["masks_digest_checked"] == 1
    assert report["masks_modified_count"] == 0


def test_completed_revisions_are_immutable_in_sqlite(
    client: TestClient, settings: Settings, seeded: Fixture
) -> None:
    image_id = seeded.normal_image_ids[0]
    _, etag = _open(client, image_id)
    completed = client.post(
        f"/api/images/{image_id}/annotations/complete", headers={"If-Match": etag}
    )
    revision_id = completed.json()["id"]

    with (
        connection(settings.db_path) as conn,
        pytest.raises(sqlite3.IntegrityError, match="immutable"),
    ):
        conn.execute(
            "UPDATE annotation_revision SET mask_sha256 = 'rewritten' WHERE id = ?",
            (revision_id,),
        )


def test_dataset_deletion_removes_app_annotations_but_not_source_masks(
    client: TestClient, settings: Settings, seeded: Fixture
) -> None:
    image_id = seeded.defect_image_ids[0]
    with connection(settings.db_path) as conn:
        source = masks_repo.get_mask_for_image(conn, image_id)
    assert source is not None
    source_path = Path(source.path)
    source_bytes = source_path.read_bytes()

    _, etag = _open(client, image_id)
    completed = client.post(
        f"/api/images/{image_id}/annotations/complete", headers={"If-Match": etag}
    )
    generated = Path(completed.json()["mask_path"])
    assert generated.is_file()

    preview = client.get(f"/api/datasets/{seeded.dataset_id}/deletion-preview")
    assert preview.status_code == 200
    assert preview.json()["generated_files"] >= 1

    deleted = client.delete(f"/api/datasets/{seeded.dataset_id}")
    assert deleted.status_code == 200, deleted.text
    assert not generated.exists()
    assert source_path.read_bytes() == source_bytes

    with connection(settings.db_path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM annotation_revision").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM annotation_label").fetchone()[0] == 0


def test_symlinked_annotation_storage_is_refused(
    client: TestClient, settings: Settings, seeded: Fixture, tmp_path: Path
) -> None:
    outside = tmp_path / "outside-annotation-storage"
    outside.mkdir()
    settings.annotations_dir.symlink_to(outside, target_is_directory=True)
    image_id = seeded.normal_image_ids[0]
    _, etag = _open(client, image_id)

    completed = client.post(
        f"/api/images/{image_id}/annotations/complete", headers={"If-Match": etag}
    )
    assert completed.status_code == 409
    assert list(outside.iterdir()) == []

    preview = client.get(f"/api/datasets/{seeded.dataset_id}/deletion-preview")
    assert preview.status_code == 200
    assert preview.json()["storage_locations_safe"] is False
    assert preview.json()["can_delete"] is False


def test_document_cannot_change_its_frame_base_or_use_an_unknown_label(
    client: TestClient, seeded: Fixture
) -> None:
    image_id = seeded.normal_image_ids[0]
    opened, etag = _open(client, image_id)
    document = opened["document"]
    assert isinstance(document, dict)

    wrong_size = {**document, "image_width": int(document["image_width"]) + 1}
    response = client.put(
        f"/api/images/{image_id}/annotations/draft",
        json=wrong_size,
        headers={"If-Match": etag},
    )
    assert response.status_code == 422

    wrong_base = {**document, "base": "source_mask"}
    response = client.put(
        f"/api/images/{image_id}/annotations/draft",
        json=wrong_base,
        headers={"If-Match": etag},
    )
    assert response.status_code == 422

    unknown = _triangle(document)
    unknown["shapes"][0]["label_key"] = "scratch"  # type: ignore[index]
    response = client.put(
        f"/api/images/{image_id}/annotations/draft",
        json=unknown,
        headers={"If-Match": etag},
    )
    assert response.status_code == 422
