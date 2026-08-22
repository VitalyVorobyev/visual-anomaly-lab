"""Sample-scoped annotation editing, fanned out to per-image truth (ADR-0036).

Every fixture here is a generated PNG. The point of the file is the seam: one document
edited once must land as N image-keyed revisions that every existing consumer reads
without knowing scope exists.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from anomaly_lab.config import Settings
from anomaly_lab.db.connection import connection
from anomaly_lab.db.repositories import annotations as annotations_repo
from anomaly_lab.db.repositories import datasets, images, masks, samples
from anomaly_lab.domain.entities import Label
from anomaly_lab.media.decode import sha256_of

from .conftest import FIXTURE_SIZE, write_normal_image

CHANNELS = ("bright", "dark", "dome")
# The recorded frame is the generated file's real size, so a test that does reach for the
# pixels finds the dimensions the catalogue promised.
FRAME = (FIXTURE_SIZE, FIXTURE_SIZE)


def _multishot_dataset(
    settings: Settings,
    root: Path,
    *,
    name: str = "multishot",
    samples_count: int = 2,
    channels: tuple[str, ...] = CHANNELS,
) -> tuple[int, list[int], dict[int, list[int]]]:
    """A dataset of multi-channel samples with no imported masks.

    The `seeded` fixture cannot be used for these tests: it pins a source mask to every
    defect image, which is exactly the condition sample scope refuses.
    """
    with connection(settings.db_path) as conn:
        dataset = datasets.create_dataset(conn, name=name, root_path=str(root))
        channel_ids = [
            datasets.upsert_channel(conn, dataset.id, name=channel, position=index).id
            for index, channel in enumerate(channels)
        ]
        sample_ids: list[int] = []
        images_by_sample: dict[int, list[int]] = {}
        for index in range(samples_count):
            sample, _ = samples.upsert_sample(
                conn,
                dataset.id,
                group_key="all",
                external_id=f"part-{index}",
                label=Label.DEFECT,
            )
            sample_ids.append(sample.id)
            images_by_sample[sample.id] = []
            for channel_id, channel in zip(channel_ids, channels, strict=True):
                path = root / channel / f"{index}.png"
                write_normal_image(path, index * 10 + channel_id)
                image, _ = images.upsert_image(
                    conn,
                    sample.id,
                    channel_id=channel_id,
                    path=str(path),
                    width=FRAME[0],
                    height=FRAME[1],
                    bit_depth=24,
                    file_size=path.stat().st_size,
                    sha256=f"sha-{index}-{channel}",
                )
                images_by_sample[sample.id].append(image.id)
    return dataset.id, sample_ids, images_by_sample


def _use_sample_scope(client: TestClient, dataset_id: int) -> dict[str, Any]:
    response = client.put(f"/api/datasets/{dataset_id}/annotation-scope", json={"scope": "sample"})
    assert response.status_code == 200, response.text
    payload: dict[str, Any] = response.json()
    return payload


def _open_sample_draft(client: TestClient, sample_id: int) -> tuple[dict[str, Any], str]:
    response = client.post(f"/api/samples/{sample_id}/annotations/draft")
    assert response.status_code == 200, response.text
    etag = response.headers["etag"]
    assert etag.startswith('"annotation-sample-draft-')
    payload: dict[str, Any] = response.json()
    return payload, etag


def _triangle(document: dict[str, Any]) -> dict[str, Any]:
    return {
        **document,
        "shapes": [
            {
                "id": "defect-1",
                "label_key": "defect",
                "kind": "polygon",
                "operation": "add",
                "points": [{"x": 2, "y": 2}, {"x": 14, "y": 2}, {"x": 2, "y": 12}],
            }
        ],
    }


def test_scope_defaults_to_image_and_reports_what_sample_scope_would_gain(
    client: TestClient, settings: Settings, tmp_path: Path
) -> None:
    dataset_id, _, _ = _multishot_dataset(settings, tmp_path / "src")

    state = client.get(f"/api/datasets/{dataset_id}/annotation-scope")

    assert state.status_code == 200
    assert state.json()["scope"] == "image"
    assert state.json()["samples"] == 2
    assert state.json()["multi_image_samples"] == 2
    assert state.json()["can_use_sample_scope"] is True
    assert state.json()["blockers"] == []


def test_imported_masks_and_mixed_frames_are_reported_together_not_one_at_a_time(
    client: TestClient, settings: Settings, tmp_path: Path
) -> None:
    root = tmp_path / "src"
    dataset_id, sample_ids, images_by_sample = _multishot_dataset(settings, root)
    with connection(settings.db_path) as conn:
        mask_path = root / "masks" / "0.png"
        write_normal_image(mask_path, 3)
        masks.upsert_mask(conn, images_by_sample[sample_ids[0]][0], path=str(mask_path))
        conn.execute(
            "UPDATE image SET width = 8, height = 8 WHERE id = ?",
            (images_by_sample[sample_ids[1]][2],),
        )

    state = client.get(f"/api/datasets/{dataset_id}/annotation-scope").json()

    assert state["can_use_sample_scope"] is False
    assert len(state["blockers"]) == 2, state["blockers"]
    assert any("source masks" in blocker for blocker in state["blockers"])
    assert any("part-1" in blocker for blocker in state["blockers"])

    refused = client.put(f"/api/datasets/{dataset_id}/annotation-scope", json={"scope": "sample"})
    assert refused.status_code == 409
    # Both reasons travel in the refusal, not just the first one found.
    assert "source masks" in refused.json()["detail"]
    assert "part-1" in refused.json()["detail"]


def test_one_completion_writes_one_revision_per_channel_with_identical_bytes(
    client: TestClient, settings: Settings, tmp_path: Path
) -> None:
    dataset_id, sample_ids, images_by_sample = _multishot_dataset(settings, tmp_path / "src")
    _use_sample_scope(client, dataset_id)
    sample_id = sample_ids[0]

    draft, etag = _open_sample_draft(client, sample_id)
    assert draft["document"]["image_width"] == FRAME[0]
    assert draft["document"]["base"] == "empty"

    saved = client.put(
        f"/api/samples/{sample_id}/annotations/draft",
        json=_triangle(draft["document"]),
        headers={"If-Match": etag},
    )
    assert saved.status_code == 200, saved.text

    completed = client.post(
        f"/api/samples/{sample_id}/annotations/complete",
        headers={"If-Match": saved.headers["etag"]},
    )
    assert completed.status_code == 200, completed.text
    revisions = completed.json()

    assert [revision["image_id"] for revision in revisions] == images_by_sample[sample_id]
    # One render, three files: the shared digest is what makes "these channels carry the
    # same truth" checkable rather than merely intended.
    assert len({revision["mask_sha256"] for revision in revisions}) == 1
    assert len({revision["document_sha256"] for revision in revisions}) == 1
    assert len({revision["mask_path"] for revision in revisions}) == 3
    for revision in revisions:
        path = Path(revision["mask_path"])
        assert path.is_file()
        assert sha256_of(path) == revision["mask_sha256"]
        assert path.parent == settings.annotation_image_dir(revision["image_id"])

    # The draft is consumed, and the existing image-keyed resolver sees every channel.
    assert client.get(f"/api/samples/{sample_id}/annotations/draft").status_code == 404
    with connection(settings.db_path) as conn:
        resolved = annotations_repo.resolve_ground_truth_masks(
            conn, images_by_sample[sample_id], verify_bytes=True
        )
    assert sorted(resolved) == sorted(images_by_sample[sample_id])
    assert {truth.kind for truth in resolved.values()} == {"revision"}


def test_every_channel_exports_the_shared_truth_through_the_image_routes(
    client: TestClient, settings: Settings, tmp_path: Path
) -> None:
    dataset_id, sample_ids, images_by_sample = _multishot_dataset(settings, tmp_path / "src")
    _use_sample_scope(client, dataset_id)
    sample_id = sample_ids[0]
    draft, etag = _open_sample_draft(client, sample_id)
    saved = client.put(
        f"/api/samples/{sample_id}/annotations/draft",
        json=_triangle(draft["document"]),
        headers={"If-Match": etag},
    )
    client.post(
        f"/api/samples/{sample_id}/annotations/complete",
        headers={"If-Match": saved.headers["etag"]},
    )

    exported = [
        client.get(f"/api/images/{image_id}/annotations/export/png")
        for image_id in images_by_sample[sample_id]
    ]

    assert [response.status_code for response in exported] == [200, 200, 200]
    assert len({response.content for response in exported}) == 1


def test_a_stale_token_is_refused_and_the_image_routes_point_at_the_sample_route(
    client: TestClient, settings: Settings, tmp_path: Path
) -> None:
    dataset_id, sample_ids, images_by_sample = _multishot_dataset(settings, tmp_path / "src")
    _use_sample_scope(client, dataset_id)
    sample_id = sample_ids[0]
    draft, etag = _open_sample_draft(client, sample_id)

    first = client.put(
        f"/api/samples/{sample_id}/annotations/draft",
        json=_triangle(draft["document"]),
        headers={"If-Match": etag},
    )
    assert first.status_code == 200
    stale = client.put(
        f"/api/samples/{sample_id}/annotations/draft",
        json=draft["document"],
        headers={"If-Match": etag},
    )
    assert stale.status_code == 412
    assert (
        client.put(
            f"/api/samples/{sample_id}/annotations/draft", json=draft["document"]
        ).status_code
        == 428
    )

    image_id = images_by_sample[sample_id][0]
    opened = client.post(f"/api/images/{image_id}/annotations/draft")
    assert opened.status_code == 409
    assert "/api/samples/" in opened.json()["detail"]


def test_a_sample_scoped_document_may_not_claim_a_base_layer_or_a_foreign_frame(
    client: TestClient, settings: Settings, tmp_path: Path
) -> None:
    dataset_id, sample_ids, _ = _multishot_dataset(settings, tmp_path / "src")
    _use_sample_scope(client, dataset_id)
    sample_id = sample_ids[0]
    draft, etag = _open_sample_draft(client, sample_id)

    based = client.put(
        f"/api/samples/{sample_id}/annotations/draft",
        json={**draft["document"], "base": "source_mask"},
        headers={"If-Match": etag},
    )
    assert based.status_code == 422

    resized = client.put(
        f"/api/samples/{sample_id}/annotations/draft",
        json={**draft["document"], "image_width": FRAME[0] + 1},
        headers={"If-Match": etag},
    )
    assert resized.status_code == 422


def test_reopening_a_draft_resumes_the_sample_s_completed_truth(
    client: TestClient, settings: Settings, tmp_path: Path
) -> None:
    dataset_id, sample_ids, _ = _multishot_dataset(settings, tmp_path / "src")
    _use_sample_scope(client, dataset_id)
    sample_id = sample_ids[0]
    draft, etag = _open_sample_draft(client, sample_id)
    saved = client.put(
        f"/api/samples/{sample_id}/annotations/draft",
        json=_triangle(draft["document"]),
        headers={"If-Match": etag},
    )
    client.post(
        f"/api/samples/{sample_id}/annotations/complete",
        headers={"If-Match": saved.headers["etag"]},
    )

    reopened, reopened_etag = _open_sample_draft(client, sample_id)

    assert [shape["id"] for shape in reopened["document"]["shapes"]] == ["defect-1"]
    assert reopened["version"] == 1
    assert reopened_etag.endswith('-v1"')
    # A second completion is a second revision on every image, never a rewrite of the first.
    again = client.post(
        f"/api/samples/{sample_id}/annotations/complete",
        headers={"If-Match": reopened_etag},
    )
    assert again.status_code == 200
    assert [revision["revision_no"] for revision in again.json()] == [2, 2, 2]


def test_scope_cannot_change_while_a_draft_is_open_in_either_direction(
    client: TestClient, settings: Settings, tmp_path: Path
) -> None:
    dataset_id, sample_ids, images_by_sample = _multishot_dataset(settings, tmp_path / "src")

    client.post(f"/api/images/{images_by_sample[sample_ids[0]][0]}/annotations/draft")
    blocked = client.put(f"/api/datasets/{dataset_id}/annotation-scope", json={"scope": "sample"})
    assert blocked.status_code == 409
    assert "drafts are open" in blocked.json()["detail"]

    with connection(settings.db_path) as conn:
        annotations_repo.get_draft(conn, images_by_sample[sample_ids[0]][0])
        conn.execute("DELETE FROM annotation_draft")
    _use_sample_scope(client, dataset_id)

    _open_sample_draft(client, sample_ids[1])
    back = client.put(f"/api/datasets/{dataset_id}/annotation-scope", json={"scope": "image"})
    assert back.status_code == 409
    # Re-asserting the current scope is not a change and is therefore never blocked.
    assert (
        client.put(
            f"/api/datasets/{dataset_id}/annotation-scope", json={"scope": "sample"}
        ).status_code
        == 200
    )


def test_deleting_the_dataset_removes_its_shared_drafts(
    client: TestClient, settings: Settings, tmp_path: Path
) -> None:
    dataset_id, sample_ids, _ = _multishot_dataset(settings, tmp_path / "src")
    _use_sample_scope(client, dataset_id)
    _open_sample_draft(client, sample_ids[0])

    with connection(settings.db_path) as conn:
        assert annotations_repo.count_open_sample_drafts(conn, dataset_id) == 1
        datasets.delete_dataset(conn, dataset_id)
        remaining = conn.execute("SELECT COUNT(*) FROM annotation_sample_draft").fetchone()[0]

    assert remaining == 0
