"""Copying regions between the channels of one part.

The target workflow this exists for: draw the defect on the illumination that shows it,
then put the same outline on the others and nudge each into place. The rig's exposures are
milliseconds apart, so the geometry is nearly right on every channel and completely right
on none — which is why this copies rather than shares, and why every copy stays editable.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from anomaly_lab.config import Settings
from anomaly_lab.db.connection import connection

from .conftest import FRAME, multishot_dataset


def _open(client: TestClient, image_id: int) -> tuple[dict[str, Any], str]:
    seed = client.get(f"/api/images/{image_id}/annotations/draft")
    assert seed.status_code == 200, seed.text
    response = client.post(
        f"/api/images/{image_id}/annotations/draft",
        json=seed.json()["document"],
        headers={"If-None-Match": "*"},
    )
    assert response.status_code == 201, response.text
    return response.json(), response.headers["etag"]


def _with_shapes(document: dict[str, Any], count: int, prefix: str = "region") -> dict[str, Any]:
    return {
        **document,
        "shapes": [
            {
                "id": f"{prefix}-{index}",
                "label_key": "defect",
                "kind": "polygon",
                "operation": "add",
                "points": [{"x": 2, "y": 2}, {"x": 9, "y": 2}, {"x": 2, "y": 9}],
            }
            for index in range(count)
        ],
    }


def _save(client: TestClient, image_id: int, document: dict[str, Any], etag: str) -> str:
    response = client.put(
        f"/api/images/{image_id}/annotations/draft",
        json=document,
        headers={"If-Match": etag},
    )
    assert response.status_code == 200, response.text
    return str(response.headers["etag"])


def _drawn(client: TestClient, image_id: int, count: int = 2) -> str:
    """A source draft holding `count` regions, and the token that owns it."""
    draft, etag = _open(client, image_id)
    return _save(client, image_id, _with_shapes(draft["document"], count), etag)


def test_regions_are_appended_to_each_channel_with_fresh_ids(
    client: TestClient, settings: Settings, tmp_path: Path
) -> None:
    _, sample_ids, images_by_sample = multishot_dataset(settings, tmp_path / "src")
    bright, dark, dome = images_by_sample[sample_ids[0]]
    etag = _drawn(client, bright, count=2)

    # `dark` already carries work of its own, so appending is the whole point: it must
    # still be there afterwards.
    dark_draft, dark_etag = _open(client, dark)
    _save(client, dark, _with_shapes(dark_draft["document"], 1, prefix="own"), dark_etag)

    response = client.post(
        f"/api/images/{bright}/annotations/copy-regions",
        json={"target_image_ids": [dark, dome]},
        headers={"If-Match": etag},
    )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["copied"] == 2
    assert {item["image_id"]: item["shape_count"] for item in payload["targets"]} == {
        dark: 3,
        dome: 2,
    }

    for image_id, expected in ((dark, 3), (dome, 2)):
        document = client.get(f"/api/images/{image_id}/annotations/draft").json()["document"]
        assert len(document["shapes"]) == expected
        # Ids are unique *within* a document, so a copy has to mint its own -- otherwise
        # the second copy into one channel is a duplicate-id 422 half way through a fan-out.
        assert len({shape["id"] for shape in document["shapes"]}) == expected
        assert not any(shape["id"].startswith("region-") for shape in document["shapes"])
        assert document["shapes"][-1]["points"] == [
            {"x": 2.0, "y": 2.0},
            {"x": 9.0, "y": 2.0},
            {"x": 2.0, "y": 9.0},
        ]

    # The source is untouched, and still holds the token the caller was given.
    source = client.get(f"/api/images/{bright}/annotations/draft")
    assert source.headers["etag"] == etag
    assert len(source.json()["document"]["shapes"]) == 2


def test_copying_twice_into_one_channel_keeps_both_copies(
    client: TestClient, settings: Settings, tmp_path: Path
) -> None:
    """The case that fails if ids are copied verbatim rather than minted."""
    _, sample_ids, images_by_sample = multishot_dataset(settings, tmp_path / "src")
    bright, dark, _ = images_by_sample[sample_ids[0]]
    etag = _drawn(client, bright, count=1)

    for _ in range(2):
        response = client.post(
            f"/api/images/{bright}/annotations/copy-regions",
            json={"target_image_ids": [dark]},
            headers={"If-Match": etag},
        )
        assert response.status_code == 200, response.text

    document = client.get(f"/api/images/{dark}/annotations/draft").json()["document"]
    assert len(document["shapes"]) == 2
    assert len({shape["id"] for shape in document["shapes"]}) == 2


def test_a_channel_named_twice_is_copied_into_once(
    client: TestClient, settings: Settings, tmp_path: Path
) -> None:
    _, sample_ids, images_by_sample = multishot_dataset(settings, tmp_path / "src")
    bright, dark, _ = images_by_sample[sample_ids[0]]
    etag = _drawn(client, bright, count=1)

    response = client.post(
        f"/api/images/{bright}/annotations/copy-regions",
        json={"target_image_ids": [dark, dark]},
        headers={"If-Match": etag},
    )

    assert response.status_code == 200, response.text
    assert response.json()["targets"] == [{"image_id": dark, "version": 2, "shape_count": 1}]


def test_the_source_precondition_is_the_draft_the_caller_read(
    client: TestClient, settings: Settings, tmp_path: Path
) -> None:
    """Copying a stale document into three channels is the mistake this refuses."""
    _, sample_ids, images_by_sample = multishot_dataset(settings, tmp_path / "src")
    bright, dark, _ = images_by_sample[sample_ids[0]]
    stale = _drawn(client, bright, count=1)
    fresh = _save(
        client,
        bright,
        _with_shapes(client.get(f"/api/images/{bright}/annotations/draft").json()["document"], 3),
        stale,
    )

    refused = client.post(
        f"/api/images/{bright}/annotations/copy-regions",
        json={"target_image_ids": [dark]},
        headers={"If-Match": stale},
    )
    assert refused.status_code == 412

    missing = client.post(
        f"/api/images/{bright}/annotations/copy-regions",
        json={"target_image_ids": [dark]},
    )
    assert missing.status_code == 428

    accepted = client.post(
        f"/api/images/{bright}/annotations/copy-regions",
        json={"target_image_ids": [dark]},
        headers={"If-Match": fresh},
    )
    assert accepted.status_code == 200
    assert accepted.json()["copied"] == 3


def test_only_a_sibling_of_the_same_size_can_receive_a_copy(
    client: TestClient, settings: Settings, tmp_path: Path
) -> None:
    _, sample_ids, images_by_sample = multishot_dataset(
        settings, tmp_path / "src", frames={"dome": (FRAME[0] + 4, FRAME[1])}
    )
    bright, dark, dome = images_by_sample[sample_ids[0]]
    stranger = images_by_sample[sample_ids[1]][0]
    etag = _drawn(client, bright, count=1)

    def copy(targets: list[int]) -> Any:
        return client.post(
            f"/api/images/{bright}/annotations/copy-regions",
            json={"target_image_ids": targets},
            headers={"If-Match": etag},
        )

    other_part = copy([stranger])
    assert other_part.status_code == 409
    assert "another channel of this sample" in other_part.json()["detail"]

    mixed = copy([dome])
    assert mixed.status_code == 409
    assert "never leaves its source frame" in mixed.json()["detail"]

    assert copy([bright]).status_code == 422

    # A refused target takes the whole request down: a partial fan-out would leave the
    # person to work out which channels received the copy and which did not.
    assert copy([dark, dome]).status_code == 409
    assert client.get(f"/api/images/{dark}/annotations/draft").json()["persisted"] is False


def test_an_empty_draft_has_nothing_to_copy(
    client: TestClient, settings: Settings, tmp_path: Path
) -> None:
    _, sample_ids, images_by_sample = multishot_dataset(settings, tmp_path / "src")
    bright, dark, _ = images_by_sample[sample_ids[0]]
    _, etag = _open(client, bright)

    response = client.post(
        f"/api/images/{bright}/annotations/copy-regions",
        json={"target_image_ids": [dark]},
        headers={"If-Match": etag},
    )

    assert response.status_code == 422
    assert "no regions to copy" in response.json()["detail"]


def test_sample_scope_is_pointed_at_its_own_routes(
    client: TestClient, settings: Settings, tmp_path: Path
) -> None:
    """Under sample scope one document already covers every channel, so copying is nonsense."""
    dataset_id, sample_ids, images_by_sample = multishot_dataset(settings, tmp_path / "src")
    bright, dark, _ = images_by_sample[sample_ids[0]]
    etag = _drawn(client, bright, count=1)

    blocked = client.put(f"/api/datasets/{dataset_id}/annotation-scope", json={"scope": "sample"})
    assert blocked.status_code == 409

    client.delete(f"/api/images/{bright}/annotations/draft", headers={"If-Match": etag})
    moved = client.put(f"/api/datasets/{dataset_id}/annotation-scope", json={"scope": "sample"})
    assert moved.status_code == 200, moved.text

    response = client.post(
        f"/api/images/{bright}/annotations/copy-regions",
        json={"target_image_ids": [dark]},
        headers={"If-Match": etag},
    )
    assert response.status_code == 409
    assert "/api/samples/" in response.json()["detail"]


def test_a_copy_target_that_has_never_been_opened_is_created_from_its_own_seed(
    client: TestClient, settings: Settings, tmp_path: Path
) -> None:
    """The target's draft is materialised by the copy, which is a write and may create one."""
    _, sample_ids, images_by_sample = multishot_dataset(settings, tmp_path / "src")
    bright, dark, _ = images_by_sample[sample_ids[0]]
    etag = _drawn(client, bright, count=1)

    with connection(settings.db_path) as conn:
        before = conn.execute("SELECT COUNT(*) FROM annotation_draft").fetchone()[0]
    assert before == 1

    response = client.post(
        f"/api/images/{bright}/annotations/copy-regions",
        json={"target_image_ids": [dark]},
        headers={"If-Match": etag},
    )
    assert response.status_code == 200, response.text

    with connection(settings.db_path) as conn:
        after = conn.execute("SELECT COUNT(*) FROM annotation_draft").fetchone()[0]
    assert after == 2
