"""Explore's Text mode — SAM 3 by phrase — with the resident stubbed: no torch, no weights."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

from anomaly_lab.api.routers import explore as explore_routes
from anomaly_lab.config import Settings
from anomaly_lab.db.connection import connection
from anomaly_lab.db.repositories import images as images_repo
from anomaly_lab.db.repositories import jobs as jobs_repo
from anomaly_lab.domain.entities import JobKind
from anomaly_lab.explore.store import (
    MapKind,
    explore_dir,
    instance_labels,
    mask_of,
    read_grid,
    to_source,
    write_grid,
)
from anomaly_lab.explore.text import (
    MAX_INSTANCES,
    TextExploreError,
    clean_phrase,
    rank_instances,
)
from anomaly_lab.jobs.queue import JobQueue
from anomaly_lab.model_assets.catalog import SAM3_KEY, gated_message, get_spec
from anomaly_lab.model_assets.store import ResolvedAsset
from anomaly_lab.regions.transform import SpatialTransform

from .conftest import Fixture

# ------------------------------------------------------------------ ranking and bounds


def _masks(count: int, height: int = 6, width: int = 5) -> np.ndarray:
    masks = np.zeros((count, height, width), dtype=np.bool_)
    for index in range(count):
        masks[index, index % height, : (index % width) + 1] = True
    return masks


def test_instances_are_ranked_best_first_and_empty_masks_are_dropped() -> None:
    masks = _masks(3)
    masks[1] = False  # an instance with no pixels is not an instance
    scores = np.asarray([0.6, 0.99, 0.9])
    boxes = np.arange(12, dtype=np.float64).reshape(3, 4)
    stack, rows, dropped = rank_instances(masks, scores, boxes)
    assert [row.score for row in rows] == [0.9, 0.6]
    assert rows[0].box == (8.0, 9.0, 10.0, 11.0)
    assert rows[0].area == int(masks[2].sum())
    assert np.array_equal(stack, masks[[2, 0]])
    assert dropped == 0


def test_instances_past_the_bound_are_counted_not_silently_lost() -> None:
    count = MAX_INSTANCES + 5
    scores = np.linspace(0.5, 0.99, count)
    stack, rows, dropped = rank_instances(_masks(count), scores, np.zeros((count, 4)))
    assert len(rows) == MAX_INSTANCES == stack.shape[0]
    assert dropped == 5
    assert rows[0].score == pytest.approx(0.99)


def test_nothing_found_is_an_empty_stack_at_the_source_size() -> None:
    stack, rows, dropped = rank_instances(
        np.zeros((0, 7, 9), dtype=np.bool_), np.zeros(0), np.zeros((0, 4))
    )
    assert stack.shape == (0, 7, 9) and rows == [] and dropped == 0


def test_a_phrase_is_trimmed_and_bounded() -> None:
    assert clean_phrase("  the   cap ") == "the cap"
    with pytest.raises(TextExploreError):
        clean_phrase("   ")
    with pytest.raises(TextExploreError):
        clean_phrase("x" * 81)


# ------------------------------------------------------------------ the instance map


def _identity(width: int, height: int) -> SpatialTransform:
    return SpatialTransform.resolve(source_size=(width, height), prepared_size=(width, height))


def test_an_instance_map_round_trips_bit_packed_and_the_best_is_drawn_on_top(
    tmp_path: Path,
) -> None:
    masks = np.zeros((2, 4, 11), dtype=np.bool_)
    masks[0, 1:3, 2:6] = True
    masks[1, 0:4, 4:10] = True  # overlaps the first at columns 4-5
    map_id = write_grid(tmp_path, MapKind.INSTANCES, masks, _identity(11, 4), 1)
    stored = read_grid(tmp_path / f"{map_id}.npz")
    assert stored.kind is MapKind.INSTANCES
    assert np.array_equal(stored.grid, masks)

    labels = to_source(stored)
    assert labels.shape == (4, 11) and labels.dtype == np.uint8
    assert labels[1, 4] == 1  # the better instance wins the shared pixel
    assert labels[0, 4] == 2
    assert labels[3, 0] == 0
    assert np.array_equal(labels, instance_labels(masks))

    # One instance is its whole mask, including where a better one covers it.
    assert np.array_equal(mask_of(stored, instance=2), masks[1])
    with pytest.raises(ValueError):
        mask_of(stored, instance=3)


# ------------------------------------------------------------------ the HTTP edge


def _app(client: TestClient) -> FastAPI:
    app = client.app
    assert isinstance(app, FastAPI)
    return app


class _StubResident:
    """Answers as the SAM 3 child would, writing a real instance map, spawning nothing."""

    def __init__(self, settings: Settings, found: int = 2) -> None:
        self.settings = settings
        self.found = found
        self.calls: list[dict[str, object]] = []

    async def segment_text(
        self, *, asset_key: str, asset_path: Path, payload: dict[str, object]
    ) -> tuple[dict[str, object], bool]:
        self.calls.append({"asset_key": asset_key, **payload})
        with connection(self.settings.db_path) as conn:
            image = images_repo.get_image(conn, int(str(payload["image_id"])))
        assert image is not None
        masks = np.zeros((self.found, image.height, image.width), dtype=np.bool_)
        for index in range(self.found):
            masks[index, index : index + 4, 2:9] = True
        map_id = (
            write_grid(
                explore_dir(self.settings),
                MapKind.INSTANCES,
                masks,
                _identity(image.width, image.height),
                1,
            )
            if self.found
            else None
        )
        return (
            {
                "image_id": payload["image_id"],
                "phrase": payload["phrase"],
                "threshold": payload["threshold"],
                "device": "cpu",
                "cached": len(self.calls) > 1,
                "encode_ms": 0.0 if len(self.calls) > 1 else 900.0,
                "prompt_ms": 120.0,
                "map_id": map_id,
                "instances": [
                    {
                        "score": 0.9 - 0.1 * index,
                        "box": {"x0": 2.0, "y0": float(index), "x1": 9.0, "y1": index + 4.0},
                        "area": int(masks[index].sum()),
                    }
                    for index in range(self.found)
                ],
                "dropped": 0,
            },
            len(self.calls) > 1,
        )

    def stderr_tail(self) -> str:
        return ""


def _ready(settings: Settings) -> ResolvedAsset:
    spec = get_spec(SAM3_KEY)
    assert spec is not None
    return ResolvedAsset(
        path=settings.model_assets_dir / spec.key / spec.filename,
        source="managed",
        ready=True,
        size=spec.expected_size,
    )


@pytest.fixture
def ready(client: TestClient, settings: Settings, monkeypatch: pytest.MonkeyPatch) -> _StubResident:
    stub = _StubResident(settings)
    monkeypatch.setattr(_app(client).state, "resident", stub)
    monkeypatch.setattr(explore_routes, "text_runtime_available", lambda: True)
    monkeypatch.setattr(explore_routes, "resolve_asset", lambda _settings, _spec: _ready(settings))
    return stub


def test_without_the_dl_extra_text_says_so(
    client: TestClient, seeded: Fixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(explore_routes, "text_runtime_available", lambda: False)
    text = client.get("/api/explore/capability").json()["text"]
    assert text["available"] is False and text["installable"] is False
    assert "dl" in text["reason"]
    response = client.post(
        f"/api/images/{seeded.normal_image_ids[0]}/explore/text", json={"phrase": "candle"}
    )
    assert response.status_code == 409
    assert "dl" in response.json()["detail"]


def test_a_missing_checkpoint_explains_the_gate_until_an_account_is_signed_in(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(explore_routes, "text_runtime_available", lambda: True)
    monkeypatch.setattr(explore_routes, "hub_token_present", lambda: False)
    text = client.get("/api/explore/capability").json()["text"]
    assert text["available"] is False
    assert text["installable"] is True
    assert text["gated"] is True
    assert text["asset_key"] == SAM3_KEY
    assert text["access_url"] == "https://huggingface.co/facebook/sam3"
    assert "HF_TOKEN" in text["reason"] and text["access_url"] in text["reason"]
    assert "Traceback" not in text["reason"]

    monkeypatch.setattr(explore_routes, "hub_token_present", lambda: True)
    signed_in = client.get("/api/explore/capability").json()["text"]
    assert "not downloaded" in signed_in["reason"]
    assert "HF_TOKEN" not in signed_in["reason"]


def test_text_is_refused_while_a_job_runs(
    client: TestClient, settings: Settings, seeded: Fixture, ready: _StubResident
) -> None:
    queue: JobQueue = _app(client).state.job_queue
    job = queue.enqueue(kind=JobKind.PREWARM, params={"dataset_id": seeded.dataset_id})
    with connection(settings.db_path) as conn:
        jobs_repo.mark_running(conn, job.id, log_path="/dev/null")
    response = client.post(
        f"/api/images/{seeded.normal_image_ids[0]}/explore/text", json={"phrase": "candle"}
    )
    assert response.status_code == 409
    assert "prewarm" in response.text
    assert ready.calls == []


def test_bad_phrases_never_reach_the_resident(
    client: TestClient, seeded: Fixture, ready: _StubResident
) -> None:
    image_id = seeded.normal_image_ids[0]
    assert (
        client.post(f"/api/images/{image_id}/explore/text", json={"phrase": ""}).status_code == 422
    )
    blank = client.post(f"/api/images/{image_id}/explore/text", json={"phrase": "   "})
    assert blank.status_code == 400
    long = client.post(f"/api/images/{image_id}/explore/text", json={"phrase": "x" * 81})
    assert long.status_code == 422
    bad_cut = client.post(
        f"/api/images/{image_id}/explore/text", json={"phrase": "cap", "threshold": 1.0}
    )
    assert bad_cut.status_code == 422
    missing = client.post("/api/images/999999/explore/text", json={"phrase": "cap"})
    assert missing.status_code == 404
    assert ready.calls == []


def test_a_phrase_returns_ranked_instances_an_overlay_and_a_candidate(
    client: TestClient, seeded: Fixture, ready: _StubResident
) -> None:
    capability = client.get("/api/explore/capability").json()["text"]
    assert capability["available"] is True and capability["reason"] is None

    image_id = seeded.defect_image_ids[0]
    response = client.post(
        f"/api/images/{image_id}/explore/text", json={"phrase": "  the  cap ", "threshold": 0.4}
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert ready.calls[0] == {
        "asset_key": SAM3_KEY,
        "image_id": image_id,
        "phrase": "the cap",
        "threshold": 0.4,
    }
    assert [entry["index"] for entry in body["instances"]] == [1, 2]
    assert body["instances"][0]["score"] == pytest.approx(0.9)
    assert body["cached"] is False and body["warm"] is False
    assert body["map_url"] == f"/api/explore/maps/{body['map_id']}.png"

    overlay = client.get(body["map_url"], params={"colours": "3bc9db,f0883e"})
    assert overlay.status_code == 200
    with Image.open(io.BytesIO(overlay.content)) as image:
        assert image.size == (16, 16) and image.mode == "RGBA"
    one = client.get(body["map_url"], params={"colours": "3bc9db,f0883e", "instance": 2})
    assert one.status_code == 200
    beyond = client.get(body["map_url"], params={"colours": "3bc9db", "instance": 3})
    assert beyond.status_code == 400

    shape = client.post(f"/api/explore/maps/{body['map_id']}/shape", json={"instance": 2})
    assert shape.status_code == 200, shape.text
    candidate = shape.json()
    assert candidate["shape"]["kind"] == "bitmap"
    assert candidate["area"] == body["instances"][1]["area"]
    assert (candidate["shape"]["x"], candidate["shape"]["y"]) == (2, 1)

    again = client.post(f"/api/images/{image_id}/explore/text", json={"phrase": "wick"}).json()
    assert again["cached"] is True and again["warm"] is True


def test_nothing_found_is_an_answer_without_a_map(
    client: TestClient, seeded: Fixture, ready: _StubResident
) -> None:
    ready.found = 0
    body = client.post(
        f"/api/images/{seeded.normal_image_ids[0]}/explore/text", json={"phrase": "stain"}
    ).json()
    assert body["instances"] == [] and body["map_id"] is None and body["map_url"] is None


def test_the_gated_message_names_the_licence_and_the_token_and_never_a_value() -> None:
    spec = get_spec(SAM3_KEY)
    assert spec is not None and spec.gated
    message = gated_message(spec)
    assert "SAM License" in message and "HF_TOKEN" in message
    assert "https://huggingface.co/facebook/sam3" in message
