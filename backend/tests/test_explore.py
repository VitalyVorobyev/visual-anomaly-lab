"""Explore's arithmetic, scratch maps and HTTP edge, with the resident stubbed: no torch."""

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
from anomaly_lab.explore.grid import (
    ExploreMode,
    cluster_grid,
    covered_cells,
    frame_side,
    frame_transform,
    kmeans,
    pca_grid,
    similarity,
    source_cell,
)
from anomaly_lab.explore.store import (
    MapKind,
    explore_dir,
    map_path,
    mask_of,
    read_grid,
    to_source,
    write_grid,
)
from anomaly_lab.jobs.queue import JobQueue
from anomaly_lab.models.dino_backbone import DinoBackbone
from anomaly_lab.regions.transform import SpatialTransform

from .conftest import Fixture

# ------------------------------------------------------------------ k-means


def _noise(seed: int = 3, count: int = 200, width: int = 8) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal((count, width)).astype(np.float32)


def test_kmeans_is_reproducible_in_both_directions() -> None:
    vectors = _noise()
    first = kmeans(vectors, 5, seed=0)
    assert np.array_equal(first, kmeans(vectors, 5, seed=0))
    # Structureless data, so the answer depends on the initial draw and a new seed moves it.
    assert not np.array_equal(first, kmeans(vectors, 5, seed=1))


def test_kmeans_recovers_separated_groups_largest_first() -> None:
    rng = np.random.default_rng(0)
    centres = np.eye(4, dtype=np.float32)[:3] * 10.0
    sizes = [50, 30, 20]
    vectors = np.concatenate(
        [
            centre + rng.normal(0, 0.1, (size, 4))
            for centre, size in zip(centres, sizes, strict=True)
        ]
    )
    labels = kmeans(vectors, 3, seed=4)
    assert labels.tolist() == [0] * 50 + [1] * 30 + [2] * 20


def test_kmeans_never_asks_for_more_clusters_than_points() -> None:
    labels = kmeans(_noise(count=3), 12, seed=0)
    assert sorted(labels.tolist()) == [0, 1, 2]


# ------------------------------------------------------------------ similarity


def _unit_grid(rows: int = 4, cols: int = 4, width: int = 6, seed: int = 0) -> np.ndarray:
    grid = np.random.default_rng(seed).standard_normal((rows, cols, width)).astype(np.float32)
    return grid / np.linalg.norm(grid, axis=-1, keepdims=True)


def test_similarity_is_max_positive_minus_max_negative_clipped() -> None:
    grid = _unit_grid()
    flat = grid.reshape(16, -1)
    positives, negatives = [(0, 0), (2, 3)], [(1, 1)]
    expected = np.clip(
        np.max(flat @ flat[[0, 11]].T, axis=1) - (flat @ flat[5]),
        0.0,
        1.0,
    ).reshape(4, 4)
    result = similarity(grid, positives, negatives)
    assert result.dtype == np.float32
    np.testing.assert_allclose(result, expected, rtol=1e-6, atol=1e-6)
    assert result[1, 1] == 0.0
    assert result.min() >= 0.0 and result.max() <= 1.0


def test_a_clicked_patch_is_fully_like_itself() -> None:
    result = similarity(_unit_grid(), [(3, 2)], [])
    assert result[3, 2] == pytest.approx(1.0, abs=1e-6)


def test_similarity_needs_a_positive() -> None:
    with pytest.raises(ValueError, match="positive"):
        similarity(_unit_grid(), [], [(0, 0)])


# ------------------------------------------------------------------ the frame


def test_the_frame_is_fixed_by_the_patch_size() -> None:
    assert frame_side(DinoBackbone.DINOV2_VIT_B14) == 448
    assert frame_side(DinoBackbone.DINOV3_VIT_S16) == 512


def test_letterbox_cells_are_left_out_and_clicks_land_on_the_image() -> None:
    # 2:1 landscape: the resized image is 448x224 with 112 rows of padding above and below.
    transform = frame_transform((200, 100), DinoBackbone.DINOV2_VIT_B14)
    window = covered_cells(transform, 14)
    assert (window.left, window.right) == (0, 32)
    assert (window.top, window.bottom) == (8, 24)
    assert source_cell(transform, 14, window, (0.0, 0.0)) == (8, 0)
    assert source_cell(transform, 14, window, (199.0, 99.0)) == (23, 31)

    features = _unit_grid(32, 32, 5)
    labels = cluster_grid(features, window, 3, seed=0)
    assert labels[:8].max() == 0 and labels[24:].max() == 0
    assert set(np.unique(labels[8:24]).tolist()) == {1, 2, 3}
    colour = pca_grid(features, window)
    assert colour.shape == (32, 32, 3)
    assert colour[:8].max() == 0.0
    assert colour.min() >= 0.0 and colour.max() <= 1.0


# ------------------------------------------------------------------ scratch maps


def test_a_stored_grid_projects_to_the_source_frame(tmp_path: Path) -> None:
    transform = frame_transform((60, 30), DinoBackbone.DINOV2_VIT_S14)
    values = np.zeros((32, 32), dtype=np.float32)
    values[8:24, :16] = 1.0
    map_id = write_grid(tmp_path, MapKind.VALUES, values, transform, 14)
    stored = read_grid(map_path(tmp_path, map_id))
    source = to_source(stored)
    assert source.shape == (30, 60)
    assert source[15, 5] == pytest.approx(1.0)
    assert source[15, 55] == pytest.approx(0.0)
    mask = mask_of(stored, threshold=0.5)
    assert mask[15, 5] and not mask[15, 55]

    labels = np.zeros((32, 32), dtype=np.uint8)
    labels[8:24] = 2
    labels[8:24, 16:] = 1
    stored_labels = read_grid(
        map_path(tmp_path, write_grid(tmp_path, MapKind.LABELS, labels, transform, 14))
    )
    cluster = mask_of(stored_labels, cluster=2)
    assert cluster.shape == (30, 60)
    assert cluster[15, 5] and not cluster[15, 55]
    with pytest.raises(ValueError, match="threshold"):
        mask_of(stored_labels, threshold=0.5)


def test_the_scratch_directory_is_bounded(tmp_path: Path) -> None:
    transform = frame_transform((16, 16), DinoBackbone.DINOV2_VIT_S14)
    grid = np.zeros((32, 32), dtype=np.float32)
    kept = [write_grid(tmp_path, MapKind.VALUES, grid, transform, 14, keep=3) for _ in range(5)]
    assert len(list(tmp_path.glob("*.npz"))) == 3
    assert map_path(tmp_path, kept[-1]).is_file()


def test_a_map_id_cannot_name_a_path(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        map_path(tmp_path, "../app")


# ------------------------------------------------------------------ the HTTP edge


def _app(client: TestClient) -> FastAPI:
    app = client.app
    assert isinstance(app, FastAPI)
    return app


class _StubResident:
    """Answers as the child would, writing a real grid, without spawning anything."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.calls: list[dict[str, object]] = []

    async def explore(
        self, *, backbone: str, payload: dict[str, object]
    ) -> tuple[dict[str, object], bool]:
        self.calls.append({"backbone": backbone, **payload})
        with connection(self.settings.db_path) as conn:
            image = images_repo.get_image(conn, int(str(payload["image_id"])))
        assert image is not None
        transform = frame_transform((image.width, image.height), DinoBackbone(backbone))
        grid = np.linspace(0.0, 1.0, 32 * 32, dtype=np.float32).reshape(32, 32)
        kind = MapKind.VALUES
        extra: dict[str, object] = {}
        if payload["mode"] == ExploreMode.CLUSTERS.value:
            grid = (np.arange(32 * 32).reshape(32, 32) % 3 + 1).astype(np.uint8)
            kind = MapKind.LABELS
            extra = {"cells": grid.reshape(-1).tolist(), "clusters": 3}
        elif payload["mode"] == ExploreMode.PCA.value:
            grid = np.full((32, 32, 3), 0.5, dtype=np.float32)
            kind = MapKind.RGB
        map_id = write_grid(explore_dir(self.settings), kind, grid, transform, 14)
        return (
            {
                "image_id": payload["image_id"],
                "mode": payload["mode"],
                "backbone": backbone,
                "device": "cpu",
                "grid_rows": 32,
                "grid_cols": 32,
                "feature_dim": 1536,
                "patch_size": 14,
                "cached": len(self.calls) > 1,
                "encode_ms": 1.0,
                "compute_ms": 1.0,
                "map_id": map_id,
                "map_kind": kind.value,
                "transform": transform.model_dump(mode="json"),
                **extra,
            },
            len(self.calls) > 1,
        )

    def stderr_tail(self) -> str:
        return ""


@pytest.fixture
def stubbed(
    client: TestClient, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> _StubResident:
    stub = _StubResident(settings)
    monkeypatch.setattr(_app(client).state, "resident", stub)
    monkeypatch.setattr(explore_routes, "runtime_available", lambda: True)
    return stub


def _png_size(payload: bytes) -> tuple[int, int]:
    with Image.open(io.BytesIO(payload)) as image:
        return image.size


def test_capability_lists_every_encoder_and_names_the_gate(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    body = client.get("/api/explore/capability").json()
    assert body["default_backbone"] == "dinov2_vit_b14"
    entries = {entry["key"]: entry for entry in body["backbones"]}
    assert set(entries) == {backbone.value for backbone in DinoBackbone}
    assert entries["dinov2_vit_b14"]["available"] is True
    assert entries["dinov2_vit_b14"]["title"] == "DINOv2 ViT-B/14"
    assert entries["dinov2_vit_s14_reg4"]["title"] == "DINOv2 ViT-S/14 reg4"
    gated = entries["dinov3_vit_b16"]
    assert gated["available"] is False
    assert "HF_TOKEN" in gated["reason"]

    monkeypatch.setenv("HF_TOKEN", "present")
    reopened = client.get("/api/explore/capability").json()
    assert all(entry["available"] for entry in reopened["backbones"])


def test_explore_is_refused_while_a_job_runs(
    client: TestClient, settings: Settings, seeded: Fixture, stubbed: _StubResident
) -> None:
    queue: JobQueue = _app(client).state.job_queue
    job = queue.enqueue(kind=JobKind.PREWARM, params={"dataset_id": seeded.dataset_id})
    with connection(settings.db_path) as conn:
        jobs_repo.mark_running(conn, job.id, log_path="/dev/null")
    response = client.post(
        f"/api/images/{seeded.normal_image_ids[0]}/explore",
        json={"mode": "similar", "points": [{"x": 2, "y": 2}]},
    )
    assert response.status_code == 409
    assert "prewarm" in response.text
    assert stubbed.calls == []


def test_bad_prompts_never_reach_the_resident(
    client: TestClient, seeded: Fixture, stubbed: _StubResident
) -> None:
    image_id = seeded.normal_image_ids[0]
    outside = client.post(
        f"/api/images/{image_id}/explore",
        json={"mode": "similar", "points": [{"x": 100_000, "y": 2}]},
    )
    assert outside.status_code == 400
    negative_outside = client.post(
        f"/api/images/{image_id}/explore",
        json={"mode": "similar", "points": [{"x": 1, "y": 1}], "negatives": [{"x": 1, "y": 99}]},
    )
    assert negative_outside.status_code == 400
    no_positive = client.post(
        f"/api/images/{image_id}/explore",
        json={"mode": "similar", "negatives": [{"x": 1, "y": 1}]},
    )
    assert no_positive.status_code == 400
    too_many = client.post(f"/api/images/{image_id}/explore", json={"mode": "clusters", "k": 13})
    assert too_many.status_code == 422
    missing = client.post("/api/images/999999/explore", json={"mode": "pca"})
    assert missing.status_code == 404
    assert stubbed.calls == []


def test_a_similarity_map_is_served_at_source_size_and_becomes_a_candidate(
    client: TestClient, seeded: Fixture, stubbed: _StubResident
) -> None:
    image_id = seeded.normal_image_ids[0]
    response = client.post(
        f"/api/images/{image_id}/explore",
        json={"mode": "similar", "points": [{"x": 3, "y": 4}], "negatives": [{"x": 9, "y": 9}]},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["map_kind"] == "values"
    assert stubbed.calls[0]["points"] == [{"x": 3.0, "y": 4.0}]
    assert stubbed.calls[0]["backbone"] == "dinov2_vit_b14"

    heat = client.get(body["map_url"])
    assert heat.status_code == 200
    assert heat.headers["content-type"] == "image/png"
    assert _png_size(heat.content) == (16, 16)
    mask = client.get(body["map_url"], params={"threshold": 0.5, "colours": "3bc9db"})
    assert _png_size(mask.content) == (16, 16)

    shape = client.post(
        f"/api/explore/maps/{body['map_id']}/shape", json={"threshold": 0.5, "label_key": "defect"}
    )
    assert shape.status_code == 200, shape.text
    candidate = shape.json()
    assert candidate["shape"]["kind"] == "bitmap"
    assert candidate["area"] > 0
    empty = client.post(f"/api/explore/maps/{body['map_id']}/shape", json={"threshold": 1.0})
    assert empty.status_code == 422
    wrong = client.post(f"/api/explore/maps/{body['map_id']}/shape", json={"cluster": 1})
    assert wrong.status_code == 400


def test_clusters_and_false_colour_render(
    client: TestClient, seeded: Fixture, stubbed: _StubResident
) -> None:
    image_id = seeded.normal_image_ids[0]
    clusters = client.post(f"/api/images/{image_id}/explore", json={"mode": "clusters", "k": 3})
    assert clusters.status_code == 200, clusters.text
    body = clusters.json()
    assert body["clusters"] == 3 and len(body["cells"]) == 32 * 32
    palette = {"colours": "3bc9db,f0883e,a371f7"}
    assert _png_size(client.get(body["map_url"], params=palette).content) == (16, 16)
    one = client.get(body["map_url"], params={**palette, "cluster": 2})
    assert one.status_code == 200
    bad = client.get(body["map_url"], params={"colours": "nope"})
    assert bad.status_code == 400
    shape = client.post(f"/api/explore/maps/{body['map_id']}/shape", json={"cluster": 2})
    assert shape.status_code == 200

    pca = client.post(f"/api/images/{image_id}/explore", json={"mode": "pca"}).json()
    colour = client.get(pca["map_url"])
    with Image.open(io.BytesIO(colour.content)) as image:
        assert image.mode == "RGB" and image.size == (16, 16)


def test_an_unknown_map_is_not_found(client: TestClient) -> None:
    assert client.get(f"/api/explore/maps/{'0' * 32}.png").status_code == 404
    assert client.get("/api/explore/maps/not-an-id.png").status_code == 404


def test_without_the_dl_extra_explore_says_so(
    client: TestClient, seeded: Fixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(explore_routes, "runtime_available", lambda: False)
    capability = client.get("/api/explore/capability").json()
    assert capability["available"] is False
    assert "dl" in capability["reason"]
    response = client.post(
        f"/api/images/{seeded.normal_image_ids[0]}/explore", json={"mode": "pca"}
    )
    assert response.status_code == 409
    assert "dl" in response.json()["detail"]


def test_a_transform_travels_with_the_answer(
    client: TestClient, seeded: Fixture, stubbed: _StubResident
) -> None:
    body = client.post(
        f"/api/images/{seeded.normal_image_ids[0]}/explore", json={"mode": "pca"}
    ).json()
    transform = SpatialTransform.model_validate(body["transform"])
    assert (transform.prepared_width, transform.prepared_height) == (448, 448)
