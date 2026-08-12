"""Prompt assistance is source-framed, temporary and optional."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from anomaly_lab.config import Settings
from anomaly_lab.db.connection import connection
from anomaly_lab.db.repositories import jobs as jobs_repo
from anomaly_lab.domain.entities import JobKind
from anomaly_lab.jobs.queue import JobQueue
from anomaly_lab.model_assets import mobile_sam
from anomaly_lab.model_assets.catalog import ModelAssetSpec
from anomaly_lab.model_assets.store import ResolvedAsset

from .conftest import Fixture


def _open_draft(client: TestClient, image_id: int) -> None:
    response = client.post(f"/api/images/{image_id}/annotations/draft")
    assert response.status_code == 200, response.text


def test_capability_and_missing_asset_are_actionable(client: TestClient, seeded: Fixture) -> None:
    image_id = seeded.defect_image_ids[0]
    _open_draft(client, image_id)
    capability = client.get("/api/segment-assist")
    assert capability.status_code == 200
    assert capability.json()["asset_key"] == "mobile-sam-vit-t"
    assert capability.json()["asset_status"] == "missing"
    assert capability.json()["available"] is False

    response = client.post(
        f"/api/images/{image_id}/segment-assist",
        json={
            "points": [{"x": 2, "y": 2, "kind": "positive"}],
            "label_key": "defect",
        },
    )
    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "model_asset_not_ready"


def test_prompt_geometry_and_taxonomy_are_validated(client: TestClient, seeded: Fixture) -> None:
    image_id = seeded.normal_image_ids[0]
    _open_draft(client, image_id)
    empty = client.post(
        f"/api/images/{image_id}/segment-assist",
        json={"points": [], "label_key": "defect"},
    )
    assert empty.status_code == 422
    outside = client.post(
        f"/api/images/{image_id}/segment-assist",
        json={
            "points": [{"x": 100_000, "y": 2, "kind": "positive"}],
            "label_key": "defect",
        },
    )
    assert outside.status_code == 422
    unknown = client.post(
        f"/api/images/{image_id}/segment-assist",
        json={
            "box": {"x0": 0, "y0": 0, "x1": 2, "y1": 2},
            "label_key": "unknown",
        },
    )
    assert unknown.status_code == 422


def test_assistance_is_refused_while_a_job_runs(
    client: TestClient, settings: Settings, seeded: Fixture
) -> None:
    image_id = seeded.normal_image_ids[0]
    _open_draft(client, image_id)
    app = client.app
    assert isinstance(app, FastAPI)
    queue: JobQueue = app.state.job_queue
    job = queue.enqueue(kind=JobKind.PREWARM, params={"dataset_id": seeded.dataset_id})
    with connection(settings.db_path) as conn:
        jobs_repo.mark_running(conn, job.id, log_path="/dev/null")
    response = client.post(
        f"/api/images/{image_id}/segment-assist",
        json={
            "points": [{"x": 2, "y": 2, "kind": "positive"}],
            "label_key": "defect",
        },
    )
    assert response.status_code == 409
    assert "prewarm" in response.text


class _Predictor:
    def __init__(self) -> None:
        self.image: np.ndarray | None = None

    def set_image(self, image: np.ndarray) -> None:
        self.image = image

    def predict(self, **_kwargs: object) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        assert self.image is not None
        height, width = self.image.shape[:2]
        first = np.zeros((height, width), dtype=np.bool_)
        first[1:4, 2:6] = True
        second = np.zeros((height, width), dtype=np.bool_)
        second[0:2, 0:2] = True
        return (
            np.stack([first, first, second]),
            np.asarray([0.8, 0.7, 0.6], dtype=np.float32),
            np.zeros((3, 1, 1), dtype=np.float32),
        )


class _FailingMpsPredictor(_Predictor):
    def predict(self, **_kwargs: object) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        raise RuntimeError("unsupported MPS operator")


def test_session_returns_deduplicated_tight_source_frame_candidates(
    settings: Settings,
    seeded: Fixture,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "fixture.pt"
    checkpoint.write_bytes(b"fixture")
    spec = ModelAssetSpec(
        key="fixture",
        title="Fixture",
        purpose="Test",
        filename=checkpoint.name,
        source_url="https://example.test/fixture",
        expected_size=checkpoint.stat().st_size,
        sha256="unused",
        license_name="Apache-2.0",
        license_url="https://example.test/license",
        project_url="https://example.test",
    )
    predictor = _Predictor()
    monkeypatch.setattr(mobile_sam, "get_spec", lambda _key: spec)
    monkeypatch.setattr(
        mobile_sam,
        "resolve_asset",
        lambda _settings, _spec: ResolvedAsset(checkpoint, "managed", True, 7),
    )
    monkeypatch.setattr(mobile_sam, "_preferred_device", lambda: "cpu")
    monkeypatch.setattr(mobile_sam, "_load_predictor", lambda _path, _device: (object(), predictor))
    session = mobile_sam.MobileSamSession(settings, "fixture")

    result: dict[str, Any] = session.segment(
        {
            "image_id": seeded.normal_image_ids[0],
            "points": [{"x": 3, "y": 2, "kind": "positive"}],
            "box": None,
            "label_key": "defect",
            "operation": "add",
        }
    )

    assert result["device"] == "cpu"
    candidates = result["candidates"]
    assert isinstance(candidates, list)
    assert len(candidates) == 2
    assert candidates[0]["area"] == 12
    assert candidates[0]["shape"]["x"] == 2
    assert candidates[0]["shape"]["y"] == 1
    assert candidates[0]["shape"]["width"] == 4
    assert candidates[0]["shape"]["height"] == 3


def test_session_rebuilds_on_cpu_when_prediction_fails_on_mps(
    settings: Settings,
    seeded: Fixture,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "fixture.pt"
    checkpoint.write_bytes(b"fixture")
    spec = ModelAssetSpec(
        key="fixture",
        title="Fixture",
        purpose="Test",
        filename=checkpoint.name,
        source_url="https://example.test/fixture",
        expected_size=checkpoint.stat().st_size,
        sha256="unused",
        license_name="Apache-2.0",
        license_url="https://example.test/license",
        project_url="https://example.test",
    )
    cpu_predictor = _Predictor()
    monkeypatch.setattr(mobile_sam, "get_spec", lambda _key: spec)
    monkeypatch.setattr(
        mobile_sam,
        "resolve_asset",
        lambda _settings, _spec: ResolvedAsset(checkpoint, "managed", True, 7),
    )
    monkeypatch.setattr(mobile_sam, "_preferred_device", lambda: "mps")
    monkeypatch.setattr(
        mobile_sam,
        "_load_predictor",
        lambda _path, device: (
            object(),
            _FailingMpsPredictor() if device == "mps" else cpu_predictor,
        ),
    )
    session = mobile_sam.MobileSamSession(settings, "fixture")

    result = session.segment(
        {
            "image_id": seeded.normal_image_ids[0],
            "points": [{"x": 3, "y": 2, "kind": "positive"}],
            "label_key": "defect",
            "operation": "add",
        }
    )

    assert result["device"] == "cpu"
    candidates = result["candidates"]
    assert isinstance(candidates, list)
    assert len(candidates) == 2
