"""Model assets are explicit, licensed, verified and safe to remove."""

from __future__ import annotations

import hashlib
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from anomaly_lab.api.app import create_app
from anomaly_lab.api.routers import model_assets as model_asset_routes
from anomaly_lab.config import Settings
from anomaly_lab.domain.entities import JobKind
from anomaly_lab.jobs.context import JobContext
from anomaly_lab.model_assets import download
from anomaly_lab.model_assets.catalog import ModelAssetSpec
from anomaly_lab.model_assets.store import managed_path


def _spec(payload: bytes, source_url: str = "http://127.0.0.1/asset.bin") -> ModelAssetSpec:
    return ModelAssetSpec(
        key="fixture",
        title="Fixture model",
        purpose="Test the asset boundary.",
        filename="asset.bin",
        source_url=source_url,
        expected_size=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
        license_name="Apache-2.0",
        license_url="https://example.test/license",
        project_url="https://example.test/project",
    )


def test_catalog_external_override_and_managed_deletion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = b"verified model bytes"
    spec = _spec(payload)
    monkeypatch.setattr(model_asset_routes, "SPECS", (spec,))
    monkeypatch.setattr(
        model_asset_routes, "get_spec", lambda key: spec if key == spec.key else None
    )
    external = tmp_path / "external.bin"
    external.write_bytes(payload)
    settings = Settings(data_dir=tmp_path / "data")

    with TestClient(create_app(settings)) as client:
        missing = client.get("/api/model-assets").json()["assets"][0]
        assert missing["status"] == "missing"
        assert missing["source"] == "managed"
        assert missing["license_name"] == "Apache-2.0"

        rejected = client.post(
            "/api/model-assets/fixture/install", json={"license_accepted": False}
        )
        assert rejected.status_code == 422

        selected = client.put("/api/model-assets/fixture/source", json={"path": str(external)})
        assert selected.status_code == 200, selected.text
        assert selected.json()["status"] == "ready"
        assert selected.json()["source"] == "external"
        assert client.delete("/api/model-assets/fixture").status_code == 409
        assert external.exists()

        cleared = client.delete("/api/model-assets/fixture/source")
        assert cleared.json()["status"] == "missing"

        destination = managed_path(settings, spec)
        destination.parent.mkdir(parents=True)
        destination.write_bytes(payload)
        assert client.get("/api/model-assets").json()["assets"][0]["status"] == "ready"
        removed = client.delete("/api/model-assets/fixture")
        assert removed.status_code == 200
        assert removed.json()["status"] == "missing"
        assert not destination.exists()
        assert client.delete("/api/model-assets/unknown").status_code == 404


def test_download_streams_to_a_partial_then_atomically_installs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = b"model" * 300_000
    served = tmp_path / "served"
    served.mkdir()
    (served / "asset.bin").write_bytes(payload)

    handler = partial(SimpleHTTPRequestHandler, directory=str(served))
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        spec = _spec(payload, f"http://127.0.0.1:{server.server_port}/asset.bin")
        monkeypatch.setattr(download, "get_spec", lambda key: spec if key == spec.key else None)
        settings = Settings(data_dir=tmp_path / "data")
        context = JobContext(
            job_id=7,
            kind=JobKind.MODEL_ASSET_DOWNLOAD,
            params={"asset_key": spec.key},
            settings=settings,
        )

        result = download.run_model_asset_download_job(context)

        assert result["sha256"] == spec.sha256
        assert managed_path(settings, spec).read_bytes() == payload
        assert not list(managed_path(settings, spec).parent.glob("*.part-*"))
    finally:
        server.shutdown()
        thread.join(timeout=2)


def test_invalid_external_source_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = _spec(b"right")
    monkeypatch.setattr(model_asset_routes, "SPECS", (spec,))
    monkeypatch.setattr(
        model_asset_routes, "get_spec", lambda key: spec if key == spec.key else None
    )
    wrong = tmp_path / "wrong.bin"
    wrong.write_bytes(b"wrong")
    with TestClient(create_app(Settings(data_dir=tmp_path / "data"))) as client:
        response = client.put("/api/model-assets/fixture/source", json={"path": str(wrong)})
        assert response.status_code == 422
        assert "SHA-256 mismatch" in response.text
