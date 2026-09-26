"""Model assets are explicit, licensed, verified and safe to remove."""

from __future__ import annotations

import hashlib
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from anomaly_lab.api.app import create_app
from anomaly_lab.api.routers import model_assets as model_asset_routes
from anomaly_lab.config import Settings
from anomaly_lab.domain.entities import JobKind
from anomaly_lab.jobs.context import JobContext
from anomaly_lab.model_assets import download
from anomaly_lab.model_assets.catalog import (
    SAM3_KEY,
    AssetFile,
    HubPin,
    ModelAssetSpec,
    get_spec,
)
from anomaly_lab.model_assets.store import managed_path, resolve_asset


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


def _hub_spec(files: dict[str, bytes], *, gated: bool = True) -> ModelAssetSpec:
    main = files["model.bin"]
    return ModelAssetSpec(
        key="hub-fixture",
        title="Hub fixture",
        purpose="Test a pinned Hugging Face revision.",
        filename="model.bin",
        source_url="https://huggingface.co/owner/repo/blob/abc/model.bin",
        expected_size=len(main),
        sha256=hashlib.sha256(main).hexdigest(),
        license_name="Fixture License",
        license_url="https://example.test/license",
        project_url="https://example.test/project",
        companions=tuple(
            AssetFile(name, len(payload), hashlib.sha256(payload).hexdigest())
            for name, payload in files.items()
            if name != "model.bin"
        ),
        hub=HubPin(
            repository="owner/repo",
            revision="a" * 40,
            access_url="https://huggingface.co/owner/repo" if gated else None,
        ),
    )


def _context(settings: Settings, key: str) -> JobContext:
    return JobContext(
        job_id=9,
        kind=JobKind.MODEL_ASSET_DOWNLOAD,
        params={"asset_key": key},
        settings=settings,
    )


def test_a_hub_asset_is_every_file_verified_and_moved_in_main_file_last(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    files = {"model.bin": b"weights" * 1000, "config.json": b"{}", "vocab.txt": b"a b c"}
    spec = _hub_spec(files)
    monkeypatch.setattr(download, "get_spec", lambda key: spec if key == spec.key else None)
    fetched: list[str] = []

    def fetch(repository: str, revision: str, filename: str, staging: Path, report: Any) -> Path:
        assert (repository, revision) == ("owner/repo", "a" * 40)
        fetched.append(filename)
        staging.mkdir(parents=True, exist_ok=True)
        (staging / filename).write_bytes(files[filename])
        report(len(files[filename]))
        return staging / filename

    monkeypatch.setattr(download, "fetch_from_hub", fetch)
    settings = Settings(data_dir=tmp_path / "data")
    result = download.run_model_asset_download_job(_context(settings, spec.key))

    assert fetched[-1] == "model.bin"
    assert result["size"] == spec.total_size
    resolved = resolve_asset(settings, spec)
    assert resolved.ready, resolved.reason
    assert not list(managed_path(settings, spec).parent.glob(".part-*"))

    # A directory that lost a companion is not ready, and says which file it lacks.
    (managed_path(settings, spec).parent / "vocab.txt").unlink()
    broken = resolve_asset(settings, spec)
    assert not broken.ready and broken.reason == "vocab.txt: missing"


def test_a_refused_gated_download_explains_the_token_and_installs_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = _hub_spec({"model.bin": b"weights", "config.json": b"{}"})
    monkeypatch.setattr(download, "get_spec", lambda key: spec if key == spec.key else None)

    class GatedRepoError(Exception):
        pass

    def refuse(*_args: Any) -> Path:
        raise GatedRepoError("401 Client Error")

    monkeypatch.setattr(download, "fetch_from_hub", refuse)
    settings = Settings(data_dir=tmp_path / "data")
    with pytest.raises(PermissionError) as refused:
        download.run_model_asset_download_job(_context(settings, spec.key))
    message = str(refused.value)
    assert "HF_TOKEN" in message and "https://huggingface.co/owner/repo" in message
    assert list(managed_path(settings, spec).parent.iterdir()) == []


def test_a_hub_file_that_is_not_the_pinned_one_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = _hub_spec({"model.bin": b"weights", "config.json": b"{}"}, gated=False)
    monkeypatch.setattr(download, "get_spec", lambda key: spec if key == spec.key else None)

    def tampered(
        _repository: str, _revision: str, filename: str, staging: Path, _report: Any
    ) -> Path:
        staging.mkdir(parents=True, exist_ok=True)
        (staging / filename).write_bytes(b"{}" if filename == "config.json" else b"WEIGHTS")
        return staging / filename

    monkeypatch.setattr(download, "fetch_from_hub", tampered)
    settings = Settings(data_dir=tmp_path / "data")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        download.run_model_asset_download_job(_context(settings, spec.key))
    assert not resolve_asset(settings, spec).ready
    assert list(managed_path(settings, spec).parent.iterdir()) == []


def test_sam3_is_catalogued_pinned_and_gated() -> None:
    spec = get_spec(SAM3_KEY)
    assert spec is not None
    assert spec.hub is not None and len(spec.hub.revision) == 40
    assert spec.gated and spec.license_name == "SAM License"
    assert {entry.filename for entry in spec.companions} >= {"config.json", "tokenizer.json"}
    assert spec.total_size > spec.expected_size
