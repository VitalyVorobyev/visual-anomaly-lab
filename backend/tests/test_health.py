"""The health endpoint — what the shell polls and the UI renders."""

from __future__ import annotations

from fastapi.testclient import TestClient

from anomaly_lab.config import Settings
from anomaly_lab.db.migrate import discover_migrations


def test_health_reports_version_schema_and_paths(client: TestClient, settings: Settings) -> None:
    response = client.get("/api/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["version"]
    assert body["schema_version"] == len(discover_migrations())
    assert body["db_path"] == str(settings.db_path)
    assert body["data_dir"] == str(settings.data_dir)
    assert body["started_at"]


def test_lifespan_applies_migrations(client: TestClient, settings: Settings) -> None:
    """Starting the app is enough to create the database — no separate setup step."""
    assert settings.db_path.exists()
    assert client.get("/api/health").json()["schema_version"] >= 1


def test_health_is_served_under_the_api_prefix(client: TestClient) -> None:
    """`/api/health` is the contract; the bare path is not an alias."""
    assert client.get("/health").status_code == 404


def test_dev_cors_allows_the_tauri_webview_origin(client: TestClient) -> None:
    """The desktop WebView origin is not a localhost port; it needs explicit coverage."""
    response = client.get("/api/health", headers={"Origin": "tauri://localhost"})
    assert response.headers["access-control-allow-origin"] == "tauri://localhost"


def test_dev_cors_allows_the_vite_dev_server(client: TestClient) -> None:
    response = client.get("/api/health", headers={"Origin": "http://localhost:5173"})
    assert response.headers["access-control-allow-origin"] == "http://localhost:5173"


def test_dev_cors_exposes_annotation_etags(client: TestClient) -> None:
    response = client.get(
        "/api/health",
        headers={"Origin": "http://localhost:5173"},
    )
    assert response.headers["access-control-expose-headers"] == "ETag"


def test_dev_cors_rejects_a_foreign_origin(client: TestClient) -> None:
    response = client.get("/api/health", headers={"Origin": "https://example.com"})
    assert "access-control-allow-origin" not in response.headers
