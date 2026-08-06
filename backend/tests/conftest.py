"""Shared fixtures.

Every test runs against a temporary data directory. Nothing here reads the repo-local
`data/`, and no test fixture is ever a real dataset file (ADR-0001).
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from anomaly_lab.api.app import create_app
from anomaly_lab.config import Settings, get_settings
from anomaly_lab.db.connection import connect
from anomaly_lab.db.migrate import apply_migrations


@pytest.fixture(autouse=True)
def _clear_settings_cache() -> Iterator[None]:
    """`get_settings` is process-cached; keep leakage between tests impossible."""
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    path = tmp_path / "data"
    path.mkdir()
    return path


@pytest.fixture
def settings(data_dir: Path) -> Settings:
    return Settings(data_dir=data_dir)


@pytest.fixture
def migrated_db(settings: Settings) -> Iterator[sqlite3.Connection]:
    """A database at the current schema version, with an open connection."""
    apply_migrations(settings.db_path)
    conn = connect(settings.db_path)
    try:
        yield conn
    finally:
        conn.close()


@pytest.fixture
def client(settings: Settings) -> Iterator[TestClient]:
    """A client whose lifespan has run, so migrations have been applied."""
    with TestClient(create_app(settings)) as test_client:
        yield test_client
