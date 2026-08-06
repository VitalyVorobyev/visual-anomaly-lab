"""Shared fixtures.

Every test runs against a temporary data directory. Nothing here reads the repo-local
`data/`, and no test fixture is ever a real dataset file (ADR-0001).
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from anomaly_lab.api.app import create_app
from anomaly_lab.config import Settings, get_settings
from anomaly_lab.db.connection import connect
from anomaly_lab.db.migrate import apply_migrations
from anomaly_lab.db.repositories import datasets, images, samples
from anomaly_lab.domain.entities import Label


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


@dataclass(frozen=True)
class SeededCatalog:
    """A tiny hand-built catalog: two channels, three samples, five images.

    Deliberately irregular — one sample has a single channel and one has none — so that
    any code which quietly assumes a fixed channel count fails here rather than on real
    data (ADR-0005).
    """

    dataset_id: int
    channel_ids: dict[str, int]
    sample_ids: dict[str, int]
    image_ids: list[int]


@pytest.fixture
def catalog(migrated_db: sqlite3.Connection) -> SeededCatalog:
    dataset = datasets.create_dataset(migrated_db, name="fixture", root_path="/fixture/root")
    channels = {
        name: datasets.upsert_channel(migrated_db, dataset.id, name=name, position=index).id
        for index, name in enumerate(("bright", "dark"))
    }

    plan: list[tuple[str, str, Label, list[str | None]]] = [
        ("group-a", "1", Label.NORMAL, ["bright", "dark"]),
        ("group-a", "2", Label.DEFECT, ["bright", "dark"]),
        ("group-b", "1", Label.UNLABELED, [None]),
    ]

    sample_ids: dict[str, int] = {}
    image_ids: list[int] = []
    for group_key, external_id, label, channel_names in plan:
        sample, _ = samples.upsert_sample(
            migrated_db,
            dataset.id,
            group_key=group_key,
            external_id=external_id,
            label=label,
        )
        sample_ids[f"{group_key}/{external_id}"] = sample.id
        for channel_name in channel_names:
            image, _ = images.upsert_image(
                migrated_db,
                sample.id,
                channel_id=channels[channel_name] if channel_name else None,
                path=f"/fixture/root/{group_key}/{channel_name or 'plain'}/{external_id}.png",
                width=8,
                height=8,
                bit_depth=8,
                file_size=64,
                sha256=f"hash-{group_key}-{external_id}-{channel_name}",
            )
            image_ids.append(image.id)

    return SeededCatalog(
        dataset_id=dataset.id,
        channel_ids=channels,
        sample_ids=sample_ids,
        image_ids=image_ids,
    )
