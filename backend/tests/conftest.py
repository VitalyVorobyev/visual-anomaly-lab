"""Shared fixtures.

Every test runs against a temporary data directory. Nothing here reads the repo-local
`data/`, and no test fixture is ever a real dataset file (ADR-0001).
"""

from __future__ import annotations

import sqlite3
import zlib
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from anomaly_lab.api.app import create_app
from anomaly_lab.config import Settings, get_settings
from anomaly_lab.db.connection import connect, connection
from anomaly_lab.db.migrate import apply_migrations
from anomaly_lab.db.repositories import datasets, images, masks, samples, splits
from anomaly_lab.domain.entities import JobKind, Label, Subset
from anomaly_lab.experiments.infer import run_infer_job
from anomaly_lab.experiments.train import run_train_job
from anomaly_lab.jobs.context import JobContext


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


def write_image(
    path: Path,
    *,
    mode: str = "RGB",
    size: tuple[int, int] = (8, 8),
    colour: int | None = None,
) -> Path:
    """Write a tiny synthetic image, creating parents.

    Fixtures are generated, never checked in, and never a real dataset file (ADR-0001).
    The format follows the suffix, so a test that needs the BMP decode path asks for
    `.bmp` and gets one — inside `tmp_path`, which is not the repository.

    Content varies with the filename unless `colour` is given, because otherwise every
    fixture image would be byte-identical and every scan would report duplicate hashes.
    Pass an explicit `colour` to two files when a test *wants* them to collide.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    shade = 128 if colour is None else colour
    fill: int | tuple[int, int, int] = shade if mode == "L" else (shade, shade, shade)
    image = Image.new(mode, size, fill)

    if colour is None:
        # Stamp a digest of the whole path into the first row. Real trees repeat both the
        # stem and its directory name across the label and channel dimensions — `1.png`
        # lives under `defect/Dome` *and* `no-defect/Dome` — so anything less than the
        # full path produces byte-identical fixtures and spurious duplicate-hash
        # warnings. Four bytes rather than one shade, because 256 possible images collide
        # by accident almost immediately.
        for offset, byte in enumerate(zlib.crc32(str(path).encode()).to_bytes(4, "big")):
            image.putpixel((offset, 0), byte if mode == "L" else (byte, byte, byte))

    image.save(path)
    return path


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


# ---------------------------------------------------------- a runnable experiment
#
# Shared because two test modules now need the same thing: a real dataset on disk, an
# explicit split, and a method that trains in milliseconds. `pixel_reference` is what
# makes that possible — numpy and Pillow only, no torch — which is also why it is what
# the resident worker is spiked against.

FIXTURE_SIZE = 16
TRAIN_NORMALS = 8
TEST_NORMALS = 3
TEST_DEFECTS = 3


def _gradient(seed: int) -> np.ndarray:
    generator = np.random.default_rng(seed)
    base = np.linspace(40, 200, FIXTURE_SIZE * FIXTURE_SIZE).reshape(FIXTURE_SIZE, FIXTURE_SIZE)
    return np.clip(base + generator.normal(0, 3, size=(FIXTURE_SIZE, FIXTURE_SIZE)), 0, 255)


def write_normal_image(path: Path, seed: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(_gradient(seed).astype(np.uint8), mode="L").convert("RGB").save(path)


def write_defect_image(path: Path, mask_path: Path, seed: int) -> None:
    array = _gradient(seed)
    array[5:10, 5:10] = 255
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array.astype(np.uint8), mode="L").convert("RGB").save(path)

    mask = np.zeros((FIXTURE_SIZE, FIXTURE_SIZE), dtype=np.uint8)
    mask[5:10, 5:10] = 255
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask, mode="L").save(mask_path)


@dataclass(frozen=True)
class Fixture:
    dataset_id: int
    split_id: int
    defect_image_ids: list[int]
    normal_image_ids: list[int]


def seed_synthetic_split(conn: sqlite3.Connection, root: Path) -> Fixture:
    """A dataset of single-image samples with an explicit train/test split and masks."""
    dataset = datasets.create_dataset(conn, name="synthetic", root_path=str(root))
    assignments: dict[int, Subset] = {}
    defect_image_ids: list[int] = []
    normal_image_ids: list[int] = []

    def add(external_id: str, path: Path, label: Label, subset: Subset) -> int:
        sample, _ = samples.upsert_sample(
            conn, dataset.id, group_key="all", external_id=external_id, label=label
        )
        image, _ = images.upsert_image(
            conn,
            sample.id,
            channel_id=None,
            path=str(path),
            width=FIXTURE_SIZE,
            height=FIXTURE_SIZE,
            bit_depth=24,
            file_size=path.stat().st_size,
            sha256=f"sha-{external_id}",
        )
        assignments[sample.id] = subset
        return image.id

    for index in range(TRAIN_NORMALS):
        path = root / "train" / f"n{index}.png"
        write_normal_image(path, index)
        add(f"train-{index}", path, Label.NORMAL, Subset.TRAIN)

    for index in range(TEST_NORMALS):
        path = root / "test" / f"n{index}.png"
        write_normal_image(path, 500 + index)
        normal_image_ids.append(add(f"test-n{index}", path, Label.NORMAL, Subset.TEST))

    for index in range(TEST_DEFECTS):
        path = root / "test" / f"d{index}.png"
        mask_path = root / "masks" / f"d{index}.png"
        write_defect_image(path, mask_path, 900 + index)
        image_id = add(f"test-d{index}", path, Label.DEFECT, Subset.TEST)
        masks.upsert_mask(conn, image_id, path=str(mask_path))
        defect_image_ids.append(image_id)

    split = splits.create_split(
        conn,
        dataset.id,
        name="official",
        strategy="imported",
        seed=0,
        params={"strategy": "imported"},
        assignments=assignments,
    )
    return Fixture(
        dataset_id=dataset.id,
        split_id=split.id,
        defect_image_ids=defect_image_ids,
        normal_image_ids=normal_image_ids,
    )


@pytest.fixture
def seeded(client: TestClient, settings: Settings, tmp_path: Path) -> Iterator[Fixture]:
    root = tmp_path / "source"
    with connection(settings.db_path) as conn:
        yield seed_synthetic_split(conn, root)


def create_experiment(client: TestClient, seeded: Fixture, **overrides: object) -> dict[str, Any]:
    body: dict[str, Any] = {
        "name": "baseline",
        "dataset_id": seeded.dataset_id,
        "split_id": seeded.split_id,
        "model_type": "pixel_reference",
        "config": {"smoothing_sigma": 1.0},
        "preprocessing": {"width": FIXTURE_SIZE, "height": FIXTURE_SIZE},
        "evaluation": {},
    }
    body.update(overrides)
    response = client.post("/api/experiments", json=body)
    assert response.status_code == 200, response.text
    payload: dict[str, Any] = response.json()
    return payload


def run_handler(settings: Settings, kind: JobKind, params: dict[str, Any]) -> dict[str, Any]:
    """Run a handler in-process. The subprocess path is covered in `test_job_queue`."""
    handler = run_train_job if kind is JobKind.TRAIN else run_infer_job
    return handler(JobContext(job_id=1, kind=kind, params=params, settings=settings))


@pytest.fixture
def scored(client: TestClient, settings: Settings, seeded: Fixture) -> dict[str, Any]:
    """A fully trained and scored experiment — the state most tests here start from."""
    experiment = create_experiment(client, seeded)
    run_handler(settings, JobKind.TRAIN, {"experiment_id": experiment["id"]})
    run_handler(settings, JobKind.INFER, {"experiment_id": experiment["id"], "subsets": ["test"]})
    return experiment
