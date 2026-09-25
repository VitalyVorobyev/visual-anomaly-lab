"""How an anomaly map is stored and read back (`anomaly_lab.map_files`).

A map is kept in the prepared frame beside its pinned transform and projected on read; a
float32 `.npy` written before that format must still read, and every consumer must get the
same numbers from either.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient

from anomaly_lab.config import Settings
from anomaly_lab.db.connection import connection
from anomaly_lab.db.repositories import experiments as experiments_repo
from anomaly_lab.db.repositories import results as results_repo
from anomaly_lab.eval.evaluators import evaluator_for
from anomaly_lab.map_files import (
    map_file,
    read_map,
    stored_map_file,
    write_map_file,
)
from anomaly_lab.models.base import Device, InferContext, NullReporter
from anomaly_lab.models.diagnostics import DiagnosticWriter
from anomaly_lab.models.preprocessing import PreprocessingConfig
from anomaly_lab.regions.transform import PixelBounds, SpatialTransform


def _cropped() -> SpatialTransform:
    """A real crop and a letterbox, so a frame error cannot hide behind a uniform scale."""
    return SpatialTransform.resolve(
        source_size=(40, 30),
        prepared_size=(16, 16),
        region=PixelBounds(left=6, top=4, right=34, bottom=22),
        padding_fraction=0.0,
    )


def _prepared(seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).random((16, 16), dtype=np.float32) * 14.0


def test_a_stored_map_reads_back_as_the_projection_bit_for_bit(tmp_path: Path) -> None:
    transform = _cropped()
    prepared = _prepared()
    path = map_file(tmp_path, 3)
    write_map_file(path, prepared, transform)

    values = read_map(path)
    expected = transform.project_map(prepared)
    assert values.dtype == np.float32
    assert values.shape == (30, 40)
    assert values.tobytes() == expected.tobytes()
    assert np.isnan(values[0, 0])


def test_a_map_written_before_the_format_still_reads(tmp_path: Path) -> None:
    source = _cropped().project_map(_prepared())
    legacy = tmp_path / "3.npy"
    np.save(legacy, source)

    assert stored_map_file(tmp_path, 3) == legacy
    assert read_map(legacy).tobytes() == source.tobytes()


def test_rewriting_a_legacy_map_replaces_it(tmp_path: Path) -> None:
    np.save(tmp_path / "3.npy", np.zeros((30, 40), np.float32))
    write_map_file(map_file(tmp_path, 3), _prepared(), _cropped())

    assert not (tmp_path / "3.npy").exists()
    assert stored_map_file(tmp_path, 3) == tmp_path / "3.npz"
    assert sorted(path.name for path in tmp_path.iterdir()) == ["3.npz"]


def test_a_map_without_a_transform_keeps_its_array(tmp_path: Path) -> None:
    values = _prepared()
    path = map_file(tmp_path, 1)
    write_map_file(path, values, None)
    assert read_map(path).tobytes() == values.tobytes()


def test_a_missing_or_foreign_file_fails_as_every_caller_expects(tmp_path: Path) -> None:
    with pytest.raises(OSError):
        read_map(tmp_path / "9.npz")
    (tmp_path / "8.npz").write_bytes(b"not a zip archive")
    with pytest.raises(ValueError):
        read_map(tmp_path / "8.npz")
    np.savez_compressed(tmp_path / "7.npz", other=np.zeros(2))
    with pytest.raises(ValueError):
        read_map(tmp_path / "7.npz")


def test_write_map_stores_the_prepared_frame_and_measures_the_projection(
    tmp_path: Path,
) -> None:
    transform = _cropped()
    config = PreprocessingConfig(width=16, height=16)
    ctx = InferContext(
        artifact_dir=tmp_path,
        cache_dir=tmp_path / "cache",
        preprocessing=config,
        device=Device.CPU,
        reporter=NullReporter(),
        diagnostics=DiagnosticWriter(tmp_path / "diagnostics"),
        map_transform=lambda _: transform,
    )
    prepared = _prepared()
    path = ctx.write_map(5, prepared[None])

    projected = transform.project_map(prepared)
    with np.load(path) as stored:
        assert stored["map"].shape == (16, 16)
    assert read_map(path).tobytes() == projected.tobytes()
    finite = projected[np.isfinite(projected)]
    assert ctx.display_range() == (float(finite.min()), float(np.percentile(finite, 99.9)))
    peak = ctx.peak_for(5)
    assert peak is not None
    assert projected[peak[1], peak[0]] == np.nanmax(projected)


def _evaluate(settings: Settings, experiment_id: int) -> dict[str, Any]:
    with connection(settings.db_path) as conn:
        experiment = experiments_repo.get_experiment(conn, experiment_id)
        assert experiment is not None
        return {
            subset.value: found
            for subset, found in evaluator_for(experiment.task).evaluate(conn, experiment).items()
        }


def test_legacy_and_current_maps_give_identical_metrics_and_overlays(
    client: TestClient, settings: Settings, scored: dict[str, Any]
) -> None:
    """Every consumer reads through `read_map`, so a run's numbers cannot depend on its format."""
    with connection(settings.db_path) as conn:
        images = [i for i in results_repo.list_scored_images(conn, scored["id"]) if i.map_path]
    assert images and all(str(image.map_path).endswith(".npz") for image in images)
    shown = images[0].image_id
    params = {"experiment_id": scored["id"]}
    current_metrics = _evaluate(settings, scored["id"])
    current_overlay = client.get(f"/api/images/{shown}/anomaly-map", params=params).content
    current_values = client.get(f"/api/images/{shown}/anomaly-map/values", params=params).content

    # Rewrite the run as a pre-format run would have left it: source-frame `.npy` maps.
    with connection(settings.db_path) as conn:
        for image in images:
            current = Path(str(image.map_path))
            legacy = current.with_suffix(".npy")
            np.save(legacy, read_map(current))
            current.unlink()
            conn.execute(
                "UPDATE image_result SET map_path = ? WHERE experiment_id = ? AND image_id = ?",
                (str(legacy), scored["id"], image.image_id),
            )

    assert _evaluate(settings, scored["id"]) == current_metrics
    assert client.get(f"/api/images/{shown}/anomaly-map", params=params).content == (
        current_overlay
    )
    assert client.get(f"/api/images/{shown}/anomaly-map/values", params=params).content == (
        current_values
    )
