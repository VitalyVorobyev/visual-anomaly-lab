"""Reading a run's scalar series back out of its log (ADR-0020).

`metric` events are streamed and tee'd, never stored in a column. That is enough for a
live chart and not enough for a chart that survives a reload, so the history is recovered
by re-reading the log the queue already wrote. Everything here is about that read being
tolerant of what a real log actually contains.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from fastapi.testclient import TestClient

from anomaly_lab.api.routers.jobs import read_metric_series


def _write_log(path: Path, lines: list[str]) -> str:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


def _metric(name: str, value: float, step: int | None = None) -> str:
    event: dict[str, object] = {"ev": "metric", "name": name, "value": value}
    if step is not None:
        event["step"] = step
    return json.dumps(event)


def test_series_are_grouped_by_name_and_ordered_as_emitted(tmp_path: Path) -> None:
    log = _write_log(
        tmp_path / "job.log",
        [
            _metric("loss_st", 1.0, 0),
            _metric("loss_ae", 9.0, 0),
            _metric("loss_st", 0.5, 20),
            _metric("loss_st", 0.25, 40),
        ],
    )

    series = {found.name: found for found in read_metric_series(log)}
    assert set(series) == {"loss_st", "loss_ae"}
    assert [point.value for point in series["loss_st"].points] == [1.0, 0.5, 0.25]
    assert [point.step for point in series["loss_st"].points] == [0, 20, 40]
    assert series["loss_st"].total == 3
    assert series["loss_st"].dropped == 0


def test_a_metric_without_a_step_is_kept_with_a_null_one(tmp_path: Path) -> None:
    """`pixel_reference` emits `reference_images` once, with no step. It is still a fact."""
    log = _write_log(tmp_path / "job.log", [_metric("reference_images", 8.0)])

    points = read_metric_series(log)[0].points
    assert len(points) == 1
    assert points[0].step is None
    assert points[0].value == 8.0


def test_library_chatter_and_progress_bars_are_skipped_not_fatal(tmp_path: Path) -> None:
    """The log carries every byte of worker output on purpose (ADR-0009)."""
    log = _write_log(
        tmp_path / "job.log",
        [
            "Downloading: 45%|####      | 1.2G/2.6G [00:31<00:38, 38.1MB/s]",
            '{"ev": "log", "level": "info", "message": "device=cpu"}',
            "{not json at all",
            '{"ev": "progress", "fraction": 0.5}',
            "[]",
            _metric("loss_total", 0.125, 100),
        ],
    )

    series = read_metric_series(log)
    assert len(series) == 1
    assert series[0].name == "loss_total"
    assert series[0].points[0].value == 0.125


def test_a_malformed_metric_event_is_dropped_rather_than_charted(tmp_path: Path) -> None:
    log = _write_log(
        tmp_path / "job.log",
        [
            '{"ev": "metric", "name": "ok", "value": 1.0}',
            '{"ev": "metric", "value": 2.0}',
            '{"ev": "metric", "name": "no_value"}',
            '{"ev": "metric", "name": "not_a_number", "value": "high"}',
        ],
    )

    series = read_metric_series(log)
    assert [found.name for found in series] == ["ok"]


def test_a_long_series_is_downsampled_across_the_whole_run_and_says_so(
    tmp_path: Path,
) -> None:
    """`evenly_spaced`, never the first N — a truncated curve must still show its shape."""
    log = _write_log(
        tmp_path / "job.log",
        [_metric("loss_st", float(step), step) for step in range(500)],
    )

    series = read_metric_series(log, limit=50)[0]
    assert len(series.points) == 50
    assert series.total == 500
    assert series.dropped == 450
    # The last point of the run is present, which is what "evenly spaced" buys over a head.
    assert series.points[-1].step == 499


def test_no_log_yet_is_an_empty_series_not_an_error(tmp_path: Path) -> None:
    assert read_metric_series(None) == []
    assert read_metric_series(str(tmp_path / "never-written.log")) == []


def test_the_endpoint_serves_a_job_s_series(
    client: TestClient, migrated_db: sqlite3.Connection, tmp_path: Path
) -> None:
    log = _write_log(tmp_path / "job.log", [_metric("loss_st", 0.5, 20)])
    cursor = migrated_db.execute(
        "INSERT INTO job (kind, status, log_path) VALUES ('train', 'succeeded', ?)",
        (log,),
    )
    migrated_db.commit()

    payload = client.get(f"/api/jobs/{cursor.lastrowid}/metrics").json()
    assert payload["series"] == [
        {"name": "loss_st", "points": [{"step": 20, "value": 0.5}], "total": 1, "dropped": 0}
    ]


def test_asking_for_an_unknown_job_s_metrics_is_a_404(client: TestClient) -> None:
    assert client.get("/api/jobs/9999/metrics").status_code == 404
