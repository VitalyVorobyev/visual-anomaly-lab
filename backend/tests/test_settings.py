"""Settings resolution.

The rule under test is §4's: paths come from the settings object, and the data directory
is relocatable so that nothing is pinned to a source checkout.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from anomaly_lab.config import Settings, get_settings


def test_data_dir_comes_from_the_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ANOMALY_LAB_DATA_DIR", str(tmp_path / "elsewhere"))
    assert get_settings().data_dir == tmp_path / "elsewhere"


def test_data_dir_is_absolute_and_expanded(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANOMALY_LAB_DATA_DIR", "~/anomaly-lab-data")
    data_dir = get_settings().data_dir

    assert data_dir.is_absolute()
    assert "~" not in str(data_dir)


def test_data_dir_does_not_follow_the_working_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No module resolves application paths from the CWD (§4)."""
    monkeypatch.chdir(tmp_path)
    assert get_settings().data_dir != tmp_path / "data"


def test_db_path_sits_under_the_data_dir(tmp_path: Path) -> None:
    settings = Settings(data_dir=tmp_path)
    assert settings.db_path == tmp_path / "app.sqlite3"


def test_ensure_directories_is_idempotent(tmp_path: Path) -> None:
    settings = Settings(data_dir=tmp_path / "fresh")
    settings.ensure_directories()
    settings.ensure_directories()
    assert settings.data_dir.is_dir()


def test_binds_loopback_by_default() -> None:
    """There is no authentication, so the default must never be reachable off-machine."""
    assert Settings(data_dir=Path("/tmp")).host == "127.0.0.1"


def test_parent_pid_is_absent_unless_supervised(tmp_path: Path) -> None:
    """The orphan watchdog stays off for the standalone `uv run` path."""
    assert Settings(data_dir=tmp_path).parent_pid is None


def test_parent_pid_is_read_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANOMALY_LAB_PARENT_PID", "4242")
    assert get_settings().parent_pid == 4242
