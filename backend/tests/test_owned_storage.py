"""Safety boundary for app-owned filesystem reconciliation."""

from __future__ import annotations

from pathlib import Path

from anomaly_lab.config import Settings
from anomaly_lab.owned_storage import remove_orphan_experiment_artifacts


def test_orphan_cleanup_only_removes_exact_app_owned_experiment_directories(
    settings: Settings, tmp_path: Path
) -> None:
    settings.ensure_directories()
    settings.artifacts_dir.mkdir(parents=True, exist_ok=True)
    live = settings.artifacts_dir / "exp-7"
    orphan = settings.artifacts_dir / "exp-8"
    unrelated = settings.artifacts_dir / "exports"
    suspicious = settings.artifacts_dir / "exp-09-extra"
    for path in (live, orphan, unrelated, suspicious):
        path.mkdir()
        (path / "payload.bin").write_bytes(b"synthetic")
    external = tmp_path / "external"
    external.mkdir()
    sentinel = external / "sentinel"
    sentinel.write_bytes(b"keep")
    (settings.artifacts_dir / "exp-10").symlink_to(external, target_is_directory=True)

    removed = remove_orphan_experiment_artifacts(settings, {7})

    assert removed == [orphan]
    assert live.is_dir()
    assert not orphan.exists()
    assert unrelated.is_dir()
    assert suspicious.is_dir()
    assert (settings.artifacts_dir / "exp-10").is_symlink()
    assert sentinel.read_bytes() == b"keep"
