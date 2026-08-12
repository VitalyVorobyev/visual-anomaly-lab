"""Inventory and remove paths that are owned by the application.

Source datasets are referenced in place and must never enter this module.  Callers pass
only paths derived from :class:`Settings`; exact-path checks happen before a destructive
operation is offered to the API.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from anomaly_lab.config import Settings
from anomaly_lab.domain.entities import Experiment


@dataclass(frozen=True)
class StorageUsage:
    files: int = 0
    bytes: int = 0

    def __add__(self, other: StorageUsage) -> StorageUsage:
        return StorageUsage(self.files + other.files, self.bytes + other.bytes)


def experiment_artifact_path(settings: Settings, experiment: Experiment) -> Path | None:
    """Return the exact app-owned directory, or ``None`` for an unsafe stored path."""
    if not experiment.artifact_dir:
        return None
    stored = Path(experiment.artifact_dir).expanduser()
    expected = settings.experiment_dir(experiment.id)
    if not stored.is_absolute() or ".." in stored.parts:
        return None
    if settings.artifacts_dir.is_symlink() or expected.is_symlink():
        return None
    return stored if stored == expected else None


def path_usage(path: Path | None) -> StorageUsage:
    """Count payload files and bytes without following symlinks out of a directory."""
    if path is None or (not path.exists() and not path.is_symlink()):
        return StorageUsage()
    if path.is_symlink() or not path.is_dir():
        try:
            stat = path.stat(follow_symlinks=False)
        except FileNotFoundError:
            return StorageUsage()
        return StorageUsage(files=1, bytes=stat.st_size)

    usage = StorageUsage()
    pending = [path]
    while pending:
        directory = pending.pop()
        try:
            entries = os.scandir(directory)
        except FileNotFoundError:
            continue
        with entries:
            for entry in entries:
                try:
                    if entry.is_dir(follow_symlinks=False):
                        pending.append(Path(entry.path))
                    else:
                        stat = entry.stat(follow_symlinks=False)
                        usage += StorageUsage(files=1, bytes=stat.st_size)
                except FileNotFoundError:
                    continue
    return usage
