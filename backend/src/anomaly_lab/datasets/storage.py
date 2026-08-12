"""Where manifests live on disk.

A manifest is kept for two reasons (ADR-0006): the review step needs somewhere to read
the proposal back from between the scan job finishing and the operator pressing commit,
and the *accepted* manifest is the reproducibility record — "how did this dataset come to
look like this?" is answerable months later only because the answer was written down.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from pathlib import Path

from anomaly_lab.config import Settings
from anomaly_lab.datasets.manifest import MANIFEST_VERSION, Manifest

# Manifest ids reach the filesystem through a URL path segment, so they are constrained
# to something that cannot escape the directory rather than merely checked for `..`.
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class ManifestNotFoundError(Exception):
    """No manifest with that id, or an id that is not one."""


class UnsupportedManifestVersionError(Exception):
    """The file was written by a newer version of this application."""


def scan_manifest_id(job_id: int) -> str:
    return f"scan-{job_id}"


def committed_manifest_id(dataset_id: int) -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    return f"dataset-{dataset_id}-{stamp}"


def manifest_path(settings: Settings, manifest_id: str) -> Path:
    if not _SAFE_ID.match(manifest_id):
        msg = f"{manifest_id!r} is not a manifest id"
        raise ManifestNotFoundError(msg)
    return settings.manifests_dir / f"{manifest_id}.json"


def save_manifest(settings: Settings, manifest: Manifest, *, manifest_id: str) -> Path:
    path = manifest_path(settings, manifest_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    return path


def load_manifest(settings: Settings, manifest_id: str) -> Manifest:
    return load_manifest_file(manifest_path(settings, manifest_id), label=repr(manifest_id))


def load_manifest_file(path: Path, *, label: str | None = None) -> Manifest:
    """Read a manifest from a path the application recorded for itself.

    Separate from `load_manifest` because a dataset stores the *path* of the manifest it
    was committed from, not its id — which is what lets a split be built from the import
    that produced the dataset, months later, without the id having to be kept anywhere.
    The id-based entry point stays the only one reachable from a URL.
    """
    if not path.is_file():
        msg = f"no manifest {label or str(path)}"
        raise ManifestNotFoundError(msg)

    manifest = Manifest.model_validate_json(path.read_text(encoding="utf-8"))
    if manifest.manifest_version > MANIFEST_VERSION:
        msg = (
            f"manifest {label or str(path)} is version {manifest.manifest_version}; "
            f"this build understands up to {MANIFEST_VERSION}"
        )
        raise UnsupportedManifestVersionError(msg)
    return manifest
