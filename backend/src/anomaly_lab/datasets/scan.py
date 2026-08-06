"""The scan job.

Stage one of import (ADR-0006): an adapter walks a tree and proposes a dataset, and
**nothing is written to the database**. The proposal lands in `data/manifests/` for the
review screen to read back.

Scanning is a job because it reads every byte of the source tree to hash it, and because
reusing the job system means there is no second progress mechanism anywhere in the
application (ADR-0009).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from anomaly_lab.datasets.adapters import channel_folders as _default_adapter
from anomaly_lab.datasets.adapters.base import get_adapter, parse_options
from anomaly_lab.datasets.storage import save_manifest, scan_manifest_id
from anomaly_lab.jobs.context import JobContext

DEFAULT_ADAPTER = _default_adapter.ChannelFoldersAdapter.name()


class ScanParams(BaseModel):
    """What a scan job was asked to do. Stored verbatim in `job.params`."""

    model_config = ConfigDict(frozen=True)

    root_path: str
    dataset_name: str
    adapter: str = DEFAULT_ADAPTER
    options: dict[str, Any] = Field(default_factory=dict)


def run_scan_job(context: JobContext) -> dict[str, Any]:
    params = ScanParams.model_validate(dict(context.params))
    root = Path(params.root_path).expanduser()

    if not root.is_dir():
        msg = f"{root} is not a directory"
        raise NotADirectoryError(msg)

    adapter = get_adapter(params.adapter)
    options = parse_options(adapter, params.options)
    context.log(f"scanning {root} with the {params.adapter} adapter")

    def report(fraction: float, message: str | None) -> None:
        # Raising here is what stops a cancelled scan; the adapter needs no knowledge of
        # jobs, and no polling loop has to be threaded through the walk.
        context.raise_if_cancelled()
        context.progress(fraction, message)

    manifest = adapter.scan(root, options, dataset_name=params.dataset_name, progress=report)

    manifest_id = scan_manifest_id(context.job_id)
    save_manifest(context.settings, manifest, manifest_id=manifest_id)

    for warning in manifest.warnings:
        context.log(f"{warning.code.value}: {warning.message}", level="warning")

    context.progress(1.0, f"{manifest.stats.samples} samples proposed")
    return {
        "manifest_id": manifest_id,
        "samples": manifest.stats.samples,
        "images": manifest.stats.images,
        "channels": len(manifest.channels),
        "warnings": len(manifest.warnings),
        "files_excluded": manifest.stats.files_excluded,
        "files_skipped": manifest.stats.files_skipped,
    }
