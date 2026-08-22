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
from anomaly_lab.datasets.probe import measure as probe_manifest
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
    dataset_root: str | None = Field(
        default=None,
        description=(
            "The path recorded as this dataset's identity, when it is not where the walk "
            "starts. Must be the scan root or a directory inside it."
        ),
    )


def dataset_identity(root: Path, dataset_root: str | None) -> Path:
    """Resolve the path this dataset will be known by, refusing a claim it cannot back.

    Constrained to the scan root or a directory beneath it so an identity always names
    somewhere the images actually came from. Without that, two unrelated imports could
    claim each other's root and a re-import would update the wrong dataset.
    """
    if dataset_root is None:
        return root
    identity = Path(dataset_root).expanduser()
    if not identity.is_dir():
        msg = f"{identity} is not a directory"
        raise NotADirectoryError(msg)
    resolved, resolved_root = identity.resolve(), root.resolve()
    if resolved != resolved_root and resolved_root not in resolved.parents:
        msg = f"{identity} is not inside the scan root {root}"
        raise ValueError(msg)
    return identity


def run_scan_job(context: JobContext) -> dict[str, Any]:
    params = ScanParams.model_validate(dict(context.params))
    root = Path(params.root_path).expanduser()

    if not root.is_dir():
        msg = f"{root} is not a directory"
        raise NotADirectoryError(msg)

    identity = dataset_identity(root, params.dataset_root)

    adapter = get_adapter(params.adapter)
    options = parse_options(adapter, params.options)
    context.log(f"scanning {root} with the {params.adapter} adapter")

    def report(fraction: float, message: str | None) -> None:
        # Raising here is what stops a cancelled scan; the adapter needs no knowledge of
        # jobs, and no polling loop has to be threaded through the walk.
        context.raise_if_cancelled()
        context.progress(fraction, message)

    manifest = adapter.scan(root, options, dataset_name=params.dataset_name, progress=report)
    if identity != root:
        # `root_path` is the dataset's stable identity -- it is `UNIQUE`, and commit
        # resolves against it to make a re-import idempotent (ADR-0013). One capture tree
        # can legitimately hold several datasets: VisA's twelve object classes already do
        # this internally, and a rig that photographs two product variants into one
        # channel-first tree needs exactly the same separation. Recording the walk root
        # for both would collide them into one dataset.
        context.log(f"recording {identity} as this dataset's root")
        manifest = manifest.model_copy(update={"root_path": str(identity)})

    # The measured half of the scan, after the adapter has decided what the files *are*.
    # Kept here rather than inside any adapter so that every layout gets the same answers
    # and no adapter has to know the probe exists.
    context.raise_if_cancelled()
    context.progress(0.97, "measuring colour and channel registration")
    probe, probe_warnings = probe_manifest(manifest)
    manifest = manifest.model_copy(
        update={
            "probe": probe.model_dump(mode="json"),
            "warnings": [*manifest.warnings, *probe_warnings],
        }
    )

    manifest_id = scan_manifest_id(context.job_id)
    save_manifest(context.settings, manifest, manifest_id=manifest_id)

    for warning in manifest.warnings:
        context.log(f"{warning.code.value}: {warning.message}", level="warning")

    context.progress(1.0, f"{manifest.stats.samples} samples proposed")
    return {
        "manifest_id": manifest_id,
        "samples": manifest.stats.samples,
        "images": manifest.stats.images,
        "masks": manifest.stats.masks,
        "channels": len(manifest.channels),
        "warnings": len(manifest.warnings),
        "files_excluded": manifest.stats.files_excluded,
        "files_skipped": manifest.stats.files_skipped,
    }
