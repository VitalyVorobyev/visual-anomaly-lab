"""Cancellable, streaming download handler for catalogued model assets."""

from __future__ import annotations

import hashlib
import os
from contextlib import closing
from typing import Any
from urllib.request import Request, urlopen

from anomaly_lab.jobs.context import JobCancelledError, JobContext
from anomaly_lab.model_assets.catalog import get_spec
from anomaly_lab.model_assets.store import managed_path

CHUNK_SIZE = 1024 * 1024


def run_model_asset_download_job(ctx: JobContext) -> dict[str, Any]:
    key = str(ctx.params.get("asset_key", ""))
    spec = get_spec(key)
    if spec is None:
        raise ValueError(f"unknown model asset {key!r}")

    destination = managed_path(ctx.settings, spec)
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(f".part-{ctx.job_id}")
    digest = hashlib.sha256()
    received = 0
    request = Request(spec.source_url, headers={"User-Agent": "visual-anomaly-lab/0.1"})

    ctx.log(f"Downloading {spec.title} ({spec.expected_size:,} bytes).")
    try:
        with closing(urlopen(request, timeout=30)) as response, partial.open("wb") as output:
            while chunk := response.read(CHUNK_SIZE):
                ctx.raise_if_cancelled()
                output.write(chunk)
                digest.update(chunk)
                received += len(chunk)
                ctx.progress(
                    received / spec.expected_size,
                    f"{received / 1_048_576:.1f} / {spec.expected_size / 1_048_576:.1f} MiB",
                )
            output.flush()
            os.fsync(output.fileno())

        if received != spec.expected_size:
            raise ValueError(f"expected {spec.expected_size} bytes, received {received}")
        actual_digest = digest.hexdigest()
        if actual_digest != spec.sha256:
            raise ValueError(f"SHA-256 mismatch: expected {spec.sha256}, found {actual_digest}")
        partial.replace(destination)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise

    ctx.progress(1.0, "Verified and ready")
    ctx.log(f"Verified SHA-256 {spec.sha256}.")
    return {
        "asset_key": spec.key,
        "path": str(destination),
        "size": received,
        "sha256": spec.sha256,
    }


__all__ = ["JobCancelledError", "run_model_asset_download_job"]
