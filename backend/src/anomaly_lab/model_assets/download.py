"""Cancellable, streaming download handler for catalogued model assets.

Two sources, one outcome. An asset with a plain `source_url` streams from it; an asset
pinned to a Hugging Face revision (`spec.hub`) is fetched file by file through
`huggingface_hub`, which carries the account token a gated repository needs and follows
its storage redirects without handing that token to a third host. Either way every file is
checked against its catalogued size and SHA-256 before it is moved into place, and the main
file is moved last, so a cancelled or failed download never leaves a directory the catalogue
would call ready.
"""

from __future__ import annotations

import hashlib
import os
import shutil
from contextlib import closing
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

from anomaly_lab.jobs.context import JobCancelledError, JobContext
from anomaly_lab.model_assets.catalog import ModelAssetSpec, gated_message, get_spec
from anomaly_lab.model_assets.store import asset_files, managed_path, sha256_file

CHUNK_SIZE = 1024 * 1024


def run_model_asset_download_job(ctx: JobContext) -> dict[str, Any]:
    key = str(ctx.params.get("asset_key", ""))
    spec = get_spec(key)
    if spec is None:
        raise ValueError(f"unknown model asset {key!r}")
    if spec.hub is not None:
        return _download_from_hub(ctx, spec)

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


def _download_from_hub(ctx: JobContext, spec: ModelAssetSpec) -> dict[str, Any]:
    """Every file of a pinned revision into a staging directory, verified, then moved in."""
    assert spec.hub is not None
    destination = managed_path(ctx.settings, spec)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.parent / f".part-{ctx.job_id}"
    # Companions first and the main file last: small files fail fast on a missing grant,
    # and the file whose presence the catalogue reads is the last one to arrive.
    files = list(reversed(asset_files(spec, destination)))
    total = spec.total_size
    done = 0

    ctx.log(
        f"Downloading {spec.title} from {spec.hub.repository}@{spec.hub.revision[:12]} "
        f"({total:,} bytes in {len(files)} files)."
    )
    try:
        for target, size, digest in files:
            ctx.raise_if_cancelled()

            # A file can have more than one progress bar (Xet reports reconstruction and
            # transfer separately), so only the furthest reading counts, and a step of
            # under half a percent of the whole is not worth a progress line.
            reported = {"bytes": 0}

            def report(
                received: int, *, before: int = done, seen: dict[str, int] = reported
            ) -> None:
                ctx.raise_if_cancelled()
                if received - seen["bytes"] < total // 200:
                    return
                seen["bytes"] = received
                fraction = min((before + received) / total, 1.0)
                ctx.progress(
                    fraction,
                    f"{(before + received) / 1_048_576:.1f} / {total / 1_048_576:.1f} MiB",
                )

            try:
                fetched = fetch_from_hub(
                    spec.hub.repository, spec.hub.revision, target.name, staging, report
                )
            except JobCancelledError:
                raise
            except Exception as exc:
                if spec.gated and _refused(exc):
                    raise PermissionError(gated_message(spec, exc)) from exc
                raise
            actual_size = fetched.stat().st_size
            if actual_size != size:
                raise ValueError(f"{target.name}: expected {size} bytes, received {actual_size}")
            actual = sha256_file(fetched)
            if actual != digest:
                raise ValueError(
                    f"{target.name}: SHA-256 mismatch: expected {digest}, found {actual}"
                )
            done += size
            ctx.progress(done / total, f"{done / 1_048_576:.1f} / {total / 1_048_576:.1f} MiB")
        for target, _, _ in files:
            (staging / target.name).replace(target)
    finally:
        shutil.rmtree(staging, ignore_errors=True)

    ctx.progress(1.0, "Verified and ready")
    ctx.log(f"Verified {len(files)} files; {spec.filename} SHA-256 {spec.sha256}.")
    return {
        "asset_key": spec.key,
        "path": str(destination),
        "size": total,
        "sha256": spec.sha256,
    }


def fetch_from_hub(
    repository: str,
    revision: str,
    filename: str,
    staging: Path,
    report: Any,
) -> Path:
    """One file of a pinned revision into `staging`, reporting bytes as they arrive.

    `huggingface_hub` is imported here, not at module import: it ships with the `dl`
    extra, and a torch-free backend must still import this module to run a plain download.
    """
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils.tqdm import tqdm as hub_tqdm

    # Silent on the terminal; every update becomes job progress, and a cancel point. Built
    # with `type` rather than a class statement: the base is typed where the `dl` extra is
    # installed and `Any` where it is not, and a class statement cannot satisfy both.
    def display(self: Any, *args: Any, **kwargs: Any) -> None:
        return None

    def update(self: Any, n: float | None = 1) -> Any:
        result = hub_tqdm.update(self, n)
        report(int(self.n))
        return result

    progress = type("Progress", (hub_tqdm,), {"display": display, "update": update})

    return Path(
        hf_hub_download(
            repo_id=repository,
            filename=filename,
            revision=revision,
            local_dir=staging,
            tqdm_class=progress,
        )
    )


def _refused(exc: Exception) -> bool:
    """A 401/403 from the hub, or its own gated-repository error."""
    if type(exc).__name__ in {"GatedRepoError", "RepositoryNotFoundError"}:
        return True
    response = getattr(exc, "response", None)
    return getattr(response, "status_code", None) in {401, 403}


__all__ = ["JobCancelledError", "fetch_from_hub", "run_model_asset_download_job"]
