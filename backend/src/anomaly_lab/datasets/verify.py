"""The verify job.

Referencing images in place rather than copying them (ADR-0001) buys privacy and disk,
and costs the guarantee that what the catalog points at is still what it pointed at. A
renamed folder breaks every path; a re-exported file keeps its path and changes its
bytes. Verify is what makes that discoverable rather than something you learn from a
model that suddenly scores differently.

It **detects drift and never repairs it**. Deciding that a file which changed should be
re-hashed into the catalog, or that a missing one should be dropped, is the operator's
call, and the honest failure of an unmounted disk is a report rather than a deletion.

Masks are walked too. Migration 005 added a nullable digest: masks that have been pinned
as annotation provenance get a byte-for-byte check, while older rows still receive an
explicit presence-only check. The report separates that coverage so it never implies a
hash comparison it did not perform.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from anomaly_lab.db.connection import connection
from anomaly_lab.db.repositories import datasets as datasets_repo
from anomaly_lab.db.repositories import images as images_repo
from anomaly_lab.db.repositories import masks as masks_repo
from anomaly_lab.jobs.context import JobContext
from anomaly_lab.media.decode import sha256_of

# Same reasoning as the commit report: a moved directory is one problem, not nine hundred.
MAX_PATHS_REPORTED = 50


class VerifyParams(BaseModel):
    model_config = ConfigDict(frozen=True)

    dataset_id: int


def run_verify_job(context: JobContext) -> dict[str, Any]:
    params = VerifyParams.model_validate(dict(context.params))

    with connection(context.settings.db_path) as conn:
        dataset = datasets_repo.get_dataset(conn, params.dataset_id)
        if dataset is None:
            msg = f"no dataset with id {params.dataset_id}"
            raise LookupError(msg)
        recorded = images_repo.list_images_for_dataset(conn, dataset.id)
        recorded_masks = masks_repo.list_masks_for_dataset(conn, dataset.id)

    context.log(
        f"verifying {len(recorded)} images and {len(recorded_masks)} masks "
        f"of dataset {dataset.name!r}"
    )

    missing: list[str] = []
    modified: list[str] = []
    unreadable: list[str] = []
    verified = 0

    total = len(recorded) + len(recorded_masks)
    for index, image in enumerate(recorded):
        context.raise_if_cancelled()
        path = Path(image.path)

        if not path.is_file():
            missing.append(image.path)
        else:
            try:
                digest = sha256_of(path)
            except OSError:
                unreadable.append(image.path)
            else:
                if digest == image.sha256:
                    verified += 1
                else:
                    modified.append(image.path)

        if total:
            context.progress((index + 1) / total, f"{index + 1} of {total}")

    masks_missing: list[str] = []
    masks_modified: list[str] = []
    masks_unreadable: list[str] = []
    masks_verified = 0
    masks_digest_checked = 0
    masks_unhashed = 0
    for offset, mask in enumerate(recorded_masks):
        context.raise_if_cancelled()
        path = Path(mask.path)
        if not path.is_file():
            masks_missing.append(mask.path)
        elif mask.sha256 is None:
            masks_unhashed += 1
            masks_verified += 1
        else:
            try:
                digest = sha256_of(path)
            except OSError:
                masks_unreadable.append(mask.path)
            else:
                masks_digest_checked += 1
                if digest == mask.sha256:
                    masks_verified += 1
                else:
                    masks_modified.append(mask.path)

        if total:
            position = len(recorded) + offset + 1
            context.progress(position / total, f"{position} of {total}")

    for label, paths in (
        ("missing", missing),
        ("modified", modified),
        ("unreadable", unreadable),
        ("missing mask", masks_missing),
        ("modified mask", masks_modified),
        ("unreadable mask", masks_unreadable),
    ):
        if paths:
            context.log(f"{len(paths)} {label} files", level="warning")

    return {
        "dataset_id": dataset.id,
        "checked": len(recorded),
        "verified": verified,
        "missing": missing[:MAX_PATHS_REPORTED],
        "modified": modified[:MAX_PATHS_REPORTED],
        "unreadable": unreadable[:MAX_PATHS_REPORTED],
        "missing_count": len(missing),
        "modified_count": len(modified),
        "unreadable_count": len(unreadable),
        "masks_checked": len(recorded_masks),
        "masks_verified": masks_verified,
        "masks_missing": masks_missing[:MAX_PATHS_REPORTED],
        "masks_missing_count": len(masks_missing),
        "masks_modified": masks_modified[:MAX_PATHS_REPORTED],
        "masks_modified_count": len(masks_modified),
        "masks_unreadable": masks_unreadable[:MAX_PATHS_REPORTED],
        "masks_unreadable_count": len(masks_unreadable),
        "masks_digest_checked": masks_digest_checked,
        "masks_unhashed": masks_unhashed,
    }
