"""The verify job.

Referencing images in place rather than copying them (ADR-0001) buys privacy and disk,
and costs the guarantee that what the catalog points at is still what it pointed at. A
renamed folder breaks every path; a re-exported file keeps its path and changes its
bytes. Verify is what makes that discoverable rather than something you learn from a
model that suddenly scores differently.

It **detects drift and never repairs it**. Deciding that a file which changed should be
re-hashed into the catalog, or that a missing one should be dropped, is the operator's
call, and the honest failure of an unmounted disk is a report rather than a deletion.

Masks are walked too, but only for presence: schema v1 records no hash for them and the
schema is frozen (ADR-0004). They are counted apart from the images so that a clean report
does not quietly claim a check it did not perform.
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

    # Masks are checked for presence only. Schema v1 has no `mask.sha256` and the schema
    # is frozen (ADR-0004), so a mask that was re-exported in place is drift this job
    # cannot see. Counted separately so the report never implies otherwise.
    masks_missing: list[str] = []
    masks_verified = 0
    for offset, mask in enumerate(recorded_masks):
        context.raise_if_cancelled()
        if Path(mask.path).is_file():
            masks_verified += 1
        else:
            masks_missing.append(mask.path)

        if total:
            position = len(recorded) + offset + 1
            context.progress(position / total, f"{position} of {total}")

    for label, paths in (
        ("missing", missing),
        ("modified", modified),
        ("unreadable", unreadable),
        ("missing mask", masks_missing),
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
    }
