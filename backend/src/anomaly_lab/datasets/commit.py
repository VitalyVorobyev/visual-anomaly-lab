"""Stage two of import: turning an accepted manifest into rows (ADR-0006, ADR-0013).

Commit is **synchronous**, not a job. It is a few hundred INSERTs in one transaction —
milliseconds — because all the expensive work happened during the scan. Making it a job
would add a progress bar to something that finishes before the first frame renders.

It is also **idempotent**. Re-scanning and re-committing the same tree updates the rows
it produced last time rather than creating a parallel set:

  * the dataset is resolved by `root_path`, not by name;
  * a sample is identified by `(dataset_id, group_key, external_id)`;
  * an image is identified by `(sample_id, path)`;
  * a label the operator set by hand is never overwritten by an imported guess;
  * a recorded file that the manifest no longer mentions is **reported, never deleted** —
    a vanished file may be an unmounted disk, and quietly dropping catalog rows would
    turn a mount problem into data loss.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field

from anomaly_lab.config import Settings
from anomaly_lab.datasets.manifest import Manifest
from anomaly_lab.datasets.storage import committed_manifest_id, save_manifest
from anomaly_lab.db.repositories import datasets as datasets_repo
from anomaly_lab.db.repositories import images as images_repo
from anomaly_lab.db.repositories import masks as masks_repo
from anomaly_lab.db.repositories import samples as samples_repo
from anomaly_lab.domain.entities import Dataset


class CommitError(Exception):
    """The manifest cannot be committed as asked."""


@dataclass(frozen=True)
class CommitResult:
    dataset_id: int
    dataset_created: bool
    manifest_id: str
    samples_created: int = 0
    samples_updated: int = 0
    images_created: int = 0
    images_updated: int = 0
    channels: int = 0
    masks: int = 0
    """Images that now carry pixel-level ground truth."""
    missing_paths: list[str] = field(default_factory=list)
    """Recorded images the manifest no longer mentions. Reported, not removed."""


# A commit that reports nine hundred missing files is reporting a moved directory, not
# nine hundred separate problems.
MAX_MISSING_REPORTED = 50


def commit_manifest(
    conn: sqlite3.Connection,
    settings: Settings,
    manifest: Manifest,
    *,
    dataset_id: int | None = None,
) -> CommitResult:
    """Write an accepted manifest into the catalog, and store it verbatim."""
    dataset, created = _resolve_dataset(conn, manifest, dataset_id)

    conn.execute("BEGIN")
    try:
        channel_ids = {
            name: datasets_repo.upsert_channel(conn, dataset.id, name=name, position=index).id
            for index, name in enumerate(manifest.channels)
        }

        samples_created = samples_updated = images_created = images_updated = masks = 0
        seen_paths: set[str] = set()

        for proposed in manifest.samples:
            sample, sample_is_new = samples_repo.upsert_sample(
                conn,
                dataset.id,
                group_key=proposed.group_key,
                external_id=proposed.external_id,
                label=proposed.label,
                notes=proposed.notes,
            )
            samples_created += int(sample_is_new)
            samples_updated += int(not sample_is_new)

            for image in proposed.images:
                row, image_is_new = images_repo.upsert_image(
                    conn,
                    sample.id,
                    # An unmapped channel name means no channel, not a new one: the
                    # dictionary is the manifest's `channels` list, which the operator
                    # has just reviewed.
                    channel_id=channel_ids.get(image.channel or ""),
                    path=image.path,
                    width=image.width,
                    height=image.height,
                    bit_depth=image.bit_depth,
                    file_size=image.file_size,
                    sha256=image.sha256,
                )
                images_created += int(image_is_new)
                images_updated += int(not image_is_new)
                seen_paths.add(image.path)

                # A mask the manifest no longer mentions is left alone, for the same
                # reason a missing image is reported rather than deleted: a re-scan run
                # with the mask options forgotten must not silently discard annotations.
                if image.mask_path is not None:
                    masks_repo.upsert_mask(conn, row.id, path=image.mask_path)
                    masks += 1

        missing = [
            image.path
            for image in images_repo.list_images_for_dataset(conn, dataset.id)
            if image.path not in seen_paths
        ]
    except Exception:
        conn.execute("ROLLBACK")
        raise
    conn.execute("COMMIT")

    manifest_id = committed_manifest_id(dataset.id)
    path = save_manifest(settings, manifest, manifest_id=manifest_id)
    datasets_repo.record_manifest(
        conn, dataset.id, adapter=manifest.adapter, manifest_path=str(path)
    )

    return CommitResult(
        dataset_id=dataset.id,
        dataset_created=created,
        manifest_id=manifest_id,
        samples_created=samples_created,
        samples_updated=samples_updated,
        images_created=images_created,
        images_updated=images_updated,
        channels=len(channel_ids),
        masks=masks,
        missing_paths=sorted(missing)[:MAX_MISSING_REPORTED],
    )


def _resolve_dataset(
    conn: sqlite3.Connection,
    manifest: Manifest,
    dataset_id: int | None,
) -> tuple[Dataset, bool]:
    """Explicit id, else the dataset already rooted here, else a new one."""
    if dataset_id is not None:
        existing = datasets_repo.get_dataset(conn, dataset_id)
        if existing is None:
            msg = f"no dataset with id {dataset_id}"
            raise CommitError(msg)
        return existing, False

    by_root = datasets_repo.find_dataset_by_root(conn, manifest.root_path)
    if by_root is not None:
        return by_root, False

    try:
        created = datasets_repo.create_dataset(
            conn,
            name=manifest.dataset_name,
            root_path=manifest.root_path,
            adapter=manifest.adapter,
        )
    except sqlite3.IntegrityError as exc:
        msg = (
            f"a different dataset is already named {manifest.dataset_name!r}; "
            "choose another name or commit into the existing dataset by id"
        )
        raise CommitError(msg) from exc
    return created, True
