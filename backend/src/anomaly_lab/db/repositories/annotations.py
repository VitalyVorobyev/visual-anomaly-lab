"""Versioned annotation repository."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from anomaly_lab.domain.annotations import (
    AnnotationDocument,
    AnnotationDraft,
    AnnotationLabel,
    AnnotationRevision,
    AnnotationSampleDraft,
)
from anomaly_lab.domain.entities import AnnotationState
from anomaly_lab.media.decode import sha256_of


class GroundTruthDriftError(RuntimeError):
    """Pinned truth bytes no longer match their recorded identity."""


# "This image has ground truth", spelled as SQL over an in-scope `image` row. It is exactly
# what `resolve_ground_truth_masks` resolves -- a completed revision, else an imported
# source mask -- and it is shared rather than restated so a queue filter can never disagree
# with what evaluation will actually read.
IMAGE_HAS_TRUTH = (
    "(EXISTS (SELECT 1 FROM annotation_revision"
    "         WHERE annotation_revision.image_id = image.id)"
    " OR EXISTS (SELECT 1 FROM mask"
    "            WHERE mask.image_id = image.id AND mask.kind = 'ground_truth'))"
)


@dataclass(frozen=True)
class GroundTruthMask:
    image_id: int
    record_id: int
    path: str
    sha256: str | None
    kind: Literal["revision", "source"]

    @property
    def identity(self) -> str:
        digest = self.sha256 or "unhashed"
        return f"{self.kind}:{self.record_id}:{digest}"


def _document(value: str) -> AnnotationDocument:
    return AnnotationDocument.model_validate_json(value)


def _draft(row: sqlite3.Row) -> AnnotationDraft:
    values = dict(row)
    values["document"] = _document(str(values["document"]))
    return AnnotationDraft.model_validate(values)


def _revision(row: sqlite3.Row) -> AnnotationRevision:
    values = dict(row)
    values["document"] = _document(str(values["document"]))
    return AnnotationRevision.model_validate(values)


def _sample_draft(row: sqlite3.Row) -> AnnotationSampleDraft:
    values = dict(row)
    values["document"] = _document(str(values["document"]))
    return AnnotationSampleDraft.model_validate(values)


#: The seeded colour of the `defect` class. Migration 017 says why it is not red.
DEFAULT_LABEL_COLOR = "#c026d3"


def ensure_default_label(conn: sqlite3.Connection, dataset_id: int) -> None:
    conn.execute(
        """
        INSERT INTO annotation_label (dataset_id, key, name, color, position)
             VALUES (?, 'defect', 'Defect', ?, 0)
        ON CONFLICT (dataset_id, key) DO NOTHING
        """,
        (dataset_id, DEFAULT_LABEL_COLOR),
    )


def list_labels(conn: sqlite3.Connection, dataset_id: int) -> list[AnnotationLabel]:
    rows = conn.execute(
        "SELECT * FROM annotation_label WHERE dataset_id = ? ORDER BY position, id",
        (dataset_id,),
    ).fetchall()
    return [AnnotationLabel.model_validate(dict(row)) for row in rows]


def create_label(
    conn: sqlite3.Connection,
    dataset_id: int,
    *,
    key: str,
    name: str,
    color: str,
    position: int,
) -> AnnotationLabel:
    cursor = conn.execute(
        """
        INSERT INTO annotation_label (dataset_id, key, name, color, position)
             VALUES (?, ?, ?, ?, ?)
        """,
        (dataset_id, key, name, color.lower(), position),
    )
    row = conn.execute(
        "SELECT * FROM annotation_label WHERE id = ?", (int(cursor.lastrowid or 0),)
    ).fetchone()
    if row is None:  # pragma: no cover
        raise RuntimeError("annotation label vanished after insertion")
    return AnnotationLabel.model_validate(dict(row))


def update_label(
    conn: sqlite3.Connection,
    dataset_id: int,
    key: str,
    *,
    name: str,
    color: str,
    position: int,
) -> AnnotationLabel | None:
    cursor = conn.execute(
        """
        UPDATE annotation_label
           SET name = ?, color = ?, position = ?
         WHERE dataset_id = ? AND key = ?
        """,
        (name, color.lower(), position, dataset_id, key),
    )
    if cursor.rowcount != 1:
        return None
    row = conn.execute(
        "SELECT * FROM annotation_label WHERE dataset_id = ? AND key = ?",
        (dataset_id, key),
    ).fetchone()
    return AnnotationLabel.model_validate(dict(row)) if row is not None else None


def get_draft(conn: sqlite3.Connection, image_id: int) -> AnnotationDraft | None:
    row = conn.execute("SELECT * FROM annotation_draft WHERE image_id = ?", (image_id,)).fetchone()
    return _draft(row) if row is not None else None


def latest_revision(conn: sqlite3.Connection, image_id: int) -> AnnotationRevision | None:
    row = conn.execute(
        "SELECT * FROM annotation_revision WHERE image_id = ? ORDER BY revision_no DESC LIMIT 1",
        (image_id,),
    ).fetchone()
    return _revision(row) if row is not None else None


def create_draft(
    conn: sqlite3.Connection,
    image_id: int,
    document: AnnotationDocument,
    *,
    base_revision_id: int | None,
    source_mask_id: int | None,
    source_mask_path: str | None,
    source_mask_sha256: str | None,
) -> AnnotationDraft:
    conn.execute(
        """
        INSERT INTO annotation_draft
               (image_id, base_revision_id, document, source_mask_id,
                source_mask_path, source_mask_sha256)
             VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            image_id,
            base_revision_id,
            document.canonical_json(),
            source_mask_id,
            source_mask_path,
            source_mask_sha256,
        ),
    )
    created = get_draft(conn, image_id)
    if created is None:  # pragma: no cover
        raise RuntimeError("annotation draft vanished after insertion")
    return created


def update_draft(
    conn: sqlite3.Connection,
    image_id: int,
    expected_version: int,
    document: AnnotationDocument,
) -> AnnotationDraft | None:
    cursor = conn.execute(
        """
        UPDATE annotation_draft
           SET document = ?, version = version + 1,
               updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
         WHERE image_id = ? AND version = ?
        """,
        (document.canonical_json(), image_id, expected_version),
    )
    if cursor.rowcount != 1:
        return None
    return get_draft(conn, image_id)


def delete_draft(conn: sqlite3.Connection, image_id: int, expected_version: int | None) -> bool:
    """Discard a draft, optionally only if the caller still owns the version it read.

    `expected_version=None` is the `If-Match: *` force path: the caller has been shown that
    the draft moved under them and has chosen to throw it away anyway.
    """
    cursor = conn.execute(
        "DELETE FROM annotation_draft WHERE image_id = ? AND (? IS NULL OR version = ?)",
        (image_id, expected_version, expected_version),
    )
    return cursor.rowcount == 1


def insert_revision(
    conn: sqlite3.Connection,
    draft: AnnotationDraft,
    *,
    document_sha256: str,
    mask_path: str,
    mask_sha256: str,
) -> AnnotationRevision:
    next_no = int(
        conn.execute(
            "SELECT COALESCE(MAX(revision_no), 0) + 1 FROM annotation_revision WHERE image_id = ?",
            (draft.image_id,),
        ).fetchone()[0]
    )
    cursor = conn.execute(
        """
        INSERT INTO annotation_revision
               (image_id, revision_no, document, document_sha256, mask_path, mask_sha256,
                source_mask_id, source_mask_path, source_mask_sha256)
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            draft.image_id,
            next_no,
            draft.document.canonical_json(),
            document_sha256,
            mask_path,
            mask_sha256,
            draft.source_mask_id,
            draft.source_mask_path,
            draft.source_mask_sha256,
        ),
    )
    conn.execute(
        "DELETE FROM annotation_draft WHERE image_id = ? AND version = ?",
        (draft.image_id, draft.version),
    )
    row = conn.execute(
        "SELECT * FROM annotation_revision WHERE id = ?", (int(cursor.lastrowid or 0),)
    ).fetchone()
    if row is None:  # pragma: no cover
        raise RuntimeError("annotation revision vanished after insertion")
    return _revision(row)


def list_revisions(conn: sqlite3.Connection, image_id: int) -> list[AnnotationRevision]:
    rows = conn.execute(
        "SELECT * FROM annotation_revision WHERE image_id = ? ORDER BY revision_no DESC",
        (image_id,),
    ).fetchall()
    return [_revision(row) for row in rows]


def get_revision(conn: sqlite3.Connection, revision_id: int) -> AnnotationRevision | None:
    row = conn.execute("SELECT * FROM annotation_revision WHERE id = ?", (revision_id,)).fetchone()
    return _revision(row) if row is not None else None


def get_sample_draft(conn: sqlite3.Connection, sample_id: int) -> AnnotationSampleDraft | None:
    row = conn.execute(
        "SELECT * FROM annotation_sample_draft WHERE sample_id = ?", (sample_id,)
    ).fetchone()
    return _sample_draft(row) if row is not None else None


def create_sample_draft(
    conn: sqlite3.Connection, sample_id: int, document: AnnotationDocument
) -> AnnotationSampleDraft:
    conn.execute(
        "INSERT INTO annotation_sample_draft (sample_id, document) VALUES (?, ?)",
        (sample_id, document.canonical_json()),
    )
    created = get_sample_draft(conn, sample_id)
    if created is None:  # pragma: no cover
        raise RuntimeError("annotation sample draft vanished after insertion")
    return created


def update_sample_draft(
    conn: sqlite3.Connection,
    sample_id: int,
    expected_version: int,
    document: AnnotationDocument,
) -> AnnotationSampleDraft | None:
    cursor = conn.execute(
        """
        UPDATE annotation_sample_draft
           SET document = ?, version = version + 1,
               updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
         WHERE sample_id = ? AND version = ?
        """,
        (document.canonical_json(), sample_id, expected_version),
    )
    if cursor.rowcount != 1:
        return None
    return get_sample_draft(conn, sample_id)


def delete_sample_draft(
    conn: sqlite3.Connection, sample_id: int, expected_version: int | None = None
) -> bool:
    """See `delete_draft`. `None` means "whatever version is there", which is what the
    completion path wants — it has already consumed the draft it read."""
    cursor = conn.execute(
        "DELETE FROM annotation_sample_draft WHERE sample_id = ? AND (? IS NULL OR version = ?)",
        (sample_id, expected_version, expected_version),
    )
    return cursor.rowcount == 1


def latest_revision_for_sample(
    conn: sqlite3.Connection, sample_id: int
) -> AnnotationRevision | None:
    """The newest completed revision across a sample's images, for seeding a shared draft.

    Under sample scope every image of a sample was written the same document, so any of
    them answers the question. `completed_at` then `id` orders a fan-out written inside one
    transaction deterministically, and an image-scoped history left behind by a scope flip
    resolves to whichever image was edited last -- which is the only defensible reading of
    "what did this part look like".
    """
    row = conn.execute(
        """
        SELECT annotation_revision.*
          FROM annotation_revision
          JOIN image ON image.id = annotation_revision.image_id
         WHERE image.sample_id = ?
         ORDER BY annotation_revision.completed_at DESC, annotation_revision.id DESC
         LIMIT 1
        """,
        (sample_id,),
    ).fetchone()
    return _revision(row) if row is not None else None


def insert_shared_revision(
    conn: sqlite3.Connection,
    image_id: int,
    document: AnnotationDocument,
    *,
    document_sha256: str,
    mask_path: str,
    mask_sha256: str,
) -> AnnotationRevision:
    """Append one image's revision from a sample-scoped completion.

    Deliberately separate from `insert_revision`: there is no image draft to consume and
    no source-mask provenance to carry forward. `revision_no` is still per image, so an
    image that has its own earlier history keeps counting from where it was.
    """
    next_no = int(
        conn.execute(
            "SELECT COALESCE(MAX(revision_no), 0) + 1 FROM annotation_revision WHERE image_id = ?",
            (image_id,),
        ).fetchone()[0]
    )
    cursor = conn.execute(
        """
        INSERT INTO annotation_revision
               (image_id, revision_no, document, document_sha256, mask_path, mask_sha256)
             VALUES (?, ?, ?, ?, ?, ?)
        """,
        (image_id, next_no, document.canonical_json(), document_sha256, mask_path, mask_sha256),
    )
    row = conn.execute(
        "SELECT * FROM annotation_revision WHERE id = ?", (int(cursor.lastrowid or 0),)
    ).fetchone()
    if row is None:  # pragma: no cover
        raise RuntimeError("annotation revision vanished after insertion")
    return _revision(row)


def next_revision_no(conn: sqlite3.Connection, image_id: int) -> int:
    """What `insert_revision`/`insert_shared_revision` will number the next revision.

    Read separately because the materialised PNG's filename embeds it, and the file has to
    be written before the row that points at it.
    """
    return int(
        conn.execute(
            "SELECT COALESCE(MAX(revision_no), 0) + 1 FROM annotation_revision WHERE image_id = ?",
            (image_id,),
        ).fetchone()[0]
    )


def count_open_image_drafts(conn: sqlite3.Connection, dataset_id: int) -> int:
    """How many images hold annotation work that has not been completed.

    **The absence of a predicate here is load-bearing.** A draft row exists only because
    somebody saved one, so counting rows is counting work. It was not always so: creation used
    to happen when the editor opened an image, which is what made every image ever looked at a
    permanent blocker on `annotation_scope` and what migration 016 cleaned up. Do not be
    tempted to filter on `version` — under the current write path `version = 1` means "saved
    once", not "untouched".
    """
    return int(
        conn.execute(
            """
            SELECT COUNT(*)
              FROM annotation_draft
              JOIN image  ON image.id = annotation_draft.image_id
              JOIN sample ON sample.id = image.sample_id
             WHERE sample.dataset_id = ?
            """,
            (dataset_id,),
        ).fetchone()[0]
    )


@dataclass(frozen=True)
class OpenDraft:
    """One unit whose draft holds saved-but-uncompleted work.

    A count on its own is a dead end. "2 images hold annotation work that has not been
    completed" is true, and leaves an operator with a dataset of several hundred images and no
    way to find the two. A scope change needs an empty desk (ADR-0036), so the
    desk has to say what is on it.

    `image_id` is always present, including for a sample draft, where it is the sample's first
    image: the editor is addressed by the pair either way, so a caller can always build a link.
    """

    sample_id: int
    sample_key: str
    image_id: int
    channel: str | None


#: How many open drafts are named before the rest are only counted. A dataset with hundreds of
#: them is not going to be cleared from a list, and the count still tells the whole truth.
OPEN_DRAFT_SAMPLE = 24


def list_open_image_drafts(
    conn: sqlite3.Connection, dataset_id: int, limit: int = OPEN_DRAFT_SAMPLE
) -> list[OpenDraft]:
    """Which images hold work, ordered as the queue orders them."""
    rows = conn.execute(
        """
        SELECT sample.id           AS sample_id,
               sample.external_id  AS sample_key,
               image.id            AS image_id,
               channel.name        AS channel
          FROM annotation_draft
          JOIN image   ON image.id = annotation_draft.image_id
          JOIN sample  ON sample.id = image.sample_id
     LEFT JOIN channel ON channel.id = image.channel_id
         WHERE sample.dataset_id = ?
         ORDER BY sample.id, channel.position, image.id
         LIMIT ?
        """,
        (dataset_id, limit),
    ).fetchall()
    return [OpenDraft(**dict(row)) for row in rows]


def list_open_sample_drafts(
    conn: sqlite3.Connection, dataset_id: int, limit: int = OPEN_DRAFT_SAMPLE
) -> list[OpenDraft]:
    """Which samples hold work, each addressed through its first image."""
    rows = conn.execute(
        """
        SELECT sample.id          AS sample_id,
               sample.external_id AS sample_key,
               (SELECT image.id FROM image WHERE image.sample_id = sample.id
                 ORDER BY image.id LIMIT 1) AS image_id,
               NULL               AS channel
          FROM annotation_sample_draft
          JOIN sample ON sample.id = annotation_sample_draft.sample_id
         WHERE sample.dataset_id = ?
         ORDER BY sample.id
         LIMIT ?
        """,
        (dataset_id, limit),
    ).fetchall()
    # A sample with no images cannot be opened, and cannot have acquired a draft either;
    # skipping it keeps `image_id` a promise rather than a hope.
    return [OpenDraft(**dict(row)) for row in rows if row["image_id"] is not None]


def count_open_sample_drafts(conn: sqlite3.Connection, dataset_id: int) -> int:
    """See `count_open_image_drafts` for why this counts rows and nothing else."""
    return int(
        conn.execute(
            """
            SELECT COUNT(*)
              FROM annotation_sample_draft
              JOIN sample ON sample.id = annotation_sample_draft.sample_id
             WHERE sample.dataset_id = ?
            """,
            (dataset_id,),
        ).fetchone()[0]
    )


def annotation_state_for_samples(
    conn: sqlite3.Connection, sample_ids: list[int]
) -> dict[int, AnnotationState]:
    """Whether each sample's images all have truth, some do, or none do.

    Three states rather than a boolean, because a multi-channel sample can genuinely be
    half done: image scope lets a part's bright view be annotated while its dark view is
    not, and calling that "annotated" would lose the only detail the queue exists to show.
    """
    states: dict[int, AnnotationState] = dict.fromkeys(sample_ids, AnnotationState.NONE)
    for start in range(0, len(sample_ids), 900):
        chunk = sample_ids[start : start + 900]
        if not chunk:
            continue
        placeholders = ",".join("?" * len(chunk))
        rows = conn.execute(
            f"""
            SELECT image.sample_id AS sample_id,
                   COUNT(*)        AS total,
                   SUM(CASE WHEN {IMAGE_HAS_TRUTH} THEN 1 ELSE 0 END) AS covered
              FROM image
             WHERE image.sample_id IN ({placeholders})
             GROUP BY image.sample_id
            """,
            chunk,
        ).fetchall()
        for row in rows:
            covered = int(row["covered"])
            total = int(row["total"])
            states[int(row["sample_id"])] = (
                AnnotationState.COMPLETE
                if covered == total
                else AnnotationState.PARTIAL
                if covered
                else AnnotationState.NONE
            )
    return states


def count_multi_image_samples(conn: sqlite3.Connection, dataset_id: int) -> int:
    """Samples carrying more than one image — how much sample scope would actually save.

    Not a threshold anything is gated on: a dataset of single-view samples may still be
    put in sample scope and behaves identically to image scope. It is shown so the choice
    is made against the data rather than against a guess.
    """
    return int(
        conn.execute(
            """
            SELECT COUNT(*) FROM (
                SELECT sample.id
                  FROM sample
                  JOIN image ON image.sample_id = sample.id
                 WHERE sample.dataset_id = ?
                 GROUP BY sample.id
                HAVING COUNT(image.id) > 1
            )
            """,
            (dataset_id,),
        ).fetchone()[0]
    )


def samples_with_mixed_dimensions(conn: sqlite3.Connection, dataset_id: int) -> list[str]:
    """External ids of samples whose images are not all the same size, newest first.

    A shared document pins one `image_width`/`image_height`, so a sample whose channels
    were captured at different resolutions has no single frame to annotate in. Reported
    rather than silently skipped: the operator has to know which parts are affected.
    """
    rows = conn.execute(
        """
        SELECT sample.external_id AS external_id
          FROM sample
          JOIN image ON image.sample_id = sample.id
         WHERE sample.dataset_id = ?
         GROUP BY sample.id
        HAVING COUNT(DISTINCT image.width || 'x' || image.height) > 1
         ORDER BY sample.group_key, sample.external_id
        """,
        (dataset_id,),
    ).fetchall()
    return [str(row["external_id"]) for row in rows]


def resolve_ground_truth_masks(
    conn: sqlite3.Connection,
    image_ids: list[int],
    *,
    verify_bytes: bool = False,
) -> dict[int, GroundTruthMask]:
    """Resolve newest completed revision, then imported source mask, per image.

    `verify_bytes` is for consumers that will use the pixels. Metadata-only reads use
    recorded digests so experiment detail stays O(rows), not O(source bytes).
    """
    found: dict[int, GroundTruthMask] = {}
    for start in range(0, len(image_ids), 900):
        chunk = image_ids[start : start + 900]
        if not chunk:
            continue
        placeholders = ",".join("?" * len(chunk))
        revisions = conn.execute(
            f"""
            SELECT id, image_id, mask_path, mask_sha256
              FROM annotation_revision
             WHERE image_id IN ({placeholders})
             ORDER BY image_id, revision_no DESC
            """,
            chunk,
        ).fetchall()
        for row in revisions:
            image_id = int(row["image_id"])
            found.setdefault(
                image_id,
                GroundTruthMask(
                    image_id=image_id,
                    record_id=int(row["id"]),
                    path=str(row["mask_path"]),
                    sha256=str(row["mask_sha256"]),
                    kind="revision",
                ),
            )

        unresolved = [image_id for image_id in chunk if image_id not in found]
        if unresolved:
            source_placeholders = ",".join("?" * len(unresolved))
            sources = conn.execute(
                f"""
                SELECT id, image_id, path, sha256
                  FROM mask
                 WHERE kind = 'ground_truth' AND image_id IN ({source_placeholders})
                 ORDER BY image_id, id
                """,
                unresolved,
            ).fetchall()
            for row in sources:
                image_id = int(row["image_id"])
                found.setdefault(
                    image_id,
                    GroundTruthMask(
                        image_id=image_id,
                        record_id=int(row["id"]),
                        path=str(row["path"]),
                        sha256=str(row["sha256"]) if row["sha256"] is not None else None,
                        kind="source",
                    ),
                )

    if not verify_bytes:
        return found

    verified: dict[int, GroundTruthMask] = {}
    for image_id, truth in found.items():
        path = Path(truth.path)
        if not path.is_file():
            if truth.kind == "revision":
                raise GroundTruthDriftError(
                    f"completed annotation ground truth for image {image_id} is unavailable"
                )
            verified[image_id] = truth
            continue
        try:
            actual = sha256_of(path)
        except OSError as exc:
            if truth.kind == "revision":
                raise GroundTruthDriftError(
                    f"completed annotation ground truth for image {image_id} cannot be read"
                ) from exc
            verified[image_id] = truth
            continue
        if truth.sha256 is not None and actual != truth.sha256:
            raise GroundTruthDriftError(
                f"{truth.kind} ground truth for image {image_id} changed after it was pinned"
            )
        if truth.kind == "source" and truth.sha256 is None:
            conn.execute("UPDATE mask SET sha256 = ? WHERE id = ?", (actual, truth.record_id))
        verified[image_id] = GroundTruthMask(
            image_id=truth.image_id,
            record_id=truth.record_id,
            path=truth.path,
            sha256=actual,
            kind=truth.kind,
        )
    return verified


def delete_for_dataset(conn: sqlite3.Connection, dataset_id: int) -> None:
    image_select = (
        "SELECT image.id FROM image JOIN sample ON sample.id = image.sample_id "
        "WHERE sample.dataset_id = ?"
    )
    conn.execute(f"DELETE FROM annotation_draft WHERE image_id IN ({image_select})", (dataset_id,))
    conn.execute(
        f"DELETE FROM annotation_revision WHERE image_id IN ({image_select})", (dataset_id,)
    )
    conn.execute(
        "DELETE FROM annotation_sample_draft "
        "WHERE sample_id IN (SELECT id FROM sample WHERE dataset_id = ?)",
        (dataset_id,),
    )
    conn.execute("DELETE FROM annotation_label WHERE dataset_id = ?", (dataset_id,))
