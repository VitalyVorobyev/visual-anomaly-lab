"""One ground-truth snapshot shared by metrics, overlays and freshness checks."""

from __future__ import annotations

import hashlib
import sqlite3
from collections.abc import Sequence

from anomaly_lab.db.repositories import annotations as annotations_repo
from anomaly_lab.db.repositories import results as results_repo
from anomaly_lab.db.repositories.results import ScoredImage
from anomaly_lab.domain.entities import Subset


def resolved_masks(
    conn: sqlite3.Connection,
    images: Sequence[ScoredImage],
    *,
    verify_bytes: bool = False,
) -> dict[int, annotations_repo.GroundTruthMask]:
    return annotations_repo.resolve_ground_truth_masks(
        conn, [image.image_id for image in images], verify_bytes=verify_bytes
    )


def digest(
    images: Sequence[ScoredImage],
    masks: dict[int, annotations_repo.GroundTruthMask],
) -> str:
    """Digest labels and resolved mask identities in a deterministic order."""
    lines = ["ground-truth-v1"]
    labels = {image.sample_id: image.label.value for image in images}
    lines.extend(f"sample:{sample_id}:{labels[sample_id]}" for sample_id in sorted(labels))
    for image in sorted(images, key=lambda item: item.image_id):
        truth = masks.get(image.image_id)
        identity = truth.identity if truth is not None else "none"
        lines.append(f"image:{image.image_id}:{identity}")
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()


def current_digest(
    conn: sqlite3.Connection,
    experiment_id: int,
    subset: Subset,
) -> str:
    """Current metadata digest for a scored subset; never opens source files."""
    images = results_repo.list_scored_images(conn, experiment_id, subset=subset)
    return digest(images, resolved_masks(conn, images))
