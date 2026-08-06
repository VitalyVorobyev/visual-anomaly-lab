"""Building a seeded, sample-level split (ADR-0011).

Three properties make a split worth trusting, and all three are structural rather than
conventions someone has to remember:

  * **Assignment is per sample, never per image.** The repository offers no image-level
    assignment, so every view of a part necessarily shares a subset and a model cannot be
    tested on the dark-field view of a part it was trained on in bright-field.
  * **Training is normals only.** Anomaly detection learns what normal looks like; a
    defect in the training set teaches it that defects are normal.
  * **Stratification is by capture group**, so a batch effect — a lighting change, a
    fixturing change between acquisition sessions — cannot end up entirely on one side of
    the split and be mistaken for detection performance.

Unlabelled samples are assigned to a subset (`test` by default) rather than left out.
They are excluded from every metric, but they must be *scored* for the ranked
most-anomalous list to include them, which is what turns the model into a labelling aid
for unlabelled data (§8).
"""

from __future__ import annotations

import random
import sqlite3
from collections import defaultdict
from enum import StrEnum

from pydantic import BaseModel, Field, model_validator

from anomaly_lab.db.repositories import samples as samples_repo
from anomaly_lab.domain.entities import Label, Sample, Subset
from anomaly_lab.schemas import API_MODEL_CONFIG


class SplitStrategy(StrEnum):
    NORMAL_ONLY_TRAIN = "normal_only_train"


class SplitPlanError(Exception):
    """The requested split cannot be built from this dataset."""


class SplitParams(BaseModel):
    """How a split is drawn. Stored with the split so it can be rebuilt exactly."""

    model_config = API_MODEL_CONFIG

    strategy: SplitStrategy = SplitStrategy.NORMAL_ONLY_TRAIN
    train_normal_fraction: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Share of normal samples used for fitting. Defects never train.",
    )
    val_normal_fraction: float = Field(
        default=0.2,
        ge=0.0,
        le=1.0,
        description="Share of normal samples held out for threshold selection.",
    )
    val_defect_fraction: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Share of defect samples in validation; the rest are reported on.",
    )
    unlabeled_subset: Subset | None = Field(
        default=Subset.TEST,
        description=(
            "Where unlabelled samples go. They are excluded from every metric but must be "
            "scored to appear in the ranked lists. `null` leaves them out of the split."
        ),
    )

    @model_validator(mode="after")
    def _normals_must_add_up(self) -> SplitParams:
        if self.train_normal_fraction + self.val_normal_fraction > 1.0:
            msg = "train_normal_fraction + val_normal_fraction cannot exceed 1.0"
            raise ValueError(msg)
        return self


def plan_split(
    conn: sqlite3.Connection,
    dataset_id: int,
    *,
    seed: int,
    params: SplitParams,
) -> dict[int, Subset]:
    """Decide every sample's subset. Deterministic for a given dataset, seed and params.

    The shuffle is seeded once and consumed in a fixed order — groups sorted by key,
    samples sorted by id — so the same inputs always produce the same assignment. Without
    that ordering the seed would record nothing useful.
    """
    everything = samples_repo.list_samples(conn, dataset_id, limit=_ALL, offset=0)
    if not everything:
        msg = f"dataset {dataset_id} has no samples to split"
        raise SplitPlanError(msg)

    by_group: dict[str, dict[Label, list[Sample]]] = defaultdict(
        lambda: {label: [] for label in Label}
    )
    for sample in everything:
        by_group[sample.group_key][sample.label].append(sample)

    rng = random.Random(seed)
    assignments: dict[int, Subset] = {}

    for group_key in sorted(by_group):
        buckets = by_group[group_key]

        normals = sorted(buckets[Label.NORMAL], key=lambda sample: sample.id)
        rng.shuffle(normals)
        train_count = round(len(normals) * params.train_normal_fraction)
        val_count = round(len(normals) * params.val_normal_fraction)
        for index, sample in enumerate(normals):
            if index < train_count:
                assignments[sample.id] = Subset.TRAIN
            elif index < train_count + val_count:
                assignments[sample.id] = Subset.VAL
            else:
                assignments[sample.id] = Subset.TEST

        defects = sorted(buckets[Label.DEFECT], key=lambda sample: sample.id)
        rng.shuffle(defects)
        defect_val_count = round(len(defects) * params.val_defect_fraction)
        for index, sample in enumerate(defects):
            assignments[sample.id] = Subset.VAL if index < defect_val_count else Subset.TEST

        if params.unlabeled_subset is not None:
            for sample in buckets[Label.UNLABELED]:
                assignments[sample.id] = params.unlabeled_subset

    if not any(subset is Subset.TRAIN for subset in assignments.values()):
        msg = (
            "the split would have an empty training set: this dataset has no normal "
            "samples, or the training fraction rounds to zero in every capture group"
        )
        raise SplitPlanError(msg)

    return assignments


# `list_samples` pages by default; a split needs the whole dataset at once, and a dataset
# that does not fit in memory here would not fit in a training run either.
_ALL = 1_000_000
