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

The `imported` strategy is the deliberate exception to all of the above. When a benchmark
publishes both its numbers and the partition behind them, drawing our own would produce a
figure that resembles theirs without being comparable to it, so `plan_imported_split`
reproduces the source's partition verbatim — including a partition with no `val` subset at
all, which the official one-class protocols generally are. Everything downstream therefore
has to treat an empty `val` as ordinary rather than as a broken split.

The `manual` and `few_shot` strategies partition for a few-shot task (ADR-0040), where
`train` is a handful of references and `test` is every other sample. `manual` takes the
references as listed; `few_shot` draws `shots` of the samples that show one class, under the
seed, so the same request with three seeds is three reference draws. Neither has a `val`.
"""

from __future__ import annotations

import random
import sqlite3
from collections import defaultdict
from enum import StrEnum

from pydantic import BaseModel, Field, model_validator

from anomaly_lab.datasets.manifest import Manifest
from anomaly_lab.db.repositories import annotations as annotations_repo
from anomaly_lab.db.repositories import samples as samples_repo
from anomaly_lab.domain.entities import ClassPresence, Label, Sample, Subset
from anomaly_lab.schemas import API_MODEL_CONFIG


class SplitStrategy(StrEnum):
    NORMAL_ONLY_TRAIN = "normal_only_train"
    IMPORTED = "imported"
    """Take the partition the source dataset published, rather than drawing one."""
    MANUAL = "manual"
    """Few-shot references chosen by hand: `sample_ids` train, everything else tests."""
    FEW_SHOT = "few_shot"
    """Few-shot references drawn under the seed from the samples that show `label_key`."""


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

    holdout_from_train: float = Field(
        default=0.0,
        ge=0.0,
        lt=1.0,
        description=(
            "For `imported` only: move this share of the published *training* normals to "
            "`val`. The official one-class protocols publish no validation subset, but "
            "methods that calibrate on held-out normals need one. Taking it out of train "
            "leaves the published test set untouched, so the reported number stays "
            "comparable; 0 reproduces the source's partition exactly."
        ),
    )

    manifest_id: str | None = Field(
        default=None,
        description=(
            "Which import proposal an `imported` split was materialized from. Recorded so "
            "the partition stays traceable to the file that asserted it; ignored by every "
            "other strategy."
        ),
    )

    sample_ids: list[int] = Field(
        default_factory=list,
        description="For `manual` only: the reference samples, which form `train`.",
    )
    label_key: str | None = Field(
        default=None,
        description="For `few_shot` only: the annotation class the references must show.",
    )
    shots: int | None = Field(
        default=None,
        ge=1,
        description="For `few_shot` only: how many references to draw.",
    )

    @model_validator(mode="after")
    def _normals_must_add_up(self) -> SplitParams:
        # The fractions are meaningless for `imported`, `manual` and `few_shot`, where
        # something other than a ratio decides. They keep their defaults rather than being
        # rejected, so that the stored params do not read as if ratios were applied.
        if self.strategy is SplitStrategy.MANUAL:
            if not self.sample_ids:
                raise ValueError("a manual split needs at least one reference in sample_ids")
            if len(set(self.sample_ids)) != len(self.sample_ids):
                raise ValueError("sample_ids lists a sample twice")
            return self
        if self.strategy is SplitStrategy.FEW_SHOT:
            if not self.label_key:
                raise ValueError("a few_shot split needs the label_key its references show")
            if self.shots is None:
                raise ValueError("a few_shot split needs the number of shots to draw")
            return self
        if self.strategy is SplitStrategy.IMPORTED:
            return self
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


def plan_imported_split(
    conn: sqlite3.Connection,
    dataset_id: int,
    manifest: Manifest,
    *,
    seed: int = 0,
    holdout_from_train: float = 0.0,
) -> dict[int, Subset]:
    """Materialize the partition a source dataset published, as recorded in its manifest.

    No seed, no fractions, no stratification: the point is to reproduce someone else's
    partition exactly, so that a number computed here is comparable to the number they
    published. Drawing our own would produce a figure that looks like theirs and is not.

    Samples the manifest does not place are **left out of the split** rather than swept
    into `test`. A benchmark's protocol says what belongs in its test set; adding samples
    it never scored would change the denominator of every metric.

    `holdout_from_train` is the one permitted departure, and it is careful about which
    number it is allowed to move. The official one-class protocols publish no validation
    subset, but EfficientAD — and any method that calibrates on held-out normals — has to
    fit those statistics on *something*, and falling back to the training normals means
    calibrating on data the model has already memorized. Carving the holdout out of the
    published **train** subset leaves the published **test** subset byte-identical, so the
    reported figure remains the one the protocol defines. Zero reproduces the source
    exactly, and is the default.
    """
    if not manifest.has_imported_split():
        msg = (
            "this dataset was imported without split information, so there is no "
            "published partition to materialize. Use a seeded strategy, or re-import "
            "with an adapter that reads the source's split column."
        )
        raise SplitPlanError(msg)

    published = {
        (sample.group_key, sample.external_id): sample.subset
        for sample in manifest.samples
        if sample.subset is not None
    }

    assignments: dict[int, Subset] = {}
    for sample in samples_repo.list_samples(conn, dataset_id, limit=_ALL, offset=0):
        subset = published.get((sample.group_key, sample.external_id))
        if subset is not None:
            assignments[sample.id] = subset

    if not assignments:
        msg = (
            f"the manifest places {len(published)} samples, none of which are in dataset "
            f"{dataset_id}. It is probably a manifest for a different dataset."
        )
        raise SplitPlanError(msg)

    if not any(subset is Subset.TRAIN for subset in assignments.values()):
        msg = "the published partition has no training samples"
        raise SplitPlanError(msg)

    if holdout_from_train > 0.0:
        _hold_out_from_train(conn, dataset_id, assignments, seed, holdout_from_train)

    return assignments


def plan_manual_split(
    conn: sqlite3.Connection, dataset_id: int, sample_ids: list[int]
) -> dict[int, Subset]:
    """The listed samples are the references; every other sample is a query."""
    everything = samples_repo.list_samples(conn, dataset_id, limit=_ALL, offset=0)
    known = {sample.id for sample in everything}
    foreign = sorted(set(sample_ids) - known)
    if foreign:
        msg = f"samples {foreign} are not in dataset {dataset_id}"
        raise SplitPlanError(msg)
    references = set(sample_ids)
    return {
        sample.id: Subset.TRAIN if sample.id in references else Subset.TEST for sample in everything
    }


def plan_few_shot_split(
    conn: sqlite3.Connection,
    dataset_id: int,
    *,
    seed: int,
    label_key: str,
    shots: int,
) -> dict[int, Subset]:
    """Draw `shots` references from the samples that show the class; the rest are queries.

    Seeded and consumed in sorted id order, so the same request reproduces the same draw,
    and a different seed is a different draw of the same size — which is what makes
    sensitivity to the choice of references measurable.
    """
    presence = annotations_repo.class_presence(conn, dataset_id, label_key)
    if not presence:
        msg = f"dataset {dataset_id} has no samples to split"
        raise SplitPlanError(msg)
    candidates = sorted(
        sample_id for sample_id, found in presence.items() if found is ClassPresence.PRESENT
    )
    if len(candidates) < shots:
        msg = (
            f"{shots} references of {label_key!r} were asked for, but only "
            f"{len(candidates)} samples show it in a completed annotation"
        )
        raise SplitPlanError(msg)

    rng = random.Random(seed)
    rng.shuffle(candidates)
    references = set(candidates[:shots])
    return {
        sample_id: Subset.TRAIN if sample_id in references else Subset.TEST
        for sample_id in presence
    }


def _hold_out_from_train(
    conn: sqlite3.Connection,
    dataset_id: int,
    assignments: dict[int, Subset],
    seed: int,
    fraction: float,
) -> None:
    """Move a seeded share of the *normal* training samples into `val`, in place.

    Normals only, because validation for a one-class method means "more of what normal
    looks like"; a defect moved here would be a defect the model is calibrated against
    but never scored on. Seeded and consumed in sorted id order, so the same request
    reproduces the same holdout.
    """
    labels = {
        sample.id: sample.label
        for sample in samples_repo.list_samples(conn, dataset_id, limit=_ALL, offset=0)
    }
    candidates = sorted(
        sample_id
        for sample_id, subset in assignments.items()
        if subset is Subset.TRAIN and labels.get(sample_id) is Label.NORMAL
    )

    count = round(len(candidates) * fraction)
    if count == 0 or count == len(candidates):
        msg = (
            f"holdout_from_train={fraction} would move {count} of {len(candidates)} "
            "training normals, leaving either no validation set or no training set"
        )
        raise SplitPlanError(msg)

    rng = random.Random(seed)
    rng.shuffle(candidates)
    for sample_id in candidates[:count]:
        assignments[sample_id] = Subset.VAL


# `list_samples` pages by default; a split needs the whole dataset at once, and a dataset
# that does not fit in memory here would not fit in a training run either.
_ALL = 1_000_000
