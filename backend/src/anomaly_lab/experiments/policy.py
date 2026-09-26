"""Which images a task fits on — the training-set half of a task (ADR-0039).

`anomaly` learns what normal looks like, so it fits on the train subset's normals and
calibrates on the val subset's. A targeted task (ADR-0040) fits on its references — every
image of the train subset whose ground truth answers for the target class, present or
absent — and has no val subset. Supervised segmentation and detection fit on every image of
the train subset whose truth answers for all of their pinned classes, and have no val subset
either. The train handler asks here instead of deciding, so the answer has one home per task.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field

from anomaly_lab.annotations.class_truth import (
    BoxTruth,
    ClassTruth,
    LabelTruth,
    resolve_box_truth,
    resolve_class_truth,
    resolve_label_truth,
)
from anomaly_lab.db.repositories import images as images_repo
from anomaly_lab.db.repositories.images import SplitImage
from anomaly_lab.domain.entities import Experiment, Label, Subset, Task


class NoTrainingPolicyError(Exception):
    """The task has no rule for what it fits on."""


@dataclass(frozen=True)
class TrainingSet:
    train: list[SplitImage]
    val: list[SplitImage]
    excluded: int
    """Train-subset images the policy left out."""
    excluded_because: str
    empty_because: str
    """What to tell the operator when `train` is empty."""
    truths: dict[int, ClassTruth] = field(default_factory=dict)
    """For a targeted task: the ground truth of every image in `train`."""
    label_truths: dict[int, LabelTruth] = field(default_factory=dict)
    """For supervised segmentation: the label truth of every image in `train`."""
    box_truths: dict[int, BoxTruth] = field(default_factory=dict)
    """For detection: the box truth of every image in `train`."""


def training_set(conn: sqlite3.Connection, experiment: Experiment) -> TrainingSet:
    everything = images_repo.list_images_for_split(
        conn, experiment.split_id, subsets=[Subset.TRAIN], channels=experiment.channels
    )
    if experiment.task is Task.ANOMALY:
        # Normals only, and by *sample* label rather than by image, because the label lives
        # on the sample (ADR-0041). A defect in the training set teaches that defects are
        # normal. Held-out normals, where the split has any, calibrate; VisA's official
        # protocol has no `val` subset, so an empty one is routine.
        train = [image for image in everything if image.label is Label.NORMAL]
        val = images_repo.list_images_for_split(
            conn,
            experiment.split_id,
            subsets=[Subset.VAL],
            labels=[Label.NORMAL],
            channels=experiment.channels,
        )
        return TrainingSet(
            train=train,
            val=val,
            excluded=len(everything) - len(train),
            excluded_because="are not labelled normal",
            empty_because=(
                f"split {experiment.split_id} has no normal-labelled samples in its train "
                "subset, so there is nothing to learn from. Label some samples normal, or "
                "create a split whose train subset holds them."
            ),
        )

    if experiment.task is Task.FEW_SHOT_SEGMENTATION and experiment.target_label is not None:
        key = experiment.target_label
        truths = resolve_class_truth(
            conn, experiment.dataset_id, [image.image_id for image in everything], key
        )
        train = [image for image in everything if image.image_id in truths]
        return TrainingSet(
            train=train,
            val=[],
            excluded=len(everything) - len(train),
            excluded_because=f"have no completed annotation that answers for {key!r}",
            empty_because=(
                f"no reference in split {experiment.split_id} has ground truth for {key!r}. "
                "Complete an annotation of each reference, or choose other references."
            ),
            truths={image.image_id: truths[image.image_id] for image in train},
        )

    if experiment.task is Task.SEMANTIC_SEGMENTATION and experiment.classes:
        # Annotated images only. `Sample.label` is the anomaly verdict and says nothing
        # about classes; an image with a gap in its truth would teach that the gap is
        # background.
        classes = experiment.classes
        labelled = resolve_label_truth(
            conn, experiment.dataset_id, [image.image_id for image in everything], classes
        )
        train = [image for image in everything if image.image_id in labelled]
        return TrainingSet(
            train=train,
            val=[],
            excluded=len(everything) - len(train),
            excluded_because=(
                "have no completed annotation that answers for every class of the run"
            ),
            empty_because=(
                f"no image in the train subset of split {experiment.split_id} has ground "
                f"truth for every one of {', '.join(classes)}. Complete an annotation of "
                "the training images, or choose a split whose train subset holds annotated "
                "samples."
            ),
            label_truths={image.image_id: labelled[image.image_id] for image in train},
        )

    if experiment.task is Task.OBJECT_DETECTION and experiment.classes:
        # The images segmentation would label, read as boxes: an image with a gap in its
        # truth would teach that an unanswered object is not there.
        classes = experiment.classes
        boxed = resolve_box_truth(
            conn, experiment.dataset_id, [image.image_id for image in everything], classes
        )
        train = [image for image in everything if image.image_id in boxed]
        return TrainingSet(
            train=train,
            val=[],
            excluded=len(everything) - len(train),
            excluded_because=(
                "have no completed annotation that answers for every class of the run"
            ),
            empty_because=(
                f"no image in the train subset of split {experiment.split_id} has ground "
                f"truth for every one of {', '.join(classes)}. Complete an annotation of "
                "the training images, or choose a split whose train subset holds annotated "
                "samples."
            ),
            box_truths={image.image_id: boxed[image.image_id] for image in train},
        )

    raise NoTrainingPolicyError(f"the task {experiment.task.value!r} has no training policy")
