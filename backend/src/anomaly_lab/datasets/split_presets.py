"""Planning a split without writing it: presets, dry runs, composition and names.

A split is immutable (ADR-0041), so the moment to see what it will contain is *before* it is
created. Every entry point — a preset card, the custom form's live preview, the create call
itself — goes through `plan`, which dispatches to the `plan_*` functions in `splitting` and
writes nothing. `compose` then counts the result the way a stored split reports itself, so a
preview and the split it becomes cannot disagree.

**A preset is a request that can work on this dataset**, not a new strategy: a name, the
params of an existing strategy, the task it serves and its dry-run composition. The anomaly
presets need anomaly verdicts (ADR-0041); the class presets need class truth, and a few-shot
preset lists the classes with enough samples to draw from rather than being repeated per class.
A preset whose dry run fails is not offered, because a card that errors on "Create" is worse
than no card.

**Names are derived.** `describe` turns params into the label a person would give them —
"Standard · 60/20/20, normals only", "5-shot · screw" — and an unnamed split is
`<label> · seed <n>`. When no seed is given, the first seed no split of the same description
already uses is taken, so pressing "Create" on the same preset twice draws a second, different
split rather than a copy of the first.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, Field

from anomaly_lab.datasets.manifest import Manifest
from anomaly_lab.datasets.splitting import (
    SplitParams,
    SplitPlanError,
    SplitStrategy,
    plan_class_stratified_split,
    plan_few_shot_split,
    plan_imported_split,
    plan_manual_split,
    plan_split,
)
from anomaly_lab.datasets.storage import (
    ManifestNotFoundError,
    UnsupportedManifestVersionError,
    load_manifest_file,
)
from anomaly_lab.db.repositories import annotations as annotations_repo
from anomaly_lab.db.repositories import datasets as datasets_repo
from anomaly_lab.db.repositories import samples as samples_repo
from anomaly_lab.domain.entities import ClassPresence, Dataset, Label, Split, Subset, Task
from anomaly_lab.models.base import evenly_spaced
from anomaly_lab.schemas import API_MODEL_CONFIG

# Everything that means "this request cannot be planned on this dataset".
PLAN_ERRORS = (SplitPlanError, ManifestNotFoundError, UnsupportedManifestVersionError)

SUPERVISED_TASKS = [Task.SEMANTIC_SEGMENTATION, Task.OBJECT_DETECTION]


def tasks_served(strategy: str) -> list[Task]:
    """The tasks a split of this strategy trains (ADR-0039, ADR-0040).

    An anomaly run trains on normals, so on a drawn or adopted partition; a few-shot run on a
    split of references; a supervised run on annotated samples drawn by class. A hand-picked
    (`manual`) split is a list of references, which a supervised run can also fit on.
    """
    if strategy in (SplitStrategy.NORMAL_ONLY_TRAIN, SplitStrategy.IMPORTED):
        return [Task.ANOMALY]
    if strategy == SplitStrategy.FEW_SHOT:
        return [Task.FEW_SHOT_SEGMENTATION]
    if strategy == SplitStrategy.CLASS_STRATIFIED:
        return list(SUPERVISED_TASKS)
    if strategy == SplitStrategy.MANUAL:
        return [Task.FEW_SHOT_SEGMENTATION, *SUPERVISED_TASKS]
    return []


# --- composition ---------------------------------------------------------------------------


class ClassShare(BaseModel):
    """How many samples of a subset show one class."""

    model_config = API_MODEL_CONFIG

    key: str
    samples: int


# Enough to see what a subset holds; a subset of a thousand is not a thousand thumbnails.
EXAMPLES_PER_SUBSET = 4


class SubsetComposition(BaseModel):
    """What a subset actually contains, so a split can report itself honestly.

    `normal`, `defect` and `unlabeled` count anomaly verdicts; `classes` counts the samples
    whose completed annotation shows each class, for a dataset with class truth (ADR-0041).
    """

    model_config = API_MODEL_CONFIG

    subset: Subset
    total: int
    normal: int
    defect: int
    unlabeled: int
    classes: list[ClassShare] = Field(default_factory=list)
    examples: list[int] = Field(
        default_factory=list,
        description=(
            f"Up to {EXAMPLES_PER_SUBSET} of the subset's sample ids, spread evenly over it in "
            "browse order: a picture of what it holds, never a list of its members."
        ),
    )


@dataclass(frozen=True)
class DatasetTruth:
    """Every sample's verdict and every class's presence, read once per request."""

    labels: Mapping[int, Label]
    classes: Mapping[str, Mapping[int, ClassPresence]]
    """Per class the dataset's completed annotations show, in class order."""

    def present(self, key: str) -> list[int]:
        return sorted(
            sample_id
            for sample_id, found in self.classes.get(key, {}).items()
            if found is ClassPresence.PRESENT
        )


def read_truth(conn: sqlite3.Connection, dataset_id: int) -> DatasetTruth:
    everything = samples_repo.list_samples(conn, dataset_id, limit=_ALL, offset=0)
    keys = list(datasets_repo.class_counts(conn, dataset_id))
    return DatasetTruth(
        labels={sample.id: sample.label for sample in everything},
        classes=annotations_repo.presence_by_class(conn, dataset_id, keys) if keys else {},
    )


def compose(assignments: Mapping[int, Subset], truth: DatasetTruth) -> list[SubsetComposition]:
    """Per-subset counts by verdict and by class, with every subset present even when empty."""
    rows: list[SubsetComposition] = []
    # Browse order, which `truth.labels` keeps, so the examples spread over the dataset's own
    # order rather than over whatever order the planner happened to assign in.
    order = {sample_id: index for index, sample_id in enumerate(truth.labels)}
    for subset in Subset:
        members = sorted(
            (sample_id for sample_id, placed in assignments.items() if placed is subset),
            key=lambda sample_id: order.get(sample_id, len(order)),
        )
        verdicts = [truth.labels.get(sample_id, Label.UNLABELED) for sample_id in members]
        rows.append(
            SubsetComposition(
                subset=subset,
                total=len(members),
                normal=verdicts.count(Label.NORMAL),
                defect=verdicts.count(Label.DEFECT),
                unlabeled=verdicts.count(Label.UNLABELED),
                classes=[
                    ClassShare(
                        key=key,
                        samples=sum(
                            presence.get(sample_id) is ClassPresence.PRESENT
                            for sample_id in members
                        ),
                    )
                    for key, presence in truth.classes.items()
                ],
                examples=[
                    members[index] for index in evenly_spaced(len(members), EXAMPLES_PER_SUBSET)
                ],
            )
        )
    return rows


# --- planning ------------------------------------------------------------------------------


def committed_manifest(dataset: Dataset) -> tuple[Manifest, str]:
    """The manifest this dataset was last committed from, and its id.

    A dataset with no manifest path was never imported through the two-stage flow, so
    there is nothing to adopt — and saying that plainly is more useful than an empty
    split that looks like a successful one.
    """
    if not dataset.manifest_path:
        msg = (
            f"dataset {dataset.id} has no committed manifest, so there is no published "
            "partition to import. Re-import it with an adapter that reads a split column."
        )
        raise ManifestNotFoundError(msg)

    path = Path(dataset.manifest_path)
    return load_manifest_file(path), path.stem


def plan(
    conn: sqlite3.Connection, dataset: Dataset, *, seed: int, params: SplitParams
) -> tuple[dict[int, Subset], SplitParams]:
    """Every sample's subset under this request, and the params to store with it.

    Writes nothing. Raises one of `PLAN_ERRORS` when the request cannot work on this dataset.
    The returned params record what the draw was judged against — the manifest an `imported`
    split came from, the taxonomy a `class_stratified` one stratified by.
    """
    if params.strategy is SplitStrategy.IMPORTED:
        manifest, manifest_id = committed_manifest(dataset)
        assignments = plan_imported_split(
            conn, dataset.id, manifest, seed=seed, holdout_from_train=params.holdout_from_train
        )
        return assignments, params.model_copy(update={"manifest_id": manifest_id})
    if params.strategy is SplitStrategy.MANUAL:
        return plan_manual_split(conn, dataset.id, params.sample_ids), params
    if params.strategy is SplitStrategy.FEW_SHOT:
        # The params validator requires both for this strategy.
        assert params.label_key is not None and params.shots is not None
        known = {label.key for label in annotations_repo.list_labels(conn, dataset.id)}
        if params.label_key not in known:
            raise SplitPlanError(
                f"dataset {dataset.id} has no annotation class {params.label_key!r}"
            )
        assignments = plan_few_shot_split(
            conn, dataset.id, seed=seed, label_key=params.label_key, shots=params.shots
        )
        return assignments, params
    if params.strategy is SplitStrategy.CLASS_STRATIFIED:
        # Every class, in taxonomy order — the list a supervised run pins.
        classes = [label.key for label in annotations_repo.list_labels(conn, dataset.id)]
        assignments = plan_class_stratified_split(
            conn,
            dataset.id,
            seed=seed,
            classes=classes,
            train_fraction=params.train_fraction,
            unlabeled_subset=params.unlabeled_subset,
        )
        return assignments, params.model_copy(update={"classes": classes})
    return plan_split(conn, dataset.id, seed=seed, params=params), params


# --- names ---------------------------------------------------------------------------------


def _percent(fraction: float) -> int:
    return round(fraction * 100)


def describe(params: SplitParams) -> str:
    """The label a person would give these params; the stem of an unnamed split's name."""
    if params.strategy is SplitStrategy.IMPORTED:
        if params.holdout_from_train > 0:
            return f"Published · {_percent(params.holdout_from_train)}% train held out"
        return "Published"
    if params.strategy is SplitStrategy.FEW_SHOT:
        return f"{params.shots}-shot · {params.label_key}"
    if params.strategy is SplitStrategy.MANUAL:
        count = len(params.sample_ids)
        return f"{count} hand-picked reference{'' if count == 1 else 's'}"
    if params.strategy is SplitStrategy.CLASS_STRATIFIED:
        train = _percent(params.train_fraction)
        return f"{train}/{100 - train} by class"
    train = _percent(params.train_normal_fraction)
    val = _percent(params.val_normal_fraction)
    ratios = f"{train}/{val}/{100 - train - val}, normals only"
    standard = SplitParams()
    if (
        params.train_normal_fraction == standard.train_normal_fraction
        and params.val_normal_fraction == standard.val_normal_fraction
        and params.val_defect_fraction == standard.val_defect_fraction
    ):
        return f"Standard · {ratios}"
    return ratios


def _seed_matters(params: SplitParams) -> bool:
    """A published partition with no holdout ignores the seed, so no name should cite one."""
    if params.strategy is SplitStrategy.MANUAL:
        return False
    return not (params.strategy is SplitStrategy.IMPORTED and params.holdout_from_train == 0)


def next_seed(existing: Iterable[Split], params: SplitParams) -> int:
    """The first seed no split of the same description has used yet."""
    stem = describe(params)
    used = {
        split.seed
        for split in existing
        if describe(SplitParams.model_validate(split.params)) == stem
    }
    seed = 0
    while seed in used:
        seed += 1
    return seed


def auto_name(existing: Iterable[Split], params: SplitParams, seed: int) -> str:
    """`<label> · seed <n>`, with a numeric suffix if a split of that name already exists."""
    stem = describe(params)
    base = f"{stem} · seed {seed}" if _seed_matters(params) else stem
    taken = {split.name for split in existing}
    if base not in taken:
        return base
    suffix = 2
    while f"{base} ({suffix})" in taken:
        suffix += 1
    return f"{base} ({suffix})"


# --- presets -------------------------------------------------------------------------------


class PresetClass(BaseModel):
    """A class a few-shot preset can draw references of, and how many samples show it."""

    model_config = API_MODEL_CONFIG

    key: str
    name: str
    samples: int


class SplitPreset(BaseModel):
    """A zero-configuration split that works on this dataset, with its dry-run composition."""

    model_config = API_MODEL_CONFIG

    key: str
    label: str
    meaning: str = Field(description="One line saying what the split is for.")
    tasks: list[Task]
    params: SplitParams
    seed: int = Field(description="The seed Create would draw with: the first one not yet used.")
    name: str = Field(description="The name Create would give the split.")
    composition: list[SubsetComposition]
    classes: list[PresetClass] = Field(
        default_factory=list,
        description=(
            "For a few-shot preset: the classes with enough samples to draw from, most "
            "frequent first. `params.label_key` is the first of them."
        ),
    )


@dataclass(frozen=True)
class _Candidate:
    key: str
    label: str
    meaning: str
    params: SplitParams
    classes: Sequence[PresetClass] = ()


def _candidates(
    conn: sqlite3.Connection, dataset: Dataset, truth: DatasetTruth
) -> list[_Candidate]:
    verdicts = set(truth.labels.values())
    found: list[_Candidate] = []
    if Label.NORMAL in verdicts or Label.DEFECT in verdicts:
        found.append(
            _Candidate(
                key="standard",
                label="Standard · 60/20/20, normals only",
                meaning=(
                    "Train on 60% of the normals, calibrate on 20%, and test on the rest with "
                    "every defect — drawn per capture group."
                ),
                params=SplitParams(strategy=SplitStrategy.NORMAL_ONLY_TRAIN),
            )
        )
        if _has_published_partition(dataset):
            found.append(
                _Candidate(
                    key="published",
                    label="Published",
                    meaning=(
                        "The partition the source published, so a number here is comparable to "
                        "the one it reports."
                    ),
                    params=SplitParams(strategy=SplitStrategy.IMPORTED),
                )
            )
    if not truth.classes:
        return found

    names = {label.key: label.name for label in annotations_repo.list_labels(conn, dataset.id)}
    counted = sorted(
        (
            PresetClass(key=key, name=names.get(key, key), samples=len(truth.present(key)))
            for key in truth.classes
        ),
        # Most frequent first; class order breaks ties, as `sorted` is stable.
        key=lambda entry: -entry.samples,
    )
    total = len(truth.labels)
    for shots in (1, 5):
        # References to draw, and at least one other sample to query.
        eligible = [entry for entry in counted if entry.samples >= shots and total > shots]
        if eligible:
            found.append(
                _Candidate(
                    key=f"few_shot_{shots}",
                    label=f"{shots}-shot",
                    meaning=(
                        f"{'One reference' if shots == 1 else f'{shots} references'} of a class "
                        "drawn under the seed; every other sample is a query."
                    ),
                    params=SplitParams(
                        strategy=SplitStrategy.FEW_SHOT, label_key=eligible[0].key, shots=shots
                    ),
                    classes=eligible,
                )
            )
    found.append(
        _Candidate(
            key="by_class",
            label="70/30 by class",
            meaning=(
                "Annotated samples drawn 70/30 into train and test by the classes each shows; "
                "the rest are scored in test."
            ),
            params=SplitParams(strategy=SplitStrategy.CLASS_STRATIFIED),
        )
    )
    return found


def _has_published_partition(dataset: Dataset) -> bool:
    try:
        manifest, _ = committed_manifest(dataset)
    except (ManifestNotFoundError, UnsupportedManifestVersionError, OSError, ValueError):
        return False
    return manifest.has_imported_split()


def presets(
    conn: sqlite3.Connection, dataset: Dataset, existing: Sequence[Split]
) -> list[SplitPreset]:
    """The presets whose dry run succeeds on this dataset, in the order they are shown."""
    truth = read_truth(conn, dataset.id)
    offered: list[SplitPreset] = []
    for candidate in _candidates(conn, dataset, truth):
        seed = next_seed(existing, candidate.params)
        try:
            assignments, _ = plan(conn, dataset, seed=seed, params=candidate.params)
        except PLAN_ERRORS:
            continue
        offered.append(
            SplitPreset(
                key=candidate.key,
                label=candidate.label,
                meaning=candidate.meaning,
                tasks=tasks_served(candidate.params.strategy),
                params=candidate.params,
                seed=seed,
                name=auto_name(existing, candidate.params, seed),
                composition=compose(assignments, truth),
                classes=list(candidate.classes),
            )
        )
    return offered


# `list_samples` pages by default; a split needs the whole dataset at once.
_ALL = 1_000_000
