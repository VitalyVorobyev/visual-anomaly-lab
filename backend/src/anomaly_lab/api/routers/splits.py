"""Splits.

A split is created once and never edited — changing one means creating another (ADR-0041)
— so everything needed to reproduce it is written down at creation: the seed, the strategy
and the fractions. A seed alone reproduces nothing.

Because it cannot be edited, what a split will contain is shown before it exists: the presets
this dataset can serve and the custom form's preview are dry runs of the same planner the
create call uses (`datasets/split_presets.py`). A split no experiment ran on can be deleted;
one an experiment ran on cannot, because the run's numbers are only meaningful against it.
"""

from __future__ import annotations

import sqlite3

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, Field

from anomaly_lab.config import Settings
from anomaly_lab.datasets.split_presets import (
    PLAN_ERRORS,
    DatasetTruth,
    SplitPreset,
    SubsetComposition,
    auto_name,
    compose,
    next_seed,
    plan,
    presets,
    read_truth,
    tasks_served,
)
from anomaly_lab.datasets.splitting import SplitParams
from anomaly_lab.db.connection import connection, transaction
from anomaly_lab.db.repositories import datasets as datasets_repo
from anomaly_lab.db.repositories import splits as splits_repo
from anomaly_lab.domain.entities import Dataset, Split, Task
from anomaly_lab.schemas import API_MODEL_CONFIG

router = APIRouter(tags=["splits"])


class SplitHolder(BaseModel):
    """An experiment that ran on a split, and so holds it."""

    model_config = API_MODEL_CONFIG

    experiment_id: int
    name: str


class SplitSummary(BaseModel):
    model_config = API_MODEL_CONFIG

    id: int
    dataset_id: int
    name: str
    strategy: str
    seed: int
    created_at: str
    tasks: list[Task] = Field(default_factory=list, description="The tasks this split trains.")
    composition: list[SubsetComposition] = Field(default_factory=list)
    experiments: list[SplitHolder] = Field(
        default_factory=list, description="The experiments that ran on this split, newest first."
    )


class SplitDetail(SplitSummary):
    params: SplitParams


class CreateSplitRequest(BaseModel):
    model_config = API_MODEL_CONFIG

    dataset_id: int
    name: str | None = Field(
        default=None,
        description="Leave empty for `<label> · seed <n>`, derived from the params.",
    )
    seed: int | None = Field(
        default=None,
        description=(
            "Same seed and params reproduce this split. Leave empty for the first seed no "
            "split of the same params has used, so a repeated request is a new draw."
        ),
    )
    params: SplitParams = Field(default_factory=SplitParams)


class SplitPreviewRequest(BaseModel):
    model_config = API_MODEL_CONFIG

    seed: int | None = Field(default=None, description="As on create.")
    params: SplitParams = Field(default_factory=SplitParams)


class SplitPreview(BaseModel):
    """What a create call with these params would produce, computed without writing it."""

    model_config = API_MODEL_CONFIG

    seed: int
    name: str
    tasks: list[Task]
    composition: list[SubsetComposition] = Field(default_factory=list)
    error: str | None = Field(
        default=None,
        description="Why the request cannot be drawn on this dataset; composition is then empty.",
    )


class SplitDeletionPreview(BaseModel):
    """What deleting a split removes, and what blocks it."""

    model_config = API_MODEL_CONFIG

    split_id: int
    name: str
    assignments: int
    experiments: list[SplitHolder] = Field(default_factory=list)
    can_delete: bool
    blocker: str | None = None


class SplitDeletionResult(BaseModel):
    model_config = API_MODEL_CONFIG

    deleted: bool
    assignments_removed: int


def _detail(
    conn: sqlite3.Connection, split: Split, truth: DatasetTruth | None = None
) -> SplitDetail:
    if truth is None:
        truth = read_truth(conn, split.dataset_id)
    return SplitDetail(
        id=split.id,
        dataset_id=split.dataset_id,
        name=split.name,
        strategy=split.strategy,
        seed=split.seed,
        created_at=split.created_at,
        tasks=tasks_served(split.strategy),
        params=SplitParams.model_validate(split.params),
        composition=compose(splits_repo.assignments(conn, split.id), truth),
        experiments=[
            SplitHolder(experiment_id=identifier, name=name)
            for identifier, name in splits_repo.experiments_using(conn, split.id)
        ],
    )


def _require_dataset(conn: sqlite3.Connection, dataset_id: int) -> Dataset:
    dataset = datasets_repo.get_dataset(conn, dataset_id)
    if dataset is None:
        raise HTTPException(status_code=404, detail=f"no dataset with id {dataset_id}")
    return dataset


@router.post("/api/splits", summary="Create a sample-level split, seeded or imported")
def create_split(request: Request, body: CreateSplitRequest) -> SplitDetail:
    """Draw a split, or adopt a published one, and store what reproduces it.

    A seeded split is per sample, so no two views of one part can straddle the boundary;
    training gets normals only; and the draw is stratified by capture group so an
    acquisition-batch effect cannot land entirely on one side (handbook evaluation.md).

    The `imported` strategy instead reads the partition out of the manifest the dataset
    was committed from, because a benchmark's published number is only comparable against
    the benchmark's own partition. `manual` and `few_shot` hold a few-shot task's
    references in `train` and everything else in `test` (ADR-0040). `class_stratified`
    draws the samples annotated for every class into `train` and `test`, stratified by the
    classes each shows, for a supervised task (ADR-0039).
    """
    settings: Settings = request.app.state.settings
    with connection(settings.db_path) as conn:
        dataset = _require_dataset(conn, body.dataset_id)
        existing = splits_repo.list_splits(conn, dataset.id)
        seed = body.seed if body.seed is not None else next_seed(existing, body.params)
        try:
            assignments, params = plan(conn, dataset, seed=seed, params=body.params)
        except PLAN_ERRORS as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

        name = (body.name or "").strip() or auto_name(existing, params, seed)
        try:
            split = splits_repo.create_split(
                conn,
                dataset.id,
                name=name,
                strategy=params.strategy.value,
                seed=seed,
                params=params.model_dump(mode="json"),
                assignments=assignments,
            )
        except sqlite3.IntegrityError as exc:
            raise HTTPException(
                status_code=409,
                detail=f"dataset {dataset.id} already has a split named {name!r}",
            ) from exc

        return _detail(conn, split)


@router.get("/api/splits", summary="Splits of a dataset, with their composition")
def list_splits(
    request: Request, dataset_id: int = Query(description="Dataset to list splits for.")
) -> list[SplitDetail]:
    """List splits. Composition is included so the picker can show it without a fan-out."""
    settings: Settings = request.app.state.settings
    with connection(settings.db_path) as conn:
        _require_dataset(conn, dataset_id)
        truth = read_truth(conn, dataset_id)
        return [_detail(conn, split, truth) for split in splits_repo.list_splits(conn, dataset_id)]


@router.get(
    "/api/datasets/{dataset_id}/split-presets",
    summary="The zero-configuration splits this dataset can serve, with dry-run compositions",
)
def list_split_presets(request: Request, dataset_id: int) -> list[SplitPreset]:
    """Presets per task, each with the composition Create would produce. Writes nothing.

    Anomaly presets need anomaly verdicts; class presets need class truth (ADR-0041). A
    preset whose dry run fails on this dataset is left out.
    """
    settings: Settings = request.app.state.settings
    with connection(settings.db_path) as conn:
        dataset = _require_dataset(conn, dataset_id)
        return presets(conn, dataset, splits_repo.list_splits(conn, dataset_id))


@router.post(
    "/api/datasets/{dataset_id}/splits/preview",
    summary="What a split with these params would contain, without creating it",
)
def preview_split(request: Request, dataset_id: int, body: SplitPreviewRequest) -> SplitPreview:
    """The dry run the custom form shows as it is edited.

    A request that cannot be drawn answers 200 with `error` set, because an unfinished form
    is an ordinary state, not a failed call.
    """
    settings: Settings = request.app.state.settings
    with connection(settings.db_path) as conn:
        dataset = _require_dataset(conn, dataset_id)
        existing = splits_repo.list_splits(conn, dataset_id)
        seed = body.seed if body.seed is not None else next_seed(existing, body.params)
        name = auto_name(existing, body.params, seed)
        tasks = tasks_served(body.params.strategy)
        try:
            assignments, _ = plan(conn, dataset, seed=seed, params=body.params)
        except PLAN_ERRORS as exc:
            return SplitPreview(seed=seed, name=name, tasks=tasks, error=str(exc))
        return SplitPreview(
            seed=seed,
            name=name,
            tasks=tasks,
            composition=compose(assignments, read_truth(conn, dataset_id)),
        )


@router.get("/api/splits/{split_id}", summary="One split")
def get_split(request: Request, split_id: int) -> SplitDetail:
    """One split and its exact composition."""
    settings: Settings = request.app.state.settings
    with connection(settings.db_path) as conn:
        split = splits_repo.get_split(conn, split_id)
        if split is None:
            raise HTTPException(status_code=404, detail=f"no split with id {split_id}")
        return _detail(conn, split)


def _blocker(holders: list[tuple[int, str]]) -> str | None:
    if not holders:
        return None
    named = ", ".join(f"#{identifier} {name}" for identifier, name in holders[:3])
    rest = "" if len(holders) <= 3 else f" and {len(holders) - 3} more"
    return (
        f"{len(holders)} experiment{'' if len(holders) == 1 else 's'} ran on this split "
        f"({named}{rest}); their numbers are only meaningful against it. Delete them first."
    )


@router.get(
    "/api/splits/{split_id}/deletion-preview",
    summary="What deleting a split removes, and the experiments that block it",
)
def preview_split_deletion(request: Request, split_id: int) -> SplitDeletionPreview:
    settings: Settings = request.app.state.settings
    with connection(settings.db_path) as conn:
        split = splits_repo.get_split(conn, split_id)
        if split is None:
            raise HTTPException(status_code=404, detail=f"no split with id {split_id}")
        holders = splits_repo.experiments_using(conn, split_id)
        blocker = _blocker(holders)
        return SplitDeletionPreview(
            split_id=split.id,
            name=split.name,
            assignments=len(splits_repo.assignments(conn, split_id)),
            experiments=[
                SplitHolder(experiment_id=identifier, name=name) for identifier, name in holders
            ],
            can_delete=blocker is None,
            blocker=blocker,
        )


@router.delete("/api/splits/{split_id}", summary="Delete a split no experiment ran on")
def delete_split(request: Request, split_id: int) -> SplitDeletionResult:
    """Remove a split and its assignments; refuse while an experiment holds it.

    `experiment.split_id` is `ON DELETE RESTRICT` and nothing cascades: an experiment's
    results are only meaningful against the partition it ran on, so the experiments are
    deleted first, deliberately, or the split stays. The check and the delete share one
    write transaction, so an experiment cannot be created on the split in between.
    """
    settings: Settings = request.app.state.settings
    with connection(settings.db_path) as conn, transaction(conn, immediate=True):
        if splits_repo.get_split(conn, split_id) is None:
            raise HTTPException(status_code=404, detail=f"no split with id {split_id}")
        blocker = _blocker(splits_repo.experiments_using(conn, split_id))
        if blocker is not None:
            raise HTTPException(status_code=409, detail=blocker)
        removed = len(splits_repo.assignments(conn, split_id))
        deleted = splits_repo.delete_split(conn, split_id)
    return SplitDeletionResult(deleted=deleted, assignments_removed=removed)
