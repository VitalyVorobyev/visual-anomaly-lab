"""Splits.

A split is created once and never edited — changing one means creating another (ADR-0005)
— so everything needed to reproduce it is written down at creation: the seed, the strategy
and the fractions. A seed alone reproduces nothing.
"""

from __future__ import annotations

import sqlite3

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, ConfigDict, Field

from anomaly_lab.config import Settings
from anomaly_lab.datasets.splitting import SplitParams, SplitPlanError, plan_split
from anomaly_lab.db.connection import connection
from anomaly_lab.db.repositories import datasets as datasets_repo
from anomaly_lab.db.repositories import splits as splits_repo
from anomaly_lab.domain.entities import Split, Subset

router = APIRouter(prefix="/api/splits", tags=["splits"])


class SubsetComposition(BaseModel):
    """What a subset actually contains, so a split can report itself honestly."""

    model_config = ConfigDict(frozen=True)

    subset: Subset
    total: int
    normal: int
    defect: int
    unlabeled: int


class SplitSummary(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: int
    dataset_id: int
    name: str
    strategy: str
    seed: int
    created_at: str
    composition: list[SubsetComposition] = Field(default_factory=list)


class SplitDetail(SplitSummary):
    params: SplitParams


class CreateSplitRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    dataset_id: int
    name: str
    seed: int = Field(default=0, description="Same seed and params reproduce this split.")
    params: SplitParams = Field(default_factory=SplitParams)


def _detail(conn: sqlite3.Connection, split: Split) -> SplitDetail:
    return SplitDetail(
        id=split.id,
        dataset_id=split.dataset_id,
        name=split.name,
        strategy=split.strategy,
        seed=split.seed,
        created_at=split.created_at,
        params=SplitParams.model_validate(split.params),
        composition=[
            SubsetComposition(
                subset=row.subset,
                total=row.total,
                normal=row.normal,
                defect=row.defect,
                unlabeled=row.unlabeled,
            )
            for row in splits_repo.composition(conn, split.id)
        ],
    )


@router.post("", summary="Create a seeded, sample-level split")
def create_split(request: Request, body: CreateSplitRequest) -> SplitDetail:
    """Draw a split and store it with everything needed to redraw it.

    Assignment is per sample, so no two views of one part can straddle the boundary;
    training gets normals only; and the draw is stratified by capture group so an
    acquisition-batch effect cannot land entirely on one side (ADR-0011).
    """
    settings: Settings = request.app.state.settings
    with connection(settings.db_path) as conn:
        if datasets_repo.get_dataset(conn, body.dataset_id) is None:
            raise HTTPException(status_code=404, detail=f"no dataset with id {body.dataset_id}")

        try:
            assignments = plan_split(conn, body.dataset_id, seed=body.seed, params=body.params)
        except SplitPlanError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

        try:
            split = splits_repo.create_split(
                conn,
                body.dataset_id,
                name=body.name,
                strategy=body.params.strategy.value,
                seed=body.seed,
                params=body.params.model_dump(mode="json"),
                assignments=assignments,
            )
        except sqlite3.IntegrityError as exc:
            raise HTTPException(
                status_code=409,
                detail=f"dataset {body.dataset_id} already has a split named {body.name!r}",
            ) from exc

        return _detail(conn, split)


@router.get("", summary="Splits of a dataset, with their composition")
def list_splits(
    request: Request, dataset_id: int = Query(description="Dataset to list splits for.")
) -> list[SplitDetail]:
    """List splits. Composition is included so the picker can show it without a fan-out."""
    settings: Settings = request.app.state.settings
    with connection(settings.db_path) as conn:
        if datasets_repo.get_dataset(conn, dataset_id) is None:
            raise HTTPException(status_code=404, detail=f"no dataset with id {dataset_id}")
        return [_detail(conn, split) for split in splits_repo.list_splits(conn, dataset_id)]


@router.get("/{split_id}", summary="One split")
def get_split(request: Request, split_id: int) -> SplitDetail:
    """One split and its exact composition."""
    settings: Settings = request.app.state.settings
    with connection(settings.db_path) as conn:
        split = splits_repo.get_split(conn, split_id)
        if split is None:
            raise HTTPException(status_code=404, detail=f"no split with id {split_id}")
        return _detail(conn, split)
