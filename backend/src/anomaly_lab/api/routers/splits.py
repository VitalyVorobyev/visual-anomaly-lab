"""Splits.

A split is created once and never edited — changing one means creating another (ADR-0005)
— so everything needed to reproduce it is written down at creation: the seed, the strategy
and the fractions. A seed alone reproduces nothing.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, Field

from anomaly_lab.config import Settings
from anomaly_lab.datasets.manifest import Manifest
from anomaly_lab.datasets.splitting import (
    SplitParams,
    SplitPlanError,
    SplitStrategy,
    plan_imported_split,
    plan_split,
)
from anomaly_lab.datasets.storage import (
    ManifestNotFoundError,
    UnsupportedManifestVersionError,
    load_manifest_file,
)
from anomaly_lab.db.connection import connection
from anomaly_lab.db.repositories import datasets as datasets_repo
from anomaly_lab.db.repositories import splits as splits_repo
from anomaly_lab.domain.entities import Dataset, Split, Subset
from anomaly_lab.schemas import API_MODEL_CONFIG

router = APIRouter(prefix="/api/splits", tags=["splits"])


class SubsetComposition(BaseModel):
    """What a subset actually contains, so a split can report itself honestly."""

    model_config = API_MODEL_CONFIG

    subset: Subset
    total: int
    normal: int
    defect: int
    unlabeled: int


class SplitSummary(BaseModel):
    model_config = API_MODEL_CONFIG

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
    model_config = API_MODEL_CONFIG

    dataset_id: int
    name: str
    seed: int = Field(default=0, description="Same seed and params reproduce this split.")
    params: SplitParams = Field(default_factory=SplitParams)


def _committed_manifest(dataset: Dataset) -> tuple[Manifest, str]:
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


@router.post("", summary="Create a sample-level split, seeded or imported")
def create_split(request: Request, body: CreateSplitRequest) -> SplitDetail:
    """Draw a split, or adopt a published one, and store what reproduces it.

    A seeded split is per sample, so no two views of one part can straddle the boundary;
    training gets normals only; and the draw is stratified by capture group so an
    acquisition-batch effect cannot land entirely on one side (ADR-0011).

    The `imported` strategy instead reads the partition out of the manifest the dataset
    was committed from, because a benchmark's published number is only comparable against
    the benchmark's own partition.
    """
    settings: Settings = request.app.state.settings
    with connection(settings.db_path) as conn:
        dataset = datasets_repo.get_dataset(conn, body.dataset_id)
        if dataset is None:
            raise HTTPException(status_code=404, detail=f"no dataset with id {body.dataset_id}")

        params = body.params
        try:
            if params.strategy is SplitStrategy.IMPORTED:
                manifest, manifest_id = _committed_manifest(dataset)
                assignments = plan_imported_split(
                    conn,
                    body.dataset_id,
                    manifest,
                    seed=body.seed,
                    holdout_from_train=params.holdout_from_train,
                )
                # Record which import asserted this partition, so the split stays
                # traceable to the file that decided it rather than to a seed it ignored.
                params = params.model_copy(update={"manifest_id": manifest_id})
            else:
                assignments = plan_split(conn, body.dataset_id, seed=body.seed, params=params)
        except SplitPlanError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ManifestNotFoundError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except UnsupportedManifestVersionError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

        try:
            split = splits_repo.create_split(
                conn,
                body.dataset_id,
                name=body.name,
                strategy=params.strategy.value,
                seed=body.seed,
                params=params.model_dump(mode="json"),
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
