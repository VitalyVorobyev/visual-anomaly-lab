"""Region extractor catalogue and immutable dataset profile revisions."""

from __future__ import annotations

import sqlite3
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field, ValidationError, field_validator

from anomaly_lab.config import Settings
from anomaly_lab.db.connection import connection
from anomaly_lab.db.repositories import datasets as datasets_repo
from anomaly_lab.db.repositories import region_profiles as profiles_repo
from anomaly_lab.domain.entities import RegionProfileRevision
from anomaly_lab.regions.base import RegionExtractorDescription
from anomaly_lab.regions.registry import (
    UnknownRegionExtractorError,
    describe_all,
    validate_config,
)
from anomaly_lab.schemas import API_MODEL_CONFIG

router = APIRouter(tags=["region-profiles"])


class RegionProfileCreate(BaseModel):
    model_config = API_MODEL_CONFIG

    name: str = Field(min_length=1, max_length=120)
    extractor_type: str = Field(min_length=1, max_length=80)
    extractor_config: dict[str, Any] = Field(default_factory=dict)
    prepared_width: int = Field(default=256, ge=8, le=2048)
    prepared_height: int = Field(default=256, ge=8, le=2048)
    padding_fraction: float = Field(default=0.05, ge=0.0, le=1.0)
    seed: int = 17

    @field_validator("name")
    @classmethod
    def _meaningful_name(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("profile name cannot be blank")
        return stripped


@router.get("/api/region-extractors", summary="Every registered region extractor and its schema")
def list_region_extractors() -> list[RegionExtractorDescription]:
    return describe_all()


@router.get(
    "/api/datasets/{dataset_id}/region-profiles",
    summary="Immutable region profile revisions owned by one dataset",
)
def list_region_profiles(request: Request, dataset_id: int) -> list[RegionProfileRevision]:
    settings: Settings = request.app.state.settings
    with connection(settings.db_path) as conn:
        if datasets_repo.get_dataset(conn, dataset_id) is None:
            raise HTTPException(status_code=404, detail=f"no dataset with id {dataset_id}")
        return profiles_repo.list_profiles(conn, dataset_id)


@router.post(
    "/api/datasets/{dataset_id}/region-profiles",
    summary="Append an immutable region profile revision",
)
def create_region_profile(
    request: Request, dataset_id: int, body: RegionProfileCreate
) -> RegionProfileRevision:
    try:
        validated = validate_config(body.extractor_type, body.extractor_config)
    except UnknownRegionExtractorError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.errors(include_url=False)) from exc

    settings: Settings = request.app.state.settings
    with connection(settings.db_path) as conn:
        conn.execute("BEGIN IMMEDIATE")
        try:
            if datasets_repo.get_dataset(conn, dataset_id) is None:
                raise HTTPException(status_code=404, detail=f"no dataset with id {dataset_id}")
            created = profiles_repo.create_revision(
                conn,
                dataset_id=dataset_id,
                name=body.name,
                extractor_type=body.extractor_type,
                extractor_config=validated.model_dump(mode="json"),
                prepared_width=body.prepared_width,
                prepared_height=body.prepared_height,
                padding_fraction=body.padding_fraction,
                seed=body.seed,
            )
            conn.execute("COMMIT")
        except sqlite3.IntegrityError as exc:
            conn.execute("ROLLBACK")
            raise HTTPException(
                status_code=409, detail="region profile revision conflicts"
            ) from exc
        except BaseException:
            if conn.in_transaction:
                conn.execute("ROLLBACK")
            raise
    return created


@router.get("/api/region-profiles/{profile_id}", summary="One immutable region profile revision")
def get_region_profile(request: Request, profile_id: int) -> RegionProfileRevision:
    settings: Settings = request.app.state.settings
    with connection(settings.db_path) as conn:
        profile = profiles_repo.get_profile(conn, profile_id)
    if profile is None:
        raise HTTPException(status_code=404, detail=f"no region profile with id {profile_id}")
    return profile
