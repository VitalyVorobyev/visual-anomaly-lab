"""Health endpoint.

The Tauri shell polls this until the sidecar is ready, and the UI renders it as the
proof that the whole chain — window, HTTP, sidecar, SQLite — is live.
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
from typing import Literal

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

from anomaly_lab.config import Settings
from anomaly_lab.db.connection import connection
from anomaly_lab.db.migrate import current_schema_version
from anomaly_lab.jobs.resident import ResidentWorker

router = APIRouter(prefix="/api", tags=["health"])


class ResidentHealth(BaseModel):
    """The one long-lived compute process nothing else would show (ADR-0026).

    The resident holds either an experiment checkpoint or a promptable segmentation model
    — and on this machine, the MPS device — between interactive requests. Without this
    block the only evidence it exists is a second Python process in Activity Monitor,
    which is the wrong place to discover that something is holding your accelerator.
    """

    kind: str = Field(description="The resident target family.")
    key: str = Field(description="The experiment id or model-asset key.")
    experiment_id: int | None = None
    generation: str = Field(description="Checkpoint fingerprint; a retrain changes it.")
    evicted_in_seconds: float = Field(description="Until the idle timeout takes it down.")
    requests_served: int


class HealthResponse(BaseModel):
    status: Literal["ok"]
    version: str
    schema_version: int
    db_path: str
    data_dir: str
    started_at: datetime
    resident: ResidentHealth | None = Field(
        default=None, description="The resident compute worker, when one is live."
    )


# Declared `def`, not `async def`: it opens SQLite, and a blocking driver call inside a
# coroutine would stall the event loop. FastAPI runs synchronous handlers in a threadpool
# instead. Every DB-touching handler in this application follows the same rule.
#
# Docstrings on route functions become the endpoint description in the OpenAPI schema, and
# from there the generated TypeScript — so they describe the contract, not the implementation.
@router.get("/health", summary="Backend liveness, version and database state")
def read_health(request: Request) -> HealthResponse:
    """Report the backend's version, its schema version, and where its data lives."""
    settings: Settings = request.app.state.settings
    with connection(settings.db_path) as conn:
        schema_version = current_schema_version(conn)

    # A plain field read, never the resident's lock: a health check that can block behind
    # a model load is not a health check.
    resident: ResidentWorker | None = getattr(request.app.state, "resident", None)
    snapshot = resident.snapshot() if resident is not None else None

    return HealthResponse(
        status="ok",
        version=request.app.state.version,
        schema_version=schema_version,
        db_path=str(settings.db_path),
        data_dir=str(settings.data_dir),
        started_at=request.app.state.started_at,
        resident=ResidentHealth(**asdict(snapshot)) if snapshot is not None else None,
    )
