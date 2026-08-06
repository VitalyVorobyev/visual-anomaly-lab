"""Health endpoint.

The Tauri shell polls this until the sidecar is ready, and the UI renders it as the
proof that the whole chain — window, HTTP, sidecar, SQLite — is live.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from fastapi import APIRouter, Request
from pydantic import BaseModel

from anomaly_lab.config import Settings
from anomaly_lab.db.connection import connection
from anomaly_lab.db.migrate import current_schema_version

router = APIRouter(prefix="/api", tags=["health"])


class HealthResponse(BaseModel):
    status: Literal["ok"]
    version: str
    schema_version: int
    db_path: str
    data_dir: str
    started_at: datetime


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

    return HealthResponse(
        status="ok",
        version=request.app.state.version,
        schema_version=schema_version,
        db_path=str(settings.db_path),
        data_dir=str(settings.data_dir),
        started_at=request.app.state.started_at,
    )
