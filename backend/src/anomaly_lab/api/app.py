"""FastAPI application factory.

`create_app` is referenced by the documented development command:

    uv run --directory backend uvicorn anomaly_lab.api.app:create_app --factory --reload --port 8000
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from anomaly_lab import __version__
from anomaly_lab.api.routers import (
    compare,
    datasets,
    experiments,
    health,
    images,
    import_,
    jobs,
    reference_packs,
    splits,
    ws,
)
from anomaly_lab.config import Settings, get_settings
from anomaly_lab.db.connection import connection
from anomaly_lab.db.migrate import apply_migrations_to
from anomaly_lab.db.repositories.experiments import fail_stale_training_experiments
from anomaly_lab.db.repositories.jobs import fail_stale_running_jobs
from anomaly_lab.jobs.queue import JobQueue
from anomaly_lab.jobs.resident import ResidentWorker

logger = logging.getLogger(__name__)

# The Tauri v2 WebView serves the app from `tauri://localhost` on macOS and
# `http://tauri.localhost` on Windows — neither is a localhost *port*, so a regex that
# only covers `http://localhost:*` lets the browser path work while the desktop path
# fails on CORS. Development affordance only; disabled by ANOMALY_LAB_DEV_CORS=false.
DEV_ORIGIN_REGEX = (
    r"^(https?://(localhost|127\.0\.0\.1)(:\d+)?|tauri://localhost|https?://tauri\.localhost)$"
)


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings: Settings = app.state.settings
    settings.ensure_directories()

    with connection(settings.db_path) as conn:
        # Idempotent, and also run by `serve.py` before the port is announced — this call
        # is what makes the plain `uv run uvicorn` path migrate as well.
        schema_version = apply_migrations_to(conn)
        stale_jobs = fail_stale_running_jobs(conn)
        stale_experiments = fail_stale_training_experiments(conn)

    logger.info("Database %s ready at schema version %d", settings.db_path, schema_version)
    if stale_jobs:
        logger.warning(
            "Marked %d job(s) left running by a previous process as failed: %s",
            len(stale_jobs),
            ", ".join(str(job.id) for job in stale_jobs),
        )
    if stale_experiments:
        logger.warning(
            "Marked %d experiment(s) left mid-training as failed: %s",
            len(stale_experiments),
            ", ".join(str(experiment_id) for experiment_id in stale_experiments),
        )

    # Started after reconciliation, so the runner never sees a row the previous process
    # left behind, and stopped before the process exits, so no worker outlives us.
    queue: JobQueue = app.state.job_queue
    resident: ResidentWorker = app.state.resident
    await queue.start()
    try:
        yield
    finally:
        # The resident first: it is the process the queue's own teardown knows nothing
        # about, and leaving one behind is the orphan ADR-0026 is careful about.
        await resident.stop()
        await queue.stop()


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()

    app = FastAPI(
        title="visual-anomaly-lab",
        version=__version__,
        summary="Local workbench for training and comparing visual anomaly detectors.",
        lifespan=_lifespan,
    )
    app.state.settings = settings
    app.state.version = __version__
    app.state.started_at = datetime.now(UTC)
    # Wired here and nowhere else. The queue must not import the resident and the resident
    # must not import the queue's manager; the composition root points at both, which is
    # the same discipline ADR-0014 applies to shell capabilities. The hook is what makes a
    # resident and a job worker unable to coexist (ADR-0026).
    app.state.resident = ResidentWorker(settings)
    app.state.job_queue = JobQueue(settings, before_spawn=app.state.resident.evict)

    if settings.dev_cors:
        app.add_middleware(
            CORSMiddleware,
            allow_origin_regex=DEV_ORIGIN_REGEX,
            allow_credentials=False,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    app.include_router(compare.router)
    app.include_router(datasets.router)
    app.include_router(experiments.router)
    app.include_router(health.router)
    app.include_router(images.router)
    app.include_router(import_.router)
    app.include_router(jobs.router)
    app.include_router(jobs.ws_router)
    app.include_router(reference_packs.router)
    app.include_router(splits.router)
    app.include_router(ws.router)
    return app
