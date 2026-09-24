"""What a run recorded about itself, what it is asked on demand, and the pixels it read."""

from __future__ import annotations

from fastapi import APIRouter, Query, Request, Response

from anomaly_lab.api.routers.experiments.views import (
    DiagnoseRequest,
    DiagnoseResponse,
    PayloadFormat,
    load,
)
from anomaly_lab.config import Settings
from anomaly_lab.experiments import service
from anomaly_lab.jobs.queue import JobQueue
from anomaly_lab.jobs.resident import ResidentWorker
from anomaly_lab.media.values import encode_plane
from anomaly_lab.models.diagnostics import (
    DiagnosticIndex,
    PruneResult,
    PruneScope,
    load_index,
)

router = APIRouter(prefix="/api/experiments", tags=["experiments"])


@router.get("/{experiment_id}/diagnostics", summary="What this run recorded about itself")
def get_diagnostics(request: Request, experiment_id: int) -> DiagnosticIndex:
    """The self-describing index a model wrote (ADR-0018).

    Returned verbatim. The UI renders by `kind` and never by method name, which is what
    makes a future method's diagnostics work here with no change.
    """
    experiment, settings = load(request, experiment_id)
    return load_index(settings.experiment_dir(experiment.id) / "diagnostics")


@router.post("/{experiment_id}/diagnose", summary="Record diagnostics for one image, now")
async def diagnose(request: Request, experiment_id: int, body: DiagnoseRequest) -> DiagnoseResponse:
    """Ask the method what it saw in one image, outside any job (ADR-0026).

    An inference run records per-image diagnostics for a bounded sample of what it scored,
    so most images have none. This answers for any image in the split, served from a
    resident worker that keeps the checkpoint loaded — the first request pays the model
    load, the rest do not.

    **It does not change this image's score, its map, or any metric.** Those come from a
    job and stay the run's (ADR-0011); what persists here is the diagnostics, marked
    `on_demand` in the index (handbook diagnostics.md).

    Refused with 409 while a job is running: one machine, one device, and a browse request
    must not queue behind a two-hour train.
    """
    settings: Settings = request.app.state.settings
    queue: JobQueue = request.app.state.job_queue
    resident: ResidentWorker = request.app.state.resident
    outcome = await service.diagnose(settings, queue, resident, experiment_id, body.image_id)
    return DiagnoseResponse(keys=outcome.keys, elapsed_ms=outcome.elapsed_ms, warm=outcome.warm)


@router.delete("/{experiment_id}/diagnostics", summary="Delete diagnostics to reclaim disk")
async def clear_diagnostics(
    request: Request,
    experiment_id: int,
    scope: PruneScope = Query(
        default=PruneScope.IMAGE,
        description=(
            "`image` drops every per-image entry and keeps the model-scoped ones; "
            "`on_demand` drops only what was asked for while browsing; `all` drops "
            "everything this run recorded about itself."
        ),
    ),
) -> PruneResult:
    """Remove stored diagnostics and report what it reclaimed (handbook diagnostics.md).

    **Anomaly maps are never touched.** They live in a sibling directory and each is
    referenced by an `ImageResult` row: deleting one orphans that row and silently breaks
    the overlay, `has_map` and the map scale. Reclaiming their space means deleting the
    experiment.

    Refused with 409 while a job is running, because an inference job's index write merges
    with what is on disk and would put back exactly what this removed. The resident worker
    can write the same file, so it is evicted first rather than raced — the delete waits
    for any request in flight, which is bounded and is the same guarantee the queue gets.
    """
    settings: Settings = request.app.state.settings
    queue: JobQueue = request.app.state.job_queue
    resident: ResidentWorker = request.app.state.resident
    return await service.clear_diagnostics(settings, queue, resident, experiment_id, scope)


@router.get(
    "/{experiment_id}/images/{image_id}/source-values",
    summary="The preprocessed pixels the model actually read",
    response_class=Response,
    responses={
        200: {"content": {"application/octet-stream": {}}},
        304: {"description": "The client's copy is current."},
    },
)
def read_source_values(request: Request, experiment_id: int, image_id: int) -> Response:
    """The pinned prepared pixels as float32, every colour plane (handbook diagnostics.md).

    **The preprocessed array, not the display tier.** A readout taken from the rendered
    preview would report what the browser is showing — 8-bit, resampled for display — when
    the question is what the *method* consumed. Every method loads its pixels through this
    same function, so this is exactly the number that went into the model.

    **Every plane in one response**, with the count in the header. How many there are is a
    property of the experiment's colour mode, and a client that had to know it in advance
    would either encode that in the UI or discover it by asking until something 404s.

    The planes are projected back through the pinned transform before encoding. They
    therefore share the source frame with the photograph and anomaly map; pixels outside
    the selected crop are NaN rather than invented model input.

    Cacheable forever in practice: the prepared artifact is immutable and the model-input
    config is frozen at creation, so the ETag can never go stale for one experiment. It is
    one fetch per image, ever.
    """
    experiment, settings = load(request, experiment_id)
    values = service.source_values(settings, experiment, image_id)
    if request.headers.get("if-none-match") == values.etag:
        return Response(status_code=304, headers=_diagnostic_headers(values.etag))

    return Response(
        content=encode_plane(values.load()),
        media_type="application/octet-stream",
        headers=_diagnostic_headers(values.etag),
    )


@router.get(
    "/{experiment_id}/diagnostics/payload",
    summary="One diagnostic array, rendered",
    response_class=Response,
    responses={
        200: {"content": {"image/png": {}, "application/octet-stream": {}}},
        304: {"description": "The client's copy is current."},
    },
)
def read_diagnostic_payload(
    request: Request,
    experiment_id: int,
    key: str = Query(description="The diagnostic's key, as the index reports it."),
    image_id: int | None = Query(
        default=None,
        description="For a per-image diagnostic. Omit for a run-scoped one.",
    ),
    frame: int = Query(default=0, ge=0, description="Which cell of a `grid` payload."),
    # Aliased because `format` is a builtin; the wire name is what a caller writes.
    payload_format: PayloadFormat = Query(
        default=PayloadFormat.PNG,
        alias="format",
        description=(
            "`png` to draw it; `raw` for the float32 values behind it (handbook diagnostics.md)."
        ),
    ),
) -> Response:
    """Render one stored diagnostic array as a PNG.

    **Addressed through the index, never through `entry.path`.** The client names a
    `(key, image_id)` pair and this resolves it against the index the model wrote; there
    is no request that can name a file. That is the same rule the image routes follow
    (§11), and it means path traversal is impossible by construction rather than by
    sanitising a query parameter after the fact.

    Colormapped kinds are stretched over the **run-wide** range the writer recorded, so
    every image's student-teacher error is drawn on one scale and two images can be
    compared by eye. An index written before ranges were recorded has none, and each
    array then falls back to its own extremes — visibly worse, and better than refusing.
    """
    experiment, settings = load(request, experiment_id)
    stored = service.stored_diagnostic(settings, experiment, key, image_id)

    etag = stored.etag(frame, payload_format.value)
    if etag is not None and request.headers.get("if-none-match") == etag:
        return Response(status_code=304, headers=_diagnostic_headers(etag))
    headers = {} if etag is None else _diagnostic_headers(etag)

    array = stored.load()
    if payload_format is PayloadFormat.RAW:
        # The same `(key, image_id)` resolution, so the per-branch panes inherit the hover
        # readout with no code written per method — which is what ADR-0018 is for.
        return Response(
            content=encode_plane(stored.raw_plane(array, frame)),
            media_type="application/octet-stream",
            headers=headers,
        )
    return Response(
        content=stored.render_png(array, frame), media_type="image/png", headers=headers
    )


def _diagnostic_headers(etag: str) -> dict[str, str]:
    # `no-cache` means "revalidate", not "do not store": the client keeps the bytes and
    # asks whether they are still current, which the ETag answers with a 304.
    return {"ETag": etag, "Cache-Control": "no-cache"}
