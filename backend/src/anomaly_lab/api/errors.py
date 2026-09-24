"""The one place a domain refusal becomes an HTTP status.

Services raise `anomaly_lab.errors.DomainError` subclasses. This module maps each to a
status code and renders it exactly the way FastAPI renders an `HTTPException` —
`{"detail": ...}` — so a client cannot tell which layer refused, and moving a refusal from
a router into a service changes nothing on the wire.

HTTP-only concerns stay in the routers as `HTTPException`: a missing `If-Match` header
(428) is a fact about the request's headers, not about the domain.
"""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from anomaly_lab.errors import (
    ConflictError,
    DomainError,
    GoneError,
    InvalidInputError,
    NotFoundError,
    StaleVersionError,
    UnavailableError,
    UnsupportedRequestError,
)

STATUS_BY_ERROR: dict[type, int] = {
    UnsupportedRequestError: 400,
    NotFoundError: 404,
    ConflictError: 409,
    GoneError: 410,
    StaleVersionError: 412,
    InvalidInputError: 422,
    UnavailableError: 503,
}

# A `DomainError` raised directly, or a subclass nobody mapped, is still a refusal of the
# request rather than a server fault.
DEFAULT_STATUS = 400


def status_for(error: DomainError) -> int:
    """The status of the nearest mapped class in `error`'s MRO."""
    for cls in type(error).__mro__:
        if cls in STATUS_BY_ERROR:
            return STATUS_BY_ERROR[cls]
    return DEFAULT_STATUS


async def _render_domain_error(_request: Request, exc: Exception) -> JSONResponse:
    if not isinstance(exc, DomainError):  # pragma: no cover - registered for DomainError only
        raise exc
    return JSONResponse({"detail": exc.detail}, status_code=status_for(exc))


def install_error_handlers(app: FastAPI) -> None:
    app.add_exception_handler(DomainError, _render_domain_error)
