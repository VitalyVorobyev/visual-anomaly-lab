"""Domain refusals reach the wire exactly as an `HTTPException` with the same code would."""

from __future__ import annotations

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from anomaly_lab.api.errors import DEFAULT_STATUS, install_error_handlers, status_for
from anomaly_lab.db.repositories.annotations import GroundTruthDriftError
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


@pytest.mark.parametrize(
    ("error", "status"),
    [
        (UnsupportedRequestError("x"), 400),
        (NotFoundError("x"), 404),
        (ConflictError("x"), 409),
        (GoneError("x"), 410),
        (StaleVersionError("x"), 412),
        (InvalidInputError("x"), 422),
        (UnavailableError("x"), 503),
        (DomainError("x"), DEFAULT_STATUS),
    ],
)
def test_each_category_has_one_status(error: DomainError, status: int) -> None:
    assert status_for(error) == status


def test_a_subclass_takes_its_nearest_mapped_ancestor() -> None:
    class DraftMissingError(NotFoundError):
        pass

    assert status_for(DraftMissingError("x")) == 404
    # An existing domain exception joins a category by inheriting from it.
    assert status_for(GroundTruthDriftError("drifted")) == 409
    assert isinstance(GroundTruthDriftError("drifted"), RuntimeError)


def _app() -> FastAPI:
    app = FastAPI()
    install_error_handlers(app)

    @app.get("/domain/{code}")
    def domain(code: int) -> None:
        by_code = {404: NotFoundError, 409: ConflictError, 412: StaleVersionError}
        raise by_code[code](f"refused with {code}")

    @app.get("/http/{code}")
    def http(code: int) -> None:
        raise HTTPException(status_code=code, detail=f"refused with {code}")

    return app


@pytest.mark.parametrize("code", [404, 409, 412])
def test_the_response_is_indistinguishable_from_an_http_exception(code: int) -> None:
    """Moving a refusal from a router into a service must change nothing a client sees."""
    with TestClient(_app()) as client:
        from_domain = client.get(f"/domain/{code}")
        from_http = client.get(f"/http/{code}")

    assert from_domain.status_code == from_http.status_code == code
    assert from_domain.json() == from_http.json() == {"detail": f"refused with {code}"}
    assert from_domain.headers["content-type"] == from_http.headers["content-type"]
