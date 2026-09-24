"""Refusals a service can raise without knowing it is behind HTTP.

A service says *what kind* of refusal it is making — the thing does not exist, it is in a
state that forbids the request, the caller's copy is stale — and `detail` says it in words
a person can act on. The HTTP status is decided once, in `api/errors.py`, so no module
below the routers imports FastAPI to say "no".

The categories are deliberately few. A new one earns its place only when it needs a status
code none of these has; a new *reason* is a new `detail`, not a new class.
"""

from __future__ import annotations


class DomainError(Exception):
    """A request the domain refuses, with a reason the caller can read."""

    def __init__(self, detail: str) -> None:
        super().__init__(detail)
        self.detail = detail


class NotFoundError(DomainError):
    """The thing the request names does not exist, or not where the request looked."""


class ConflictError(DomainError):
    """The request is well formed but the current state refuses it.

    A job is running, a blocker stands, stored bytes no longer match their recorded
    identity, the dataset is edited in the other annotation scope.
    """


class StaleVersionError(DomainError):
    """The caller's version of a document is no longer the current one.

    Optimistic concurrency: the precondition the caller sent has failed.
    """


class InvalidInputError(DomainError):
    """The request's content cannot be accepted as it is.

    An unknown name, a configuration that fails its schema, geometry in the wrong frame.
    """


class UnsupportedRequestError(DomainError):
    """The request asks for something this kind of resource never offers."""


class GoneError(DomainError):
    """The thing is recorded but its bytes are no longer readable.

    Artifacts and source files are deletable by design, so this is an expected state
    rather than corruption.
    """


class UnavailableError(DomainError):
    """The request was valid, but the process that answers it did not survive."""
