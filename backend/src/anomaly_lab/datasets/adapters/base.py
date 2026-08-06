"""The import adapter interface and registry (ADR-0006).

Adding support for a directory layout means adding an adapter and a registry entry —
never editing the importer. The shape deliberately mirrors the model plugin registry
(ADR-0007): a name, a pydantic options model the UI renders as a form from its JSON
Schema, and one method that does the work.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Protocol, runtime_checkable

from pydantic import BaseModel

from anomaly_lab.datasets.manifest import Manifest

ProgressCallback = Callable[[float, str | None], None]


def _ignore_progress(_fraction: float, _message: str | None) -> None:
    """Default for callers that are not running inside a job."""


@runtime_checkable
class ImportAdapter(Protocol):
    """Walks a directory tree and proposes a dataset, writing nothing."""

    @classmethod
    def name(cls) -> str:
        """The registry key, persisted on the dataset it produced."""
        ...

    @classmethod
    def summary(cls) -> str:
        """One line, shown next to the adapter in the import screen."""
        ...

    @classmethod
    def options_model(cls) -> type[BaseModel]:
        """The adapter's options. Its JSON Schema drives the UI form."""
        ...

    @classmethod
    def scan(
        cls,
        root: Path,
        options: BaseModel,
        *,
        dataset_name: str,
        progress: ProgressCallback | None = None,
    ) -> Manifest:
        """Produce a manifest. Must not touch the database or the source tree.

        `progress` is called as the walk advances and **may raise** to abort — that is
        how a cancelled job stops a scan without adapters needing to know jobs exist.
        """
        ...


class UnknownAdapterError(Exception):
    """The requested adapter name is not registered."""


_REGISTRY: dict[str, type[ImportAdapter]] = {}


def register(adapter: type[ImportAdapter]) -> type[ImportAdapter]:
    _REGISTRY[adapter.name()] = adapter
    return adapter


def get_adapter(name: str) -> type[ImportAdapter]:
    adapter = _REGISTRY.get(name)
    if adapter is None:
        known = ", ".join(sorted(_REGISTRY)) or "none"
        msg = f"unknown import adapter {name!r}; registered adapters: {known}"
        raise UnknownAdapterError(msg)
    return adapter


def registered_adapters() -> list[type[ImportAdapter]]:
    return [_REGISTRY[name] for name in sorted(_REGISTRY)]


def parse_options(adapter: type[ImportAdapter], options: dict[str, object] | None) -> BaseModel:
    """Validate raw options against the adapter's own model."""
    return adapter.options_model().model_validate(options or {})


def default_progress() -> ProgressCallback:
    return _ignore_progress
