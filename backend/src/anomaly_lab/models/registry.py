"""The method registry (ADR-0007).

An explicit table of lazy loaders, for the same two reasons as `jobs/handlers.py`:
reading this file tells you every method the application has, and the API process imports
a method's dependencies only when it is about to build one — opening the method picker
should not cost three seconds of torch.

That laziness is only real if each plugin module keeps its heavy imports inside its
functions. `describe_all` imports every registered module to read its config schema and
capabilities, which are pure pydantic; a plugin that imports torch at module scope
silently breaks the property for everyone.
"""

from __future__ import annotations

from collections.abc import Callable

from anomaly_lab.models.base import AnomalyModel, ModelDescription


class UnknownModelError(Exception):
    """No method is registered under this key — a stale experiment or a typo."""


def _pixel_reference() -> type[AnomalyModel]:
    from anomaly_lab.models.pixel_reference import PixelReferenceModel

    return PixelReferenceModel


def _efficientad_anomalib() -> type[AnomalyModel]:
    from anomaly_lab.models.efficientad_anomalib import EfficientAdAnomalibModel

    return EfficientAdAnomalibModel


def _efficientad_custom() -> type[AnomalyModel]:
    from anomaly_lab.models.efficientad_custom import EfficientAdCustomModel

    return EfficientAdCustomModel


# `patchcore_anomalib` (M7) and `classical_circular` (optional M8) each cost exactly one
# more entry here and one more module. `efficientad_custom` cost exactly that in M6 — no
# route, no schema, no line of TypeScript — which is the prediction ADR-0007 made.
LOADERS: dict[str, Callable[[], type[AnomalyModel]]] = {
    "pixel_reference": _pixel_reference,
    "efficientad_anomalib": _efficientad_anomalib,
    "efficientad_custom": _efficientad_custom,
}


def registered_keys() -> tuple[str, ...]:
    return tuple(LOADERS)


def get_model_class(key: str) -> type[AnomalyModel]:
    loader = LOADERS.get(key)
    if loader is None:
        known = ", ".join(sorted(LOADERS))
        msg = f"no method is registered under {key!r}; known methods are {known}"
        raise UnknownModelError(msg)
    return loader()


def describe(key: str) -> ModelDescription:
    """Everything the method picker shows for one method."""
    model_class = get_model_class(key)
    return ModelDescription(
        key=key,
        title=model_class.title or key,
        summary=model_class.summary,
        capabilities=model_class.capabilities(),
        availability=model_class.availability(),
        config_schema=model_class.config_model().model_json_schema(),
    )


def describe_all() -> list[ModelDescription]:
    """Every method, in registration order — cheapest and most useful listed first."""
    return [describe(key) for key in LOADERS]
