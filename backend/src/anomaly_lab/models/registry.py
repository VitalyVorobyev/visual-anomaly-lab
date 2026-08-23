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


def _efficientad_custom() -> type[AnomalyModel]:
    from anomaly_lab.models.efficientad_custom import EfficientAdCustomModel

    return EfficientAdCustomModel


def _patchcore_anomalib() -> type[AnomalyModel]:
    from anomaly_lab.models.patchcore_anomalib import PatchcoreAnomalibModel

    return PatchcoreAnomalibModel


def _dinomaly_anomalib() -> type[AnomalyModel]:
    from anomaly_lab.models.dinomaly_anomalib import DinomalyAnomalibModel

    return DinomalyAnomalibModel


def _glass_anomalib() -> type[AnomalyModel]:
    from anomaly_lab.models.glass_anomalib import GlassAnomalibModel

    return GlassAnomalibModel


def _dino_memory() -> type[AnomalyModel]:
    from anomaly_lab.models.dino_memory import DinoMemoryModel

    return DinoMemoryModel


# `efficientad_custom` cost exactly one entry and one module in M6 — no route, no schema, no
# line of TypeScript — which is the prediction ADR-0007 made. It started as a second
# implementation measured against the anomalib-wrapped `efficientad_anomalib`, which has
# since been retired now that the in-house implementation is the one the workbench carries
# forward (ADR-0008, ADR-0029). `patchcore_anomalib` cost the same in M7, and it is the
# stronger test of the two: PatchCore trains nothing and holds a memory bank instead of
# weights. Dinomaly adds reconstruction training and exact continuation without changing
# the boundary. GLASS adds learned anomaly synthesis and a bounded reference-frame pass
# under that same contract: each method still costs one module and one entry here.
# `dino_memory` is the first in-house method built on the shared frozen-encoder table, and it
# holds three different memories behind one `scoring` axis — a coreset bank, a per-position
# bank and a per-position Gaussian — which still cost one module and this one line.
LOADERS: dict[str, Callable[[], type[AnomalyModel]]] = {
    "pixel_reference": _pixel_reference,
    "efficientad_custom": _efficientad_custom,
    "patchcore_anomalib": _patchcore_anomalib,
    "dinomaly_anomalib": _dinomaly_anomalib,
    "glass_anomalib": _glass_anomalib,
    "dino_memory": _dino_memory,
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
