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


def _glass_anomalib() -> type[AnomalyModel]:
    from anomaly_lab.models.glass_anomalib import GlassAnomalibModel

    return GlassAnomalibModel


def _dino_memory() -> type[AnomalyModel]:
    from anomaly_lab.models.dino_memory import DinoMemoryModel

    return DinoMemoryModel


def _dinomaly_custom() -> type[AnomalyModel]:
    from anomaly_lab.models.dinomaly_custom import DinomalyCustomModel

    return DinomalyCustomModel


def _color_prototype() -> type[AnomalyModel]:
    from anomaly_lab.models.color_prototype import ColorPrototypeModel

    return ColorPrototypeModel


def _color_classifier() -> type[AnomalyModel]:
    from anomaly_lab.models.color_classifier import ColorClassifierModel

    return ColorClassifierModel


def _fss_dino() -> type[AnomalyModel]:
    from anomaly_lab.models.fss_dino import FssDinoModel

    return FssDinoModel


def _proto_seg() -> type[AnomalyModel]:
    from anomaly_lab.models.proto_seg import ProtoSegModel

    return ProtoSegModel


def _dino_linear_seg() -> type[AnomalyModel]:
    from anomaly_lab.models.dino_linear_seg import DinoLinearSegModel

    return DinoLinearSegModel


def _subspace_ad() -> type[AnomalyModel]:
    from anomaly_lab.models.subspace_ad import SubspaceAdModel

    return SubspaceAdModel


# `efficientad_custom` cost exactly one entry and one module in M6 — no route, no schema, no
# line of TypeScript — which is the prediction ADR-0007 made. It started as a second
# implementation measured against the anomalib-wrapped `efficientad_anomalib`, which has
# since been retired now that the in-house implementation is the one the workbench carries
# forward (ADR-0008, ADR-0029). `patchcore_anomalib` cost the same in M7, and it is the
# stronger test of the two: PatchCore trains nothing and holds a memory bank instead of
# weights. `dinomaly_custom` adds reconstruction training and exact continuation without
# changing the boundary — it started as a second implementation measured against the
# anomalib-wrapped `dinomaly_anomalib`, which has since been retired the same way
# `efficientad_anomalib` was, now that the in-house implementation reached VisA parity
# (ADR-0008, ADR-0029) and is the one the workbench carries forward: a configurable encoder
# and a configurable decoder depth, neither of which the wrapper could offer. GLASS adds
# learned anomaly synthesis and a bounded reference-frame pass under that same contract: each
# method still costs one module and one entry here.
# `dino_memory` is the first in-house method built on the shared frozen-encoder table, and it
# holds three different memories behind one `scoring` axis — a coreset bank, a per-position
# bank and a per-position Gaussian — which still cost one module and this one line.
# `subspace_ad` is the first method whose defaults were *measured* rather than chosen: a sweep
# outside the application picked its backbone, layer window and two thresholds, and what
# shipped was one module, this one line, and an entry in the measurements record (ADR-0038).
# It trains nothing at all — the fit is a covariance and an eigendecomposition — which makes
# it the cheapest strong baseline in the table and the one to run first on a new dataset.
# `color_prototype` is the first method of another task (few-shot segmentation, ADR-0040): it
# fits on references' masks through `TrainContext.targets`, and still cost one module and
# this one line. `fss_dino` reproduces a published few-shot baseline on the shared frozen-DINO
# blocks, and writes its own mask beside its map; the same cost. `proto_seg` is ours on those
# blocks, with debiasing, the bank, the adaptation and the refinement each a field.
# `color_classifier` is the first method of supervised segmentation (ADR-0039): it fits on
# label maps through `TrainContext.label_targets` and writes label maps through
# `InferContext.write_label_map`, and still cost one module and this one line.
# `dino_linear_seg` is its first deep method, a softmax head on the shared frozen-DINO blocks
# trained at sampled pixels, and it needed nothing the floor had not already put in place.
LOADERS: dict[str, Callable[[], type[AnomalyModel]]] = {
    "pixel_reference": _pixel_reference,
    "efficientad_custom": _efficientad_custom,
    "patchcore_anomalib": _patchcore_anomalib,
    "dinomaly_custom": _dinomaly_custom,
    "glass_anomalib": _glass_anomalib,
    "dino_memory": _dino_memory,
    "subspace_ad": _subspace_ad,
    "color_prototype": _color_prototype,
    "fss_dino": _fss_dino,
    "proto_seg": _proto_seg,
    "color_classifier": _color_classifier,
    "dino_linear_seg": _dino_linear_seg,
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
