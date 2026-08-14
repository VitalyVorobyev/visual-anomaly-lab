"""Diagnosing one image on request, inside the resident worker (ADR-0026).

The third layer of the resident, mirroring `experiments/infer.py`: the manager owns the
process, `jobs/inspector.py` is the entrypoint, and this is the work. It is the only
module of the three that knows what a request means, so the process machinery above it
stays a transport.

**This scores an image and records nothing about the score.** `ImageResult`, the sample
aggregation and every metric come from an inference job, and a browse request that quietly
updated one row would put a number on the results screen that no run produced (ADR-0011).
What survives a request is the diagnostics the model emitted, in the index, marked
`on_demand` (ADR-0027).
"""

from __future__ import annotations

import shutil
from pathlib import Path

from anomaly_lab.config import Settings
from anomaly_lab.db.connection import connection
from anomaly_lab.db.repositories import images as images_repo
from anomaly_lab.domain.entities import ExperimentStatus, Subset
from anomaly_lab.experiments.context import LoadedExperiment, load_experiment, to_records
from anomaly_lab.experiments.train import MODEL_SUBDIR
from anomaly_lab.models.base import InferContext, NullReporter
from anomaly_lab.models.diagnostics import DiagnosticOrigin, DiagnosticWriter

# Not `maps`. The whole reason `InferContext.maps_subdir` exists (ADR-0026).
SCRATCH_MAPS_SUBDIR = "scratch-maps"


class DiagnoseError(Exception):
    """The request cannot be served, for a reason the caller can act on."""


def prepare(settings: Settings, experiment_id: int) -> LoadedExperiment:
    """Build the plugin and load its checkpoint, once per resident.

    The expensive half — the model load is what makes the first request take seconds and
    every later one fast — and separated for that reason: a failure here is a failure to
    start, which the manager reports differently from a failure to answer.
    """
    with connection(settings.db_path) as conn:
        loaded = load_experiment(conn, settings, experiment_id)

    capabilities = loaded.model_class.capabilities()
    if not capabilities.produces_diagnostics:
        msg = f"{loaded.experiment.model_type} records no diagnostics"
        raise DiagnoseError(msg)

    model_dir = loaded.artifact_dir / MODEL_SUBDIR
    if model_dir.is_dir():
        loaded.model.load(model_dir)
    elif capabilities.requires_training:
        msg = (
            f"experiment {experiment_id} has no saved model; train it before asking about an image"
        )
        raise DiagnoseError(msg)
    elif loaded.experiment.status is not ExperimentStatus.TRAINED:
        msg = f"experiment {experiment_id} is {loaded.experiment.status.value}, not trained"
        raise DiagnoseError(msg)

    return loaded


def diagnose_image(loaded: LoadedExperiment, settings: Settings, image_id: int) -> list[str]:
    """Score one image for its diagnostics alone; return the keys that were recorded.

    The record handed to the model is built by the same query a run uses, which does two
    things at once: the model sees exactly what it would have seen during inference —
    including the channel name — and an image outside this experiment's split cannot be
    asked about at all, because it is not in the answer.
    """
    with connection(settings.db_path) as conn:
        selected = images_repo.list_images_for_split(
            conn,
            loaded.experiment.split_id,
            subsets=list(Subset),
            channels=loaded.experiment.channels,
        )

    found = [image for image in selected if image.image_id == image_id]
    if not found:
        msg = f"image {image_id} is not in this experiment's split"
        raise DiagnoseError(msg)

    writer = DiagnosticWriter(
        loaded.artifact_dir / "diagnostics",
        # No budget: the caller asked about this image by name. A cap here would be a
        # silent refusal of an explicit request, which is the opposite of what the run's
        # budget note exists to make visible.
        origin=DiagnosticOrigin.ON_DEMAND,
    )
    ctx = InferContext(
        artifact_dir=loaded.artifact_dir,
        cache_dir=loaded.cache_dir,
        preprocessing=loaded.preprocessing,
        device=loaded.device.device,
        # No reporter: there is no job row, no log file and no socket to fan out to.
        reporter=NullReporter(),
        diagnostics=writer,
        maps_subdir=SCRATCH_MAPS_SUBDIR,
        map_projector=lambda target_id, values: loaded.region_build.transform_for(
            target_id
        ).project_map(values),
    )

    try:
        loaded.model.predict(to_records(found, loaded.region_build), ctx)
        index = writer.flush()
    finally:
        # `predict` writes a map whether or not anybody wants it. Removed rather than
        # kept: maps nothing references would grow with every browsed image and show up in
        # the artifact listing as though a run had produced them.
        _discard(loaded.artifact_dir / SCRATCH_MAPS_SUBDIR)

    return sorted(
        {
            entry.key
            for entry in index.entries
            if entry.image_id == image_id and entry.origin is DiagnosticOrigin.ON_DEMAND
        }
    )


def _discard(path: Path) -> None:
    shutil.rmtree(path, ignore_errors=True)
