"""The `infer` job handler: score a split's images and evaluate the result.

Kept separate from `train` rather than chained onto it, because they fail for different
reasons and are re-run for different reasons. Re-scoring a trained model after changing
the preprocessing is nonsense; re-scoring it over a subset it has not seen is routine.

Inference ends by running the evaluation layer, so a finished job means finished results.
Evaluation is cheap, model-independent, and re-runnable on its own (ADR-0011).
"""

from __future__ import annotations

import json
import time
from typing import Any

from pydantic import BaseModel, Field

from anomaly_lab.db.connection import connection
from anomaly_lab.db.repositories import experiments as experiments_repo
from anomaly_lab.db.repositories import images as images_repo
from anomaly_lab.db.repositories import results as results_repo
from anomaly_lab.domain.entities import ExperimentStatus, ImageResult, Subset
from anomaly_lab.eval.runner import evaluate_and_store
from anomaly_lab.experiments.context import (
    ExperimentJobError,
    diagnostics_writer,
    load_experiment,
    to_records,
)
from anomaly_lab.experiments.train import MODEL_SUBDIR
from anomaly_lab.jobs.context import JobCancelledError, JobContext
from anomaly_lab.models.base import InferContext, ModelCancelledError
from anomaly_lab.schemas import API_MODEL_CONFIG

RANGE_FILENAME = "range.json"


class InferParams(BaseModel):
    model_config = API_MODEL_CONFIG

    experiment_id: int
    subsets: list[Subset] = Field(
        default_factory=lambda: [Subset.VAL, Subset.TEST],
        description=(
            "Which subsets to score. Train is excluded by default because scoring the "
            "images a model was fitted on costs the most time and tells you the least."
        ),
    )
    diagnostics: bool = Field(
        default=True,
        description="Record per-image diagnostics, spread evenly across the scored set.",
    )
    diagnostic_images: int = Field(
        # `default_factory`, like `subsets` above, so the JSON Schema carries no `default`
        # and the field is *optional* in the generated client. A literal default is emitted
        # into the schema, which makes the property required in TypeScript and forces every
        # caller to restate the number — which is how the frontend came to pin 12 and
        # silently override this line.
        default_factory=lambda: 64,
        ge=0,
        le=500,
        description=(
            "How many images keep per-image diagnostics, chosen evenly across the run. "
            "Each costs a few float32 maps on disk — measured at about half a megabyte "
            "for a two-branch method at 256x256."
        ),
    )
    """
    Raised from 12, which was too tight to be useful and was justified by an estimate
    rather than a measurement.

    Measured on the reference run: two branch maps per image at 256x256 float32 is 512 KB,
    so 64 images is ~33 MB against the 127 MB of combined anomaly maps the same run already
    writes. The default answer to "show me what the branches did on this image" was "that
    image was not one of the first twelve", which made the decomposition — the thing that
    makes a two-branch method legible at all — effectively unreachable.

    Still bounded, and still reported. A run over a few thousand images at the ceiling is
    real disk, and the index says what it dropped.
    """


def run_infer_job(ctx: JobContext) -> dict[str, Any]:
    """Score images with a trained experiment, persist results, and evaluate."""
    params = InferParams.model_validate(dict(ctx.params))

    with connection(ctx.settings.db_path) as conn:
        loaded = load_experiment(conn, ctx.settings, params.experiment_id)
        experiment = loaded.experiment
        selected = images_repo.list_images_for_split(
            conn, experiment.split_id, subsets=params.subsets, channels=experiment.channels
        )

    if experiment.status is not ExperimentStatus.TRAINED:
        capabilities = loaded.model_class.capabilities()
        if capabilities.requires_training:
            msg = (
                f"experiment {experiment.id} is {experiment.status.value}, not trained. "
                f"{experiment.model_type} needs a train job to run first."
            )
            raise ExperimentJobError(msg)

    if not selected:
        names = ", ".join(subset.value for subset in params.subsets)
        msg = (
            f"split {experiment.split_id} has no samples in subset(s) {names}. "
            "An imported split may legitimately have no val subset — choose test."
        )
        raise ExperimentJobError(msg)

    model_dir = loaded.artifact_dir / MODEL_SUBDIR
    if model_dir.is_dir():
        loaded.model.load(model_dir)
    elif loaded.model_class.capabilities().requires_training:
        msg = f"experiment {experiment.id} has no saved model at {model_dir}"
        raise ExperimentJobError(msg)

    ctx.log(f"scoring {len(selected)} image(s) on device {loaded.device.device.value}")
    ctx.log(f"device: {loaded.device.reason}")

    writer = diagnostics_writer(
        loaded,
        enabled=params.diagnostics,
        image_budget=params.diagnostic_images,
        # The handler knows the whole scored set before the model sees any of it, so the
        # budget is spent across the run rather than on whichever images arrive first.
        over_images=[image.image_id for image in selected],
    )
    infer_ctx = InferContext(
        artifact_dir=loaded.artifact_dir,
        cache_dir=loaded.cache_dir,
        preprocessing=loaded.preprocessing,
        device=loaded.device.device,
        reporter=ctx,
        diagnostics=writer,
        map_projector=lambda image_id, values: loaded.region_build.transform_for(
            image_id
        ).project_map(values),
    )

    started = time.perf_counter()
    try:
        predictions = loaded.model.predict(to_records(selected, loaded.region_build), infer_ctx)
    except ModelCancelledError as exc:
        raise JobCancelledError from exc
    elapsed = time.perf_counter() - started

    if len(predictions) != len(selected):
        msg = (
            f"{experiment.model_type} returned {len(predictions)} prediction(s) for "
            f"{len(selected)} image(s); a plugin must return one per input, in order"
        )
        raise ExperimentJobError(msg)

    index = writer.flush()
    if index.truncated_images:
        ctx.log(
            f"per-image diagnostics kept for {params.diagnostic_images} image(s); "
            f"{index.truncated_images} more were scored without them",
        )

    display_range = infer_ctx.display_range()
    if display_range is not None:
        low, high = display_range
        (loaded.artifact_dir / "maps").mkdir(parents=True, exist_ok=True)
        (loaded.artifact_dir / "maps" / RANGE_FILENAME).write_text(
            json.dumps({"low": low, "high": high}, indent=2),
            encoding="utf-8",
        )

    image_rows = [
        ImageResult(
            experiment_id=experiment.id,
            image_id=prediction.image_id,
            score=prediction.score,
            map_path=str(prediction.anomaly_map) if prediction.anomaly_map else None,
            inference_ms=prediction.inference_ms,
        )
        for prediction in predictions
    ]

    with connection(ctx.settings.db_path) as conn:
        results_repo.replace_image_results(conn, experiment.id, image_rows)

        # `evaluate_and_store` re-derives the sample scores itself, so this handler stores
        # what the model produced and nothing else. That split is what makes `reevaluate`
        # able to apply a changed `eval_config` without re-running inference.
        ctx.progress(0.95, "computing metrics")
        metrics = evaluate_and_store(conn, experiment)
        sample_count = len(results_repo.list_scored_samples(conn, experiment.id))
        experiments_repo.set_status(conn, experiment.id, ExperimentStatus.TRAINED)

    headline = {subset.value: found.get("sample_roc_auc") for subset, found in metrics.items()}
    ctx.log(f"sample ROC-AUC by subset: {headline}")
    ctx.progress(1.0, "scored and evaluated")

    return {
        "experiment_id": experiment.id,
        "images": len(image_rows),
        "samples": sample_count,
        "subsets": [subset.value for subset in params.subsets],
        "maps": sum(1 for row in image_rows if row.map_path),
        "inference_seconds": elapsed,
        "metrics": {subset.value: found for subset, found in metrics.items()},
        "diagnostics": len(index.entries),
    }
