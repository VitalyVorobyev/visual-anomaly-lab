"""Generic, atomic, parity-checked portable export job (ADR-0034)."""

from __future__ import annotations

import hashlib
import importlib
import shutil
import sqlite3
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel

from anomaly_lab.db.connection import connection
from anomaly_lab.db.repositories import results as results_repo
from anomaly_lab.db.repositories.results import ScoredSample
from anomaly_lab.deployment.protocol import SupportsOnnxExport
from anomaly_lab.deployment.schema import (
    CoordinateFrame,
    DeploymentManifest,
    FileDigest,
    MapOutputContract,
    OperatingPointContract,
    ParityFixture,
    PercentileReducer,
    PixelInputContract,
    RegionContract,
    SourceExperiment,
    TensorDtype,
    TensorLayout,
    TensorSpec,
)
from anomaly_lab.domain.entities import ExperimentStatus, Subset
from anomaly_lab.eval.threshold import suggest_threshold
from anomaly_lab.experiments.context import ExperimentJobError, load_experiment
from anomaly_lab.experiments.train import MODEL_SUBDIR
from anomaly_lab.jobs.context import JobContext
from anomaly_lab.models.base import PortableFormat
from anomaly_lab.schemas import API_MODEL_CONFIG

GRAPH_FILENAME = "model.onnx"
MANIFEST_FILENAME = "manifest.json"
EXPORTS_SUBDIR = "exports"
FIXTURE_INPUT = "fixture/input.f32"
FIXTURE_MAP = "fixture/expected-map.f32"


class ExportParams(BaseModel):
    model_config = API_MODEL_CONFIG

    experiment_id: int
    format: PortableFormat = PortableFormat.ONNX


def run_export_job(ctx: JobContext) -> dict[str, Any]:
    """Export one fitted experiment, publishing nothing until parity succeeds."""
    params = ExportParams.model_validate(dict(ctx.params))
    with connection(ctx.settings.db_path) as conn:
        loaded = load_experiment(conn, ctx.settings, params.experiment_id)
        operating_subset, scored = _operating_point_samples(conn, params.experiment_id)

    capabilities = loaded.model_class.capabilities()
    if params.format not in capabilities.portable_formats:
        raise ExperimentJobError(
            f"method {loaded.experiment.model_type} does not support {params.format.value} export"
        )
    if loaded.experiment.status is not ExperimentStatus.TRAINED:
        raise ExperimentJobError(
            f"experiment {params.experiment_id} is {loaded.experiment.status.value}; train it first"
        )
    if not isinstance(loaded.model, SupportsOnnxExport):
        raise ExperimentJobError(
            f"method {loaded.experiment.model_type} declares ONNX export but does not implement it"
        )

    model_dir = loaded.artifact_dir / MODEL_SUBDIR
    if not model_dir.is_dir():
        raise ExperimentJobError("the experiment has no stored fitted model")
    loaded.model.load(model_dir)

    exports_dir = loaded.artifact_dir / EXPORTS_SUBDIR
    exports_dir.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".export-", dir=exports_dir))
    published = exports_dir / (
        f"onnx-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-job-{ctx.job_id}"
    )
    try:
        ctx.progress(0.1, "writing ONNX graph")
        graph = loaded.model.export_onnx(staging / GRAPH_FILENAME, loaded.preprocessing)
        ctx.raise_if_cancelled()

        fixture = _fixture(
            loaded.preprocessing.channels,
            loaded.preprocessing.height,
            loaded.preprocessing.width,
        )
        expected_map, expected_score = loaded.model.portable_reference(fixture)
        (staging / FIXTURE_INPUT).parent.mkdir(parents=True, exist_ok=True)
        (staging / FIXTURE_INPUT).write_bytes(np.asarray(fixture, dtype="<f4").tobytes(order="C"))
        (staging / FIXTURE_MAP).write_bytes(
            np.asarray(expected_map, dtype="<f4").tobytes(order="C")
        )

        ctx.progress(0.45, "checking portable-runtime parity")
        actual_map = _run_onnx(staging / GRAPH_FILENAME, graph.input_name, fixture)
        actual_score = float(np.percentile(actual_map, graph.score_percentile))
        absolute_tolerance = 2e-5
        relative_tolerance = 2e-5
        max_absolute_error = float(np.max(np.abs(actual_map - expected_map)))
        score_absolute_error = abs(actual_score - expected_score)
        if (
            not np.allclose(
                actual_map,
                expected_map,
                atol=absolute_tolerance,
                rtol=relative_tolerance,
            )
            or score_absolute_error > absolute_tolerance
        ):
            raise ExperimentJobError(
                "ONNX parity failed: "
                f"map max abs error {max_absolute_error:.8g}, "
                f"score abs error {score_absolute_error:.8g}"
            )

        operating_point = None
        if scored:
            value, rationale = suggest_threshold(scored)
            operating_point = OperatingPointContract(
                rule=rationale,
                value=value,
                subset=operating_subset.value if operating_subset is not None else None,
            )

        files = [_digest(staging, path) for path in (GRAPH_FILENAME, FIXTURE_INPUT, FIXTURE_MAP)]
        manifest = DeploymentManifest(
            created_at=datetime.now(UTC).isoformat(timespec="seconds"),
            source=SourceExperiment(
                id=loaded.experiment.id,
                model_type=loaded.experiment.model_type,
                method_config=loaded.experiment.model_config_,
                preprocessing=loaded.experiment.preprocessing_config,
            ),
            graph_path=GRAPH_FILENAME,
            opset=graph.opset,
            input=PixelInputContract(
                color=loaded.preprocessing.color,
                width=loaded.preprocessing.width,
                height=loaded.preprocessing.height,
                tensor=TensorSpec(
                    name=graph.input_name,
                    dtype=TensorDtype.FLOAT32,
                    layout=TensorLayout.NCHW,
                    shape=list(fixture.shape),
                ),
            ),
            anomaly_map=MapOutputContract(
                coordinate_frame=CoordinateFrame.PREPARED,
                tensor=TensorSpec(
                    name=graph.output_name,
                    dtype=TensorDtype.FLOAT32,
                    layout=TensorLayout.NCHW,
                    shape=[1, 1, loaded.preprocessing.height, loaded.preprocessing.width],
                ),
            ),
            score=PercentileReducer(percentile=graph.score_percentile),
            operating_point=operating_point,
            region=RegionContract(
                profile_revision_id=loaded.experiment.region_profile_id,
                manifest_sha256=loaded.experiment.region_manifest_sha256,
            ),
            files=files,
            parity=ParityFixture(
                input_path=FIXTURE_INPUT,
                expected_map_path=FIXTURE_MAP,
                expected_score=expected_score,
                absolute_tolerance=absolute_tolerance,
                relative_tolerance=relative_tolerance,
                max_absolute_error=max_absolute_error,
                score_absolute_error=score_absolute_error,
            ),
        )
        (staging / MANIFEST_FILENAME).write_text(
            manifest.model_dump_json(indent=2), encoding="utf-8"
        )
        ctx.raise_if_cancelled()
        staging.replace(published)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    ctx.progress(1.0, "portable bundle verified")
    return {
        "experiment_id": loaded.experiment.id,
        "format": params.format.value,
        "bundle_path": str(published),
        "manifest_path": str(published / MANIFEST_FILENAME),
        "files": len(files) + 1,
        "bytes": sum(item.bytes for item in files) + (published / MANIFEST_FILENAME).stat().st_size,
        "map_max_absolute_error": max_absolute_error,
        "score_absolute_error": score_absolute_error,
    }


def _operating_point_samples(
    conn: sqlite3.Connection,
    experiment_id: int,
) -> tuple[Subset | None, list[ScoredSample]]:
    """Choose one named evaluation subset; never calibrate on a silent mixture.

    Test is the workbench's canonical reported subset. Validation is the next-best
    deployment calibration source, and train is an explicit last resort. Results with
    no split assignment remain usable for legacy/import edge cases, but the manifest
    records that their subset is unknown.
    """
    for subset in (Subset.TEST, Subset.VAL, Subset.TRAIN):
        scored = results_repo.list_scored_samples(conn, experiment_id, subset=subset)
        if scored:
            return subset, scored
    return None, results_repo.list_scored_samples(conn, experiment_id)


def _fixture(channels: int, height: int, width: int) -> np.ndarray:
    """Deterministic, non-dataset parity input spanning the legal pixel range."""
    y = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
    x = np.linspace(0.0, 1.0, width, dtype=np.float32)[None, :]
    planes = [np.mod(x * (index + 1) + y * (index + 2), 1.0) for index in range(channels)]
    return np.ascontiguousarray(np.stack(planes, axis=0)[np.newaxis], dtype=np.float32)


def _run_onnx(graph: Path, input_name: str, fixture: np.ndarray) -> np.ndarray:
    ort: Any = importlib.import_module("onnxruntime")

    session = ort.InferenceSession(str(graph), providers=["CPUExecutionProvider"])
    output: Any = session.run(None, {input_name: fixture})[0]
    result = np.asarray(output, dtype=np.float32).squeeze()
    if result.ndim != 2:
        raise ExperimentJobError(f"exported anomaly map has invalid shape {np.shape(output)}")
    return result


def _digest(root: Path, relative: str) -> FileDigest:
    path = root / relative
    content = path.read_bytes()
    return FileDigest(
        path=relative,
        bytes=len(content),
        sha256=hashlib.sha256(content).hexdigest(),
    )
