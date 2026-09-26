#!/usr/bin/env -S uv run --project backend --extra dl python
"""Python-versus-portable parity for one method, on real prepared pixels after a real fit.

The export job already refuses a bundle whose graph disagrees with the method on a
dataset-free fixture. That fixture is a smooth ramp: it proves the graph runs and computes
the method's arithmetic, not that it does so on the activations a trained network sees on
real parts. This gate is that second claim, and it is what a method must pass before its
`Capabilities.portable_formats` lists a format (ADR-0034, docs/measurements.md).

Method-agnostic: any key in `m11_public_gate.CANDIDATES` whose plugin declares ONNX.

    ./scripts/export-parity-gate.py --data-dir /tmp/export-parity --candidate dinomaly_custom

For each VisA class: import the official 1cls split, build an identity profile at the
candidate's prepared size, fit and score the test subset through the application's own
train and infer jobs, export through the ordinary export job, then hand every test image's
prepared tensor to both the method's `portable_reference` (torch, CPU) and the bundle's graph
(ONNX Runtime, CPU), reading the score through the manifest's contract exactly as a host
would. Source images stay read-only under ``/datasets``.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from m11_public_gate import (
    CANDIDATES,
    REPOSITORY,
    SEED,
    _create_experiment,
    _empty_destination,
    _identity_profile,
    _official_dataset,
    _packages,
)

from anomaly_lab.config import Settings
from anomaly_lab.db.connection import connection
from anomaly_lab.db.migrate import apply_schema
from anomaly_lab.db.repositories import results as results_repo
from anomaly_lab.deployment.export import MANIFEST_FILENAME, run_export_job
from anomaly_lab.deployment.parity import (
    ParityReading,
    compare_outputs,
    portable_score,
    summarize,
)
from anomaly_lab.deployment.protocol import SupportsOnnxExport
from anomaly_lab.deployment.schema import DeploymentManifest
from anomaly_lab.domain.entities import JobKind, Label, Subset
from anomaly_lab.eval.metrics import roc_auc
from anomaly_lab.experiments.context import load_experiment
from anomaly_lab.experiments.infer import run_infer_job
from anomaly_lab.experiments.train import MODEL_SUBDIR, run_train_job
from anomaly_lab.jobs.context import JobContext
from anomaly_lab.models.base import PortableFormat
from anomaly_lab.models.preprocessing import load_array, to_chw

DEFAULT_CATEGORIES = ("candle", "pcb1")

ABSOLUTE_TOLERANCE = 1e-4
RELATIVE_TOLERANCE = 1e-4
"""Predeclared in docs/measurements.md before the first run. Every test image's map must be
`allclose` to the Python map at these bounds and its score within the absolute one."""

AUC_TOLERANCE = 1e-3
"""Image ROC-AUC over the portable scores may differ from the Python scores' by at most this.
Per-image parity already bounds it; stated separately because ranking is what a score is for."""


def _job(settings: Any, job_id: int, kind: JobKind, params: dict[str, Any]) -> JobContext:
    return JobContext(job_id=job_id, kind=kind, params=params, settings=settings)


def _leg(
    settings: Any,
    index: int,
    category: str,
    candidate_key: str,
    steps: int | None,
    log: Any,
) -> dict[str, Any]:
    candidate = CANDIDATES[candidate_key]
    config = dict(candidate.config)
    if steps is not None:
        if candidate.step_field is None:
            raise ValueError(f"{candidate.label} has no step budget; --steps does not apply")
        config[candidate.step_field] = steps

    dataset_id, split_id = _official_dataset(settings, category, log)
    build = _identity_profile(
        settings,
        category=category,
        dataset_id=dataset_id,
        prepared_size=candidate.prepared_size,
        job_id=index,
        log=log,
    )
    experiment_id = _create_experiment(
        settings,
        category=category,
        dataset_id=dataset_id,
        split_id=split_id,
        method=candidate.key,
        method_config=config,
        build=build,
        candidate=candidate,
    )

    print(f"Fitting and scoring {category} / {candidate.key}...", file=sys.stderr)
    base = 30_000 + experiment_id * 3
    started = time.perf_counter()
    with contextlib.redirect_stdout(log):
        run_train_job(
            _job(
                settings,
                base,
                JobKind.TRAIN,
                {"experiment_id": experiment_id, "diagnostics": False},
            )
        )
        infer = run_infer_job(
            _job(
                settings,
                base + 1,
                JobKind.INFER,
                {
                    "experiment_id": experiment_id,
                    "subsets": ["test"],
                    "diagnostics": False,
                    "diagnostic_images": 0,
                },
            )
        )
        fit_seconds = time.perf_counter() - started
        print(f"Exporting {category} / {candidate.key}...", file=sys.stderr)
        exported = run_export_job(
            _job(
                settings,
                base + 2,
                JobKind.EXPORT,
                {"experiment_id": experiment_id, "format": PortableFormat.ONNX.value},
            )
        )
    bundle = Path(exported["bundle_path"])
    manifest = DeploymentManifest.model_validate_json(
        (bundle / MANIFEST_FILENAME).read_text(encoding="utf-8")
    )

    with connection(settings.db_path) as conn:
        loaded = load_experiment(conn, settings, experiment_id)
        scored = results_repo.list_scored_images(conn, experiment_id, subset=Subset.TEST)
    loaded.model.load(loaded.artifact_dir / MODEL_SUBDIR)
    if not isinstance(loaded.model, SupportsOnnxExport):
        raise TypeError(f"{candidate.key} does not implement the ONNX export protocol")
    reference_path = loaded.model
    ort: Any = importlib.import_module("onnxruntime")
    session = ort.InferenceSession(
        str(bundle / manifest.graph_path), providers=["CPUExecutionProvider"]
    )
    output_names = [output.name for output in session.get_outputs()]

    print(f"Comparing {len(scored)} test images...", file=sys.stderr)
    readings: list[ParityReading] = []
    labels: list[bool] = []
    reference_scores: list[float] = []
    portable_scores: list[float] = []
    stored_scores: list[float] = []
    stored_map_errors: list[float] = []
    for image in scored:
        prepared = load_array(loaded.region_build.image_path(image.image_id), loaded.preprocessing)
        tensor = np.ascontiguousarray(to_chw(prepared)[np.newaxis], dtype=np.float32)
        expected_map, expected_score = reference_path.portable_reference(tensor)
        outputs = dict(
            zip(
                output_names,
                session.run(None, {manifest.input.tensor.name: tensor}),
                strict=True,
            )
        )
        actual_map = np.asarray(outputs[manifest.anomaly_map.tensor.name], np.float32).squeeze()
        actual_score = portable_score(outputs, actual_map, manifest.score)
        readings.append(
            compare_outputs(
                expected_map,
                expected_score,
                actual_map,
                actual_score,
                absolute_tolerance=ABSOLUTE_TOLERANCE,
                relative_tolerance=RELATIVE_TOLERANCE,
            )
        )
        labels.append(image.label is Label.DEFECT)
        reference_scores.append(expected_score)
        portable_scores.append(actual_score)
        stored_scores.append(image.score)
        if image.map_path is not None:
            with np.load(image.map_path, allow_pickle=False) as stored:
                prepared_map = np.asarray(stored["map"], dtype=np.float32)
            stored_map_errors.append(float(np.max(np.abs(prepared_map - actual_map))))

    truth = np.asarray(labels)
    auc = {
        "python_reference": roc_auc(truth, np.asarray(reference_scores)),
        "portable": roc_auc(truth, np.asarray(portable_scores)),
        "workbench_stored": roc_auc(truth, np.asarray(stored_scores)),
    }
    summary = summarize(readings)
    auc_delta = (
        abs(auc["portable"] - auc["python_reference"])
        if auc["portable"] is not None and auc["python_reference"] is not None
        else None
    )
    passed = summary.passed and auc_delta is not None and auc_delta <= AUC_TOLERANCE
    return {
        "experiment_id": experiment_id,
        "config": config,
        "prepared_size": candidate.prepared_size,
        "fit_and_score_seconds": fit_seconds,
        "test_metrics": infer.get("metrics", {}).get("test", {}),
        "bundle": {
            "path": str(bundle),
            "opset": manifest.opset,
            "score_contract": manifest.score.model_dump(mode="json"),
            "declared_absolute_tolerance": manifest.parity.absolute_tolerance,
            "declared_relative_tolerance": manifest.parity.relative_tolerance,
            "fixture_map_max_absolute_error": manifest.parity.max_absolute_error,
            "fixture_score_absolute_error": manifest.parity.score_absolute_error,
            "graph_bytes": next(
                item.bytes for item in manifest.files if item.path == manifest.graph_path
            ),
        },
        "parity": {
            "images": summary.inputs,
            "defects": int(truth.sum()),
            "failures": summary.failures,
            "worst_map_absolute_error": summary.worst_map_absolute_error,
            "worst_score_absolute_error": summary.worst_score_absolute_error,
            "median_map_absolute_error": float(
                np.median([reading.map_max_absolute_error for reading in readings])
            )
            if readings
            else None,
        },
        "image_roc_auc": auc,
        "auc_delta": auc_delta,
        "reported_not_gated": {
            "portable_vs_workbench_worst_score_error": float(
                np.max(np.abs(np.asarray(portable_scores) - np.asarray(stored_scores)))
            )
            if stored_scores
            else None,
            "portable_vs_workbench_worst_map_error": max(stored_map_errors, default=None),
        },
        "passed": passed,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--candidate", choices=tuple(CANDIDATES), default="dinomaly_custom")
    parser.add_argument(
        "--category",
        action="append",
        dest="categories",
        help="VisA class; repeat for more than one (default: candle and pcb1).",
    )
    parser.add_argument(
        "--datasets",
        type=Path,
        default=REPOSITORY / "datasets",
        help="Directory holding the public reference datasets (default: the repository's).",
    )
    parser.add_argument(
        "--steps",
        type=int,
        help="Shorten the fit for a smoke run only; the recorded gate uses the default budget.",
    )
    args = parser.parse_args(argv)
    categories = tuple(args.categories or DEFAULT_CATEGORIES)

    _empty_destination(args.data_dir)
    settings = Settings(
        data_dir=args.data_dir.resolve(),
        reference_datasets_dir=args.datasets.resolve(),
        dev_cors=False,
    )
    settings.ensure_directories()
    apply_schema(settings.db_path)

    report: dict[str, Any] = {
        "schema_version": 1,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source": "VisA official 1cls split, test subset",
        "candidate": args.candidate,
        "seed": SEED,
        "steps_override": args.steps,
        "tolerance": {
            "absolute": ABSOLUTE_TOLERANCE,
            "relative": RELATIVE_TOLERANCE,
            "image_roc_auc": AUC_TOLERANCE,
        },
        "packages": {
            **_packages(),
            "onnxruntime": importlib.import_module("onnxruntime").__version__,
        },
        "categories": {},
    }
    with (args.data_dir / "gate.log").open("a", encoding="utf-8") as log:
        for index, category in enumerate(categories, start=1):
            report["categories"][category] = _leg(
                settings, index, category, args.candidate, args.steps, log
            )
            (args.data_dir / "result.partial.json").write_text(
                json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
            )
    report["passed"] = all(leg["passed"] for leg in report["categories"].values())
    output = args.data_dir / "result.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"Full evidence: {output}", file=sys.stderr)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
