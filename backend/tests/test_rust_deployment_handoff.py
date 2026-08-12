"""Cross-language proof that a Python export is sufficient for the Rust consumer."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

import pytest

from anomaly_lab.config import Settings
from anomaly_lab.deployment.export import run_export_job
from anomaly_lab.domain.entities import JobKind
from anomaly_lab.jobs.context import JobContext


def test_python_bundle_passes_rust_runtime_parity(
    settings: Settings,
    scored: dict[str, Any],
) -> None:
    runner_value = os.environ.get("ANOMALY_LAB_RUST_RUNNER")
    if not runner_value:
        pytest.skip("ANOMALY_LAB_RUST_RUNNER is not set")
    runner = Path(runner_value)
    if not runner.is_file():
        pytest.fail(f"Rust deployment runner does not exist: {runner}")

    exported = run_export_job(
        JobContext(
            job_id=81,
            kind=JobKind.EXPORT,
            params={"experiment_id": scored["id"], "format": "onnx"},
            settings=settings,
        )
    )
    completed = subprocess.run(
        [str(runner), "verify", exported["bundle_path"]],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stderr
    report = json.loads(completed.stdout)
    assert report["status"] == "ok"
    assert report["format_version"] == 2
    assert report["checked_files"] == 3
    assert report["map_max_absolute_error"] <= 2e-5
    assert report["score_absolute_error"] <= 2e-5
