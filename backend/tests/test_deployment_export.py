"""Portable bundles prove semantics, not merely that ONNX can load (ADR-0034)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from anomaly_lab.config import Settings
from anomaly_lab.deployment.export import MANIFEST_FILENAME, run_export_job
from anomaly_lab.deployment.schema import DeploymentManifest, FileDigest
from anomaly_lab.domain.entities import JobKind
from anomaly_lab.jobs.context import JobContext


def test_pixel_reference_exports_an_atomic_verified_bundle(
    settings: Settings,
    scored: dict[str, Any],
) -> None:
    result = run_export_job(
        JobContext(
            job_id=43,
            kind=JobKind.EXPORT,
            params={"experiment_id": scored["id"], "format": "onnx"},
            settings=settings,
        )
    )

    bundle = Path(result["bundle_path"])
    manifest = DeploymentManifest.model_validate_json(
        (bundle / MANIFEST_FILENAME).read_text(encoding="utf-8")
    )

    assert bundle.name.startswith("onnx-")
    assert not any(path.name.startswith(".export-") for path in bundle.parent.iterdir())
    assert manifest.source.id == scored["id"]
    assert manifest.source.model_type == "pixel_reference"
    assert manifest.input.tensor.shape == [1, 3, 16, 16]
    assert manifest.anomaly_map.tensor.shape == [1, 1, 16, 16]
    assert manifest.region.runtime_input_is_prepared is True
    assert manifest.parity.max_absolute_error <= manifest.parity.absolute_tolerance
    assert manifest.parity.score_absolute_error <= manifest.parity.absolute_tolerance
    assert manifest.operating_point is not None
    assert manifest.operating_point.subset == "test"

    for item in manifest.files:
        content = (bundle / item.path).read_bytes()
        assert len(content) == item.bytes
        assert hashlib.sha256(content).hexdigest() == item.sha256

    values = np.frombuffer((bundle / manifest.parity.expected_map_path).read_bytes(), dtype="<f4")
    assert values.size == 16 * 16
    assert np.isfinite(values).all()


def test_bundle_paths_cannot_escape_the_export_root() -> None:
    digest = "0" * 64
    with pytest.raises(ValueError, match="safe POSIX-relative"):
        FileDigest(path="../model.onnx", bytes=1, sha256=digest)
    with pytest.raises(ValueError, match="safe POSIX-relative"):
        FileDigest(path="/tmp/model.onnx", bytes=1, sha256=digest)


def test_manifest_is_plain_json_with_no_python_type_tags(
    settings: Settings,
    scored: dict[str, Any],
) -> None:
    result = run_export_job(
        JobContext(
            job_id=44,
            kind=JobKind.EXPORT,
            params={"experiment_id": scored["id"]},
            settings=settings,
        )
    )
    payload = json.loads(Path(result["manifest_path"]).read_text(encoding="utf-8"))

    assert payload["format_version"] == 2
    assert payload["portable_format"] == "onnx"
    assert payload["score"]["kind"] == "percentile_linear"
    assert not any(key.startswith("__") for key in payload)
