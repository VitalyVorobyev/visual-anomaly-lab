"""AnomalyVFM's torch-free half: the frame plan, the asset contract and the recorded run.

Everything here runs without the `dl` extra. The checkpoint is replaced by a small file
under a stand-in pin, so resolution, verification and refusal are exercised for real.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from anomaly_lab.models import anomalyvfm_anomalib as vfm
from anomaly_lab.models.anomalyvfm_anomalib import (
    STATE_FILENAME,
    AnomalyVfmAnomalibModel,
    AnomalyVfmConfig,
    AssetError,
    PinnedAsset,
    pinned_download,
    plan_inference,
    resolve_weights,
)
from anomaly_lab.models.base import (
    Device,
    ImageRecord,
    NullReporter,
    TrainContext,
)
from anomaly_lab.models.diagnostics import DiagnosticWriter
from anomaly_lab.models.preprocessing import PreprocessingConfig
from anomaly_lab.models.registry import describe, get_model_class

PAYLOAD = b"not a real checkpoint, but a pinned one"


def _asset(payload: bytes = PAYLOAD) -> PinnedAsset:
    return PinnedAsset(
        repository="someone/fake-radio",
        filename="model.safetensors",
        revision="0123456789abcdef0123456789abcdef01234567",
        sha256=hashlib.sha256(payload).hexdigest(),
        size=len(payload),
    )


def _cache(tmp_path: Path, asset: PinnedAsset, payload: bytes = PAYLOAD) -> Path:
    path = asset.snapshot_path(tmp_path / "huggingface" / "hub")
    path.parent.mkdir(parents=True)
    path.write_bytes(payload)
    return path


def _train_ctx(tmp_path: Path, size: int = 768) -> TrainContext:
    return TrainContext(
        artifact_dir=tmp_path / "run",
        cache_dir=tmp_path,
        preprocessing=PreprocessingConfig(width=size, height=size),
        device=Device.CPU,
        reporter=NullReporter(),
        diagnostics=DiagnosticWriter(tmp_path / "diagnostics", enabled=False),
    )


def test_it_is_registered_as_a_zero_shot_method_with_no_portable_format() -> None:
    assert get_model_class("anomalyvfm_anomalib") is AnomalyVfmAnomalibModel
    description = describe("anomalyvfm_anomalib")
    assert description.capabilities.requires_training is False
    assert description.capabilities.portable_formats == []
    assert description.capabilities.supports_resume is False
    assert description.capabilities.dataset_specific is False
    fields = description.config_schema["properties"]
    assert set(fields) == {"allow_downloads"}
    assert fields["allow_downloads"]["description"]


def test_the_plan_at_the_measured_size() -> None:
    plan = plan_inference(768, 768)
    assert plan.grid == (48, 48)
    assert plan.tokens == 48 * 48 + 8
    assert plan.attention_bytes == 16 * plan.tokens**2 * 4
    text = plan.describe()
    assert "48x48 patch grid" in text
    assert "not that one" not in text
    assert "not that one" in plan_inference(512, 768).describe()


@pytest.mark.parametrize(("width", "height"), [(770, 768), (768, 100), (8, 8)])
def test_a_frame_the_patch_size_does_not_divide_is_refused_at_creation(
    width: int, height: int
) -> None:
    with pytest.raises(ValueError, match="multiples of 16"):
        AnomalyVfmAnomalibModel.check_input(
            AnomalyVfmConfig(), PreprocessingConfig(width=width, height=height)
        )


def test_a_frame_whose_attention_would_not_fit_is_refused_by_the_bound() -> None:
    AnomalyVfmAnomalibModel.check_input(
        AnomalyVfmConfig(), PreprocessingConfig(width=1024, height=1024)
    )
    with pytest.raises(ValueError, match="GiB bound"):
        AnomalyVfmAnomalibModel.check_input(
            AnomalyVfmConfig(), PreprocessingConfig(width=2048, height=2048)
        )


def test_a_cached_checkpoint_is_found_by_its_revision_and_verified(tmp_path: Path) -> None:
    asset = _asset()
    expected = _cache(tmp_path, asset)
    messages: list[str] = []
    found = resolve_weights(tmp_path, allow_downloads=False, log=messages.append, asset=asset)
    assert found == expected
    assert any("verified" in message for message in messages)


def test_a_missing_checkpoint_with_downloads_off_names_the_pin_and_the_switch(
    tmp_path: Path,
) -> None:
    asset = _asset()
    with pytest.raises(AssetError, match="allow_downloads") as refused:
        resolve_weights(tmp_path, allow_downloads=False, log=print, asset=asset)
    assert asset.revision[:12] in str(refused.value)


@pytest.mark.parametrize("payload", [PAYLOAD + b"!", PAYLOAD[:-1] + b"?"])
def test_a_file_that_is_not_the_pinned_one_is_refused(tmp_path: Path, payload: bytes) -> None:
    asset = _asset()
    _cache(tmp_path, asset, payload)
    with pytest.raises(AssetError, match=r"bytes|SHA-256"):
        resolve_weights(tmp_path, allow_downloads=True, log=print, asset=asset)


def test_a_download_is_verified_before_it_is_used(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    asset = _asset()
    fetched: list[PinnedAsset] = []

    def download(requested: PinnedAsset, cache_dir: Path) -> Path:
        fetched.append(requested)
        return _cache(cache_dir, requested, b"truncated")

    monkeypatch.setattr(vfm, "_download", download)
    with pytest.raises(AssetError, match="bytes"):
        resolve_weights(tmp_path, allow_downloads=True, log=print, asset=asset)
    assert fetched == [asset]


def test_the_constructor_is_answered_only_for_the_pinned_request(tmp_path: Path) -> None:
    asset = _asset()
    weights = tmp_path / "weights.safetensors"
    download = pinned_download(asset, weights)
    answered = download(
        repo_id=asset.repository,
        filename=asset.filename,
        revision=asset.revision,
        local_files_only=False,
    )
    assert answered == str(weights)
    with pytest.raises(AssetError, match="update the pin"):
        download(repo_id=asset.repository, filename=asset.filename, revision="main")


def test_a_train_job_verifies_and_records_the_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    asset = _asset()
    _cache(tmp_path, asset)
    monkeypatch.setattr(vfm, "ASSET", asset)
    ctx = _train_ctx(tmp_path)
    model = AnomalyVfmAnomalibModel(AnomalyVfmConfig(allow_downloads=False))
    # The images are never opened: a path that does not exist proves it.
    model.fit([ImageRecord(image_id=1, sample_id=1, path=tmp_path / "absent.png")], ctx)
    model.save(ctx.artifact_dir)

    stored: dict[str, Any] = json.loads((ctx.artifact_dir / STATE_FILENAME).read_text())
    assert stored["revision"] == asset.revision
    assert stored["sha256"] == asset.sha256
    assert (stored["width"], stored["height"]) == (768, 768)

    AnomalyVfmAnomalibModel(AnomalyVfmConfig()).load(ctx.artifact_dir)


def test_the_module_imports_without_torch() -> None:
    """The registry imports every plugin to draw the picker; this one must stay cheap."""
    import subprocess
    import sys

    probe = (
        "import sys; import anomaly_lab.models.anomalyvfm_anomalib; "
        "sys.exit(int(any(m in sys.modules for m in ('torch', 'anomalib', 'huggingface_hub'))))"
    )
    assert subprocess.run([sys.executable, "-c", probe], check=False).returncode == 0


def test_a_run_recorded_against_another_checkpoint_refuses_to_load(tmp_path: Path) -> None:
    (tmp_path / STATE_FILENAME).write_text(
        json.dumps(
            {
                "format": 1,
                "repository": vfm.ASSET.repository,
                "revision": "f" * 40,
                "sha256": "0" * 64,
            }
        )
    )
    with pytest.raises(RuntimeError, match="create a new experiment"):
        AnomalyVfmAnomalibModel(AnomalyVfmConfig()).load(tmp_path)


def test_there_is_nothing_to_save_before_a_train_job(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="nothing to save"):
        AnomalyVfmAnomalibModel(AnomalyVfmConfig()).save(tmp_path)
