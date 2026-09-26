"""A method refuses, at creation, an input it could only fail on at fit — and its own size
never needs refusing.

Torch-free: the check is the method's own `validate_prepared_size`, which needs no encoder.
The fixture's prepared frame is 16 x 16, which no 14-pixel ViT patch divides.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from anomaly_lab.config import Settings
from anomaly_lab.db.connection import connection
from anomaly_lab.models.base import Availability
from anomaly_lab.models.dino_backbone import DinoBackbone
from anomaly_lab.models.fss_dino import FssDinoConfig, FssDinoModel
from anomaly_lab.models.pixel_reference import PixelReferenceConfig, PixelReferenceModel
from anomaly_lab.models.preprocessing import PreprocessingConfig
from anomaly_lab.models.registry import get_model_class, registered_keys

from .conftest import Fixture, create_experiment


def test_a_patch_the_frame_does_not_divide_is_refused_by_the_method() -> None:
    size = PreprocessingConfig(width=16, height=16)
    try:
        FssDinoModel.check_input(FssDinoConfig(backbone=DinoBackbone.DINOV2_VIT_B14), size)
    except ValueError as exc:
        assert "divisible by 14" in str(exc)
    else:  # pragma: no cover - the assertion is that it raises
        raise AssertionError("a 16 px frame passed a 14 px patch")
    FssDinoModel.check_input(FssDinoConfig(), PreprocessingConfig(width=448, height=448))
    PixelReferenceModel.check_input(PixelReferenceConfig(), size)


@pytest.mark.parametrize("key", registered_keys())
def test_every_method_reads_its_own_native_size(key: str) -> None:
    """A run that names no size must never be refused for the size it was given."""
    model_class = get_model_class(key)
    defaults = model_class.config_model().model_validate({})
    width, height = model_class.native_size(defaults)
    multiple = model_class.size_multiple(defaults)

    assert multiple >= 1
    assert width % multiple == 0 and height % multiple == 0
    model_class.check_input(defaults, PreprocessingConfig(width=width, height=height))


def test_a_dino_method_s_size_follows_its_backbone_s_patch() -> None:
    for backbone in DinoBackbone:
        config = FssDinoConfig(backbone=backbone)
        width, height = FssDinoModel.native_size(config)
        assert width % FssDinoModel.size_multiple(config) == 0
        FssDinoModel.check_input(config, PreprocessingConfig(width=width, height=height))


def test_creation_refuses_it_on_the_create_screen(client: TestClient, seeded: Fixture) -> None:
    body = {
        "name": "memory",
        "dataset_id": seeded.dataset_id,
        "split_id": seeded.split_id,
        "model_type": "dino_memory",
    }
    refused = client.post("/api/experiments", json={**body, "width": 16, "height": 16})
    assert refused.status_code == 422
    assert "divisible by" in refused.text
    half = client.post("/api/experiments", json={**body, "width": 448})
    assert half.status_code == 422

    # Named by nobody, the size is the method's own — and the profile the dataset's own.
    native = client.post("/api/experiments", json=body)
    assert native.status_code == 200, native.text
    created = native.json()
    frozen = created["preprocessing"]
    assert (frozen["width"], frozen["height"]) == (448, 448)
    assert created["region_profile_id"] == seeded.region_profile_id
    assert created["region_manifest_sha256"] is None

    answer = client.post("/api/experiments/input-size", json={"model_type": "dino_memory"})
    assert answer.json() == {
        "model_type": "dino_memory",
        "width": 448,
        "height": 448,
        "multiple": 14,
    }
    v3 = client.post(
        "/api/experiments/input-size",
        json={"model_type": "dino_memory", "config": {"backbone": "dinov3_vit_s16"}},
    )
    assert v3.status_code == 200, v3.text
    assert v3.json()["multiple"] == 16


def test_the_studio_preview_names_the_size_it_needs_built(
    client: TestClient, settings: Settings, seeded: Fixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Available or not, the frame is wrong; without torch the availability refusal would
    # answer first, so it is taken out of the way.
    monkeypatch.setattr(FssDinoModel, "availability", classmethod(lambda cls: Availability()))
    profile = create_experiment(client, seeded)["region_profile_id"]
    with connection(settings.db_path) as conn:
        reference = int(
            conn.execute(
                "SELECT sample_id FROM image WHERE id = ?", (seeded.defect_image_ids[0],)
            ).fetchone()[0]
        )
    refused = client.post(
        f"/api/datasets/{seeded.dataset_id}/studio/preview",
        json={
            "class_key": "defect",
            "method": "fss_dino",
            "profile_id": profile,
            "references": [reference],
            "image_id": seeded.defect_image_ids[1],
        },
    )
    assert refused.status_code == 422
    assert "448x448" in refused.text
