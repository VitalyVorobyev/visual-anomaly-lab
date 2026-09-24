"""A method refuses, at creation, an input it could only fail on at fit.

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


def test_creation_refuses_it_on_the_create_screen(client: TestClient, seeded: Fixture) -> None:
    base = create_experiment(client, seeded)
    refused = client.post(
        "/api/experiments",
        json={
            "name": "memory",
            "dataset_id": seeded.dataset_id,
            "split_id": seeded.split_id,
            "region_profile_id": base["region_profile_id"],
            "model_type": "dino_memory",
        },
    )
    assert refused.status_code == 422
    assert "divisible by" in refused.text


def test_the_studio_preview_refuses_it_too(
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
    assert "divisible by" in refused.text
