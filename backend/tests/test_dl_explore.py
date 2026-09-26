"""Explore end to end: the real resident child, a real DINOv2 encoder, a tiny synthetic PNG.

Pretrained weights are required — Explore has no untrained mode — and they are never
downloaded here. The test runs only when `ANOMALY_LAB_DINO_CACHE` names an app model cache
(`<data dir>/model-cache`) that already holds `dinov2_vit_s14`, and it runs offline against
it; everywhere else it is skipped, as `test_dl_anomalyvfm.py` skips its published checkpoint.
"""

from __future__ import annotations

import io
import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

pytest.importorskip("torch")
pytest.importorskip("timm")

from anomaly_lab.config import Settings
from anomaly_lab.models.dino_backbone import BACKBONES, DinoBackbone

from .conftest import Fixture

BACKBONE = DinoBackbone.DINOV2_VIT_S14
REAL_CACHE = os.environ.get("ANOMALY_LAB_DINO_CACHE")


def _holds_weights(cache: str | None) -> bool:
    if not cache:
        return False
    hub = Path(cache) / "huggingface" / "hub"
    return (hub / f"models--timm--{BACKBONES[BACKBONE].timm_name}").is_dir()


pytestmark = pytest.mark.skipif(
    not _holds_weights(REAL_CACHE),
    reason="ANOMALY_LAB_DINO_CACHE does not name a model cache holding dinov2_vit_s14",
)


def test_the_real_child_encodes_once_and_answers_every_mode(
    client: TestClient,
    settings: Settings,
    seeded: Fixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert REAL_CACHE is not None
    settings.model_cache_dir.symlink_to(Path(REAL_CACHE), target_is_directory=True)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    image_id = seeded.defect_image_ids[0]

    first = client.post(
        f"/api/images/{image_id}/explore",
        json={"mode": "similar", "backbone": BACKBONE.value, "points": [{"x": 7, "y": 7}]},
    )
    assert first.status_code == 200, first.text
    cold = first.json()
    assert cold["warm"] is False and cold["cached"] is False
    assert (cold["grid_rows"], cold["grid_cols"]) == (32, 32)
    assert cold["feature_dim"] == 2 * BACKBONES[BACKBONE].embedding_dim
    heat = client.get(cold["map_url"])
    with Image.open(io.BytesIO(heat.content)) as picture:
        assert picture.size == (16, 16)

    clusters = client.post(
        f"/api/images/{image_id}/explore",
        json={"mode": "clusters", "backbone": BACKBONE.value, "k": 4, "seed": 1},
    )
    assert clusters.status_code == 200, clusters.text
    warm = clusters.json()
    assert warm["warm"] is True and warm["cached"] is True
    assert warm["encode_ms"] == 0.0
    assert 2 <= warm["clusters"] <= 4
    again = client.post(
        f"/api/images/{image_id}/explore",
        json={"mode": "clusters", "backbone": BACKBONE.value, "k": 4, "seed": 1},
    ).json()
    assert again["cells"] == warm["cells"]

    pca = client.post(
        f"/api/images/{image_id}/explore", json={"mode": "pca", "backbone": BACKBONE.value}
    )
    assert pca.status_code == 200, pca.text
    with Image.open(io.BytesIO(client.get(pca.json()["map_url"]).content)) as picture:
        assert picture.mode == "RGB" and picture.size == (16, 16)

    other = client.post(
        f"/api/images/{seeded.normal_image_ids[0]}/explore",
        json={"mode": "pca", "backbone": BACKBONE.value},
    ).json()
    assert other["cached"] is False and other["warm"] is True
