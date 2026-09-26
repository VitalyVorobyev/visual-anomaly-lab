"""Explore's Text mode end to end: the real resident child, the real SAM 3, a synthetic PNG.

SAM 3's weights are gated and 3.4 GB, so they are never downloaded here. The test runs only
when `ANOMALY_LAB_SAM3_DIR` names a directory that already holds the catalogued files — the
app's `<data dir>/model-cache/assets/sam3`, or the pinned revision's snapshot in a Hugging
Face cache — which it selects as the asset's external source, exactly as a person would; it
is skipped everywhere else, as `test_dl_explore.py` skips without its encoder cache.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

pytest.importorskip("torch")
pytest.importorskip("transformers")

from anomaly_lab.config import Settings
from anomaly_lab.db.connection import connection
from anomaly_lab.db.repositories import images as images_repo
from anomaly_lab.model_assets.catalog import SAM3_KEY

from .conftest import Fixture

SAM3_DIR = os.environ.get("ANOMALY_LAB_SAM3_DIR")

pytestmark = pytest.mark.skipif(
    not SAM3_DIR or not (Path(SAM3_DIR) / "model.safetensors").is_file(),
    reason="ANOMALY_LAB_SAM3_DIR does not name a directory holding SAM 3's files",
)


def _discs(path: Path) -> None:
    image = Image.new("RGB", (256, 256), (228, 224, 214))
    draw = ImageDraw.Draw(image)
    for x, y, radius in ((64, 64, 36), (190, 80, 28), (120, 180, 44)):
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=(40, 70, 150))
    image.save(path)


def test_the_real_child_encodes_once_and_answers_each_phrase(
    client: TestClient, settings: Settings, seeded: Fixture, tmp_path: Path
) -> None:
    assert SAM3_DIR is not None
    image_id = seeded.normal_image_ids[0]
    with connection(settings.db_path) as conn:
        image = images_repo.get_image(conn, image_id)
        assert image is not None
        _discs(Path(image.path))
        conn.execute("UPDATE image SET width = 256, height = 256 WHERE id = ?", (image_id,))

    source = client.put(
        f"/api/model-assets/{SAM3_KEY}/source",
        json={"path": str(Path(SAM3_DIR) / "model.safetensors")},
    )
    assert source.status_code == 200, source.text
    assert client.get("/api/explore/capability").json()["text"]["available"] is True

    first = client.post(f"/api/images/{image_id}/explore/text", json={"phrase": "circle"})
    assert first.status_code == 200, first.text
    cold = first.json()
    assert cold["cached"] is False and cold["warm"] is False
    assert len(cold["instances"]) == 3
    assert all(entry["score"] >= 0.5 for entry in cold["instances"])
    assert cold["map_url"] is not None

    second = client.post(f"/api/images/{image_id}/explore/text", json={"phrase": "triangle"})
    assert second.status_code == 200, second.text
    warm = second.json()
    assert warm["cached"] is True and warm["warm"] is True
    assert warm["encode_ms"] == 0.0

    shape = client.post(f"/api/explore/maps/{cold['map_id']}/shape", json={"instance": 1})
    assert shape.status_code == 200, shape.text
    assert shape.json()["area"] == cold["instances"][0]["area"]
