"""The Prepare screen's live stage: an unsaved recipe on one image, answered synchronously."""

from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

from anomaly_lab.config import Settings
from anomaly_lab.db.connection import connection
from anomaly_lab.db.repositories import jobs as jobs_repo
from anomaly_lab.domain.entities import JobKind
from anomaly_lab.jobs.context import JobContext
from anomaly_lab.jobs.queue import JobQueue
from anomaly_lab.model_assets import mobile_sam as mobile_sam_session
from anomaly_lab.model_assets.catalog import ModelAssetSpec
from anomaly_lab.model_assets.store import ResolvedAsset
from anomaly_lab.regions import live
from anomaly_lab.regions.base import RegionAvailability
from anomaly_lab.regions.mobile_sam import MobileSamRegionExtractor
from anomaly_lab.regions.preparation import run_region_prepare_job
from tests.conftest import SeededCatalog

SOURCE = (40, 30)


def _square(path: Path, box: tuple[int, int, int, int]) -> Path:
    """A dark frame with one bright rectangle: a foreground the threshold extractor finds."""
    pixels = np.full((SOURCE[1], SOURCE[0], 3), 20, dtype=np.uint8)
    left, top, right, bottom = box
    pixels[top:bottom, left:right] = 220
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pixels).save(path)
    return path


@pytest.fixture
def on_disk(catalog: SeededCatalog, settings: Settings, tmp_path: Path) -> SeededCatalog:
    """The catalog's five images as real PNGs, each channel of a part boxed differently."""
    boxes = [(8, 6, 20, 18), (14, 10, 30, 24), (5, 5, 15, 15), (20, 8, 34, 22), (10, 10, 30, 20)]
    with connection(settings.db_path) as conn:
        for image_id, box in zip(catalog.image_ids, boxes, strict=True):
            path = _square(tmp_path / "sources" / f"{image_id}.png", box)
            conn.execute(
                "UPDATE image SET path = ?, width = ?, height = ? WHERE id = ?",
                (str(path), SOURCE[0], SOURCE[1], image_id),
            )
    return catalog


def _preview(client: TestClient, catalog: SeededCatalog, image_id: int, **recipe: Any) -> Any:
    body = {"extractor_type": "identity", "image_id": image_id, "width": 32, "height": 32}
    return client.post(
        f"/api/datasets/{catalog.dataset_id}/region-preview", json={**body, **recipe}
    )


def _png(data_url: str) -> Image.Image:
    prefix = "data:image/png;base64,"
    assert data_url.startswith(prefix)
    return Image.open(io.BytesIO(base64.b64decode(data_url[len(prefix) :])))


def test_foreground_threshold_finds_the_part_and_returns_the_prepared_frame(
    client: TestClient, on_disk: SeededCatalog
) -> None:
    image_id = on_disk.image_ids[0]
    response = _preview(
        client,
        on_disk,
        image_id,
        extractor_type="foreground_threshold",
        padding_fraction=0.0,
        width=48,
        height=24,
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "succeeded"
    assert body["region"] == {"left": 8.0, "top": 6.0, "right": 20.0, "bottom": 18.0}
    transform = body["transform"]
    assert (transform["crop_left"], transform["crop_top"]) == (8, 6)
    assert (transform["crop_right"], transform["crop_bottom"]) == (20, 18)
    assert body["extractor_metadata"]["area_pixels"] == 144
    assert body["united"] == 1
    prepared = _png(body["prepared_png"])
    assert prepared.size == (48, 24)


def test_identity_and_centred_crop_answer_without_a_saved_revision(
    client: TestClient, on_disk: SeededCatalog, settings: Settings
) -> None:
    image_id = on_disk.image_ids[0]
    identity = _preview(client, on_disk, image_id, padding_fraction=0.0)
    centred = _preview(
        client,
        on_disk,
        image_id,
        extractor_type="center_crop",
        extractor_config={"width_fraction": 0.5, "height_fraction": 0.5},
        padding_fraction=0.0,
    )

    assert identity.json()["transform"]["crop_right"] == SOURCE[0]
    assert centred.json()["transform"]["crop_left"] == 10
    assert centred.json()["transform"]["crop_right"] == 30
    with connection(settings.db_path) as conn:
        saved = conn.execute("SELECT COUNT(*) FROM region_profile_revision").fetchone()[0]
    # Only the implicit Full frame profile: tuning writes nothing.
    assert saved == 1


def test_a_failed_extraction_is_an_answer_with_the_extractor_s_reason(
    client: TestClient, on_disk: SeededCatalog, settings: Settings, tmp_path: Path
) -> None:
    image_id = on_disk.image_ids[0]
    flat = tmp_path / "flat.png"
    Image.new("RGB", SOURCE, (90, 90, 90)).save(flat)
    with connection(settings.db_path) as conn:
        conn.execute("UPDATE image SET path = ? WHERE id = ?", (str(flat), image_id))

    response = _preview(client, on_disk, image_id, extractor_type="foreground_threshold")

    assert response.status_code == 200
    assert response.json()["status"] == "failed"
    assert "foreground component" in response.json()["error"]
    assert response.json()["prepared_png"] is None


def test_a_shared_crop_is_the_union_over_the_sample(
    client: TestClient, on_disk: SeededCatalog
) -> None:
    # Images 0 and 1 are the two channels of one part, boxed at (8,6,20,18) and (14,10,30,24).
    response = _preview(
        client,
        on_disk,
        on_disk.image_ids[0],
        extractor_type="foreground_threshold",
        padding_fraction=0.0,
        sample_alignment="union",
    )

    body = response.json()
    assert body["status"] == "succeeded", body
    assert body["united"] == 2
    # Its own box is still reported; the crop is the union.
    assert body["region"]["right"] == 20.0
    crop = body["transform"]
    assert (crop["crop_left"], crop["crop_top"], crop["crop_right"], crop["crop_bottom"]) == (
        8,
        6,
        30,
        24,
    )


def test_bad_recipes_and_foreign_images_are_refused(
    client: TestClient,
    on_disk: SeededCatalog,
    catalog: SeededCatalog,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image_id = on_disk.image_ids[0]
    unknown = _preview(client, on_disk, image_id, extractor_type="magic")
    invalid = _preview(
        client,
        on_disk,
        image_id,
        extractor_type="foreground_threshold",
        extractor_config={"min_contrast": 1000},
    )
    tiny = _preview(client, on_disk, image_id, width=4)
    missing = _preview(client, on_disk, 999_999)
    elsewhere = client.post(
        "/api/datasets/999/region-preview",
        json={"extractor_type": "identity", "image_id": image_id, "width": 32, "height": 32},
    )

    assert unknown.status_code == 422
    assert invalid.status_code == 422
    assert tiny.status_code == 422
    assert missing.status_code == 404
    assert elsewhere.status_code == 404

    monkeypatch.setattr(live, "MAX_LIVE_SOURCE_PIXELS", 100)
    too_large = _preview(client, on_disk, image_id)
    assert too_large.status_code == 422
    assert "Check 24" in too_large.json()["detail"]


def test_mobile_sam_needs_its_asset_and_an_idle_device(
    client: TestClient,
    on_disk: SeededCatalog,
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image_id = on_disk.image_ids[0]
    monkeypatch.setattr(
        MobileSamRegionExtractor, "availability", classmethod(lambda _cls: RegionAvailability())
    )

    no_asset = _preview(client, on_disk, image_id, extractor_type="mobile_sam")
    assert no_asset.status_code == 409
    assert "install it" in no_asset.json()["detail"]

    app = client.app
    assert isinstance(app, FastAPI)
    queue: JobQueue = app.state.job_queue
    job = queue.enqueue(kind=JobKind.PREWARM, params={"dataset_id": on_disk.dataset_id})
    with connection(settings.db_path) as conn:
        jobs_repo.mark_running(conn, job.id, log_path="/dev/null")
    busy = _preview(client, on_disk, image_id, extractor_type="mobile_sam")
    assert busy.status_code == 409
    assert "prewarm" in busy.text
    # The classical extractors do not touch the device, so they answer through a job.
    assert _preview(client, on_disk, image_id).status_code == 200


def test_the_stage_steps_through_images_spread_over_the_dataset(
    client: TestClient, on_disk: SeededCatalog
) -> None:
    per_image = client.get(f"/api/datasets/{on_disk.dataset_id}/region-preview/images")
    union = client.get(
        f"/api/datasets/{on_disk.dataset_id}/region-preview/images",
        params={"alignment": "union"},
    )
    two = client.get(
        f"/api/datasets/{on_disk.dataset_id}/region-preview/images", params={"limit": 2}
    )
    random = client.get(f"/api/datasets/{on_disk.dataset_id}/region-preview/random")

    assert per_image.json()["total"] == 5
    assert [item["image_id"] for item in per_image.json()["images"]] == on_disk.image_ids
    first = per_image.json()["images"][0]
    assert (first["group_key"], first["external_id"], first["channel"]) == (
        "group-a",
        "1",
        "bright",
    )
    # One image per sample when the crop is shared.
    assert len({item["sample_id"] for item in union.json()["images"]}) == 3
    assert len(union.json()["images"]) == 3
    assert len(two.json()["images"]) == 2
    assert random.json()["image_id"] in on_disk.image_ids


def test_check_24_runs_the_sampled_preview_on_an_unsaved_recipe(
    client: TestClient, on_disk: SeededCatalog, settings: Settings
) -> None:
    response = client.post(
        f"/api/datasets/{on_disk.dataset_id}/region-check",
        json={
            "extractor_type": "foreground_threshold",
            "padding_fraction": 0.0,
            "width": 32,
            "height": 32,
        },
    )
    assert response.status_code == 200, response.text
    job_id = response.json()["id"]
    with connection(settings.db_path) as conn:
        job = jobs_repo.get_job(conn, job_id)
    assert job is not None
    assert "profile_id" not in job.params
    # Defaults are filled in before the job is queued, so the job runs what was validated.
    assert job.params["recipe"]["extractor_config"]["min_contrast"] == 12

    result = run_region_prepare_job(
        JobContext(job_id=job_id, kind=JobKind.REGION_PREPARE, params=job.params, settings=settings)
    )
    assert result["mode"] == "preview"
    assert result["profile_id"] is None
    assert result["sampled"] == 5
    assert result["succeeded"] == 5
    assert result["recipe"]["extractor_type"] == "foreground_threshold"

    invalid = client.post(
        f"/api/datasets/{on_disk.dataset_id}/region-check",
        json={"extractor_type": "magic", "width": 32, "height": 32},
    )
    assert invalid.status_code == 422


def test_an_unsaved_recipe_cannot_be_built(settings: Settings, on_disk: SeededCatalog) -> None:
    params = {
        "dataset_id": on_disk.dataset_id,
        "mode": "build",
        "recipe": {"extractor_type": "identity"},
        "width": 16,
        "height": 16,
    }
    with pytest.raises(ValueError, match="unsaved"):
        run_region_prepare_job(
            JobContext(job_id=1, kind=JobKind.REGION_PREPARE, params=params, settings=settings)
        )


def _session(
    settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, device: str = "cpu"
) -> mobile_sam_session.MobileSamSession:
    checkpoint = tmp_path / "fixture.pt"
    checkpoint.write_bytes(b"fixture")
    spec = ModelAssetSpec(
        key="fixture",
        title="Fixture",
        purpose="Test",
        filename=checkpoint.name,
        source_url="https://example.test/fixture",
        expected_size=checkpoint.stat().st_size,
        sha256="unused",
        license_name="Apache-2.0",
        license_url="https://example.test/license",
        project_url="https://example.test",
    )
    monkeypatch.setattr(mobile_sam_session, "get_spec", lambda _key: spec)
    monkeypatch.setattr(
        mobile_sam_session,
        "resolve_asset",
        lambda _settings, _spec: ResolvedAsset(checkpoint, "managed", True, 7),
    )
    monkeypatch.setattr(mobile_sam_session, "_preferred_device", lambda: device)
    monkeypatch.setattr(
        mobile_sam_session, "_load_predictor", lambda _path, where: (f"{where}-model", object())
    )
    return mobile_sam_session.MobileSamSession(settings, "fixture")


def _mask_record(width: int, height: int, box: tuple[int, int, int, int]) -> dict[str, Any]:
    left, top, right, bottom = box
    mask = np.zeros((height, width), dtype=bool)
    mask[top:bottom, left:right] = True
    return {
        "segmentation": mask,
        "area": int(mask.sum()),
        "bbox": [left, top, right - left, bottom - top],
        "predicted_iou": 0.9,
        "stability_score": 0.95,
    }


def test_the_resident_answers_a_region_with_the_build_s_selection(
    settings: Settings,
    on_disk: SeededCatalog,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = _session(settings, tmp_path, monkeypatch)
    records = [_mask_record(*SOURCE, (8, 6, 20, 18))]
    monkeypatch.setattr(session, "_generate", lambda _image, _config: records)

    found = session.region({"op": "region", "image_id": on_disk.image_ids[0], "config": {}})
    extraction = found["extraction"]
    assert isinstance(extraction, dict)
    assert extraction["bounds"] == {"left": 8.0, "top": 6.0, "right": 20.0, "bottom": 18.0}
    assert extraction["metadata"]["device"] == "cpu"

    # A window no mask survives is an answer, not a protocol failure that kills the model.
    none = session.region({"image_id": on_disk.image_ids[0], "config": {"min_area_fraction": 0.9}})
    assert "no mask" in str(none["error"])


def test_the_generator_alone_leaves_the_accelerator_when_it_refuses_float64(
    settings: Settings,
    on_disk: SeededCatalog,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = _session(settings, tmp_path, monkeypatch, device="mps")
    records = [_mask_record(*SOURCE, (8, 6, 20, 18))]
    used: list[object] = []

    def generate(image: np.ndarray, config: Any) -> list[dict[str, Any]]:
        model = session.auto_model if session.auto_model is not None else session.model
        used.append(model)
        if model == "mps-model":
            raise TypeError("Cannot convert a MPS Tensor to float64 dtype")
        return records

    monkeypatch.setattr(session, "_generate", generate)

    found = session.region({"image_id": on_disk.image_ids[0], "config": {}})

    assert used == ["mps-model", "cpu-model"]
    assert found["device"] == "cpu"
    # The prompt predictor the annotation editor uses stays where it was.
    assert session.device == "mps"
    assert session.model == "mps-model"


def test_the_segmenter_routes_a_region_request_to_the_region_answer(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from anomaly_lab.jobs import segmenter

    class _Session:
        def region(self, request: dict[str, Any]) -> dict[str, object]:
            return {"answered": "region"}

        def segment(self, request: dict[str, Any]) -> dict[str, object]:
            return {"answered": "segment"}

    session: Any = _Session()
    segmenter._answer(session, {"rid": 1, "op": "region", "image_id": 3})
    segmenter._answer(session, {"rid": 2, "image_id": 3, "points": []})

    lines = capsys.readouterr().out.replace(" ", "").strip().splitlines()
    assert '"answered":"region"' in lines[0]
    assert '"answered":"segment"' in lines[1]
