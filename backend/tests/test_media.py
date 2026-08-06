"""Decoding, the tier cache, image delivery, and the pre-warm job."""

from __future__ import annotations

import io
import time
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from PIL import Image as PILImage

from anomaly_lab.config import Settings
from anomaly_lab.db.connection import connection
from anomaly_lab.db.repositories import datasets as datasets_repo
from anomaly_lab.db.repositories import images as images_repo
from anomaly_lab.db.repositories import samples as samples_repo
from anomaly_lab.domain.entities import Label
from anomaly_lab.media import cache
from anomaly_lab.media.cache import ImageTier
from anomaly_lab.media.decode import UnreadableImageError, bits_for_mode, load, probe
from tests.conftest import write_image

TERMINAL_WAIT_SECONDS = 60.0


@pytest.fixture
def catalogued(client: TestClient, settings: Settings, tmp_path: Path) -> dict[str, Any]:
    """A committed dataset of four images: two 24-bit colour, two 8-bit grayscale.

    Mixed bit depths on purpose — §9 promises they are handled with no special casing,
    and this is where that claim gets exercised rather than restated.
    """
    root = tmp_path / "src"
    paths = [
        write_image(root / "g" / "Bright" / "1.bmp", mode="RGB", size=(60, 40)),
        write_image(root / "g" / "Dark" / "1.bmp", mode="L", size=(60, 40)),
        write_image(root / "g" / "Bright" / "2.bmp", mode="RGB", size=(40, 60)),
        write_image(root / "g" / "Dark" / "2.bmp", mode="L", size=(40, 60)),
    ]

    with connection(settings.db_path) as conn:
        dataset = datasets_repo.create_dataset(conn, name="d", root_path=str(root))
        channels = {
            name: datasets_repo.upsert_channel(conn, dataset.id, name=name, position=index).id
            for index, name in enumerate(("bright", "dark"))
        }
        image_ids = []
        for path in paths:
            external_id = path.stem
            sample, _ = samples_repo.upsert_sample(
                conn, dataset.id, group_key="g", external_id=external_id, label=Label.NORMAL
            )
            info = probe(path)
            image, _ = images_repo.upsert_image(
                conn,
                sample.id,
                channel_id=channels["bright" if "Bright" in str(path) else "dark"],
                path=str(path),
                width=info.width,
                height=info.height,
                bit_depth=info.bit_depth,
                file_size=info.file_size,
                sha256=info.sha256,
            )
            image_ids.append(image.id)

    return {"dataset_id": dataset.id, "image_ids": image_ids, "paths": paths}


def _await_job(client: TestClient, job_id: int) -> dict[str, Any]:
    deadline = time.monotonic() + TERMINAL_WAIT_SECONDS
    while time.monotonic() < deadline:
        payload: dict[str, Any] = client.get(f"/api/jobs/{job_id}").json()
        if payload["status"] in {"succeeded", "failed", "cancelled"}:
            return payload
        time.sleep(0.05)
    msg = f"job {job_id} never finished"
    raise AssertionError(msg)


# -- decoding ------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("mode", "expected"),
    [("L", 8), ("RGB", 24), ("RGBA", 32), ("1", 1), ("I;16", 16), ("nonsense", 8)],
)
def test_bit_depth_is_derived_from_the_mode(mode: str, expected: int) -> None:
    assert bits_for_mode(mode) == expected


def test_probe_reads_a_header_and_a_hash_without_decoding(tmp_path: Path) -> None:
    path = write_image(tmp_path / "a.bmp", mode="L", size=(32, 16))

    info = probe(path)

    assert (info.width, info.height) == (32, 16)
    assert info.bit_depth == 8
    assert info.file_size == path.stat().st_size
    assert len(info.sha256) == 64


def test_decoding_normalizes_every_mode_onto_l_or_rgb(tmp_path: Path) -> None:
    """So no renderer downstream needs a mode switch (§9)."""
    grey = load(write_image(tmp_path / "g.bmp", mode="L"))
    colour = load(write_image(tmp_path / "c.bmp", mode="RGB"))

    assert grey.mode == "L"
    assert colour.mode == "RGB"


def test_a_file_that_is_not_an_image_raises_rather_than_returning_junk(tmp_path: Path) -> None:
    broken = tmp_path / "broken.png"
    broken.write_bytes(b"definitely not a png")

    with pytest.raises(UnreadableImageError):
        probe(broken)
    with pytest.raises(UnreadableImageError):
        load(broken)


# -- the cache -----------------------------------------------------------------------


def test_tiers_are_rendered_to_their_documented_sizes(
    settings: Settings, catalogued: dict[str, Any]
) -> None:
    with connection(settings.db_path) as conn:
        landscape = images_repo.get_image(conn, catalogued["image_ids"][0])
    assert landscape is not None

    thumb = PILImage.open(io.BytesIO(cache.render(landscape, ImageTier.THUMB)))
    full = PILImage.open(io.BytesIO(cache.render(landscape, ImageTier.FULL)))

    # 256 px on the long edge, aspect preserved; the source is smaller here so it is not
    # enlarged — `thumbnail` only shrinks.
    assert max(thumb.size) <= 256
    assert thumb.format == "WEBP"
    # Full is native resolution and lossless, because it is used to judge defects.
    assert full.size == (landscape.width, landscape.height)
    assert full.format == "PNG"


def test_the_cached_tiers_are_written_once_and_reused(
    settings: Settings, catalogued: dict[str, Any]
) -> None:
    with connection(settings.db_path) as conn:
        image = images_repo.get_image(conn, catalogued["image_ids"][0])
    assert image is not None

    first = cache.ensure_cached(settings, image, ImageTier.THUMB)
    stamp = first.stat().st_mtime_ns
    second = cache.ensure_cached(settings, image, ImageTier.THUMB)

    assert first == second == settings.thumbnails_dir / "thumb" / f"{image.id}.webp"
    assert second.stat().st_mtime_ns == stamp, "the second call re-rendered"
    # No partial files left behind by the atomic write.
    assert not list(first.parent.glob(".*partial"))


def test_the_full_tier_is_never_cached(settings: Settings, catalogued: dict[str, Any]) -> None:
    """Caching it would cost most of a gigabyte per dataset to save one render."""
    with connection(settings.db_path) as conn:
        image = images_repo.get_image(conn, catalogued["image_ids"][0])
    assert image is not None

    with pytest.raises(ValueError, match="never cached"):
        cache.ensure_cached(settings, image, ImageTier.FULL)


def test_the_etag_is_the_content_hash_and_the_tier(
    settings: Settings, catalogued: dict[str, Any]
) -> None:
    with connection(settings.db_path) as conn:
        image = images_repo.get_image(conn, catalogued["image_ids"][0])
        other = images_repo.get_image(conn, catalogued["image_ids"][1])
    assert image is not None
    assert other is not None

    assert cache.etag_for(image, ImageTier.THUMB) != cache.etag_for(image, ImageTier.PREVIEW)
    assert cache.etag_for(image, ImageTier.THUMB) != cache.etag_for(other, ImageTier.THUMB)
    assert image.sha256 in cache.etag_for(image, ImageTier.THUMB)


# -- delivery ------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("tier", "media_type"),
    [("thumb", "image/webp"), ("preview", "image/webp"), ("full", "image/png")],
)
def test_every_tier_is_served_with_its_media_type_and_an_immutable_etag(
    client: TestClient, catalogued: dict[str, Any], tier: str, media_type: str
) -> None:
    image_id = catalogued["image_ids"][0]

    response = client.get(f"/api/images/{image_id}/{tier}")

    assert response.status_code == 200
    assert response.headers["content-type"] == media_type
    assert response.headers["cache-control"] == "public, max-age=31536000, immutable"
    assert response.headers["etag"].startswith('"')
    assert len(response.content) > 0


def test_a_client_that_already_has_the_bytes_gets_a_304(
    client: TestClient, catalogued: dict[str, Any]
) -> None:
    image_id = catalogued["image_ids"][0]
    first = client.get(f"/api/images/{image_id}/thumb")

    second = client.get(
        f"/api/images/{image_id}/thumb", headers={"If-None-Match": first.headers["etag"]}
    )

    assert second.status_code == 304
    assert second.content == b""
    assert second.headers["etag"] == first.headers["etag"]


def test_an_eight_bit_and_a_twenty_four_bit_image_both_render(
    client: TestClient, catalogued: dict[str, Any]
) -> None:
    """The mixed-bit-depth promise of §9, exercised rather than asserted in prose."""
    for image_id in catalogued["image_ids"][:2]:
        response = client.get(f"/api/images/{image_id}/thumb")
        assert response.status_code == 200
        assert PILImage.open(io.BytesIO(response.content)).format == "WEBP"


def test_an_unknown_image_is_a_404(client: TestClient) -> None:
    assert client.get("/api/images/4242/thumb").status_code == 404


def test_an_unknown_tier_is_rejected_by_the_schema(
    client: TestClient, catalogued: dict[str, Any]
) -> None:
    """The tier is an enum, so `../../app.sqlite3` is not a tier."""
    image_id = catalogued["image_ids"][0]

    assert client.get(f"/api/images/{image_id}/original").status_code == 422


def test_a_source_file_that_disappeared_is_reported_as_gone(
    client: TestClient, catalogued: dict[str, Any]
) -> None:
    """Images are referenced in place, so this is a state the catalog can reach."""
    image_id = catalogued["image_ids"][0]
    Path(catalogued["paths"][0]).unlink()

    response = client.get(f"/api/images/{image_id}/full")

    assert response.status_code == 410


# -- pre-warm ------------------------------------------------------------------------


def test_prewarm_renders_every_cacheable_tier(
    client: TestClient, settings: Settings, catalogued: dict[str, Any]
) -> None:
    started = client.post("/api/images/prewarm", json={"dataset_id": catalogued["dataset_id"]})
    job = _await_job(client, started.json()["id"])

    assert job["status"] == "succeeded", job["error"]
    assert job["result"]["images"] == 4
    assert job["result"]["rendered"] == 8
    assert job["result"]["failed"] == 0
    assert sorted(job["result"]["tiers"]) == ["preview", "thumb"]

    for image_id in catalogued["image_ids"]:
        for tier in (ImageTier.THUMB, ImageTier.PREVIEW):
            assert (settings.thumbnails_dir / tier.value / f"{image_id}.webp").is_file()


def test_prewarm_can_be_narrowed_to_one_tier(
    client: TestClient, settings: Settings, catalogued: dict[str, Any]
) -> None:
    started = client.post(
        "/api/images/prewarm",
        json={"dataset_id": catalogued["dataset_id"], "tiers": ["thumb"]},
    )
    job = _await_job(client, started.json()["id"])

    assert job["result"]["rendered"] == 4
    assert not (settings.thumbnails_dir / "preview").exists()


def test_prewarm_reports_an_unreadable_file_instead_of_failing_the_job(
    client: TestClient, catalogued: dict[str, Any]
) -> None:
    """One bad file must not cost the other several hundred their thumbnails."""
    Path(catalogued["paths"][0]).write_bytes(b"no longer an image")

    started = client.post("/api/images/prewarm", json={"dataset_id": catalogued["dataset_id"]})
    job = _await_job(client, started.json()["id"])

    assert job["status"] == "succeeded"
    assert job["result"]["failed"] == 2  # both tiers of the one broken file
    assert job["result"]["rendered"] == 6


def test_prewarming_an_unknown_dataset_is_a_404(client: TestClient) -> None:
    assert client.post("/api/images/prewarm", json={"dataset_id": 4242}).status_code == 404
