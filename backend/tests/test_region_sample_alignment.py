"""A region profile can give every image of a sample one shared crop (`union`)."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from anomaly_lab.config import Settings
from anomaly_lab.db.connection import connection
from anomaly_lab.db.migrate import apply_schema
from anomaly_lab.db.repositories import datasets, images, samples
from anomaly_lab.db.repositories import region_profiles as profiles_repo
from anomaly_lab.domain.entities import (
    JobKind,
    Label,
    RegionProfileRevision,
    SampleAlignment,
)
from anomaly_lab.jobs.context import JobContext
from anomaly_lab.regions import registry
from anomaly_lab.regions.identity import IdentityExtractor
from anomaly_lab.regions.preparation import (
    MANIFEST_FILENAME,
    PREVIEW_LIMIT,
    RegionBuildSummary,
    _config_digest,
    preview_sample_indices,
    run_region_prepare_job,
)
from tests.conftest import SeededCatalog

Box = tuple[int, int, int, int]


def _write(path: Path, size: tuple[int, int], box: Box | None) -> None:
    """A black frame with one bright rectangle — or none, which the threshold refuses."""
    width, height = size
    pixels = np.zeros((height, width, 3), dtype=np.uint8)
    if box is not None:
        left, top, right, bottom = box
        pixels[top:bottom, left:right] = 220
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pixels, mode="RGB").save(path)


def _seed(
    settings: Settings,
    root: Path,
    parts: Sequence[Sequence[tuple[tuple[int, int], Box | None]]],
    *,
    alignment: SampleAlignment,
) -> tuple[int, int, list[list[int]]]:
    """One sample per part, one image per channel; returns dataset, profile, image ids."""
    apply_schema(settings.db_path)
    with connection(settings.db_path) as conn:
        dataset = datasets.create_dataset(conn, name="grouped", root_path=str(root))
        channel_count = max(len(part) for part in parts)
        channels = [
            datasets.upsert_channel(conn, dataset.id, name=f"c{index}", position=index)
            for index in range(channel_count)
        ]
        ids: list[list[int]] = []
        for part_index, part in enumerate(parts):
            sample, _ = samples.upsert_sample(
                conn,
                dataset.id,
                group_key="all",
                external_id=f"part-{part_index}",
                label=Label.NORMAL,
            )
            part_ids: list[int] = []
            for channel_index, (size, box) in enumerate(part):
                path = root / f"p{part_index}" / f"c{channel_index}.png"
                _write(path, size, box)
                image, _ = images.upsert_image(
                    conn,
                    sample.id,
                    channel_id=channels[channel_index].id,
                    path=str(path),
                    width=size[0],
                    height=size[1],
                    bit_depth=24,
                    file_size=path.stat().st_size,
                    sha256=f"sha-{part_index}-{channel_index}",
                )
                part_ids.append(image.id)
            ids.append(part_ids)
        profile = profiles_repo.create_revision(
            conn,
            dataset_id=dataset.id,
            name="dominant",
            extractor_type="foreground_threshold",
            extractor_config={},
            prepared_width=32,
            prepared_height=32,
            padding_fraction=0.0,
            sample_alignment=alignment,
        )
    return dataset.id, profile.id, ids


def _run(settings: Settings, dataset_id: int, profile_id: int, mode: str) -> dict[str, Any]:
    return run_region_prepare_job(
        JobContext(
            job_id=7,
            kind=JobKind.REGION_PREPARE,
            params={"dataset_id": dataset_id, "profile_id": profile_id, "mode": mode},
            settings=settings,
        )
    )


def _crop(entry: dict[str, Any]) -> Box:
    transform = entry["transform"]
    return (
        transform["crop_left"],
        transform["crop_top"],
        transform["crop_right"],
        transform["crop_bottom"],
    )


TWO_CHANNELS = [[((64, 48), (8, 10, 24, 30)), ((64, 48), (30, 4, 50, 20))]]


def test_union_gives_both_channels_of_a_part_the_same_transform(
    settings: Settings, tmp_path: Path
) -> None:
    dataset_id, profile_id, ids = _seed(
        settings, tmp_path, TWO_CHANNELS, alignment=SampleAlignment.UNION
    )

    summary = RegionBuildSummary.model_validate(_run(settings, dataset_id, profile_id, "build"))

    assert summary.failed == 0
    by_id = {entry.image_id: entry for entry in summary.preview_entries}
    first, second = (by_id[image_id].transform for image_id in ids[0])
    assert first == second
    assert first is not None
    assert (first.crop_left, first.crop_top, first.crop_right, first.crop_bottom) == (
        8,
        4,
        50,
        30,
    )


def test_per_image_default_keeps_each_channel_its_own_crop(
    settings: Settings, tmp_path: Path
) -> None:
    dataset_id, profile_id, ids = _seed(
        settings, tmp_path, TWO_CHANNELS, alignment=SampleAlignment.PER_IMAGE
    )

    result = _run(settings, dataset_id, profile_id, "preview")

    crops = {entry["image_id"]: _crop(entry) for entry in result["entries"]}
    assert crops == {ids[0][0]: (8, 10, 24, 30), ids[0][1]: (30, 4, 50, 20)}


def test_a_sample_of_one_image_is_unchanged_by_union(settings: Settings, tmp_path: Path) -> None:
    single = [[((64, 48), (8, 10, 24, 30))]]
    dataset_id, per_image, _ = _seed(
        settings, tmp_path / "a", single, alignment=SampleAlignment.PER_IMAGE
    )
    with connection(settings.db_path) as conn:
        union = profiles_repo.create_revision(
            conn,
            dataset_id=dataset_id,
            name="dominant",
            extractor_type="foreground_threshold",
            extractor_config={},
            prepared_width=32,
            prepared_height=32,
            padding_fraction=0.0,
            sample_alignment=SampleAlignment.UNION,
        )

    own = _run(settings, dataset_id, per_image, "preview")["entries"]
    shared = _run(settings, dataset_id, union.id, "preview")["entries"]

    assert [entry["transform"] for entry in own] == [entry["transform"] for entry in shared]


def test_images_of_different_sizes_fail_by_name(settings: Settings, tmp_path: Path) -> None:
    parts = [[((64, 48), (8, 10, 24, 30)), ((60, 48), (30, 4, 50, 20))]]
    dataset_id, profile_id, ids = _seed(settings, tmp_path, parts, alignment=SampleAlignment.UNION)

    result = _run(settings, dataset_id, profile_id, "preview")

    assert result["failed"] == 2
    for entry in result["entries"]:
        assert entry["transform"] is None
        assert f"image {ids[0][0]} is 64x48" in entry["error"]
        assert f"image {ids[0][1]} is 60x48" in entry["error"]


def test_a_failed_sibling_fails_the_whole_sample(settings: Settings, tmp_path: Path) -> None:
    parts: list[list[tuple[tuple[int, int], Box | None]]] = [
        [((64, 48), (8, 10, 24, 30)), ((64, 48), None)],
        [((64, 48), (8, 10, 24, 30)), ((64, 48), (30, 4, 50, 20))],
    ]
    dataset_id, profile_id, ids = _seed(settings, tmp_path, parts, alignment=SampleAlignment.UNION)

    summary = RegionBuildSummary.model_validate(_run(settings, dataset_id, profile_id, "build"))

    assert (summary.succeeded, summary.failed) == (2, 2)
    failures = {entry.image_id: entry.error or "" for entry in summary.failure_examples}
    assert set(failures) == set(ids[0])
    # The image that failed keeps its own reason; its sibling names it.
    assert f"image {ids[0][1]} failed" in failures[ids[0][0]]
    assert "was not united" not in failures[ids[0][1]]
    # The manifest keeps the dataset's image order, as a per-image build's does.
    manifest = settings.region_profile_dir(profile_id) / MANIFEST_FILENAME
    ordered = [
        json.loads(line)["image_id"] for line in manifest.read_text(encoding="utf-8").splitlines()
    ]
    assert ordered == sorted(image_id for part_ids in ids for image_id in part_ids)


def test_union_preview_returns_complete_samples_within_budget(
    settings: Settings, tmp_path: Path
) -> None:
    part = [((32, 24), (4, 4, 12, 12)), ((32, 24), (10, 6, 20, 18)), ((32, 24), (2, 2, 8, 8))]
    dataset_id, profile_id, ids = _seed(
        settings, tmp_path, [part] * 11, alignment=SampleAlignment.UNION
    )

    result = _run(settings, dataset_id, profile_id, "preview")

    shown = {entry["image_id"] for entry in result["entries"]}
    assert len(shown) == 24
    samples_shown = [part_ids for part_ids in ids if shown & set(part_ids)]
    assert len(samples_shown) == 8
    assert all(set(part_ids) <= shown for part_ids in samples_shown)
    assert ids[0][0] in shown and ids[-1][0] in shown


def test_preview_sample_indices_take_whole_samples_evenly() -> None:
    sample_ids = [index // 2 for index in range(2 * 40)]
    chosen = preview_sample_indices(sample_ids)

    assert len(chosen) == PREVIEW_LIMIT
    chosen_samples = {sample_ids[index] for index in chosen}
    assert len(chosen_samples) == PREVIEW_LIMIT // 2
    assert {0, 39} <= chosen_samples
    # A sample larger than the budget is still shown whole to the extractor.
    assert preview_sample_indices([5] * 30) == list(range(30))
    assert preview_sample_indices([]) == []


def test_the_config_digest_covers_every_configuration_field_and_nothing_else() -> None:
    """A build is matched to its revision by configuration, not by row identity."""
    profile = RegionProfileRevision(
        id=7,
        dataset_id=3,
        name="rim",
        revision_no=2,
        extractor_type="center_crop",
        extractor_config={
            "width_fraction": 0.5,
            "height_fraction": 0.5,
            "center_x_fraction": 0.5,
            "center_y_fraction": 0.5,
        },
        prepared_width=256,
        prepared_height=256,
        padding_fraction=0.05,
        created_at="2026-01-01T00:00:00.000Z",
    )

    assert profile.sample_alignment is SampleAlignment.PER_IMAGE
    renumbered = profile.model_copy(update={"id": 8, "created_at": "2026-02-02T00:00:00.000Z"})
    assert _config_digest(renumbered) == _config_digest(profile)
    united = profile.model_copy(update={"sample_alignment": SampleAlignment.UNION})
    assert _config_digest(united) != _config_digest(profile)
    wider = profile.model_copy(update={"prepared_width": 320})
    assert _config_digest(wider) != _config_digest(profile)


def test_a_per_image_manifest_carries_no_new_field(settings: Settings, tmp_path: Path) -> None:
    dataset_id, profile_id, _ = _seed(
        settings, tmp_path, TWO_CHANNELS, alignment=SampleAlignment.PER_IMAGE
    )
    _run(settings, dataset_id, profile_id, "build")

    manifest = settings.region_profile_dir(profile_id) / MANIFEST_FILENAME
    for line in manifest.read_text(encoding="utf-8").splitlines():
        assert "sample_alignment" not in line


class _WarpExtractor(IdentityExtractor):
    crops_to_box: ClassVar[bool] = False


def test_union_is_refused_for_an_extractor_that_does_not_crop_to_a_box(
    client: TestClient, catalog: SeededCatalog, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(registry.LOADERS, "warp", lambda: _WarpExtractor)
    body: dict[str, Any] = {
        "name": "Warped",
        "extractor_type": "warp",
        "extractor_config": {},
        "prepared_width": 64,
        "prepared_height": 64,
    }
    url = f"/api/datasets/{catalog.dataset_id}/region-profiles"

    refused = client.post(url, json={**body, "sample_alignment": "union"})
    allowed = client.post(url, json=body)

    assert refused.status_code == 422
    assert "'warp'" in refused.json()["detail"]
    assert allowed.status_code == 200
    assert allowed.json()["sample_alignment"] == "per_image"


def test_the_api_stores_a_union_profile(client: TestClient, catalog: SeededCatalog) -> None:
    response = client.post(
        f"/api/datasets/{catalog.dataset_id}/region-profiles",
        json={
            "name": "Shared",
            "extractor_type": "foreground_threshold",
            "prepared_width": 64,
            "prepared_height": 64,
            "sample_alignment": "union",
        },
    )

    assert response.status_code == 200
    assert response.json()["sample_alignment"] == "union"
    fetched = client.get(f"/api/region-profiles/{response.json()['id']}")
    assert fetched.json()["sample_alignment"] == "union"
