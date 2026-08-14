"""Bounded, reproducible region preview/build from generated image fixtures."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from anomaly_lab.config import Settings
from anomaly_lab.db.connection import connection
from anomaly_lab.db.migrate import apply_migrations
from anomaly_lab.db.repositories import region_profiles as profiles_repo
from anomaly_lab.domain.entities import JobKind
from anomaly_lab.jobs.context import JobContext
from anomaly_lab.regions.preparation import (
    MANIFEST_FILENAME,
    PREVIEW_LIMIT,
    RegionBuildSummary,
    load_prepared_build,
    preview_indices,
    read_build_summary,
    run_region_prepare_job,
)
from tests.conftest import Fixture, seed_synthetic_split


@pytest.fixture
def prepared_fixture(settings: Settings, tmp_path: Path) -> tuple[Fixture, int]:
    apply_migrations(settings.db_path)
    with connection(settings.db_path) as conn:
        fixture = seed_synthetic_split(conn, tmp_path / "synthetic")
        profile = profiles_repo.create_revision(
            conn,
            dataset_id=fixture.dataset_id,
            name="full frame",
            extractor_type="identity",
            extractor_config={},
            prepared_width=24,
            prepared_height=20,
            padding_fraction=0.05,
            seed=17,
        )
    return fixture, profile.id


def _context(
    settings: Settings,
    fixture: Fixture,
    profile_id: int,
    *,
    mode: str,
    job_id: int = 41,
) -> JobContext:
    return JobContext(
        job_id=job_id,
        kind=JobKind.REGION_PREPARE,
        params={"dataset_id": fixture.dataset_id, "profile_id": profile_id, "mode": mode},
        settings=settings,
    )


def test_preview_indices_are_bounded_and_cover_both_ends() -> None:
    chosen = preview_indices(103)

    assert len(chosen) == PREVIEW_LIMIT
    assert chosen[0] == 0
    assert chosen[-1] == 102
    assert chosen == sorted(set(chosen))
    assert preview_indices(3) == [0, 1, 2]


def test_preview_reports_transforms_without_writing_prepared_pixels(
    settings: Settings,
    prepared_fixture: tuple[Fixture, int],
) -> None:
    """What the handler computes, called directly.

    Directly is the limitation worth naming: this bypasses the worker pipe, so it says
    nothing about whether the result *reaches* the parent. A 24-entry preview frame
    exceeds 16 KiB and used to be split mid-JSON, which this test could not have caught
    and did not — see `test_an_event_larger_than_a_log_line_survives_the_split`.
    """
    fixture, profile_id = prepared_fixture

    result = run_region_prepare_job(_context(settings, fixture, profile_id, mode="preview"))

    assert result["mode"] == "preview"
    assert result["sampled"] == 14
    assert result["succeeded"] == 14
    assert result["failed"] == 0
    assert all(entry["transform"]["prepared_width"] == 24 for entry in result["entries"])
    assert not settings.region_profile_dir(profile_id).exists()


def test_build_is_immutable_and_records_every_source_transform(
    settings: Settings,
    prepared_fixture: tuple[Fixture, int],
) -> None:
    fixture, profile_id = prepared_fixture
    first = RegionBuildSummary.model_validate(
        run_region_prepare_job(_context(settings, fixture, profile_id, mode="build", job_id=42))
    )
    root = settings.region_profile_dir(profile_id)
    summary_path = root / "summary.json"
    original_summary = summary_path.read_bytes()
    first_manifest = (root / MANIFEST_FILENAME).read_bytes()
    first_images = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted((root / "images").iterdir())
    }

    with pytest.raises(ValueError, match="immutable completed build"):
        run_region_prepare_job(_context(settings, fixture, profile_id, mode="build", job_id=43))
    summary_path.unlink()
    with pytest.raises(ValueError, match="immutable completed build"):
        run_region_prepare_job(_context(settings, fixture, profile_id, mode="build", job_id=44))
    summary_path.write_bytes(original_summary)

    assert first.total == 14
    assert first.succeeded == 14
    assert first.failed == 0
    assert len(first.preview_entries) == 14
    assert first_manifest == (root / MANIFEST_FILENAME).read_bytes()
    assert first_images == {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted((root / "images").iterdir())
    }
    assert read_build_summary(settings, profile_id) == first
    lines = [json.loads(line) for line in first_manifest.decode().splitlines()]
    assert len(lines) == 14
    assert [entry.image_id for entry in first.preview_entries] == [
        line["image_id"] for line in lines
    ]
    assert all(line["status"] == "succeeded" for line in lines)
    assert all("elapsed_ms" not in line for line in lines)
    assert all(line["prepared_sha256"] for line in lines)

    with connection(settings.db_path) as conn:
        profile = profiles_repo.get_profile(conn, profile_id)
    assert profile is not None
    loaded = load_prepared_build(settings, profile, manifest_sha256=first.manifest_sha256)
    summary_path.write_text(
        first.model_copy(update={"profile_id": profile_id + 100}).model_dump_json(indent=2),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="build identity"):
        load_prepared_build(settings, profile, manifest_sha256=first.manifest_sha256)
    summary_path.write_bytes(original_summary)

    tampered_id = lines[0]["image_id"]
    loaded.image_path(tampered_id).write_bytes(b"not the prepared PNG")
    with pytest.raises(ValueError, match=f"prepared image {tampered_id} changed"):
        loaded.image_path(tampered_id)


def test_a_missing_source_is_a_persisted_failure_not_identity_fallback(
    settings: Settings,
    prepared_fixture: tuple[Fixture, int],
) -> None:
    fixture, profile_id = prepared_fixture
    with connection(settings.db_path) as conn:
        row = conn.execute(
            """
            SELECT image.path FROM image JOIN sample ON sample.id = image.sample_id
             WHERE sample.dataset_id = ? ORDER BY image.id LIMIT 1
            """,
            (fixture.dataset_id,),
        ).fetchone()
    Path(row["path"]).unlink()

    summary = RegionBuildSummary.model_validate(
        run_region_prepare_job(_context(settings, fixture, profile_id, mode="build", job_id=44))
    )

    assert summary.failed == 1
    assert summary.succeeded == 13
    assert len(summary.failure_examples) == 1
    assert summary.failure_examples[0].transform is None
    assert summary.failure_examples[0].error
    assert len(list((settings.region_profile_dir(profile_id) / "images").iterdir())) == 13
