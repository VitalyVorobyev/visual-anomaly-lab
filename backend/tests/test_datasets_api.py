"""The datasets, samples and splits API."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from anomaly_lab.config import Settings
from anomaly_lab.db.connection import connection
from anomaly_lab.db.repositories import datasets as datasets_repo
from anomaly_lab.db.repositories import experiments as experiments_repo
from anomaly_lab.db.repositories import images as images_repo
from anomaly_lab.db.repositories import jobs as jobs_repo
from anomaly_lab.db.repositories import region_profiles as region_profiles_repo
from anomaly_lab.db.repositories import samples as samples_repo
from anomaly_lab.db.repositories import splits as splits_repo
from anomaly_lab.domain.entities import JobKind, JobStatus, Label, Subset
from anomaly_lab.media.cache import ImageTier, cache_path


@pytest.fixture
def dataset_id(client: TestClient, settings: Settings) -> int:
    """A dataset with two capture groups, three channels, and one two-channel sample.

    The odd sample is deliberate: every list, filter and detail response below has to
    handle it without a special case (ADR-0005).
    """
    with connection(settings.db_path) as conn:
        dataset = datasets_repo.create_dataset(
            conn, name="fixture", root_path="/roots/fixture", adapter="channel_folders"
        )
        channels = {
            name: datasets_repo.upsert_channel(conn, dataset.id, name=name, position=index).id
            for index, name in enumerate(("bright", "dark", "dome"))
        }

        plan: list[tuple[str, str, Label, tuple[str, ...]]] = [
            ("set1/defect", "1", Label.DEFECT, ("bright", "dark", "dome")),
            ("set1/defect", "2", Label.DEFECT, ("bright", "dark", "dome")),
            ("set1/no-defect", "1", Label.NORMAL, ("bright", "dark", "dome")),
            ("set1/no-defect", "10", Label.NORMAL, ("bright", "dark", "dome")),
            ("set1/no-defect", "2", Label.NORMAL, ("bright", "dark", "dome")),
            ("set2/odd", "1", Label.UNLABELED, ("bright", "dark")),
        ]
        for group_key, external_id, label, channel_names in plan:
            sample, _ = samples_repo.upsert_sample(
                conn, dataset.id, group_key=group_key, external_id=external_id, label=label
            )
            for name in channel_names:
                images_repo.upsert_image(
                    conn,
                    sample.id,
                    channel_id=channels[name],
                    path=f"/roots/fixture/{group_key}/{name}/{external_id}.bmp",
                    width=1280,
                    height=1024,
                    bit_depth=24,
                    file_size=3_932_214,
                    sha256=f"{group_key}-{external_id}-{name}",
                )
    return dataset.id


def _samples(client: TestClient, dataset_id: int, **params: Any) -> dict[str, Any]:
    response = client.get(f"/api/datasets/{dataset_id}/samples", params=params)
    assert response.status_code == 200, response.text
    return dict(response.json())


# -- datasets ------------------------------------------------------------------------


def test_the_catalog_lists_counts_per_label(client: TestClient, dataset_id: int) -> None:
    listed = client.get("/api/datasets").json()

    assert len(listed) == 1
    assert listed[0]["samples"] == 6
    assert listed[0]["images"] == 17
    assert listed[0]["label_counts"] == {"normal": 3, "defect": 2, "unlabeled": 1}


def test_dataset_detail_carries_the_channel_dictionary_and_groups(
    client: TestClient, dataset_id: int
) -> None:
    """The browser builds its filters from this, so the channel list is data, not schema."""
    detail = client.get(f"/api/datasets/{dataset_id}").json()

    assert [c["name"] for c in detail["channels"]] == ["bright", "dark", "dome"]
    assert detail["group_keys"] == ["set1/defect", "set1/no-defect", "set2/odd"]
    assert detail["adapter"] == "channel_folders"
    assert detail["splits"] == 0


def test_an_unknown_dataset_is_a_404(client: TestClient) -> None:
    assert client.get("/api/datasets/4242").status_code == 404
    assert client.get("/api/datasets/4242/samples").status_code == 404
    assert client.delete("/api/datasets/4242").status_code == 404
    assert client.get("/api/datasets/4242/deletion-preview").status_code == 404


def test_deleting_a_dataset_removes_its_rows(
    client: TestClient, settings: Settings, dataset_id: int
) -> None:
    response = client.delete(f"/api/datasets/{dataset_id}")
    assert response.status_code == 200
    assert response.json()["deleted"] is True

    assert client.get("/api/datasets").json() == []
    with connection(settings.db_path) as conn:
        assert samples_repo.count_samples(conn, dataset_id) == 0


def test_dataset_deletion_previews_and_removes_only_app_owned_state(
    client: TestClient, settings: Settings, dataset_id: int, tmp_path: Path
) -> None:
    source = tmp_path / "external-source.png"
    source.write_bytes(b"source-sentinel")

    with connection(settings.db_path) as conn:
        sample_id = int(
            conn.execute(
                "SELECT id FROM sample WHERE dataset_id = ? LIMIT 1", (dataset_id,)
            ).fetchone()[0]
        )
        conn.execute("UPDATE sample SET label_source = 'manual' WHERE id = ?", (sample_id,))
        image_ids = [
            int(row[0])
            for row in conn.execute(
                "SELECT image.id FROM image JOIN sample ON sample.id = image.sample_id "
                "WHERE sample.dataset_id = ? ORDER BY image.id",
                (dataset_id,),
            )
        ]
        conn.execute("UPDATE image SET path = ? WHERE id = ?", (str(source), image_ids[0]))
        split = splits_repo.create_split(
            conn,
            dataset_id,
            name="delete-me",
            strategy="manual",
            seed=0,
            params={},
            assignments={},
        )
        experiment = experiments_repo.create_experiment(
            conn,
            name="owned-run",
            dataset_id=dataset_id,
            split_id=split.id,
            model_type="pixel_reference",
            model_config={},
            preprocessing_config={},
            eval_config={},
            artifact_dir="pending",
        )
        artifact_dir = settings.experiment_dir(experiment.id)
        conn.execute(
            "UPDATE experiment SET artifact_dir = ? WHERE id = ?",
            (str(artifact_dir), experiment.id),
        )
        manifest = settings.manifests_dir / f"dataset-{dataset_id}-20260812T120000Z.json"
        conn.execute(
            "UPDATE dataset SET root_path = ?, manifest_path = ? WHERE id = ?",
            (str(tmp_path), str(manifest), dataset_id),
        )
        dataset_job = jobs_repo.create_job(
            conn, kind=JobKind.PREWARM, params={"dataset_id": dataset_id}
        )
        job_log = settings.jobs_log_dir / f"{dataset_job.id}.log"
        jobs_repo.mark_running(conn, dataset_job.id, log_path=str(job_log))
        jobs_repo.finish_job(conn, dataset_job.id, status=JobStatus.SUCCEEDED)
        region_profiles_repo.create_revision(
            conn,
            dataset_id=dataset_id,
            name="delete-me",
            extractor_type="identity",
            extractor_config={},
            prepared_width=256,
            prepared_height=256,
            padding_fraction=0.05,
            seed=17,
        )

    (artifact_dir / "maps").mkdir(parents=True)
    (artifact_dir / "model.bin").write_bytes(b"abc")
    (artifact_dir / "maps" / "one.npy").write_bytes(b"defg")
    manifest.parent.mkdir(parents=True)
    manifest.write_bytes(b"manifest")
    thumb = cache_path(settings, image_ids[0], ImageTier.THUMB)
    preview = cache_path(settings, image_ids[1], ImageTier.PREVIEW)
    thumb.parent.mkdir(parents=True)
    preview.parent.mkdir(parents=True)
    thumb.write_bytes(b"123")
    preview.write_bytes(b"4567")
    job_log.parent.mkdir(parents=True, exist_ok=True)
    job_log.write_bytes(b"job")

    deletion = client.get(f"/api/datasets/{dataset_id}/deletion-preview")
    assert deletion.status_code == 200
    assert deletion.json() == {
        "dataset_id": dataset_id,
        "name": "fixture",
        "samples": 6,
        "images": 17,
        "splits": 1,
        "experiments": 1,
        "jobs": 1,
        "region_profiles": 1,
        "manual_labels": 1,
        "generated_files": 6,
        "generated_bytes": 25,
        "active_jobs": [],
        "resident_loaded": False,
        "storage_locations_safe": True,
        "can_delete": True,
        "blocker": None,
    }

    result = client.delete(f"/api/datasets/{dataset_id}")
    assert result.status_code == 200
    assert result.json() == {
        "deleted": True,
        "freed_files": 6,
        "freed_bytes": 25,
        "cleanup_errors": [],
    }
    assert source.read_bytes() == b"source-sentinel"
    assert not artifact_dir.exists()
    assert not manifest.exists()
    assert not thumb.exists()
    assert not preview.exists()
    assert not job_log.exists()
    with connection(settings.db_path) as conn:
        assert datasets_repo.get_dataset(conn, dataset_id) is None
        assert experiments_repo.get_experiment(conn, experiment.id) is None
        assert jobs_repo.get_job(conn, dataset_job.id) is None
        assert region_profiles_repo.list_profiles(conn, dataset_id) == []


def test_active_dataset_work_blocks_preview_and_delete(
    client: TestClient, settings: Settings, dataset_id: int
) -> None:
    with connection(settings.db_path) as conn:
        job = jobs_repo.create_job(conn, kind=JobKind.PREWARM, params={"dataset_id": dataset_id})
        jobs_repo.mark_running(conn, job.id, log_path=str(settings.jobs_log_dir / "active.log"))

    preview = client.get(f"/api/datasets/{dataset_id}/deletion-preview")
    assert preview.json()["can_delete"] is False
    assert [job["id"] for job in preview.json()["active_jobs"]] == [job.id]
    assert client.delete(f"/api/datasets/{dataset_id}").status_code == 409
    assert client.get(f"/api/datasets/{dataset_id}").status_code == 200


def test_dataset_deletion_refuses_a_manifest_path_outside_app_storage(
    client: TestClient, settings: Settings, dataset_id: int, tmp_path: Path
) -> None:
    sentinel = tmp_path / "external-manifest.json"
    sentinel.write_text("leave me", encoding="utf-8")
    with connection(settings.db_path) as conn:
        conn.execute(
            "UPDATE dataset SET manifest_path = ? WHERE id = ?", (str(sentinel), dataset_id)
        )

    preview = client.get(f"/api/datasets/{dataset_id}/deletion-preview").json()
    assert preview["storage_locations_safe"] is False
    assert preview["can_delete"] is False
    assert client.delete(f"/api/datasets/{dataset_id}").status_code == 409
    assert sentinel.read_text(encoding="utf-8") == "leave me"


# -- samples -------------------------------------------------------------------------


def test_samples_are_paged_in_a_stable_natural_order(client: TestClient, dataset_id: int) -> None:
    """`10` follows `9`, not `1`, or the grid order looks broken to a human."""
    page = _samples(client, dataset_id, limit=100)

    identities = [(s["group_key"], s["external_id"]) for s in page["items"]]
    assert identities == [
        ("set1/defect", "1"),
        ("set1/defect", "2"),
        ("set1/no-defect", "1"),
        ("set1/no-defect", "2"),
        ("set1/no-defect", "10"),
        ("set2/odd", "1"),
    ]
    assert page["total"] == 6


def test_paging_reports_the_unfiltered_total_of_the_filtered_set(
    client: TestClient, dataset_id: int
) -> None:
    first = _samples(client, dataset_id, limit=2, offset=0)
    second = _samples(client, dataset_id, limit=2, offset=2)

    assert first["total"] == second["total"] == 6
    assert len(first["items"]) == len(second["items"]) == 2
    assert {s["id"] for s in first["items"]}.isdisjoint({s["id"] for s in second["items"]})


def test_each_sample_carries_its_images_in_channel_order(
    client: TestClient, dataset_id: int
) -> None:
    page = _samples(client, dataset_id, limit=100)
    three_channel = page["items"][0]
    two_channel = next(s for s in page["items"] if s["group_key"] == "set2/odd")

    assert [image["channel"] for image in three_channel["images"]] == ["bright", "dark", "dome"]
    # The irregular sample is described by the same shape, one entry shorter.
    assert [image["channel"] for image in two_channel["images"]] == ["bright", "dark"]


def test_filtering_by_label(client: TestClient, dataset_id: int) -> None:
    page = _samples(client, dataset_id, label="defect")

    assert page["total"] == 2
    assert {s["label"] for s in page["items"]} == {"defect"}


def test_filtering_by_channel_selects_samples_not_images(
    client: TestClient, dataset_id: int
) -> None:
    """ "Has a dome image" is a property of the sample; the response is still whole samples."""
    detail = client.get(f"/api/datasets/{dataset_id}").json()
    dome = next(c for c in detail["channels"] if c["name"] == "dome")["id"]

    page = _samples(client, dataset_id, channel_id=dome)

    assert page["total"] == 5
    assert all(len(s["images"]) == 3 for s in page["items"])
    assert all(s["group_key"] != "set2/odd" for s in page["items"])


def test_a_label_edit_is_marked_manual(client: TestClient, dataset_id: int) -> None:
    sample = _samples(client, dataset_id, label="unlabeled")["items"][0]

    updated = client.patch(
        f"/api/datasets/{dataset_id}/samples/{sample['id']}", json={"label": "defect"}
    ).json()

    assert updated["label"] == "defect"
    assert updated["label_source"] == "manual"


def test_a_sample_from_another_dataset_is_not_reachable_through_this_one(
    client: TestClient, settings: Settings, dataset_id: int
) -> None:
    with connection(settings.db_path) as conn:
        other = datasets_repo.create_dataset(conn, name="other", root_path="/roots/other")
        stranger, _ = samples_repo.upsert_sample(
            conn, other.id, group_key="g", external_id="1", label=Label.NORMAL
        )

    assert client.get(f"/api/datasets/{dataset_id}/samples/{stranger.id}").status_code == 404
    assert (
        client.patch(
            f"/api/datasets/{dataset_id}/samples/{stranger.id}", json={"label": "defect"}
        ).status_code
        == 404
    )


# -- bulk labelling ------------------------------------------------------------------


def _bulk(client: TestClient, dataset_id: int, **body: Any) -> dict[str, Any]:
    response = client.patch(f"/api/datasets/{dataset_id}/samples", json=body)
    assert response.status_code == 200, response.text
    return dict(response.json())


def test_a_selection_is_labelled_in_one_request_and_marked_manual(
    client: TestClient, dataset_id: int
) -> None:
    chosen = [s["id"] for s in _samples(client, dataset_id, label="normal")["items"]]

    assert _bulk(client, dataset_id, label="defect", sample_ids=chosen) == {"updated": 3}

    after = _samples(client, dataset_id, label="defect")["items"]
    relabelled = [s for s in after if s["id"] in set(chosen)]
    assert len(relabelled) == len(chosen)
    assert all(s["label_source"] == "manual" for s in relabelled)


def test_labelling_by_filter_touches_exactly_what_the_grid_counted(
    client: TestClient, dataset_id: int
) -> None:
    """The count the UI shows and the set that gets labelled come from one clause."""
    counted = _samples(client, dataset_id, label="unlabeled")["total"]

    result = _bulk(client, dataset_id, label="normal", filters={"label": "unlabeled"})

    assert result["updated"] == counted
    assert _samples(client, dataset_id, label="unlabeled")["total"] == 0


def test_empty_filters_mean_the_whole_dataset(client: TestClient, dataset_id: int) -> None:
    total = _samples(client, dataset_id)["total"]

    assert _bulk(client, dataset_id, label="normal", filters={})["updated"] == total
    assert _samples(client, dataset_id, label="normal")["total"] == total


def test_a_filter_that_matches_nothing_labels_nothing(client: TestClient, dataset_id: int) -> None:
    _bulk(client, dataset_id, label="normal", filters={"label": "unlabeled"})

    assert _bulk(client, dataset_id, label="defect", filters={"label": "unlabeled"}) == {
        "updated": 0
    }


def test_bulk_labelling_cannot_reach_into_another_dataset(
    client: TestClient, settings: Settings, dataset_id: int
) -> None:
    with connection(settings.db_path) as conn:
        other = datasets_repo.create_dataset(conn, name="other", root_path="/roots/other")
        stranger, _ = samples_repo.upsert_sample(
            conn, other.id, group_key="g", external_id="1", label=Label.NORMAL
        )

    assert _bulk(client, dataset_id, label="defect", sample_ids=[stranger.id]) == {"updated": 0}

    with connection(settings.db_path) as conn:
        untouched = samples_repo.get_sample(conn, stranger.id)
    assert untouched is not None
    assert untouched.label is Label.NORMAL


def test_naming_the_target_twice_or_not_at_all_is_rejected(
    client: TestClient, dataset_id: int
) -> None:
    for body in (
        {"label": "normal"},
        {"label": "normal", "sample_ids": [1], "filters": {}},
    ):
        assert client.patch(f"/api/datasets/{dataset_id}/samples", json=body).status_code == 422

    # Nothing was written by either rejected request.
    assert _samples(client, dataset_id, label="unlabeled")["total"] == 1


# -- splits --------------------------------------------------------------------------


def _create_split(client: TestClient, dataset_id: int, **body: Any) -> dict[str, Any]:
    response = client.post(
        "/api/splits", json={"dataset_id": dataset_id, "name": "s", "seed": 7, **body}
    )
    assert response.status_code == 200, response.text
    return dict(response.json())


def test_a_split_trains_on_normals_only(client: TestClient, dataset_id: int) -> None:
    """A defect in the training set teaches the model that defects are normal."""
    split = _create_split(client, dataset_id)

    train = next(row for row in split["composition"] if row["subset"] == "train")
    assert train["defect"] == 0
    assert train["normal"] > 0
    assert train["unlabeled"] == 0


def test_every_labelled_sample_lands_somewhere(client: TestClient, dataset_id: int) -> None:
    split = _create_split(client, dataset_id)

    assert sum(row["total"] for row in split["composition"]) == 6
    assert sum(row["defect"] for row in split["composition"]) == 2
    # Unlabelled samples are assigned so they can be scored, and excluded from metrics
    # later by the evaluation layer rather than by being left out here (§8).
    assert sum(row["unlabeled"] for row in split["composition"]) == 1


def test_the_same_seed_and_params_reproduce_the_same_split(
    client: TestClient, settings: Settings, dataset_id: int
) -> None:
    first = _create_split(client, dataset_id, name="a")
    second = _create_split(client, dataset_id, name="b")

    with connection(settings.db_path) as conn:
        assignments = {
            split_id: {
                subset: splits_repo.list_sample_ids(conn, split_id, subset) for subset in Subset
            }
            for split_id in (first["id"], second["id"])
        }

    assert assignments[first["id"]] == assignments[second["id"]]


def test_a_different_seed_draws_a_different_split(
    client: TestClient, settings: Settings, dataset_id: int
) -> None:
    """Needs enough normals to be meaningful.

    With the three normals of the base fixture, two seeds picking the same two of three
    is a one-in-three coincidence rather than evidence of anything.
    """
    with connection(settings.db_path) as conn:
        for index in range(30):
            samples_repo.upsert_sample(
                conn,
                dataset_id,
                group_key="set3/no-defect",
                external_id=str(index),
                label=Label.NORMAL,
            )

    first = _create_split(client, dataset_id, name="a", seed=1)
    second = _create_split(client, dataset_id, name="b", seed=999)

    with connection(settings.db_path) as conn:
        train_a = splits_repo.list_sample_ids(conn, first["id"], Subset.TRAIN)
        train_b = splits_repo.list_sample_ids(conn, second["id"], Subset.TRAIN)

    assert len(train_a) == len(train_b)
    assert train_a != train_b


def test_the_params_travel_with_the_split(client: TestClient, dataset_id: int) -> None:
    """A seed alone reproduces nothing; the fractions are part of the record."""
    split = _create_split(client, dataset_id, params={"train_normal_fraction": 0.4})

    assert split["params"]["train_normal_fraction"] == 0.4
    assert split["params"]["strategy"] == "normal_only_train"
    assert split["seed"] == 7


def test_a_split_is_readable_after_the_fact(client: TestClient, dataset_id: int) -> None:
    """Persistence is the M2 exit criterion; this is its API half."""
    created = _create_split(client, dataset_id)

    fetched = client.get(f"/api/splits/{created['id']}").json()
    listed = client.get("/api/splits", params={"dataset_id": dataset_id}).json()

    assert fetched == created
    assert listed == [created]


def test_samples_can_be_filtered_by_split_subset(client: TestClient, dataset_id: int) -> None:
    split = _create_split(client, dataset_id)
    train_total = next(r for r in split["composition"] if r["subset"] == "train")["total"]

    page = _samples(client, dataset_id, split_id=split["id"], subset="train")

    assert page["total"] == train_total
    assert all(sample["label"] == "normal" for sample in page["items"])


def test_no_samples_images_straddle_a_subset_boundary(
    client: TestClient, settings: Settings, dataset_id: int
) -> None:
    """Structural, not conventional: there is no image-level assignment table at all."""
    split = _create_split(client, dataset_id)

    with connection(settings.db_path) as conn:
        seen: dict[int, Subset] = {}
        for subset in Subset:
            for sample_id in splits_repo.list_sample_ids(conn, split["id"], subset):
                assert sample_id not in seen, "a sample was assigned to two subsets"
                seen[sample_id] = subset

    assert len(seen) == 6


def test_a_duplicate_split_name_is_a_conflict(client: TestClient, dataset_id: int) -> None:
    _create_split(client, dataset_id, name="only")

    response = client.post(
        "/api/splits", json={"dataset_id": dataset_id, "name": "only", "seed": 1}
    )

    assert response.status_code == 409


def test_a_dataset_with_no_normals_cannot_be_split(client: TestClient, settings: Settings) -> None:
    with connection(settings.db_path) as conn:
        dataset = datasets_repo.create_dataset(conn, name="all-bad", root_path="/roots/bad")
        samples_repo.upsert_sample(
            conn, dataset.id, group_key="g", external_id="1", label=Label.DEFECT
        )

    response = client.post("/api/splits", json={"dataset_id": dataset.id, "name": "s", "seed": 1})

    assert response.status_code == 409
    assert "empty training set" in response.json()["detail"]


def test_impossible_fractions_are_rejected_before_anything_is_written(
    client: TestClient, dataset_id: int
) -> None:
    response = client.post(
        "/api/splits",
        json={
            "dataset_id": dataset_id,
            "name": "s",
            "seed": 1,
            "params": {"train_normal_fraction": 0.9, "val_normal_fraction": 0.9},
        },
    )

    assert response.status_code == 422
    assert client.get("/api/splits", params={"dataset_id": dataset_id}).json() == []
