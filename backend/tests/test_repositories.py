"""Dataset, sample, image and split repositories."""

from __future__ import annotations

import sqlite3

import pytest

from anomaly_lab.datasets.split_presets import compose, read_truth
from anomaly_lab.db.repositories import (
    datasets,
    experiments,
    images,
    region_profiles,
    results,
    samples,
    splits,
)
from anomaly_lab.domain.entities import (
    Aggregation,
    ImageResult,
    Label,
    LabelSource,
    SampleResult,
    Subset,
)
from tests.conftest import SeededCatalog


def test_dataset_is_found_by_its_root_path(migrated_db: sqlite3.Connection) -> None:
    """Re-import resolves the dataset to update by root, not by name (handbook import.md)."""
    created = datasets.create_dataset(migrated_db, name="d", root_path="/roots/d")

    assert datasets.find_dataset_by_root(migrated_db, "/roots/d") == created
    assert datasets.find_dataset_by_root(migrated_db, "/roots/other") is None


def test_two_datasets_cannot_share_a_root(migrated_db: sqlite3.Connection) -> None:
    datasets.create_dataset(migrated_db, name="first", root_path="/roots/d")

    with pytest.raises(sqlite3.IntegrityError):
        datasets.create_dataset(migrated_db, name="second", root_path="/roots/d")


def test_channel_dictionary_is_per_dataset_and_ordered(
    migrated_db: sqlite3.Connection, catalog: SeededCatalog
) -> None:
    """Channels are rows in a per-dataset dictionary, never a shared enum (ADR-0041)."""
    other = datasets.create_dataset(migrated_db, name="other", root_path="/roots/other")
    datasets.upsert_channel(migrated_db, other.id, name="dome", position=0)

    ours = datasets.list_channels(migrated_db, catalog.dataset_id)

    assert [c.name for c in ours] == ["bright", "dark"]
    assert [c.name for c in datasets.list_channels(migrated_db, other.id)] == ["dome"]


def test_upserting_a_channel_reuses_the_row_and_updates_its_position(
    migrated_db: sqlite3.Connection, catalog: SeededCatalog
) -> None:
    again = datasets.upsert_channel(migrated_db, catalog.dataset_id, name="bright", position=5)

    assert again.id == catalog.channel_ids["bright"]
    assert again.position == 5
    assert len(datasets.list_channels(migrated_db, catalog.dataset_id)) == 2


def test_a_populated_dataset_deletes_cleanly(
    migrated_db: sqlite3.Connection, catalog: SeededCatalog
) -> None:
    """`image.channel_id` is RESTRICT, so cascades alone cannot do this (schema note)."""
    assert datasets.delete_dataset(migrated_db, catalog.dataset_id) is True

    assert datasets.get_dataset(migrated_db, catalog.dataset_id) is None
    for table in ("sample", "image", "channel", "split"):
        remaining = migrated_db.execute(f"SELECT COUNT(*) AS n FROM {table}").fetchone()["n"]
        assert remaining == 0, table


def test_deleting_an_unknown_dataset_reports_rather_than_raises(
    migrated_db: sqlite3.Connection,
) -> None:
    assert datasets.delete_dataset(migrated_db, 4242) is False


def test_label_counts_include_labels_with_no_samples(
    migrated_db: sqlite3.Connection, catalog: SeededCatalog
) -> None:
    counts = datasets.label_counts(migrated_db, catalog.dataset_id)

    assert counts == {Label.NORMAL: 1, Label.DEFECT: 1, Label.UNLABELED: 1}


def test_upsert_sample_is_idempotent_on_its_natural_key(
    migrated_db: sqlite3.Connection, catalog: SeededCatalog
) -> None:
    first, created_first = samples.upsert_sample(
        migrated_db,
        catalog.dataset_id,
        group_key="group-c",
        external_id="9",
        label=Label.NORMAL,
    )
    second, created_second = samples.upsert_sample(
        migrated_db,
        catalog.dataset_id,
        group_key="group-c",
        external_id="9",
        label=Label.NORMAL,
    )

    assert created_first is True
    assert created_second is False
    assert first.id == second.id


def test_a_manual_label_survives_re_import(
    migrated_db: sqlite3.Connection, catalog: SeededCatalog
) -> None:
    """An operator's correction outranks whatever the folder structure says (handbook import.md)."""
    sample_id = catalog.sample_ids["group-a/1"]
    samples.set_label(migrated_db, sample_id, Label.DEFECT)

    reimported, created = samples.upsert_sample(
        migrated_db,
        catalog.dataset_id,
        group_key="group-a",
        external_id="1",
        label=Label.NORMAL,
    )

    assert created is False
    assert reimported.label is Label.DEFECT
    assert reimported.label_source is LabelSource.MANUAL


def test_an_imported_label_is_updated_when_the_source_changes(
    migrated_db: sqlite3.Connection, catalog: SeededCatalog
) -> None:
    reimported, _ = samples.upsert_sample(
        migrated_db,
        catalog.dataset_id,
        group_key="group-a",
        external_id="1",
        label=Label.DEFECT,
    )

    assert reimported.label is Label.DEFECT
    assert reimported.label_source is LabelSource.IMPORT


def test_samples_filter_by_label_and_channel(
    migrated_db: sqlite3.Connection, catalog: SeededCatalog
) -> None:
    by_label = samples.list_samples(
        migrated_db, catalog.dataset_id, samples.SampleFilter(label=Label.DEFECT)
    )
    by_channel = samples.list_samples(
        migrated_db,
        catalog.dataset_id,
        samples.SampleFilter(channel_id=catalog.channel_ids["dark"]),
    )

    assert [s.external_id for s in by_label] == ["2"]
    # The channel filter selects samples that *have* such an image, not the images.
    assert {s.external_id for s in by_channel} == {"1", "2"}
    assert all(s.group_key == "group-a" for s in by_channel)


def test_sample_paging_is_stable_and_numeric_ids_sort_naturally(
    migrated_db: sqlite3.Connection, catalog: SeededCatalog
) -> None:
    """`10` must not sort between `1` and `2`, or the grid order looks broken."""
    for external_id in ("10", "3"):
        samples.upsert_sample(
            migrated_db,
            catalog.dataset_id,
            group_key="group-a",
            external_id=external_id,
            label=Label.NORMAL,
        )

    page = samples.list_samples(migrated_db, catalog.dataset_id, limit=10)
    group_a = [s.external_id for s in page if s.group_key == "group-a"]

    assert group_a == ["1", "2", "3", "10"]
    assert samples.count_samples(migrated_db, catalog.dataset_id) == 5


def test_images_for_a_page_of_samples_are_grouped_without_n_plus_one(
    migrated_db: sqlite3.Connection, catalog: SeededCatalog
) -> None:
    sample_ids = list(catalog.sample_ids.values())
    grouped = images.list_images_for_samples(migrated_db, sample_ids)

    assert set(grouped) == set(sample_ids)
    assert len(grouped[catalog.sample_ids["group-a/1"]]) == 2
    # A single-image sample with no channel is an ordinary case, not a special one.
    solitary = grouped[catalog.sample_ids["group-b/1"]]
    assert len(solitary) == 1
    assert solitary[0].channel_id is None


def test_upsert_image_updates_in_place_and_preserves_first_import_time(
    migrated_db: sqlite3.Connection, catalog: SeededCatalog
) -> None:
    original = images.get_image(migrated_db, catalog.image_ids[0])
    assert original is not None

    updated, created = images.upsert_image(
        migrated_db,
        original.sample_id,
        channel_id=original.channel_id,
        path=original.path,
        width=16,
        height=16,
        bit_depth=24,
        file_size=999,
        sha256="rehashed",
    )

    assert created is False
    assert updated.id == original.id
    assert updated.sha256 == "rehashed"
    assert updated.imported_at == original.imported_at


def test_duplicate_hashes_are_reported_per_dataset(
    migrated_db: sqlite3.Connection, catalog: SeededCatalog
) -> None:
    sample_id = catalog.sample_ids["group-b/1"]
    images.upsert_image(
        migrated_db,
        sample_id,
        channel_id=None,
        path="/fixture/root/group-b/plain/copy.png",
        width=8,
        height=8,
        bit_depth=8,
        file_size=64,
        sha256="hash-group-b-1-None",
    )

    duplicates = images.find_duplicate_hashes(migrated_db, catalog.dataset_id)

    assert list(duplicates) == ["hash-group-b-1-None"]
    assert len(duplicates["hash-group-b-1-None"]) == 2


def test_split_composition_reports_every_subset(
    migrated_db: sqlite3.Connection, catalog: SeededCatalog
) -> None:
    split = splits.create_split(
        migrated_db,
        catalog.dataset_id,
        name="s",
        strategy="normal_only_train",
        seed=7,
        params={"ratios": {"train": 0.5}},
        assignments={
            catalog.sample_ids["group-a/1"]: Subset.TRAIN,
            catalog.sample_ids["group-a/2"]: Subset.TEST,
        },
    )

    truth = read_truth(migrated_db, catalog.dataset_id)
    placed = splits.assignments(migrated_db, split.id)
    by_subset = {c.subset: c for c in compose(placed, truth)}

    assert split.params == {"ratios": {"train": 0.5}}
    assert by_subset[Subset.TRAIN].normal == 1
    assert by_subset[Subset.TRAIN].defect == 0
    assert by_subset[Subset.TEST].defect == 1
    assert by_subset[Subset.VAL].total == 0


def test_a_split_and_its_assignments_are_written_atomically(
    migrated_db: sqlite3.Connection, catalog: SeededCatalog
) -> None:
    """A half-written split would be silently wrong rather than visibly broken."""
    with pytest.raises(sqlite3.IntegrityError):
        splits.create_split(
            migrated_db,
            catalog.dataset_id,
            name="s",
            strategy="normal_only_train",
            seed=7,
            params={},
            assignments={catalog.sample_ids["group-a/1"]: Subset.TRAIN, 999_999: Subset.TEST},
        )

    assert splits.list_splits(migrated_db, catalog.dataset_id) == []


def _split_over_everything(migrated_db: sqlite3.Connection, catalog: SeededCatalog) -> int:
    """Every seeded sample in `test`, so a channel filter is the only thing narrowing."""
    split = splits.create_split(
        migrated_db,
        catalog.dataset_id,
        name="all",
        strategy="normal_only_train",
        seed=1,
        params={},
        assignments=dict.fromkeys(catalog.sample_ids.values(), Subset.TEST),
    )
    return split.id


def test_no_channel_filter_reads_every_channel(
    migrated_db: sqlite3.Connection, catalog: SeededCatalog
) -> None:
    split_id = _split_over_everything(migrated_db, catalog)

    rows = images.list_images_for_split(migrated_db, split_id, subsets=[Subset.TEST])

    assert len(rows) == 5  # two grouped samples x two channels, plus one unassigned
    assert {row.channel for row in rows} == {"bright", "dark", None}


def test_a_channel_filter_narrows_to_that_channel(
    migrated_db: sqlite3.Connection, catalog: SeededCatalog
) -> None:
    """The comparison three separate datasets used to require, on one split."""
    split_id = _split_over_everything(migrated_db, catalog)

    rows = images.list_images_for_split(
        migrated_db, split_id, subsets=[Subset.TEST], channels=["bright"]
    )

    assert {row.channel for row in rows} == {"bright"}
    assert len(rows) == 2


def test_an_empty_channel_filter_means_every_channel(
    migrated_db: sqlite3.Connection, catalog: SeededCatalog
) -> None:
    """`[]` is how a frozen experiment says 'unset', so it must not select nothing."""
    split_id = _split_over_everything(migrated_db, catalog)

    assert (
        len(images.list_images_for_split(migrated_db, split_id, subsets=[Subset.TEST], channels=[]))
        == 5
    )


def test_an_unassigned_image_is_excluded_by_a_named_selection(
    migrated_db: sqlite3.Connection, catalog: SeededCatalog
) -> None:
    """ "No channel" is not a synonym for "all channels"."""
    split_id = _split_over_everything(migrated_db, catalog)

    rows = images.list_images_for_split(
        migrated_db, split_id, subsets=[Subset.TEST], channels=["bright", "dark"]
    )

    assert len(rows) == 4
    assert all(row.channel is not None for row in rows)


def test_selecting_every_channel_by_name_still_drops_the_unassigned_one(
    migrated_db: sqlite3.Connection, catalog: SeededCatalog
) -> None:
    """Naming every channel is not the same request as naming none, and the difference is
    exactly the images that belong to no channel. Pinned so the two never quietly merge."""
    split_id = _split_over_everything(migrated_db, catalog)

    named = images.list_images_for_split(
        migrated_db, split_id, subsets=[Subset.TEST], channels=["bright", "dark"]
    )
    unset = images.list_images_for_split(migrated_db, split_id, subsets=[Subset.TEST])

    assert len(unset) - len(named) == 1


# ------------------------------------------------- results: three-valued localization
#
# `localized` and the map peak are the only columns in this schema whose NULL means
# something other than "unset" — it means *not applicable*, and it must not come back as
# `False`. SQLite has no boolean, so every hop between the two is a place where three
# values can silently become two.


def _experiment_over(migrated_db: sqlite3.Connection, catalog: SeededCatalog) -> int:
    profile = region_profiles.create_revision(
        migrated_db,
        dataset_id=catalog.dataset_id,
        name="full frame",
        extractor_type="identity",
        extractor_config={},
        padding_fraction=0.0,
    )
    experiment = experiments.create_experiment(
        migrated_db,
        name="run",
        dataset_id=catalog.dataset_id,
        split_id=_split_over_everything(migrated_db, catalog),
        region_profile_id=profile.id,
        region_manifest_sha256="sha",
        model_type="pixel_reference",
        model_config={},
        preprocessing_config={},
        eval_config={},
        artifact_dir="/artifacts/run",
    )
    return experiment.id


def test_a_peak_and_a_localization_verdict_round_trip_with_null_intact(
    migrated_db: sqlite3.Connection, catalog: SeededCatalog
) -> None:
    experiment_id = _experiment_over(migrated_db, catalog)
    hit, miss, unchecked = catalog.image_ids[0], catalog.image_ids[1], catalog.image_ids[2]

    results.replace_image_results(
        migrated_db,
        experiment_id,
        [
            ImageResult(
                experiment_id=experiment_id,
                image_id=hit,
                score=1.0,
                inference_ms=1.0,
                peak_x=3,
                peak_y=4,
                localized=True,
            ),
            ImageResult(
                experiment_id=experiment_id,
                image_id=miss,
                score=2.0,
                inference_ms=1.0,
                peak_x=0,
                peak_y=0,
                localized=False,
            ),
            ImageResult(experiment_id=experiment_id, image_id=unchecked, score=3.0, inference_ms=1),
        ],
    )

    by_id = {row.image_id: row for row in results.list_scored_images(migrated_db, experiment_id)}

    assert (by_id[hit].peak_x, by_id[hit].peak_y) == (3, 4)
    assert by_id[hit].localized is True
    # A peak at the origin is a measurement, not a missing value.
    assert (by_id[miss].peak_x, by_id[miss].peak_y) == (0, 0)
    assert by_id[miss].localized is False
    assert by_id[unchecked].peak_x is None
    assert by_id[unchecked].localized is None

    stored = results.get_image_result(migrated_db, experiment_id, unchecked)
    assert stored is not None
    assert stored.localized is None


def test_the_localization_write_clears_a_verdict_that_no_longer_applies(
    migrated_db: sqlite3.Connection, catalog: SeededCatalog
) -> None:
    """An annotation can be deleted after a run was scored. Writing only the hits would
    leave a stale `1` on screen beside ground truth that no longer exists."""
    experiment_id = _experiment_over(migrated_db, catalog)
    image_id = catalog.image_ids[0]
    results.replace_image_results(
        migrated_db,
        experiment_id,
        [
            ImageResult(
                experiment_id=experiment_id,
                image_id=image_id,
                score=1.0,
                inference_ms=1.0,
                localized=True,
            )
        ],
    )

    results.update_image_localization(migrated_db, experiment_id, {image_id: None})

    assert results.list_scored_images(migrated_db, experiment_id)[0].localized is None


def test_peaks_are_backfilled_without_disturbing_the_scores(
    migrated_db: sqlite3.Connection, catalog: SeededCatalog
) -> None:
    experiment_id = _experiment_over(migrated_db, catalog)
    image_id = catalog.image_ids[0]
    results.replace_image_results(
        migrated_db,
        experiment_id,
        [ImageResult(experiment_id=experiment_id, image_id=image_id, score=7.5, inference_ms=2.0)],
    )

    results.update_image_peaks(migrated_db, experiment_id, {image_id: (11, 2)})

    row = results.list_scored_images(migrated_db, experiment_id)[0]
    assert (row.peak_x, row.peak_y) == (11, 2)
    assert row.score == 7.5
    assert row.localized is None


def test_a_sample_s_localization_survives_the_sample_result_round_trip(
    migrated_db: sqlite3.Connection, catalog: SeededCatalog
) -> None:
    experiment_id = _experiment_over(migrated_db, catalog)
    identified = sorted(catalog.sample_ids.values())

    results.replace_sample_results(
        migrated_db,
        experiment_id,
        [
            SampleResult(
                experiment_id=experiment_id,
                sample_id=identified[0],
                agg_score=3.0,
                aggregation=Aggregation.MAX,
                localized=True,
            ),
            SampleResult(
                experiment_id=experiment_id,
                sample_id=identified[1],
                agg_score=2.0,
                aggregation=Aggregation.MAX,
                localized=False,
            ),
            SampleResult(
                experiment_id=experiment_id,
                sample_id=identified[2],
                agg_score=1.0,
                aggregation=Aggregation.MAX,
            ),
        ],
    )

    by_id = {row.sample_id: row for row in results.list_scored_samples(migrated_db, experiment_id)}

    assert by_id[identified[0]].localized is True
    assert by_id[identified[1]].localized is False
    assert by_id[identified[2]].localized is None
