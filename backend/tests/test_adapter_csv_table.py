"""The `csv_table` adapter: the dataset's own table decides.

The fixture below is shaped like the split tables published benchmarks ship —
`object,split,label,image,mask` — because reading one of those correctly, including its
partition, is the entire reason this adapter exists.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from anomaly_lab.datasets.adapters.base import get_adapter, parse_options
from anomaly_lab.datasets.adapters.csv_table import (
    CsvTableAdapter,
    CsvTableOptions,
    TableError,
)
from anomaly_lab.datasets.manifest import Manifest, WarningCode
from anomaly_lab.domain.entities import Label, Subset
from tests.conftest import write_image

HEADER = ["object", "split", "label", "image", "mask"]


def _write_table(path: Path, rows: list[list[str]], header: list[str] = HEADER) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


@pytest.fixture
def tree(tmp_path: Path) -> Path:
    """Two classes in one table, one of them with masks — the published-benchmark shape."""
    root = tmp_path / "source"
    rows: list[list[str]] = []

    for index in range(3):
        image = f"candle/Data/Images/Normal/{index}.png"
        write_image(root / image)
        rows.append(["candle", "train", "normal", image, ""])

    for index in range(2):
        image = f"candle/Data/Images/Anomaly/{index}.png"
        mask = f"candle/Data/Masks/Anomaly/{index}.png"
        write_image(root / image)
        write_image(root / mask, mode="L")
        rows.append(["candle", "test", "anomaly", image, mask])

    image = "candle/Data/Images/Normal/9.png"
    write_image(root / image)
    rows.append(["candle", "test", "normal", image, ""])

    # A second class, which every candle-scoped scan must leave alone.
    other = "cashew/Data/Images/Normal/0.png"
    write_image(root / other)
    rows.append(["cashew", "train", "normal", other, ""])

    _write_table(root / "split_csv" / "1cls.csv", rows)
    return root


def _scan(root: Path, **overrides: object) -> Manifest:
    options = CsvTableOptions(**{"csv_path": "split_csv/1cls.csv", **overrides})  # type: ignore[arg-type]
    return CsvTableAdapter.scan(root, options, dataset_name="fixture")


def _codes(manifest: Manifest) -> set[WarningCode]:
    return {warning.code for warning in manifest.warnings}


def test_the_table_decides_the_labels(tree: Path) -> None:
    manifest = _scan(tree)

    counts = manifest.label_counts()
    assert counts[Label.NORMAL] == 5
    assert counts[Label.DEFECT] == 2


def test_a_filter_imports_one_class_as_one_dataset(tree: Path) -> None:
    """The one-class protocol every benchmark of this family is scored under."""
    manifest = _scan(tree, filter_column="object", filter_value="candle")

    assert manifest.stats.samples == 6
    assert all("cashew" not in image.path for s in manifest.samples for image in s.images)


def test_the_published_split_is_carried_through(tree: Path) -> None:
    """A number computed on our own partition is not comparable to a published one."""
    manifest = _scan(tree, filter_column="object", filter_value="candle")

    counts = manifest.subset_counts()
    assert counts[Subset.TRAIN] == 3
    assert counts[Subset.TEST] == 3
    assert counts[Subset.VAL] == 0, "the official one-class protocol has no validation set"
    assert manifest.has_imported_split()


def test_masks_named_by_the_table_are_attached(tree: Path) -> None:
    manifest = _scan(tree, filter_column="object", filter_value="candle")

    masked = [
        image
        for sample in manifest.samples
        for image in sample.images
        if image.mask_path is not None
    ]
    assert len(masked) == 2
    assert manifest.stats.masks == 2
    assert all(Path(image.mask_path or "").is_file() for image in masked)


def test_a_mask_the_table_promises_and_does_not_deliver_is_a_warning(tmp_path: Path) -> None:
    root = tmp_path / "broken"
    write_image(root / "a.png")
    _write_table(root / "t.csv", [["x", "test", "anomaly", "a.png", "missing.png"]])

    manifest = CsvTableAdapter.scan(root, CsvTableOptions(csv_path="t.csv"), dataset_name="fixture")

    assert manifest.stats.samples == 1, "the image still imports"
    assert manifest.stats.masks == 0
    assert WarningCode.MISSING_MASK in _codes(manifest)


def test_no_channel_column_means_one_image_per_sample(tree: Path) -> None:
    manifest = _scan(tree)

    assert manifest.channels == []
    assert all(len(sample.images) == 1 for sample in manifest.samples)


def test_a_channel_column_groups_rows_into_multi_view_samples(tmp_path: Path) -> None:
    """Channel count is data: the same adapter reads a two-view dataset with no special case."""
    root = tmp_path / "multiview"
    rows = []
    for part in ("p1", "p2"):
        for channel in ("front", "back"):
            image = f"images/{channel}/{part}.png"
            write_image(root / image)
            rows.append([channel, "train", "normal", image, part])
    _write_table(root / "t.csv", rows, header=["view", "split", "label", "image", "part"])

    manifest = CsvTableAdapter.scan(
        root,
        CsvTableOptions(
            csv_path="t.csv",
            channel_column="view",
            sample_id_column="part",
            group_column="split",
            mask_column=None,
        ),
        dataset_name="fixture",
    )

    assert manifest.stats.samples == 2
    assert manifest.channels == ["back", "front"]
    assert all(len(sample.images) == 2 for sample in manifest.samples)


def test_rows_sharing_an_identity_without_a_channel_column_are_surfaced(tmp_path: Path) -> None:
    root = tmp_path / "ambiguous"
    for name in ("a.png", "b.png"):
        write_image(root / "images" / name)
    _write_table(
        root / "t.csv",
        [
            ["x", "train", "normal", "images/a.png", "s1"],
            ["x", "train", "normal", "images/b.png", "s1"],
        ],
        header=["object", "split", "label", "image", "sid"],
    )

    manifest = CsvTableAdapter.scan(
        root,
        CsvTableOptions(csv_path="t.csv", sample_id_column="sid", mask_column=None),
        dataset_name="fixture",
    )

    assert WarningCode.AMBIGUOUS_SAMPLE_ID in _codes(manifest)


def test_label_vocabulary_is_configurable(tmp_path: Path) -> None:
    root = tmp_path / "numeric"
    for name in ("a.png", "b.png"):
        write_image(root / name)
    _write_table(
        root / "t.csv",
        [["x", "train", "0", "a.png", ""], ["x", "test", "1", "b.png", ""]],
    )

    manifest = CsvTableAdapter.scan(root, CsvTableOptions(csv_path="t.csv"), dataset_name="f")

    counts = manifest.label_counts()
    assert counts[Label.NORMAL] == 1
    assert counts[Label.DEFECT] == 1


def test_an_unknown_label_value_imports_unlabelled_and_is_reported(tmp_path: Path) -> None:
    root = tmp_path / "odd"
    write_image(root / "a.png")
    _write_table(root / "t.csv", [["x", "train", "mysterious", "a.png", ""]])

    manifest = CsvTableAdapter.scan(root, CsvTableOptions(csv_path="t.csv"), dataset_name="f")

    assert manifest.label_counts()[Label.UNLABELED] == 1
    assert any("mysterious" in path for w in manifest.warnings for path in w.paths)


def test_an_unknown_subset_value_leaves_the_sample_out_of_the_partition(tmp_path: Path) -> None:
    root = tmp_path / "odd-split"
    write_image(root / "a.png")
    _write_table(root / "t.csv", [["x", "holdout", "normal", "a.png", ""]])

    manifest = CsvTableAdapter.scan(root, CsvTableOptions(csv_path="t.csv"), dataset_name="f")

    assert not manifest.has_imported_split()
    assert any("holdout" in path for w in manifest.warnings for path in w.paths)


def test_a_row_naming_a_file_that_is_not_there_is_skipped_not_fatal(tmp_path: Path) -> None:
    root = tmp_path / "gappy"
    write_image(root / "a.png")
    _write_table(
        root / "t.csv",
        [["x", "train", "normal", "a.png", ""], ["x", "train", "normal", "gone.png", ""]],
    )

    manifest = CsvTableAdapter.scan(root, CsvTableOptions(csv_path="t.csv"), dataset_name="f")

    assert manifest.stats.samples == 1
    assert manifest.stats.files_skipped == 1
    assert WarningCode.UNREADABLE_FILE in _codes(manifest)


# -- failing loudly --------------------------------------------------------------------


def test_a_missing_table_is_an_error_with_the_path_in_it(tmp_path: Path) -> None:
    with pytest.raises(TableError, match="no table at"):
        CsvTableAdapter.scan(tmp_path, CsvTableOptions(csv_path="nope.csv"), dataset_name="f")


def test_a_column_that_is_not_there_names_the_ones_that_are(tmp_path: Path) -> None:
    root = tmp_path / "wrong-columns"
    write_image(root / "a.png")
    _write_table(root / "t.csv", [["x", "train", "normal", "a.png", ""]])

    with pytest.raises(TableError, match="Columns present: object, split, label, image, mask"):
        CsvTableAdapter.scan(
            root, CsvTableOptions(csv_path="t.csv", image_column="filename"), dataset_name="f"
        )


def test_an_optional_column_that_the_table_lacks_is_tolerated(tmp_path: Path) -> None:
    """A table with no mask or split column is the common case, not a misconfiguration."""
    root = tmp_path / "minimal"
    write_image(root / "a.png")
    _write_table(root / "t.csv", [["a.png", "normal"]], header=["image", "label"])

    manifest = CsvTableAdapter.scan(root, CsvTableOptions(csv_path="t.csv"), dataset_name="f")

    assert manifest.stats.samples == 1
    assert not manifest.has_imported_split()


def test_a_filter_needs_both_halves() -> None:
    with pytest.raises(ValueError, match="must be given together"):
        CsvTableOptions(csv_path="t.csv", filter_column="object")


def test_the_adapter_is_registered_with_a_form_schema() -> None:
    adapter = get_adapter("csv_table")
    schema = adapter.options_model().model_json_schema()

    assert isinstance(parse_options(adapter, {"csv_path": "t.csv"}), CsvTableOptions)
    assert {"csv_path", "image_column", "split_column"} <= set(schema["properties"])
    assert "csv_path" in schema.get("required", [])
