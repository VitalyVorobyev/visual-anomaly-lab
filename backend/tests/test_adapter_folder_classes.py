"""The `folder_classes` adapter: labels from directories the operator names.

The fixture trees here are shaped like the public datasets this adapter exists for — a
folder of good parts and one or more folders of bad ones — built from small synthetic
PNGs, never a real dataset file (ADR-0022).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from anomaly_lab.datasets.adapters.base import get_adapter, parse_options
from anomaly_lab.datasets.adapters.folder_classes import (
    FolderClassesAdapter,
    FolderClassesOptions,
)
from anomaly_lab.datasets.manifest import Manifest, ManifestSample, WarningCode
from anomaly_lab.domain.entities import Label
from tests.conftest import write_image


@pytest.fixture
def tree(tmp_path: Path) -> Path:
    """A flat, single-view tree: good parts and two kinds of defect."""
    root = tmp_path / "source"
    for index in range(4):
        write_image(root / "Data" / "Good" / f"g{index}.png")
    for index in range(3):
        write_image(root / "Data" / "Nick" / f"n{index}.png")
    for index in range(2):
        write_image(root / "Data" / "Scratch" / f"s{index}.png")
    return root


def _scan(root: Path, **overrides: object) -> Manifest:
    options = FolderClassesOptions(**overrides)  # type: ignore[arg-type]
    return FolderClassesAdapter.scan(root, options, dataset_name="fixture")


def _codes(manifest: Manifest) -> set[WarningCode]:
    return {warning.code for warning in manifest.warnings}


def _by_label(manifest: Manifest, label: Label) -> list[ManifestSample]:
    return [sample for sample in manifest.samples if sample.label is label]


def test_named_directories_carry_the_label(tree: Path) -> None:
    manifest = _scan(
        tree,
        normal_dirs=["Data/Good"],
        defect_dirs=["Data/Nick", "Data/Scratch"],
    )

    counts = manifest.label_counts()
    assert counts[Label.NORMAL] == 4
    assert counts[Label.DEFECT] == 5
    assert counts[Label.UNLABELED] == 0


def test_one_image_is_one_sample_with_no_channel(tree: Path) -> None:
    """The point of this adapter beyond convenience: it exercises the single-view path.

    A sample with one unchannelled image has been legal since ADR-0005 and was never
    produced by anything until now.
    """
    manifest = _scan(tree, normal_dirs=["Data/Good"], defect_dirs=["Data/Nick"])

    assert manifest.channels == []
    assert all(len(sample.images) == 1 for sample in manifest.samples)
    assert all(image.channel is None for sample in manifest.samples for image in sample.images)


def test_a_pattern_covers_the_subtree_beneath_it(tmp_path: Path) -> None:
    """ "Point at the folder" has to mean the folder and everything under it."""
    root = tmp_path / "nested"
    write_image(root / "train" / "good" / "a.png")
    write_image(root / "train" / "good" / "batch2" / "b.png")

    manifest = _scan(root, normal_dirs=["train/good"])

    assert manifest.label_counts()[Label.NORMAL] == 2


def test_the_defect_directory_name_is_recorded_on_the_sample(tree: Path) -> None:
    """A per-defect-type breakdown must not require a schema that enumerates types."""
    manifest = _scan(tree, normal_dirs=["Data/Good"], defect_dirs=["Data/Nick", "Data/Scratch"])

    notes = {sample.notes for sample in _by_label(manifest, Label.DEFECT)}
    assert notes == {"Nick", "Scratch"}


def test_defect_types_can_be_left_off(tree: Path) -> None:
    manifest = _scan(tree, defect_dirs=["Data/Nick"], defect_type_from_dir=False)

    assert all(sample.notes is None for sample in manifest.samples)


def test_unnamed_directories_import_unlabelled_and_are_surfaced(tree: Path) -> None:
    """Silence would be the wrong answer: the operator has to see what was not covered."""
    manifest = _scan(tree, normal_dirs=["Data/Good"])

    assert manifest.label_counts()[Label.UNLABELED] == 5
    assert WarningCode.UNMATCHED_PATH in _codes(manifest)


def test_configuring_nothing_says_so_rather_than_importing_silently(tree: Path) -> None:
    manifest = _scan(tree)

    assert manifest.label_counts()[Label.UNLABELED] == 9
    unmatched = next(w for w in manifest.warnings if w.code is WarningCode.UNMATCHED_PATH)
    assert "No directories were configured" in unmatched.message


def test_a_directory_claimed_twice_reads_as_defective_and_is_reported(tree: Path) -> None:
    """The safe reading of an ambiguous configuration keeps defects out of training."""
    manifest = _scan(tree, normal_dirs=["Data/Nick"], defect_dirs=["Data/Nick"])

    assert manifest.label_counts()[Label.DEFECT] == 3
    assert WarningCode.CONFLICTING_DIRECTORIES in _codes(manifest)


def test_unlabeled_dirs_are_distinct_from_unmatched_ones(tree: Path) -> None:
    manifest = _scan(tree, normal_dirs=["Data/Good"], unlabeled_dirs=["Data/Nick", "Data/Scratch"])

    assert manifest.label_counts()[Label.UNLABELED] == 5
    assert WarningCode.UNMATCHED_PATH not in _codes(manifest)


# -- masks -----------------------------------------------------------------------------


def test_masks_are_attached_when_the_pattern_finds_them(tmp_path: Path) -> None:
    root = tmp_path / "masked"
    for index in range(3):
        write_image(root / "images" / "anomaly" / f"{index}.png")
        write_image(root / "masks" / f"{index}.png", mode="L")

    manifest = _scan(root, defect_dirs=["images/anomaly"], mask_dir="masks")

    masks = [image.mask_path for sample in manifest.samples for image in sample.images]
    assert len(masks) == 3
    assert all(path is not None and Path(path).is_file() for path in masks)
    assert manifest.stats.masks == 3
    assert WarningCode.MISSING_MASK not in _codes(manifest)


def test_a_mask_that_is_not_there_is_a_warning_not_a_failure(tmp_path: Path) -> None:
    root = tmp_path / "partial"
    write_image(root / "images" / "a.png")
    write_image(root / "images" / "b.png")
    write_image(root / "masks" / "a.png", mode="L")

    manifest = _scan(root, defect_dirs=["images"], mask_dir="masks")

    assert manifest.stats.masks == 1
    assert manifest.stats.samples == 2, "the image itself still imports"
    assert WarningCode.MISSING_MASK in _codes(manifest)


def test_a_mask_template_can_reach_a_sibling_directory(tmp_path: Path) -> None:
    """The MVTec shape: masks live beside the class directory, not beneath it."""
    root = tmp_path / "mvtec"
    write_image(root / "bottle" / "test" / "broken" / "000.png")
    write_image(root / "bottle" / "ground_truth" / "broken" / "000_mask.png", mode="L")

    manifest = _scan(
        root,
        defect_dirs=["bottle/test/*"],
        mask_dir="{dir}/../../ground_truth/{class}",
        mask_pattern="{stem}_mask.png",
    )

    assert manifest.stats.masks == 1


# -- scope and skips -------------------------------------------------------------------


def test_exclude_globs_narrow_the_scope(tree: Path) -> None:
    manifest = _scan(tree, normal_dirs=["Data/Good"], exclude=["Data/Scratch/*"])

    assert manifest.stats.samples == 7
    assert manifest.stats.files_excluded == 2


def test_an_excluded_file_is_not_also_reported_as_unlabelled(tree: Path) -> None:
    """Otherwise deliberately narrowing the scope produces a warning about doing so."""
    manifest = _scan(tree, normal_dirs=["Data/Good"], exclude=["Data/Nick/*", "Data/Scratch/*"])

    assert WarningCode.UNMATCHED_PATH not in _codes(manifest)


def test_files_that_are_not_images_are_ignored_entirely(tree: Path) -> None:
    (tree / "Data" / "README.txt").write_text("not an image", encoding="utf-8")

    manifest = _scan(tree, normal_dirs=["Data/Good"])

    assert manifest.stats.samples == 9


def test_two_files_sharing_a_stem_merge_and_are_surfaced(tmp_path: Path) -> None:
    root = tmp_path / "collide"
    write_image(root / "good" / "a.png")
    write_image(root / "good" / "a.bmp")

    manifest = _scan(root, normal_dirs=["good"])

    assert manifest.stats.samples == 1
    assert WarningCode.AMBIGUOUS_SAMPLE_ID in _codes(manifest)


def test_progress_may_abort_the_scan(tree: Path) -> None:
    """A cancelled job stops a scan by raising from the callback (ADR-0009)."""
    seen = 0

    def stop_after_two(_fraction: float, _message: str | None) -> None:
        nonlocal seen
        seen += 1
        if seen > 2:
            msg = "cancelled"
            raise RuntimeError(msg)

    with pytest.raises(RuntimeError, match="cancelled"):
        FolderClassesAdapter.scan(
            tree, FolderClassesOptions(), dataset_name="d", progress=stop_after_two
        )


def test_the_adapter_is_registered_with_a_form_schema() -> None:
    adapter = get_adapter("folder_classes")
    schema = adapter.options_model().model_json_schema()

    assert isinstance(parse_options(adapter, {"normal_dirs": ["a"]}), FolderClassesOptions)
    assert {"normal_dirs", "defect_dirs", "mask_dir"} <= set(schema["properties"])
