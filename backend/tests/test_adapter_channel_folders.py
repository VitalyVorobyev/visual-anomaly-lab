"""The `channel_folders` classifier and scan.

Every tree here is synthetic and built in `tmp_path`. The cases were chosen from shapes
that occur in real acquisition folders: both nesting orders, a channel name fused into a
group folder's name, colliding numeric stems under different labels, and a single-view
tree with no channels at all.
"""

from __future__ import annotations

from pathlib import Path, PurePosixPath

import pytest

from anomaly_lab.datasets.adapters.base import (
    UnknownAdapterError,
    get_adapter,
    parse_options,
    registered_adapters,
)
from anomaly_lab.datasets.adapters.channel_folders import (
    ChannelFoldersAdapter,
    ChannelFoldersOptions,
    classify,
)
from anomaly_lab.datasets.manifest import Manifest, WarningCode
from anomaly_lab.domain.entities import Label
from tests.conftest import write_image

DEFAULTS = ChannelFoldersOptions()


def _scan(root: Path, **overrides: object) -> Manifest:
    options = ChannelFoldersOptions(**overrides)  # type: ignore[arg-type]
    return ChannelFoldersAdapter.scan(root, options, dataset_name="fixture")


def _codes(manifest: Manifest) -> set[WarningCode]:
    return {warning.code for warning in manifest.warnings}


def _sample(manifest: Manifest, group_key: str, external_id: str) -> object:
    for sample in manifest.samples:
        if sample.group_key == group_key and sample.external_id == external_id:
            return sample
    msg = f"no sample {group_key}/{external_id} in {[s.group_key for s in manifest.samples]}"
    raise AssertionError(msg)


# -- classification ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("path", "expected_group", "expected_label", "expected_channel"),
    [
        ("set1/defect/Bright/12.bmp", "set1/defect", Label.DEFECT, "bright"),
        ("set1/no-defect/Dark/12.bmp", "set1/no-defect", Label.NORMAL, "dark"),
        ("set2/defect/Dome/3.bmp", "set2/defect", Label.DEFECT, "dome"),
        # The other nesting order, unchanged behaviour.
        ("set1/Bright/defect/12.bmp", "set1/defect", Label.DEFECT, "bright"),
        # Longer spellings of the same channels.
        ("batch/BrightField/1.bmp", "batch", Label.UNLABELED, "bright"),
        ("batch/Darkfield/1.bmp", "batch", Label.UNLABELED, "dark"),
        ("batch/DomeIllumination/1.bmp", "batch", Label.UNLABELED, "dome"),
        # No channel anywhere: an ordinary single-view dataset.
        ("flat/7.bmp", "flat", Label.UNLABELED, None),
    ],
)
def test_components_are_classified_regardless_of_nesting_order(
    path: str, expected_group: str, expected_label: Label, expected_channel: str | None
) -> None:
    found = classify(PurePosixPath(path), DEFAULTS)

    assert found.group_key == expected_group
    assert found.label is expected_label
    assert found.channel == expected_channel


def test_a_channel_fused_into_a_group_folder_name_is_still_found() -> None:
    """Two sibling directories named `"<Channel> <Group>"`, not `<Group>/<Channel>/`.

    A classifier that only matches whole components misses this entirely and produces
    two unrelated single-image samples instead of one two-channel sample.
    """
    bright = classify(PurePosixPath("unsorted/Brightfield Bl7/10.bmp"), DEFAULTS)
    dark = classify(PurePosixPath("unsorted/Darkfield Bl7/10.bmp"), DEFAULTS)

    assert bright.channel == "bright"
    assert dark.channel == "dark"
    assert bright.matched_by == "token"
    # Same part, so the two views must land on one identity.
    assert bright.group_key == dark.group_key == "unsorted/Bl7"
    assert bright.external_id == dark.external_id == "10"


def test_no_defect_is_not_read_as_defect() -> None:
    """`no-defect` normalizes to a string containing `defect`; matching is whole-word."""
    assert classify(PurePosixPath("s/no-defect/Bright/1.bmp"), DEFAULTS).label is Label.NORMAL
    assert classify(PurePosixPath("s/no_defect/Bright/1.bmp"), DEFAULTS).label is Label.NORMAL
    assert classify(PurePosixPath("s/nodefect/Bright/1.bmp"), DEFAULTS).label is Label.NORMAL
    assert classify(PurePosixPath("s/defect/Bright/1.bmp"), DEFAULTS).label is Label.DEFECT


def test_the_label_component_stays_in_the_group_key() -> None:
    """Otherwise `set1/defect/1` and `set1/no-defect/1` collide on one sample identity."""
    defect = classify(PurePosixPath("set1/defect/Bright/1.bmp"), DEFAULTS)
    normal = classify(PurePosixPath("set1/no-defect/Bright/1.bmp"), DEFAULTS)

    assert defect.external_id == normal.external_id
    assert defect.group_key != normal.group_key


def test_a_real_channel_directory_outranks_a_matching_word_in_a_group_name() -> None:
    """`Bright Parts/Dark/1.bmp` is a dark image, not a bright one."""
    found = classify(PurePosixPath("Bright Parts/Dark/1.bmp"), DEFAULTS)

    assert found.channel == "dark"
    assert found.group_key == "Bright Parts"


def test_an_alias_overrides_the_fuzzy_matcher() -> None:
    """The documented escape hatch when the matcher is confidently wrong (ADR-0006)."""
    options = ChannelFoldersOptions(channel_aliases={"illumb": "bright"})

    found = classify(PurePosixPath("g/IllumB/1.bmp"), options)

    assert found.channel == "bright"
    assert found.matched_by == "alias"


def test_the_channel_vocabulary_is_data() -> None:
    """A dataset with entirely unfamiliar channels imports without a code change."""
    options = ChannelFoldersOptions(channels=["uv", "ir"])

    assert classify(PurePosixPath("g/UV/1.bmp"), options).channel == "uv"
    assert classify(PurePosixPath("g/IR-long/1.bmp"), options).channel == "ir"
    assert classify(PurePosixPath("g/Bright/1.bmp"), options).channel is None


# -- scanning ------------------------------------------------------------------------


def test_both_nesting_orders_produce_the_same_grouping(tmp_path: Path) -> None:
    label_first = tmp_path / "label-first"
    channel_first = tmp_path / "channel-first"
    for external_id in ("1", "2"):
        for channel in ("Bright", "Dark", "Dome"):
            write_image(label_first / "set1" / "defect" / channel / f"{external_id}.png")
            write_image(channel_first / "set1" / channel / "defect" / f"{external_id}.png")

    first = _scan(label_first)
    second = _scan(channel_first)

    assert first.stats.samples == second.stats.samples == 2
    assert first.channels == second.channels == ["bright", "dark", "dome"]
    assert [s.group_key for s in first.samples] == [s.group_key for s in second.samples]
    assert all(len(sample.images) == 3 for sample in first.samples)
    assert not first.warnings


def test_colliding_stems_under_different_labels_stay_separate(tmp_path: Path) -> None:
    for label in ("defect", "no-defect"):
        for channel in ("Bright", "Dark"):
            write_image(tmp_path / "set1" / label / channel / "1.png")

    manifest = _scan(tmp_path)

    assert manifest.stats.samples == 2
    assert manifest.label_counts() == {Label.DEFECT: 1, Label.NORMAL: 1, Label.UNLABELED: 0}


def test_a_variable_channel_count_is_a_warning_never_an_error(tmp_path: Path) -> None:
    """Variable channel counts are legitimate data (ADR-0005)."""
    for channel in ("Bright", "Dark", "Dome"):
        write_image(tmp_path / "g" / channel / "1.png")
    for channel in ("Bright", "Dark"):
        write_image(tmp_path / "g" / channel / "2.png")

    manifest = _scan(tmp_path)

    assert manifest.stats.samples == 2
    assert WarningCode.VARIABLE_CHANNEL_COUNT in _codes(manifest)
    warning = next(w for w in manifest.warnings if w.code is WarningCode.VARIABLE_CHANNEL_COUNT)
    assert "g/2" in warning.paths[0]
    # Warned about, and imported anyway.
    assert len(_sample(manifest, "g", "2").images) == 2  # type: ignore[attr-defined]


def test_a_single_view_tree_imports_with_no_channels(tmp_path: Path) -> None:
    for external_id in ("a", "b", "c"):
        write_image(tmp_path / "good" / f"{external_id}.png")

    manifest = _scan(tmp_path, normal_labels=["good"])

    assert manifest.channels == []
    assert manifest.stats.samples == 3
    assert all(image.channel is None for sample in manifest.samples for image in sample.images)
    # No channels means nothing to warn about; an unassigned channel is only notable in a
    # dataset that has channels at all.
    assert not manifest.warnings


def test_an_unrecognized_channel_directory_is_surfaced_not_dropped(tmp_path: Path) -> None:
    for channel in ("Bright", "Dark"):
        write_image(tmp_path / "g" / channel / "1.png")
    write_image(tmp_path / "g" / "Polarized" / "1.png")

    manifest = _scan(tmp_path)

    assert WarningCode.UNASSIGNED_CHANNEL in _codes(manifest)
    unknown = next(w for w in manifest.warnings if w.code is WarningCode.UNKNOWN_CHANNEL_NAME)
    assert unknown.paths == ["Polarized"]
    # Surfaced for review, and still present in the proposal.
    assert manifest.stats.images == 3


def test_exclude_globs_narrow_the_scope_and_are_recorded(tmp_path: Path) -> None:
    for channel in ("Bright", "Dark"):
        write_image(tmp_path / "set1" / channel / "1.png")
        write_image(tmp_path / "unsorted" / channel / "9.png")

    manifest = _scan(tmp_path, exclude=["unsorted/*"])

    assert manifest.stats.files_seen == 4
    assert manifest.stats.files_excluded == 2
    assert [sample.group_key for sample in manifest.samples] == ["set1"]
    # The scope travels with the manifest, so a re-scan proposes the same thing.
    assert manifest.options["exclude"] == ["unsorted/*"]


def test_duplicate_content_is_reported(tmp_path: Path) -> None:
    write_image(tmp_path / "g" / "Bright" / "1.png", colour=64)
    write_image(tmp_path / "g" / "Dark" / "1.png", colour=64)

    manifest = _scan(tmp_path)

    assert WarningCode.DUPLICATE_HASH in _codes(manifest)


def test_empty_and_unreadable_files_are_skipped_with_their_paths(tmp_path: Path) -> None:
    write_image(tmp_path / "g" / "Bright" / "1.png")
    (tmp_path / "g" / "Bright" / "empty.png").write_bytes(b"")
    (tmp_path / "g" / "Bright" / "broken.png").write_bytes(b"this is not a png")

    manifest = _scan(tmp_path)

    assert _codes(manifest) >= {WarningCode.EMPTY_FILE, WarningCode.UNREADABLE_FILE}
    assert manifest.stats.files_skipped == 2
    assert manifest.stats.images == 1


def test_mixed_bit_depths_are_recorded_rather_than_normalized(tmp_path: Path) -> None:
    """Real datasets mix 8-bit grayscale and 24-bit colour in one tree (§9)."""
    write_image(tmp_path / "g" / "Bright" / "1.bmp", mode="RGB")
    write_image(tmp_path / "g" / "Dark" / "1.bmp", mode="L")

    manifest = _scan(tmp_path, extensions=[".bmp"])

    depths = {image.channel: image.bit_depth for image in manifest.samples[0].images}
    assert depths == {"bright": 24, "dark": 8}


def test_progress_is_reported_and_may_abort_the_scan(tmp_path: Path) -> None:
    """A cancelled job stops a scan by raising from the callback."""
    for external_id in range(5):
        write_image(tmp_path / "g" / "Bright" / f"{external_id}.png")

    seen: list[float] = []

    def stop_after_two(fraction: float, _message: str | None) -> None:
        seen.append(fraction)
        if len(seen) > 2:
            msg = "cancelled"
            raise RuntimeError(msg)

    with pytest.raises(RuntimeError, match="cancelled"):
        ChannelFoldersAdapter.scan(
            tmp_path, ChannelFoldersOptions(), dataset_name="d", progress=stop_after_two
        )

    assert seen[0] == 0.0


def test_files_that_are_not_images_are_ignored_entirely(tmp_path: Path) -> None:
    write_image(tmp_path / "g" / "Bright" / "1.png")
    (tmp_path / "g" / ".DS_Store").write_bytes(b"junk")
    (tmp_path / "g" / "notes.txt").write_text("hello")

    manifest = _scan(tmp_path)

    assert manifest.stats.files_seen == 1
    assert not manifest.warnings


# -- registry ------------------------------------------------------------------------


def test_the_adapter_is_registered_under_its_name() -> None:
    assert get_adapter("channel_folders") is ChannelFoldersAdapter
    assert ChannelFoldersAdapter in registered_adapters()


def test_an_unknown_adapter_names_the_ones_that_exist() -> None:
    with pytest.raises(UnknownAdapterError, match="channel_folders"):
        get_adapter("nope")


def test_options_are_validated_against_the_adapters_own_model() -> None:
    parsed = parse_options(ChannelFoldersAdapter, {"channels": ["uv"]})

    assert isinstance(parsed, ChannelFoldersOptions)
    assert parsed.channels == ["uv"]
    # The JSON Schema is what the import form is rendered from.
    assert "channels" in ChannelFoldersAdapter.options_model().model_json_schema()["properties"]
