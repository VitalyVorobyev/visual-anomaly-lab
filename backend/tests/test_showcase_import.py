"""Opt-in scan of the private showcase tree.

Skipped unless `ANOMALY_LAB_SHOWCASE_ROOT` points at it. CI never sets that variable, and
this file deliberately contains no path, no directory name and nothing that identifies
what the images are of — only counts, which are already public in the project docs.

Run it with::

    ANOMALY_LAB_SHOWCASE_ROOT=/path/to/tree uv run --directory backend pytest -k showcase

The numbers below are the M2 exit criteria, asserted rather than eyeballed.
"""

from __future__ import annotations

import os
from collections import Counter
from pathlib import Path

import pytest

from anomaly_lab.datasets.adapters.channel_folders import (
    ChannelFoldersAdapter,
    ChannelFoldersOptions,
)
from anomaly_lab.datasets.manifest import Manifest, WarningCode
from anomaly_lab.domain.entities import Label

SHOWCASE_ROOT_ENV = "ANOMALY_LAB_SHOWCASE_ROOT"
SHOWCASE_EXCLUDE_ENV = "ANOMALY_LAB_SHOWCASE_EXCLUDE"

# The labelled corpus, which is what experiments are run on.
EXPECTED_NORMAL = 98
EXPECTED_DEFECT = 91
EXPECTED_LABELLED = EXPECTED_NORMAL + EXPECTED_DEFECT  # 189

# The whole tree, including the unlabelled area that has no clean label split.
EXPECTED_UNLABELLED = 113
EXPECTED_SAMPLES = EXPECTED_LABELLED + EXPECTED_UNLABELLED
EXPECTED_IMAGES = 893
EXPECTED_GROUPS = 8
EXPECTED_CHANNELS = 3

# One capture group was acquired with two illuminations rather than three, and is also
# the only 8-bit grayscale data. It is the reason "channel count is data, never schema"
# is a rule and not a preference.
EXPECTED_TWO_CHANNEL_SAMPLES = 13
EXPECTED_EIGHT_BIT_IMAGES = 26


def _root() -> Path:
    configured = os.environ.get(SHOWCASE_ROOT_ENV)
    if not configured:
        pytest.skip(f"set {SHOWCASE_ROOT_ENV} to run the showcase scan")
    root = Path(configured).expanduser()
    if not root.is_dir():
        pytest.skip(f"{SHOWCASE_ROOT_ENV} does not point at a directory")
    return root


@pytest.fixture(scope="module")
def manifest() -> Manifest:
    """One scan for the whole module — it reads several gigabytes."""
    return ChannelFoldersAdapter.scan(_root(), ChannelFoldersOptions(), dataset_name="showcase")


def test_the_tree_groups_into_the_expected_samples(manifest: Manifest) -> None:
    counts = manifest.label_counts()

    assert counts[Label.NORMAL] == EXPECTED_NORMAL
    assert counts[Label.DEFECT] == EXPECTED_DEFECT
    assert counts[Label.UNLABELED] == EXPECTED_UNLABELLED
    assert manifest.stats.samples == EXPECTED_SAMPLES
    assert manifest.stats.images == EXPECTED_IMAGES
    assert manifest.stats.files_skipped == 0


def test_groups_and_channels_are_discovered_not_configured(manifest: Manifest) -> None:
    groups = {sample.group_key for sample in manifest.samples}

    assert len(groups) == EXPECTED_GROUPS
    assert len(manifest.channels) == EXPECTED_CHANNELS
    # Every spelling variant in the tree resolved onto the same three canonical names.
    assert {row.channel for row in manifest.channel_mapping} == set(manifest.channels)


def test_the_irregular_group_is_grouped_and_surfaced_not_dropped(manifest: Manifest) -> None:
    """Its channel name is fused into its directory name, and it has one channel fewer.

    A classifier matching whole path components would produce twice as many
    single-image samples here instead of correctly pairing the two views.
    """
    per_sample = Counter(len(sample.images) for sample in manifest.samples)

    assert per_sample[2] == EXPECTED_TWO_CHANNEL_SAMPLES
    assert per_sample[3] == EXPECTED_SAMPLES - EXPECTED_TWO_CHANNEL_SAMPLES
    # Warned about, never padded and never dropped (ADR-0005).
    assert WarningCode.VARIABLE_CHANNEL_COUNT in {w.code for w in manifest.warnings}


def test_mixed_bit_depths_survive_the_scan(manifest: Manifest) -> None:
    depths = Counter(image.bit_depth for sample in manifest.samples for image in sample.images)

    assert depths[8] == EXPECTED_EIGHT_BIT_IMAGES
    assert depths[24] == EXPECTED_IMAGES - EXPECTED_EIGHT_BIT_IMAGES


def test_no_two_files_share_content(manifest: Manifest) -> None:
    """A duplicate hash would mean an acquisition copy inflating the sample count."""
    hashes = [image.sha256 for sample in manifest.samples for image in sample.images]

    assert len(set(hashes)) == len(hashes)
    assert WarningCode.DUPLICATE_HASH not in {w.code for w in manifest.warnings}


def test_the_labelled_corpus_is_selected_by_an_exclude_option_alone() -> None:
    """Narrowing scope is configuration, not a code path (ADR-0013).

    The unlabelled area has no clean defect / no-defect split, so experiments run on the
    labelled corpus only. That choice travels in the manifest, so a re-scan proposes the
    same thing rather than needing the operator to drop entries again by hand.

    The patterns come from the environment so that no directory name of the private tree
    is written down here.
    """
    configured = os.environ.get(SHOWCASE_EXCLUDE_ENV)
    if not configured:
        pytest.skip(f"set {SHOWCASE_EXCLUDE_ENV} to a comma-separated list of globs")

    narrowed = ChannelFoldersAdapter.scan(
        _root(),
        ChannelFoldersOptions(exclude=[part.strip() for part in configured.split(",")]),
        dataset_name="showcase",
    )
    counts = narrowed.label_counts()

    assert counts[Label.UNLABELED] == 0
    assert counts[Label.NORMAL] == EXPECTED_NORMAL
    assert counts[Label.DEFECT] == EXPECTED_DEFECT
    assert narrowed.stats.samples == EXPECTED_LABELLED
    assert not narrowed.warnings, "the labelled corpus is regular; nothing to review"
