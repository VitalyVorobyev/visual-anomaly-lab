"""The import manifest (ADR-0006).

A scan proposes; a human accepts. The manifest is what sits between the two: everything
the adapter concluded about a directory tree, in a form that can be read, edited and
committed, and that is then stored verbatim so the import is answerable months later.

Three properties are load-bearing:

  * **It is versioned.** A manifest outlives the code that wrote it, and `manifest_version`
    is what lets a future reader know which shape it is holding.
  * **Warnings are never fatal.** A sample with a different channel count from its
    siblings is legitimate data (ADR-0005); a file that could not be read is worth
    knowing about but must not abort a scan of three thousand others.
  * **Nothing here counts channels.** `channels` is a list because a dataset has whatever
    channels it has — one, two, three, or none.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, Field

from anomaly_lab.domain.entities import Label, Subset
from anomaly_lab.schemas import API_MODEL_CONFIG

# Bump when a change to these models is not backwards compatible. Readers check it before
# trusting anything else in the file.
MANIFEST_VERSION = 1


class WarningCode(StrEnum):
    """Why the operator is being asked to look at something."""

    VARIABLE_CHANNEL_COUNT = "variable_channel_count"
    """A sample has more or fewer channels than most. Legitimate, but worth seeing."""

    UNASSIGNED_CHANNEL = "unassigned_channel"
    """An image landed in a dataset that has channels, but matched none of them."""

    UNKNOWN_CHANNEL_NAME = "unknown_channel_name"
    """A directory name that looks like it should be a channel and was not recognized."""

    DUPLICATE_HASH = "duplicate_hash"
    """Two files with identical content. Often a copy-paste during acquisition."""

    UNREADABLE_FILE = "unreadable_file"
    """Skipped: the file is not an image this application can read."""

    EMPTY_FILE = "empty_file"
    """Skipped: zero bytes."""

    AMBIGUOUS_SAMPLE_ID = "ambiguous_sample_id"
    """Two files in one group resolved to the same sample identity and were merged."""

    UNMATCHED_PATH = "unmatched_path"
    """A file matched none of the configured directories and imported as unlabelled."""

    CONFLICTING_DIRECTORIES = "conflicting_directories"
    """A file matched more than one label's directory patterns. The first list won."""

    MISSING_MASK = "missing_mask"
    """A mask was expected for this image and the file is not there."""

    REDUNDANT_COLOUR_PLANES = "redundant_colour_planes"
    """The images decode as colour, but one plane predicts the others almost exactly."""

    MIXED_IMAGE_MODES = "mixed_image_modes"
    """Not every file decodes to the same colour mode."""

    CHANNEL_OFFSET = "channel_offset"
    """A sample's channels are translated relative to one another rather than registered."""

    CHANNEL_DIMENSIONS_DIFFER = "channel_dimensions_differ"
    """A sample's channels do not share a width and height, so nothing can be read across
    them pixel-for-pixel."""


class ManifestWarning(BaseModel):
    model_config = API_MODEL_CONFIG

    code: WarningCode
    message: str
    paths: list[str] = Field(default_factory=list)


class ChannelMapping(BaseModel):
    """One source directory name and the canonical channel it was read as.

    The whole table is editable before commit: the fuzzy matcher is a convenience, not
    an authority, and a dataset with unfamiliar channel names must be importable without
    a code change (ADR-0006).
    """

    model_config = API_MODEL_CONFIG

    source: str
    channel: str
    matched_by: str
    """`exact`, `prefix`, `alias`, or `token` — how confident this row is."""


class ManifestImage(BaseModel):
    model_config = API_MODEL_CONFIG

    path: str
    channel: str | None = None
    sha256: str
    width: int
    height: int
    bit_depth: int
    file_size: int
    mask_path: str | None = Field(
        default=None,
        description=(
            "Pixel-level ground truth for this image, referenced in place like the image. "
            "`null` for the datasets that have none, which is most of them."
        ),
    )


class ManifestSample(BaseModel):
    model_config = API_MODEL_CONFIG

    group_key: str
    external_id: str
    label: Label = Label.UNLABELED
    subset: Subset | None = Field(
        default=None,
        description=(
            "The subset this sample belongs to according to the source dataset. Only "
            "adapters that read a published split fill this in; it is what a split with "
            "the `imported` strategy is built from, so that a benchmark can be run under "
            "the same partition the published numbers used."
        ),
    )
    notes: str | None = Field(
        default=None,
        description=(
            "Free text the adapter read from the source, such as a defect type. Carried "
            "onto the sample so a breakdown by defect type is possible without a schema "
            "that enumerates defect types."
        ),
    )
    images: list[ManifestImage] = Field(default_factory=list)


class ManifestStats(BaseModel):
    """What the scan actually did, so the review screen can show its scope."""

    model_config = API_MODEL_CONFIG

    files_seen: int = 0
    files_excluded: int = 0
    files_skipped: int = 0
    samples: int = 0
    images: int = 0
    masks: int = 0
    """Images that came with pixel-level ground truth. Zero for most datasets."""


class Manifest(BaseModel):
    """A complete import proposal. Nothing has been written when this is produced."""

    model_config = API_MODEL_CONFIG

    manifest_version: int = MANIFEST_VERSION
    adapter: str
    options: dict[str, object] = Field(default_factory=dict)
    dataset_name: str
    root_path: str
    channels: list[str] = Field(default_factory=list)
    channel_mapping: list[ChannelMapping] = Field(default_factory=list)
    samples: list[ManifestSample] = Field(default_factory=list)
    warnings: list[ManifestWarning] = Field(default_factory=list)
    stats: ManifestStats = Field(default_factory=ManifestStats)
    probe: dict[str, object] | None = Field(
        default=None,
        description=(
            "What the scan measured about the pixels, as opposed to what the adapter read "
            "from the paths: colour-plane redundancy and channel registration. A plain "
            "mapping here rather than the typed model, because `probe` imports this module "
            "and a manifest must stay readable without it."
        ),
    )
    scanned_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())

    def label_counts(self) -> dict[Label, int]:
        counts = dict.fromkeys(Label, 0)
        for sample in self.samples:
            counts[sample.label] += 1
        return counts

    def subset_counts(self) -> dict[Subset, int]:
        """How the source dataset partitioned these samples, if it said at all.

        Samples with no subset are absent from the total rather than bucketed somewhere,
        so a partially annotated manifest reads as partial instead of as balanced.
        """
        counts = dict.fromkeys(Subset, 0)
        for sample in self.samples:
            if sample.subset is not None:
                counts[sample.subset] += 1
        return counts

    def has_imported_split(self) -> bool:
        return any(sample.subset is not None for sample in self.samples)
