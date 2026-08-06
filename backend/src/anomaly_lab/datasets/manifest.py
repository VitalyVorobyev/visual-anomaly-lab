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

from pydantic import BaseModel, ConfigDict, Field

from anomaly_lab.domain.entities import Label

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


class ManifestWarning(BaseModel):
    model_config = ConfigDict(frozen=True)

    code: WarningCode
    message: str
    paths: list[str] = Field(default_factory=list)


class ChannelMapping(BaseModel):
    """One source directory name and the canonical channel it was read as.

    The whole table is editable before commit: the fuzzy matcher is a convenience, not
    an authority, and a dataset with unfamiliar channel names must be importable without
    a code change (ADR-0006).
    """

    model_config = ConfigDict(frozen=True)

    source: str
    channel: str
    matched_by: str
    """`exact`, `prefix`, `alias`, or `token` — how confident this row is."""


class ManifestImage(BaseModel):
    model_config = ConfigDict(frozen=True)

    path: str
    channel: str | None = None
    sha256: str
    width: int
    height: int
    bit_depth: int
    file_size: int


class ManifestSample(BaseModel):
    model_config = ConfigDict(frozen=True)

    group_key: str
    external_id: str
    label: Label = Label.UNLABELED
    images: list[ManifestImage] = Field(default_factory=list)


class ManifestStats(BaseModel):
    """What the scan actually did, so the review screen can show its scope."""

    model_config = ConfigDict(frozen=True)

    files_seen: int = 0
    files_excluded: int = 0
    files_skipped: int = 0
    samples: int = 0
    images: int = 0


class Manifest(BaseModel):
    """A complete import proposal. Nothing has been written when this is produced."""

    model_config = ConfigDict(frozen=True)

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
    scanned_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())

    def label_counts(self) -> dict[Label, int]:
        counts = dict.fromkeys(Label, 0)
        for sample in self.samples:
            counts[sample.label] += 1
        return counts
