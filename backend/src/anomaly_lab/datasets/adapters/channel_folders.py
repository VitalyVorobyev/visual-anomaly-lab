"""The `channel_folders` adapter.

Reads a tree whose directory names carry the label and the acquisition channel, and whose
filename stems identify the part within a capture group.

**It classifies path components, not path positions.** Nothing here knows whether label
folders sit above channel folders or below them; each component is tested against a label
vocabulary, then a channel vocabulary, and whatever is left becomes the group key. That
is what lets one adapter read a tree it has never seen, and it is the only thing that
copes with a real case where the channel name is fused into the group folder's own name
(`"<Channel> <Group>"` as two sibling directories, rather than `<Group>/<Channel>/`).

Two consequences of that rule are worth stating outright:

  * **Label components stay in the group key.** Real trees contain the same numeric stem
    under both a defect and a no-defect folder, so dropping the label component would
    collide two different parts onto one identity.
  * **Every match is recorded.** ADR-0006 accepts that fuzzy matching can be confidently
    wrong; the escape hatch is only useful if the operator can see what was matched, so
    the channel mapping is part of the manifest and editable before commit.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from fnmatch import fnmatch
from itertools import islice
from pathlib import Path, PurePosixPath

from pydantic import BaseModel, ConfigDict, Field

from anomaly_lab.datasets.adapters.base import ProgressCallback, default_progress, register
from anomaly_lab.datasets.manifest import (
    ChannelMapping,
    Manifest,
    ManifestImage,
    ManifestSample,
    ManifestStats,
    ManifestWarning,
    WarningCode,
)
from anomaly_lab.domain.entities import Label
from anomaly_lab.media.decode import UnreadableImageError, probe

# A warning that lists nine hundred paths is not a warning, it is a wall. The full count
# always appears in the message.
MAX_PATHS_PER_WARNING = 50

_NON_ALPHANUMERIC = re.compile(r"[^a-z0-9]")
_TOKEN_SEPARATORS = re.compile(r"[\s_\-.]+")


class ChannelFoldersOptions(BaseModel):
    """Everything the adapter knows about a layout, as data.

    `channels` is **empty by default, deliberately**. Shipping a vocabulary would make one
    acquisition setup's illumination names part of the application, and the rule is that
    channel count and channel naming are data (ADR-0005). An operator says what their
    channels are called; the adapter does the matching.

    The label vocabularies do keep defaults, and that is a different case: `good` and
    `defect` are not one dataset's private words, they are what the English-language
    convention for a labelled folder looks like across every public dataset. They are
    still options, so a tree using other words is imported by editing them.
    """

    model_config = ConfigDict(frozen=True)

    channels: list[str] = Field(
        default_factory=list,
        description=(
            "Canonical channel names. A directory or token that begins with one of these, "
            "once lower-cased and stripped of punctuation, is read as that channel. Leave "
            "empty for a single-view dataset: with no vocabulary nothing is read as a "
            "channel, so every image becomes its own sample."
        ),
    )
    channel_aliases: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Explicit source-name to canonical-channel overrides, consulted before any "
            "fuzzy matching. The escape hatch for names the matcher gets wrong."
        ),
    )
    normal_labels: list[str] = Field(
        default=["no-defect", "nodefect", "no_defect", "normal", "ok", "good", "pass"],
        description="Directory names, matched whole, that mark a sample as normal.",
    )
    defect_labels: list[str] = Field(
        default=["defect", "defects", "defective", "ng", "bad", "fail"],
        description="Directory names, matched whole, that mark a sample as defective.",
    )
    extensions: list[str] = Field(
        default=[".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp"],
        description="File extensions to consider. The adapter is not tied to one format.",
    )
    exclude: list[str] = Field(
        default_factory=list,
        description=(
            "Glob patterns matched against each file's path relative to the root. "
            "`*` crosses directory separators, so `sub/*` excludes the whole subtree. "
            "Import scope belongs here rather than in a one-off edit, so that a re-scan "
            "proposes the same thing."
        ),
    )


@dataclass(frozen=True)
class _Classification:
    group_key: str
    external_id: str
    label: Label
    channel: str | None
    source: str | None
    matched_by: str | None


def _normalize(text: str) -> str:
    return _NON_ALPHANUMERIC.sub("", text.lower())


def _tokenize(text: str) -> list[str]:
    return [token for token in _TOKEN_SEPARATORS.split(text) if token]


def _match_channel(
    text: str,
    options: ChannelFoldersOptions,
    *,
    allow_prefix: bool = True,
) -> tuple[str, str] | None:
    """Resolve one name to a canonical channel, reporting how it was resolved.

    `allow_prefix` is what keeps prefix matching honest. Normalization strips the spaces
    out of a name, so `"Brightfield Bl7"` becomes `brightfieldbl7`, which *does* start
    with `bright` — matching it as a whole would silently swallow the group name `Bl7`
    and merge that group into its parent. Prefix matching is therefore offered only for
    single-token names; anything with more than one token goes to token matching, which
    keeps the leftovers.
    """
    normalized = _normalize(text)
    if not normalized:
        return None

    alias = options.channel_aliases.get(normalized) or options.channel_aliases.get(text)
    if alias:
        return alias, "alias"

    for channel in options.channels:
        if normalized == _normalize(channel):
            return channel, "exact"

    if not allow_prefix:
        return None

    # Longest first, so that listing both `dark` and `darkfield` resolves to the more
    # specific one rather than whichever happens to come first.
    for channel in sorted(options.channels, key=len, reverse=True):
        if normalized.startswith(_normalize(channel)):
            return channel, "prefix"
    return None


def _label_for(component: str, options: ChannelFoldersOptions) -> Label | None:
    """Whole-component match only.

    Substring matching would be actively wrong here: `no-defect` normalizes to a string
    that *contains* `defect`, so a looser test would label every normal sample defective.
    """
    normalized = _normalize(component)
    if normalized in {_normalize(name) for name in options.defect_labels}:
        return Label.DEFECT
    if normalized in {_normalize(name) for name in options.normal_labels}:
        return Label.NORMAL
    return None


def classify(relative_path: PurePosixPath, options: ChannelFoldersOptions) -> _Classification:
    """Decide what one file is, from its path alone."""
    components = list(relative_path.parts[:-1])
    external_id = PurePosixPath(relative_path.name).stem

    label = Label.UNLABELED
    label_indices: set[int] = set()
    for index, component in enumerate(components):
        found = _label_for(component, options)
        if found is not None:
            label_indices.add(index)
            if label is Label.UNLABELED:
                label = found

    # Pass 1: a component that is *entirely* a channel name, innermost first. A real
    # channel directory should always win over a stray word in a group folder's name.
    channel: str | None = None
    matched_by: str | None = None
    source: str | None = None
    consumed: int | None = None
    for index in reversed(range(len(components))):
        if index in label_indices:
            continue
        component = components[index]
        match = _match_channel(component, options, allow_prefix=len(_tokenize(component)) == 1)
        if match is not None:
            channel, matched_by = match
            source = component
            consumed = index
            break

    if channel is not None:
        group_parts = [c for i, c in enumerate(components) if i != consumed]
        return _Classification(
            group_key="/".join(group_parts),
            external_id=external_id,
            label=label,
            channel=channel,
            source=source,
            matched_by=matched_by,
        )

    # Pass 2: no directory is a channel on its own, so look inside the names. This is
    # what resolves a channel fused into a group folder's name.
    group_parts = []
    for index, component in enumerate(components):
        if index in label_indices or channel is not None:
            group_parts.append(component)
            continue

        tokens = _tokenize(component)
        kept: list[str] = []
        for token in tokens:
            match = _match_channel(token, options) if channel is None else None
            if match is None:
                kept.append(token)
                continue
            channel, _ = match
            matched_by = "token"
            source = token

        if channel is not None and len(kept) != len(tokens):
            if kept:
                group_parts.append(" ".join(kept))
        else:
            group_parts.append(component)

    return _Classification(
        group_key="/".join(group_parts),
        external_id=external_id,
        label=label,
        channel=channel,
        source=source,
        matched_by=matched_by,
    )


@register
class ChannelFoldersAdapter:
    """Directory names carry the label and the channel; filename stems carry identity."""

    @classmethod
    def name(cls) -> str:
        return "channel_folders"

    @classmethod
    def summary(cls) -> str:
        return (
            "Groups images by filename stem across sibling channel directories, reading "
            "labels and channels from directory names in any nesting order."
        )

    @classmethod
    def options_model(cls) -> type[BaseModel]:
        return ChannelFoldersOptions

    @classmethod
    def scan(
        cls,
        root: Path,
        options: BaseModel,
        *,
        dataset_name: str,
        progress: ProgressCallback | None = None,
    ) -> Manifest:
        if not isinstance(options, ChannelFoldersOptions):
            options = ChannelFoldersOptions.model_validate(options.model_dump())
        report = progress or default_progress()

        candidates = _candidate_files(root, options)
        report(0.0, f"{len(candidates)} candidate files")

        builder = _ScanBuilder(root, options)
        for index, path in enumerate(candidates):
            builder.add(path)
            # The callback may raise to abort — that is how a cancelled job stops a scan
            # without this module needing to know that jobs exist.
            report((index + 1) / len(candidates), f"{index + 1} of {len(candidates)}")

        return builder.finish(dataset_name=dataset_name, adapter=cls.name())


def _candidate_files(root: Path, options: ChannelFoldersOptions) -> list[Path]:
    suffixes = {suffix.lower() for suffix in options.extensions}
    found = [
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in suffixes and not path.name.startswith(".")
    ]
    return sorted(found)


class _ScanBuilder:
    """Accumulates one scan's samples, channels and warnings."""

    def __init__(self, root: Path, options: ChannelFoldersOptions) -> None:
        self._root = root
        self._options = options
        self._samples: dict[tuple[str, str], list[ManifestImage]] = defaultdict(list)
        self._labels: dict[tuple[str, str], Label] = {}
        self._mapping: dict[str, ChannelMapping] = {}
        self._hashes: dict[str, list[str]] = defaultdict(list)
        self._unassigned: list[str] = []
        self._unreadable: list[str] = []
        self._empty: list[str] = []
        self._excluded = 0
        self._seen = 0

    def add(self, path: Path) -> None:
        relative = PurePosixPath(path.relative_to(self._root).as_posix())
        self._seen += 1

        if any(fnmatch(str(relative), pattern) for pattern in self._options.exclude):
            self._excluded += 1
            return

        if path.stat().st_size == 0:
            self._empty.append(str(relative))
            return

        try:
            info = probe(path)
        except UnreadableImageError:
            self._unreadable.append(str(relative))
            return

        found = classify(relative, self._options)
        key = (found.group_key, found.external_id)

        if found.channel is not None and found.source is not None:
            self._mapping.setdefault(
                found.source,
                ChannelMapping(
                    source=found.source,
                    channel=found.channel,
                    matched_by=found.matched_by or "exact",
                ),
            )
        else:
            self._unassigned.append(str(relative))

        # A sample's label is whatever the first labelled path said. Conflicting labels
        # across a sample's own images would be a tree problem, not something to average.
        if self._labels.get(key, Label.UNLABELED) is Label.UNLABELED:
            self._labels[key] = found.label

        self._samples[key].append(
            ManifestImage(
                path=str(path),
                channel=found.channel,
                sha256=info.sha256,
                width=info.width,
                height=info.height,
                bit_depth=info.bit_depth,
                file_size=info.file_size,
            )
        )
        self._hashes[info.sha256].append(str(relative))

    def finish(self, *, dataset_name: str, adapter: str) -> Manifest:
        samples = [
            ManifestSample(
                group_key=group_key,
                external_id=external_id,
                label=self._labels.get((group_key, external_id), Label.UNLABELED),
                images=images,
            )
            for (group_key, external_id), images in sorted(
                self._samples.items(), key=lambda item: _natural_key(*item[0])
            )
        ]

        return Manifest(
            adapter=adapter,
            options=self._options.model_dump(),
            dataset_name=dataset_name,
            root_path=str(self._root),
            channels=self._ordered_channels(),
            channel_mapping=sorted(self._mapping.values(), key=lambda row: row.source),
            samples=samples,
            warnings=self._warnings(samples),
            stats=ManifestStats(
                files_seen=self._seen,
                files_excluded=self._excluded,
                files_skipped=len(self._unreadable) + len(self._empty),
                samples=len(samples),
                images=sum(len(sample.images) for sample in samples),
            ),
        )

    def _ordered_channels(self) -> list[str]:
        """Configured order first, then anything an alias introduced."""
        found = {row.channel for row in self._mapping.values()}
        ordered = [channel for channel in self._options.channels if channel in found]
        ordered.extend(sorted(found - set(ordered)))
        return ordered

    def _warnings(self, samples: list[ManifestSample]) -> list[ManifestWarning]:
        warnings: list[ManifestWarning] = []
        channels = self._ordered_channels()

        # No vocabulary is a legitimate single-view import, and also the shape of the
        # mistake where someone points this adapter at a multi-view tree and configures
        # nothing. The review screen cannot tell the two apart, so it says what happened
        # and lets the operator decide (ADR-0006).
        if not self._options.channels and samples:
            warnings.append(
                ManifestWarning(
                    code=WarningCode.UNKNOWN_CHANNEL_NAME,
                    message=(
                        f"No channel names were configured, so all {len(samples)} samples "
                        "import as single images with no channel. If this tree has "
                        "several views per part, name them in `channels` and scan again; "
                        "if it does not, `folder_classes` is the simpler adapter for it."
                    ),
                )
            )

        if samples and channels:
            counts = Counter(len(sample.images) for sample in samples)
            modal, _ = counts.most_common(1)[0]
            odd = [sample for sample in samples if len(sample.images) != modal]
            if odd:
                warnings.append(
                    ManifestWarning(
                        code=WarningCode.VARIABLE_CHANNEL_COUNT,
                        message=(
                            f"{len(odd)} of {len(samples)} samples do not have {modal} "
                            "images. Variable channel counts are legitimate data, not an "
                            "error; check that the grouping is what you expect."
                        ),
                        paths=_capped(
                            f"{sample.group_key}/{sample.external_id} ({len(sample.images)} images)"
                            for sample in odd
                        ),
                    )
                )

        if self._unassigned and channels:
            warnings.append(
                ManifestWarning(
                    code=WarningCode.UNASSIGNED_CHANNEL,
                    message=(
                        f"{len(self._unassigned)} images matched no channel in a dataset "
                        f"that has {len(channels)}. They will import with no channel."
                    ),
                    paths=_capped(self._unassigned),
                )
            )
            unknown = sorted({PurePosixPath(path).parent.name for path in self._unassigned} - {""})
            if unknown:
                warnings.append(
                    ManifestWarning(
                        code=WarningCode.UNKNOWN_CHANNEL_NAME,
                        message=(
                            "These directory names were not recognized as channels. If any "
                            "of them is one, add it to the channel mapping before committing."
                        ),
                        paths=_capped(unknown),
                    )
                )

        duplicates = {sha: paths for sha, paths in self._hashes.items() if len(paths) > 1}
        if duplicates:
            warnings.append(
                ManifestWarning(
                    code=WarningCode.DUPLICATE_HASH,
                    message=(
                        f"{len(duplicates)} groups of files have identical content. "
                        "Often a copy made during acquisition."
                    ),
                    paths=_capped(" = ".join(paths) for paths in sorted(duplicates.values())),
                )
            )

        if self._unreadable:
            warnings.append(
                ManifestWarning(
                    code=WarningCode.UNREADABLE_FILE,
                    message=f"{len(self._unreadable)} files could not be read and were skipped.",
                    paths=_capped(self._unreadable),
                )
            )

        if self._empty:
            warnings.append(
                ManifestWarning(
                    code=WarningCode.EMPTY_FILE,
                    message=f"{len(self._empty)} files are empty and were skipped.",
                    paths=_capped(self._empty),
                )
            )

        return warnings


def _natural_key(group_key: str, external_id: str) -> tuple[str, int, str]:
    """Order samples so `10` follows `9` rather than `1`.

    Mirrors the ordering the sample repository uses, so the manifest's order and the
    browser's order are the same order.
    """
    return group_key, len(external_id), external_id


def _capped(items: Iterable[object]) -> list[str]:
    return [str(item) for item in islice(items, MAX_PATHS_PER_WARNING)]
