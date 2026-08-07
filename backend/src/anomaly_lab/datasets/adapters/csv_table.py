"""The `csv_table` adapter — the dataset's own table is the authority.

A benchmark dataset that publishes numbers also publishes the partition those numbers were
computed on. Re-drawing a split with our own seed produces a number that cannot be compared
to the paper's, and a comparison that looks rigorous while measuring something else is
worse than no comparison. So when the source ships a table, this adapter reads it: which
files, which labels, which masks, and **which subset**.

Nothing about the table's shape is assumed beyond "it has columns". Every column name is an
option, and the values that mean *normal*, *defective* and each subset are options too, so
a table using `0`/`1`, or `good`/`ng`, or `train`/`test`/`val` in any spelling, is read by
configuring rather than by patching.

`filter_column` / `filter_value` exist for the common case where one file covers a whole
benchmark family: a table with an `object` column holding twelve class names becomes twelve
datasets, each imported under the one-class protocol those benchmarks are scored under.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path, PurePosixPath

from pydantic import BaseModel, ConfigDict, Field, model_validator

from anomaly_lab.datasets.adapters.base import ProgressCallback, default_progress, register
from anomaly_lab.datasets.adapters.support import (
    FileCollector,
    capped,
    natural_key,
)
from anomaly_lab.datasets.manifest import (
    ChannelMapping,
    Manifest,
    ManifestImage,
    ManifestSample,
    ManifestStats,
    ManifestWarning,
    WarningCode,
)
from anomaly_lab.domain.entities import Label, Subset

# A table with a million rows is a different tool's problem; this bound exists so a
# mis-pointed file cannot exhaust memory before anyone notices it is the wrong file.
MAX_ROWS = 1_000_000


class CsvTableOptions(BaseModel):
    """Which columns mean what. Only `csv_path` and `image_column` are required."""

    model_config = ConfigDict(frozen=True)

    csv_path: str = Field(
        description=(
            "The table, relative to the root or absolute. Image paths inside it are "
            "resolved against the root."
        ),
    )
    image_column: str = Field(default="image", description="Column holding the image path.")
    label_column: str | None = Field(
        default="label",
        description="Column holding the label. Empty to import everything unlabelled.",
    )
    mask_column: str | None = Field(
        default="mask",
        description="Column holding the ground-truth mask path. Blank cells mean no mask.",
    )
    split_column: str | None = Field(
        default="split",
        description=(
            "Column holding the published subset. This is what a split with the "
            "`imported` strategy is built from."
        ),
    )
    group_column: str | None = Field(
        default=None,
        description=(
            "Column holding the capture group. Defaults to the image's own directory, "
            "which is the right answer whenever the tree is organized by acquisition."
        ),
    )
    sample_id_column: str | None = Field(
        default=None,
        description="Column identifying the sample. Defaults to the image's filename stem.",
    )
    channel_column: str | None = Field(
        default=None,
        description=(
            "Column naming the acquisition channel. Set it and rows sharing a sample "
            "identity become one multi-channel sample; leave it and each row is a "
            "single-image sample."
        ),
    )
    notes_column: str | None = Field(
        default=None,
        description="Column holding free text, such as a defect type, recorded on the sample.",
    )
    filter_column: str | None = Field(
        default=None,
        description="Import only rows whose value in this column equals `filter_value`.",
    )
    filter_value: str | None = Field(
        default=None, description="The value `filter_column` must have."
    )
    normal_values: list[str] = Field(
        default=["normal", "good", "ok", "pass", "0", "false"],
        description="Values in the label column, compared case-insensitively, meaning normal.",
    )
    defect_values: list[str] = Field(
        default=["anomaly", "anomalous", "defect", "defective", "bad", "ng", "1", "true"],
        description="Values in the label column meaning defective.",
    )
    train_values: list[str] = Field(
        default=["train", "training"],
        description="Values in the split column meaning the training subset.",
    )
    val_values: list[str] = Field(
        default=["val", "valid", "validation"],
        description=(
            "Values meaning the validation subset. Official one-class protocols usually "
            "have none, and an empty validation subset is expected rather than a fault."
        ),
    )
    test_values: list[str] = Field(
        default=["test", "testing"],
        description="Values in the split column meaning the evaluation subset.",
    )

    @model_validator(mode="after")
    def _filter_needs_both_halves(self) -> CsvTableOptions:
        if (self.filter_column is None) != (self.filter_value is None):
            msg = "filter_column and filter_value must be given together"
            raise ValueError(msg)
        return self


@register
class CsvTableAdapter:
    """A table in the source tree says which file is what, including its published subset."""

    @classmethod
    def name(cls) -> str:
        return "csv_table"

    @classmethod
    def summary(cls) -> str:
        return (
            "Reads a CSV shipped with the dataset: image paths, labels, masks and the "
            "published train/test split, with every column name configurable."
        )

    @classmethod
    def options_model(cls) -> type[BaseModel]:
        return CsvTableOptions

    @classmethod
    def scan(
        cls,
        root: Path,
        options: BaseModel,
        *,
        dataset_name: str,
        progress: ProgressCallback | None = None,
    ) -> Manifest:
        if not isinstance(options, CsvTableOptions):
            options = CsvTableOptions.model_validate(options.model_dump())
        report = progress or default_progress()

        table = _resolve_table(root, options.csv_path)
        rows = _read_rows(table, options)
        report(0.0, f"{len(rows)} rows selected from {table.name}")

        builder = _Builder(root, options, table)
        for index, row in enumerate(rows):
            builder.add(row)
            # The callback may raise to abort a cancelled job.
            report((index + 1) / len(rows), f"{index + 1} of {len(rows)}")

        return builder.finish(dataset_name=dataset_name, adapter=cls.name())


class TableError(Exception):
    """The table cannot be read, or does not have the columns the options name."""


def _resolve_table(root: Path, csv_path: str) -> Path:
    candidate = Path(csv_path).expanduser()
    table = candidate if candidate.is_absolute() else root / candidate
    if not table.is_file():
        msg = f"no table at {table}"
        raise TableError(msg)
    return table


def _read_rows(table: Path, options: CsvTableOptions) -> list[dict[str, str]]:
    """Read and filter the table, failing loudly on a column name that is not there.

    A missing column is raised rather than warned about: every row would be unreadable in
    the same way, and a manifest proposing nothing is a worse explanation than a message
    naming the column and listing the ones that exist.
    """
    with table.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        header = reader.fieldnames or []
        _require_columns(header, options, table)

        rows: list[dict[str, str]] = []
        for row in reader:
            if (
                options.filter_column is not None
                and (row.get(options.filter_column) or "").strip() != options.filter_value
            ):
                continue
            if not (row.get(options.image_column) or "").strip():
                continue
            rows.append({key: (value or "") for key, value in row.items() if key is not None})
            if len(rows) > MAX_ROWS:
                msg = f"{table} has more than {MAX_ROWS} matching rows"
                raise TableError(msg)
    return rows


def _require_columns(header: Sequence[str], options: CsvTableOptions, table: Path) -> None:
    named = {
        "image_column": options.image_column,
        "label_column": options.label_column,
        "mask_column": options.mask_column,
        "split_column": options.split_column,
        "group_column": options.group_column,
        "sample_id_column": options.sample_id_column,
        "channel_column": options.channel_column,
        "notes_column": options.notes_column,
        "filter_column": options.filter_column,
    }
    # Optional columns that are simply absent are tolerated by clearing them, except the
    # ones the operator typed on purpose. The distinction the code can actually make is
    # "was it left at its default": a default that does not exist in this table means the
    # table does not carry that information, which is normal.
    defaults = {"label_column", "mask_column", "split_column"}
    missing = [
        f"{option}={column!r}"
        for option, column in named.items()
        if column is not None and column not in header and option not in defaults
    ]
    if missing:
        msg = (
            f"{table.name} has no column(s) for {', '.join(missing)}. "
            f"Columns present: {', '.join(header) or 'none'}"
        )
        raise TableError(msg)


class _Builder:
    def __init__(self, root: Path, options: CsvTableOptions, table: Path) -> None:
        self._root = root
        self._options = options
        self._table = table
        self._files = FileCollector(root)
        self._images: dict[tuple[str, str], list[ManifestImage]] = defaultdict(list)
        self._labels: dict[tuple[str, str], Label] = {}
        self._subsets: dict[tuple[str, str], Subset | None] = {}
        self._notes: dict[tuple[str, str], str | None] = {}
        self._channels: dict[str, ChannelMapping] = {}
        self._unknown_labels: set[str] = set()
        self._unknown_subsets: set[str] = set()
        self._missing_masks: list[str] = []
        self._masks = 0

    def add(self, row: dict[str, str]) -> None:
        relative = PurePosixPath((row.get(self._options.image_column) or "").strip())
        path = self._root / relative

        channel = self._value(row, self._options.channel_column)
        image = self._files.probe_file(path, channel=channel)
        if image is None:
            return

        mask = self._mask_for(row, relative)
        if mask is not None:
            image = image.model_copy(update={"mask_path": mask})
            self._masks += 1

        if channel:
            self._channels.setdefault(
                channel, ChannelMapping(source=channel, channel=channel, matched_by="exact")
            )

        key = (self._group_for(row, relative), self._id_for(row, relative))
        self._images[key].append(image)
        self._labels.setdefault(key, self._label_for(row))
        self._subsets.setdefault(key, self._subset_for(row))
        self._notes.setdefault(key, self._value(row, self._options.notes_column))

    def _value(self, row: dict[str, str], column: str | None) -> str | None:
        if column is None:
            return None
        value = (row.get(column) or "").strip()
        return value or None

    def _group_for(self, row: dict[str, str], relative: PurePosixPath) -> str:
        explicit = self._value(row, self._options.group_column)
        if explicit is not None:
            return explicit
        parent = relative.parent.as_posix()
        return "" if parent == "." else parent

    def _id_for(self, row: dict[str, str], relative: PurePosixPath) -> str:
        return self._value(row, self._options.sample_id_column) or relative.stem

    def _label_for(self, row: dict[str, str]) -> Label:
        value = self._value(row, self._options.label_column)
        if value is None:
            return Label.UNLABELED
        folded = value.casefold()
        if folded in {item.casefold() for item in self._options.defect_values}:
            return Label.DEFECT
        if folded in {item.casefold() for item in self._options.normal_values}:
            return Label.NORMAL
        self._unknown_labels.add(value)
        return Label.UNLABELED

    def _subset_for(self, row: dict[str, str]) -> Subset | None:
        value = self._value(row, self._options.split_column)
        if value is None:
            return None
        folded = value.casefold()
        for subset, vocabulary in (
            (Subset.TRAIN, self._options.train_values),
            (Subset.VAL, self._options.val_values),
            (Subset.TEST, self._options.test_values),
        ):
            if folded in {item.casefold() for item in vocabulary}:
                return subset
        self._unknown_subsets.add(value)
        return None

    def _mask_for(self, row: dict[str, str], relative: PurePosixPath) -> str | None:
        value = self._value(row, self._options.mask_column)
        if value is None:
            return None
        mask = self._root / PurePosixPath(value)
        if mask.is_file():
            return str(mask)
        self._missing_masks.append(str(relative))
        return None

    def finish(self, *, dataset_name: str, adapter: str) -> Manifest:
        samples = [
            ManifestSample(
                group_key=group_key,
                external_id=external_id,
                label=self._labels.get((group_key, external_id), Label.UNLABELED),
                subset=self._subsets.get((group_key, external_id)),
                notes=self._notes.get((group_key, external_id)),
                images=images,
            )
            for (group_key, external_id), images in sorted(
                self._images.items(), key=lambda item: natural_key(*item[0])
            )
        ]

        return Manifest(
            adapter=adapter,
            options=self._options.model_dump(),
            dataset_name=dataset_name,
            root_path=str(self._root),
            channels=sorted(self._channels),
            channel_mapping=sorted(self._channels.values(), key=lambda row: row.source),
            samples=samples,
            warnings=self._warnings(samples),
            stats=ManifestStats(
                files_seen=self._files.seen,
                files_excluded=self._files.excluded,
                files_skipped=self._files.skipped,
                samples=len(samples),
                images=sum(len(sample.images) for sample in samples),
                masks=self._masks,
            ),
        )

    def _warnings(self, samples: list[ManifestSample]) -> list[ManifestWarning]:
        found: list[ManifestWarning] = []

        if self._unknown_labels:
            found.append(
                ManifestWarning(
                    code=WarningCode.UNKNOWN_CHANNEL_NAME,
                    message=(
                        f"{len(self._unknown_labels)} values in "
                        f"{self._options.label_column!r} match neither the normal nor the "
                        "defect vocabulary, and those rows import as unlabelled. Add them "
                        "to `normal_values` or `defect_values` and scan again."
                    ),
                    paths=capped(sorted(self._unknown_labels)),
                )
            )

        if self._unknown_subsets:
            found.append(
                ManifestWarning(
                    code=WarningCode.UNKNOWN_CHANNEL_NAME,
                    message=(
                        f"{len(self._unknown_subsets)} values in "
                        f"{self._options.split_column!r} match no subset vocabulary. Those "
                        "samples carry no imported subset and a split with the `imported` "
                        "strategy will leave them out."
                    ),
                    paths=capped(sorted(self._unknown_subsets)),
                )
            )

        multi = [sample for sample in samples if len(sample.images) > 1]
        if multi and self._options.channel_column is None:
            found.append(
                ManifestWarning(
                    code=WarningCode.AMBIGUOUS_SAMPLE_ID,
                    message=(
                        f"{len(multi)} samples were built from more than one row without a "
                        "channel column, so several images share one sample identity. Set "
                        "`channel_column`, or `sample_id_column`, to say which was meant."
                    ),
                    paths=capped(
                        f"{sample.group_key}/{sample.external_id} ({len(sample.images)} rows)"
                        for sample in multi
                    ),
                )
            )

        if self._missing_masks:
            found.append(
                ManifestWarning(
                    code=WarningCode.MISSING_MASK,
                    message=(
                        f"{len(self._missing_masks)} rows name a mask that is not on disk. "
                        "Those images import without ground truth, so pixel-level metrics "
                        "will not cover them."
                    ),
                    paths=capped(self._missing_masks),
                )
            )

        found.extend(self._files.warnings())
        return found
