"""The `folder_classes` adapter — one image is one sample, and folders carry the label.

This is the simplest contract a public dataset offers and the one most of them use: a
directory of defect-free images, one or more directories of defective ones, and nothing
else to know. You say where things are; the adapter does not guess.

That makes it the opposite of `channel_folders`, which infers labels and channels from a
vocabulary and can be confidently wrong. Here there is no vocabulary and nothing to be
wrong about — a file is in a directory you named, or it is not. The cost is that you have
to name the directories, which is exactly the trade the simple case wants.

**One image is one sample, with no channel.** That is the point of this adapter beyond
convenience: the domain model has always allowed a sample with a single unchannelled image
(ADR-0005), and until something exercised that path it was a claim rather than a fact.

Covers GKN, MVTec-AD, BTAD and most single-view public sets. For a dataset that ships a
published train/test split, prefer `csv_table`, which carries the split through instead of
resampling it.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path, PurePosixPath

from pydantic import BaseModel, ConfigDict, Field

from anomaly_lab.datasets.adapters.base import ProgressCallback, default_progress, register
from anomaly_lab.datasets.adapters.support import (
    DEFAULT_EXTENSIONS,
    FileCollector,
    candidate_files,
    capped,
    directory_matches,
    natural_key,
)
from anomaly_lab.datasets.manifest import (
    Manifest,
    ManifestImage,
    ManifestSample,
    ManifestStats,
    ManifestWarning,
    WarningCode,
)
from anomaly_lab.domain.entities import Label


class FolderClassesOptions(BaseModel):
    """Where the normal images are, where the defective ones are, and nothing implicit.

    Every list is a set of globs matched against a file's directory **relative to the
    root**, and a pattern covers the subtree beneath it. `Data_GKN/Good` and
    `*/test/good` are both valid ways to say the same kind of thing.
    """

    model_config = ConfigDict(frozen=True)

    normal_dirs: list[str] = Field(
        default_factory=list,
        description="Directories holding defect-free images, as globs relative to the root.",
    )
    defect_dirs: list[str] = Field(
        default_factory=list,
        description="Directories holding defective images, as globs relative to the root.",
    )
    unlabeled_dirs: list[str] = Field(
        default_factory=list,
        description=(
            "Directories to import without a label. Unlabelled samples are still scored, "
            "which is what makes a trained model useful as a labelling aid."
        ),
    )
    defect_type_from_dir: bool = Field(
        default=True,
        description=(
            "Record the matched directory's own name on the sample, so a dataset with "
            "several defect directories keeps the distinction between them without the "
            "schema having to enumerate defect types."
        ),
    )
    mask_dir: str | None = Field(
        default=None,
        description=(
            "Where to look for pixel-level ground truth, relative to the root. Accepts "
            "`{dir}` (the image's own directory), `{class}` (its last component), "
            "`{stem}` and `{name}`. Leave empty when the dataset has no masks."
        ),
    )
    mask_pattern: str = Field(
        default="{stem}.png",
        description="Mask filename within `mask_dir`. Same placeholders.",
    )
    extensions: list[str] = Field(
        default=DEFAULT_EXTENSIONS,
        description="File extensions to consider. The adapter is not tied to one format.",
    )
    exclude: list[str] = Field(
        default_factory=list,
        description=(
            "Glob patterns matched against each file's path relative to the root. "
            "`*` crosses directory separators, so `sub/*` excludes the whole subtree."
        ),
    )


@register
class FolderClassesAdapter:
    """Labels come from directories you name; one file becomes one single-image sample."""

    @classmethod
    def name(cls) -> str:
        return "folder_classes"

    @classmethod
    def summary(cls) -> str:
        return (
            "One image per sample, with labels taken from directories you name explicitly. "
            "The simple contract for public datasets: say where the good and bad images are."
        )

    @classmethod
    def options_model(cls) -> type[BaseModel]:
        return FolderClassesOptions

    @classmethod
    def scan(
        cls,
        root: Path,
        options: BaseModel,
        *,
        dataset_name: str,
        progress: ProgressCallback | None = None,
    ) -> Manifest:
        if not isinstance(options, FolderClassesOptions):
            options = FolderClassesOptions.model_validate(options.model_dump())
        report = progress or default_progress()

        candidates = candidate_files(root, options.extensions)
        builder = _Builder(root, options)

        # Masks usually live inside the tree being scanned, and they are images, so a
        # single-pass walk would import every mask as a sample of its own — and then
        # attach it to itself. Resolving them first is what keeps "where the masks are"
        # from having to be repeated as an `exclude` pattern.
        inputs = builder.without_masks(candidates)
        report(0.0, f"{len(inputs)} candidate files")

        for index, path in enumerate(inputs):
            builder.add(path)
            # The callback may raise to abort — that is how a cancelled job stops a scan
            # without this module needing to know that jobs exist.
            report((index + 1) / len(inputs), f"{index + 1} of {len(inputs)}")

        return builder.finish(dataset_name=dataset_name, adapter=cls.name())


class _Builder:
    def __init__(self, root: Path, options: FolderClassesOptions) -> None:
        self._root = root
        self._options = options
        self._files = FileCollector(root, exclude=options.exclude)
        self._images: dict[tuple[str, str], list[ManifestImage]] = defaultdict(list)
        self._labels: dict[tuple[str, str], Label] = {}
        self._notes: dict[tuple[str, str], str | None] = {}
        self._unmatched: list[str] = []
        self._conflicting: list[str] = []
        self._missing_masks: list[str] = []
        self._masks = 0
        self._mask_inputs = 0

    def without_masks(self, candidates: list[Path]) -> list[Path]:
        """Drop the files that are ground truth for other files in the same walk.

        Cheap: resolving a mask is path arithmetic and one `stat`, so this costs a stat
        per candidate and saves hashing every mask as if it were an input image.
        """
        annotations = {
            str(resolved)
            for path in candidates
            if (resolved := self._resolve_mask(path)) is not None
        }
        if not annotations:
            return candidates

        kept = [path for path in candidates if str(path.resolve()) not in annotations]
        self._mask_inputs = len(candidates) - len(kept)
        return kept

    def add(self, path: Path) -> None:
        relative = PurePosixPath(self._files.relative(path))
        directory = relative.parent.as_posix()
        if directory == ".":
            directory = ""

        # Probe before classifying, so an excluded or unreadable file does not also get
        # counted as one this configuration failed to label.
        image = self._files.probe_file(path)
        if image is None:
            return

        label, conflicted = self._label_for(directory)
        if conflicted:
            self._conflicting.append(str(relative))

        mask_path = self._mask_for(path, relative)
        if mask_path is not None:
            image = image.model_copy(update={"mask_path": mask_path})
            self._masks += 1

        key = (directory, relative.stem)
        self._images[key].append(image)
        self._labels.setdefault(key, label)
        if self._options.defect_type_from_dir and directory:
            self._notes.setdefault(key, PurePosixPath(directory).name)

    def _label_for(self, directory: str) -> tuple[Label, bool]:
        """Which list claims this directory, and whether more than one did.

        Order is defect, normal, unlabelled — a file in both a defect and a normal
        directory is a configuration mistake, and treating the defect claim as the
        stronger one keeps the mistake from putting a defect into the training set.
        """
        matches = [
            (Label.DEFECT, directory_matches(directory, self._options.defect_dirs)),
            (Label.NORMAL, directory_matches(directory, self._options.normal_dirs)),
            (Label.UNLABELED, directory_matches(directory, self._options.unlabeled_dirs)),
        ]
        claimed = [label for label, matched in matches if matched]
        if not claimed:
            self._unmatched.append(directory)
            return Label.UNLABELED, False
        return claimed[0], len(claimed) > 1

    def _resolve_mask(self, path: Path) -> Path | None:
        """Where this image's mask would be, if it is there. No side effects.

        Called once to identify the masks in the walk and again per image to attach them,
        so it must stay free of bookkeeping — the caller decides whether a miss is worth
        reporting, and during the first pass it is not.
        """
        if self._options.mask_dir is None:
            return None

        relative = PurePosixPath(self._files.relative(path))
        directory = relative.parent.as_posix()
        if directory == ".":
            directory = ""

        fields = {
            "dir": directory,
            "class": PurePosixPath(directory).name if directory else "",
            "stem": relative.stem,
            "name": path.name,
        }
        try:
            folder = self._options.mask_dir.format(**fields)
            filename = self._options.mask_pattern.format(**fields)
        except KeyError:
            # A bad placeholder is a configuration error, not a per-file one. Reporting
            # it once per file would produce one warning per image; the empty result
            # surfaces as "0 masks found" on the review screen instead.
            return None

        # `..` in a template is how MVTec-shaped trees reach a sibling `ground_truth`
        # directory, so the path is normalized rather than rejected.
        candidate = Path(str(self._root / folder / filename)).resolve()
        return candidate if candidate.is_file() else None

    def _mask_for(self, path: Path, relative: PurePosixPath) -> str | None:
        """This image's mask, recording the miss when the options say there should be one."""
        if self._options.mask_dir is None:
            return None

        resolved = self._resolve_mask(path)
        if resolved is not None:
            return str(resolved)

        self._missing_masks.append(str(relative))
        return None

    def finish(self, *, dataset_name: str, adapter: str) -> Manifest:
        samples = [
            ManifestSample(
                group_key=group_key,
                external_id=external_id,
                label=self._labels.get((group_key, external_id), Label.UNLABELED),
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
            # No channels: this adapter's whole premise is one unchannelled image per
            # sample. Channel count is data, and here the data says none.
            channels=[],
            samples=samples,
            warnings=self._warnings(samples),
            stats=ManifestStats(
                files_seen=self._files.seen + self._mask_inputs,
                # Ground-truth files are counted as excluded rather than silently
                # dropped, so the review screen's file arithmetic still adds up.
                files_excluded=self._files.excluded + self._mask_inputs,
                files_skipped=self._files.skipped,
                samples=len(samples),
                images=sum(len(sample.images) for sample in samples),
                masks=self._masks,
            ),
        )

    def _warnings(self, samples: list[ManifestSample]) -> list[ManifestWarning]:
        found: list[ManifestWarning] = []
        configured = bool(
            self._options.normal_dirs or self._options.defect_dirs or self._options.unlabeled_dirs
        )

        merged = [sample for sample in samples if len(sample.images) > 1]
        if merged:
            found.append(
                ManifestWarning(
                    code=WarningCode.AMBIGUOUS_SAMPLE_ID,
                    message=(
                        f"{len(merged)} samples were built from more than one file. This "
                        "adapter identifies a sample by its filename stem, so two files "
                        "in one directory differing only in extension become one sample."
                    ),
                    paths=capped(
                        f"{sample.group_key}/{sample.external_id} ({len(sample.images)} files)"
                        for sample in merged
                    ),
                )
            )

        if self._unmatched:
            directories = sorted(set(self._unmatched))
            found.append(
                ManifestWarning(
                    code=WarningCode.UNMATCHED_PATH,
                    message=(
                        f"{len(self._unmatched)} images are in {len(directories)} "
                        "directories that no option names, and import as unlabelled."
                        if configured
                        else (
                            f"No directories were configured, so all {len(self._unmatched)} "
                            "images import as unlabelled. Name the normal and defect "
                            "directories to label them at import."
                        )
                    ),
                    paths=capped(directories),
                )
            )

        if self._conflicting:
            found.append(
                ManifestWarning(
                    code=WarningCode.CONFLICTING_DIRECTORIES,
                    message=(
                        f"{len(self._conflicting)} images are in directories matched by "
                        "more than one option. They were read as defective, which is the "
                        "safe reading — a defect in the training set teaches the model "
                        "that defects are normal."
                    ),
                    paths=capped(self._conflicting),
                )
            )

        if self._missing_masks:
            found.append(
                ManifestWarning(
                    code=WarningCode.MISSING_MASK,
                    message=(
                        f"{len(self._missing_masks)} images have no mask where the mask "
                        "options say one should be. They import without ground truth, so "
                        "pixel-level metrics will not cover them."
                    ),
                    paths=capped(self._missing_masks),
                )
            )

        found.extend(self._files.warnings())
        return found
