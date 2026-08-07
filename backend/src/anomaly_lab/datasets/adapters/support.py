"""Pieces every import adapter needs, so that three of them do not each invent one.

Deliberately small. An adapter's *interesting* part is how it reads identity, label and
channel out of a source layout, and that stays in the adapter. What lives here is the
uninteresting part every adapter repeats identically: deciding which files to look at,
turning one file into a `ManifestImage`, and accumulating the warnings that a scan owes
its operator (ADR-0006).
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from fnmatch import fnmatch
from itertools import islice
from pathlib import Path, PurePosixPath

from anomaly_lab.datasets.manifest import ManifestImage, ManifestWarning, WarningCode
from anomaly_lab.media.decode import UnreadableImageError, probe

# A warning that lists nine hundred paths is not a warning, it is a wall. The full count
# always appears in the message.
MAX_PATHS_PER_WARNING = 50

DEFAULT_EXTENSIONS = [".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp"]


def capped(items: Iterable[object]) -> list[str]:
    return [str(item) for item in islice(items, MAX_PATHS_PER_WARNING)]


def candidate_files(root: Path, extensions: Iterable[str]) -> list[Path]:
    """Every readable-looking image under `root`, sorted for a deterministic scan order.

    Dot-files are skipped outright: on macOS a source tree is littered with `.DS_Store`,
    and reporting each of them as unreadable would bury the warnings that matter.
    """
    suffixes = {suffix.lower() for suffix in extensions}
    found = [
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in suffixes and not path.name.startswith(".")
    ]
    return sorted(found)


def directory_matches(relative_dir: str, patterns: Iterable[str]) -> bool:
    """Does a file's directory fall under any of these globs?

    A pattern names a directory and covers everything **below** it, which is what makes
    "point at the folder holding the good parts" behave the way an operator expects: the
    contract asked for is *where things are*, not a precise glob for every leaf. `*`
    crosses separators, so `data/*` still means the whole subtree when that is wanted.
    """
    subject = relative_dir.strip("/")
    for raw in patterns:
        pattern = raw.strip("/")
        if not pattern:
            # An empty pattern is the root itself, and matching everything from it would
            # silently make one label swallow the tree.
            continue
        if fnmatch(subject, pattern) or fnmatch(subject, f"{pattern}/*"):
            return True
    return False


class FileCollector:
    """Probes files, keeps the skip reasons, and turns them into warnings at the end.

    An adapter calls `probe_file` and gets either a `ManifestImage` or `None`; when it is
    `None` the reason has already been recorded. That keeps the skip-and-warn behaviour
    identical across adapters instead of each one deciding for itself whether an empty
    file is fatal. It is not: a scan of three thousand files must not abort on one.
    """

    def __init__(self, root: Path, *, exclude: Iterable[str] = ()) -> None:
        self._root = root
        self._exclude = list(exclude)
        self._hashes: dict[str, list[str]] = defaultdict(list)
        self._unreadable: list[str] = []
        self._empty: list[str] = []
        self._missing: list[str] = []
        self.seen = 0
        self.excluded = 0

    @property
    def skipped(self) -> int:
        return len(self._unreadable) + len(self._empty) + len(self._missing)

    def relative(self, path: Path) -> str:
        """The path as the operator wrote it, POSIX-style, for warnings and matching."""
        try:
            return PurePosixPath(path.relative_to(self._root).as_posix()).as_posix()
        except ValueError:
            # A table may name a file outside the root. That is legitimate — the CSV is
            # the authority on where images live — so fall back to the absolute path
            # rather than refusing to describe it.
            return path.as_posix()

    def is_excluded(self, relative_path: str) -> bool:
        return any(fnmatch(relative_path, pattern) for pattern in self._exclude)

    def probe_file(self, path: Path, *, channel: str | None = None) -> ManifestImage | None:
        """One file to one manifest entry, or `None` with the reason recorded."""
        relative_path = self.relative(path)
        self.seen += 1

        if self.is_excluded(relative_path):
            self.excluded += 1
            return None

        if not path.is_file():
            self._missing.append(relative_path)
            return None

        if path.stat().st_size == 0:
            self._empty.append(relative_path)
            return None

        try:
            info = probe(path)
        except UnreadableImageError:
            self._unreadable.append(relative_path)
            return None

        self._hashes[info.sha256].append(relative_path)
        return ManifestImage(
            path=str(path),
            channel=channel,
            sha256=info.sha256,
            width=info.width,
            height=info.height,
            bit_depth=info.bit_depth,
            file_size=info.file_size,
        )

    def warnings(self) -> list[ManifestWarning]:
        found: list[ManifestWarning] = []

        duplicates = {sha: paths for sha, paths in self._hashes.items() if len(paths) > 1}
        if duplicates:
            found.append(
                ManifestWarning(
                    code=WarningCode.DUPLICATE_HASH,
                    message=(
                        f"{len(duplicates)} groups of files have identical content. "
                        "Often a copy made during acquisition."
                    ),
                    paths=capped(" = ".join(paths) for paths in sorted(duplicates.values())),
                )
            )

        if self._missing:
            found.append(
                ManifestWarning(
                    code=WarningCode.UNREADABLE_FILE,
                    message=(
                        f"{len(self._missing)} listed files are not on disk and were "
                        "skipped. The source table names images that are not there."
                    ),
                    paths=capped(self._missing),
                )
            )

        if self._unreadable:
            found.append(
                ManifestWarning(
                    code=WarningCode.UNREADABLE_FILE,
                    message=f"{len(self._unreadable)} files could not be read and were skipped.",
                    paths=capped(self._unreadable),
                )
            )

        if self._empty:
            found.append(
                ManifestWarning(
                    code=WarningCode.EMPTY_FILE,
                    message=f"{len(self._empty)} files are empty and were skipped.",
                    paths=capped(self._empty),
                )
            )

        return found


def natural_key(group_key: str, external_id: str) -> tuple[str, int, str]:
    """Order samples so `10` follows `9` rather than `1`.

    Mirrors the ordering the sample repository uses, so the manifest's order and the
    browser's order are the same order.
    """
    return group_key, len(external_id), external_id
