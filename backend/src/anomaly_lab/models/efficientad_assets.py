"""The two files EfficientAD needs from the network, fetched by us.

EfficientAD cannot train from a dataset alone. It wants a **pretrained teacher** (40 MB),
distilled from a WideResNet on ImageNet — reproducing that distillation would need ImageNet
itself, so the published weights are an input to this method in the same way a dataset is —
and a **penalty set** of unrelated natural images (~1.5 GB of ImageNette), against which the
student is pushed toward zero so it cannot simply learn the teacher's function everywhere.

Both are fetched here rather than through anomalib, for three reasons. It is what keeps
`efficientad_custom` dependent on torch alone, so the two implementations are a real
comparison rather than two views of one library. It puts the teacher in **our** cache, so
deleting the data directory actually deletes it — the wrapper documents, as a wart, that
its teacher goes to anomalib's own platform cache and survives. And it lets an interrupted
download be a retry rather than a puzzle: the wrapper has to detect and clean up a
half-extracted tree, because the thing it calls decides whether to download by asking
whether a *directory* exists.

**There is more than one published teacher, and they are different networks.** The teacher
is not a fixed public constant the way the penalty set is: it is the output of somebody's
distillation run against a WideResNet on ImageNet, and two people who each did that
honestly ship different weights. Measured here: anomalib's `pretrained_teacher_small.pth`
and nelson1425/EfficientAD's `teacher_small.pth` have identical architecture and shapes,
and differ tensor by tensor by up to 1.4 in absolute value — they are not the same file
under two names. Which one is loaded is therefore a property of the *experiment*, chosen by
`teacher_source`, and the URLs for both live here. The reproduction repository is
Apache-2.0 and its weights are fetched at run time rather than vendored.

Every URL is checksummed, and the reproduction repository's is pinned to a commit rather
than to `main`, so "the asset changed upstream" is a checksum failure that names the file
instead of a silent change of teacher between two runs that read as comparable.

**Torch-free**, like everything else the plugin can keep out of its import path — numpy,
Pillow and the standard library. Downloading is the one thing in `src/` that touches the
network, and it never happens without a log line saying so first.

> The penalty images deliberately do **not** go through `models/preprocessing.load_array`.
> That function exists so every method sees identical *dataset* pixels (`preprocessing.py`);
> these are not dataset pixels, and the paper specifies its own transform for them — resize
> to twice the model input, drop colour with probability 0.3, then centre-crop. Routing them
> through the shared bridge would silently replace that transform with a plain resize.
"""

from __future__ import annotations

import hashlib
import shutil
import tarfile
import urllib.request
import zipfile
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from PIL import Image

from anomaly_lab.media import decode

TEACHER_SUBDIR = "efficientad-teacher"
NELSON_TEACHER_SUBDIR = "efficientad-teacher-nelson1425"
PENALTY_SUBDIR = "imagenette"

TeacherSource = Literal["anomalib", "nelson1425"]
"""Which published distillation of the teacher to load. See the module docstring."""

_NELSON_COMMIT = "a44786ccddd3697b54903394ee1361f38e65de16"
"""The last commit to touch `models/` in nelson1425/EfficientAD, pinned deliberately."""

GRAYSCALE_PROBABILITY = 0.3
"""The paper's colour-drop rate on penalty images, so the penalty is not a colour statistic."""

_IMAGE_SUFFIXES = frozenset({".jpeg", ".jpg", ".png", ".bmp", ".tif", ".tiff", ".webp"})
_CHUNK = 1 << 20
_PROGRESS_EVERY = 64 * _CHUNK


class AssetError(RuntimeError):
    """An asset is missing, refused or corrupt — always said with its name and path."""


@dataclass(frozen=True)
class Asset:
    """One downloadable archive, named so a refusal can name it."""

    name: str
    url: str
    sha256: str
    subdir: str
    purpose: str


TEACHER = Asset(
    name="efficientad_pretrained_weights.zip",
    url=(
        "https://github.com/open-edge-platform/anomalib/releases/download/"
        "efficientad_pretrained_weights/efficientad_pretrained_weights.zip"
    ),
    sha256="c09aeaa2b33f244b3261a5efdaeae8f8284a949470a4c5a526c61275fe62684a",
    subdir=TEACHER_SUBDIR,
    purpose="the pretrained EfficientAD teacher (40 MB, first run only)",
)

NELSON_TEACHERS: dict[str, Asset] = {
    size: Asset(
        name=f"teacher_{size}.pth",
        url=(
            f"https://raw.githubusercontent.com/nelson1425/EfficientAD/"
            f"{_NELSON_COMMIT}/models/teacher_{size}.pth"
        ),
        sha256=digest,
        subdir=NELSON_TEACHER_SUBDIR,
        purpose=(
            f"the nelson1425/EfficientAD reproduction's {size} teacher "
            f"({megabytes} MB, first run only)"
        ),
    )
    for size, digest, megabytes in (
        ("small", "b98a6a139087a0d743783ee59f367f0284af6003eb1a5fed0ce92bd9c0acd337", 10),
        ("medium", "a66e1513cbc252e7907531ef4536155e338a48e8edf3a5e6a66498f661414b9c", 31),
    )
}
"""The teachers from the reproduction that reports the paper's numbers.

Single `.pth` files rather than a bundle, so these are fetched directly instead of through
`ensure_asset`'s extract path.
"""

PENALTY = Asset(
    name="imagenette2.tgz",
    url="https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz",
    sha256="6cbfac238434d89fe99e651496f0812ebc7a10fa62bd42d6874042bf01de4efd",
    subdir=PENALTY_SUBDIR,
    purpose="the ImageNette penalty set (~1.5 GB, first run only)",
)


def _digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_CHUNK):
            hasher.update(chunk)
    return hasher.hexdigest()


def _download(asset: Asset, destination: Path, reporter: Any) -> None:
    """Fetch to a `.part` file, verify, then move into place.

    Nothing ever observes a partial download under its final name. That is the whole
    difference between an interrupted fetch being a retry and being a corrupt tree that the
    next run treats as ready and fails inside, several layers from the cause.
    """
    partial = destination.with_suffix(destination.suffix + ".part")
    partial.parent.mkdir(parents=True, exist_ok=True)
    partial.unlink(missing_ok=True)

    reporter.log(f"downloading {asset.name} from {asset.url}")
    downloaded = 0
    announced = 0
    try:
        with urllib.request.urlopen(asset.url) as response, partial.open("wb") as handle:
            total = int(response.headers.get("Content-Length") or 0)
            while chunk := response.read(_CHUNK):
                handle.write(chunk)
                downloaded += len(chunk)
                if downloaded - announced >= _PROGRESS_EVERY:
                    announced = downloaded
                    share = f" of {total // (1 << 20)} MB" if total else ""
                    reporter.log(f"{asset.name}: {downloaded // (1 << 20)} MB{share}")
    except OSError as exc:
        partial.unlink(missing_ok=True)
        msg = f"could not download {asset.name} from {asset.url}: {exc}"
        raise AssetError(msg) from exc

    actual = _digest(partial)
    if actual != asset.sha256:
        partial.unlink(missing_ok=True)
        msg = (
            f"{asset.name} downloaded from {asset.url} has checksum {actual}, "
            f"which is not the expected {asset.sha256}. Nothing was installed."
        )
        raise AssetError(msg)
    partial.replace(destination)


def _extract(archive: Path, destination: Path) -> None:
    """Unpack into a staging directory and move it into place atomically.

    `filter="data"` is not decoration: a tar member may name a path outside the extraction
    root, and this is the one place in the codebase that unpacks bytes off the network.
    """
    staging = destination.with_name(destination.name + ".staging")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as bundle:
            bundle.extractall(staging)
            _refuse_escapes(staging)
    else:
        with tarfile.open(archive) as bundle:
            bundle.extractall(staging, filter="data")
    staging.replace(destination)


def _refuse_escapes(root: Path) -> None:
    """Zip has no `filter=`, so the check is done afterwards, against the real paths."""
    resolved = root.resolve()
    for path in root.rglob("*"):
        if not path.resolve().is_relative_to(resolved):
            msg = f"the archive wrote {path} outside {root}; refusing it"
            raise AssetError(msg)


def ensure_asset(asset: Asset, cache_dir: Path, *, allow_downloads: bool, reporter: Any) -> Path:
    """The extracted directory for one asset, fetching it once if it is absent."""
    target = cache_dir / asset.subdir
    if target.is_dir():
        return target

    if not allow_downloads:
        msg = (
            f"EfficientAD needs {asset.purpose} at {target}, and allow_downloads is off. "
            f"Fetch {asset.url} and extract it there, or turn allow_downloads back on."
        )
        raise AssetError(msg)

    # Said before the wait, not after: on a first run this is the difference between a
    # progress bar that has not moved and a progress bar that has explained itself.
    reporter.progress(0.01, f"fetching {asset.purpose}")
    archive = cache_dir / asset.name
    cache_dir.mkdir(parents=True, exist_ok=True)
    if not (archive.is_file() and _digest(archive) == asset.sha256):
        _download(asset, archive, reporter)
    _extract(archive, target)
    archive.unlink(missing_ok=True)
    reporter.log(f"{asset.purpose} is ready at {target}")
    return target


def ensure_file(asset: Asset, cache_dir: Path, *, allow_downloads: bool, reporter: Any) -> Path:
    """One downloaded file, fetched once. The bare-file twin of `ensure_asset`.

    Kept separate rather than folded in behind a flag: the two differ in what "already
    here" means — a directory that was extracted versus a file that was verified — and a
    single function would have to be read twice to tell which branch applies.
    """
    target = cache_dir / asset.subdir / asset.name
    if target.is_file():
        return target

    if not allow_downloads:
        msg = (
            f"EfficientAD needs {asset.purpose} at {target}, and allow_downloads is off. "
            f"Fetch {asset.url} and put it there, or turn allow_downloads back on."
        )
        raise AssetError(msg)

    reporter.progress(0.01, f"fetching {asset.purpose}")
    _download(asset, target, reporter)
    reporter.log(f"{asset.purpose} is ready at {target}")
    return target


def teacher_weights(
    cache_dir: Path,
    size: str,
    *,
    source: TeacherSource = "anomalib",
    allow_downloads: bool,
    reporter: Any,
) -> Path:
    """The published teacher checkpoint for one model size, from one published source.

    The two sources are different networks, not two copies of one — see the module
    docstring. They land in different subdirectories so both can be cached at once and a
    run can be repeated against either without a refetch.
    """
    if source == "nelson1425":
        asset = NELSON_TEACHERS.get(size)
        if asset is None:
            msg = (
                f"nelson1425/EfficientAD publishes no {size} teacher; it ships small and "
                f"medium. Use teacher_source='anomalib' for this size."
            )
            raise AssetError(msg)
        return ensure_file(asset, cache_dir, allow_downloads=allow_downloads, reporter=reporter)

    root = ensure_asset(TEACHER, cache_dir, allow_downloads=allow_downloads, reporter=reporter)
    candidates = sorted(root.rglob(f"pretrained_teacher_{size}.pth"))
    if not candidates:
        msg = (
            f"the teacher bundle at {root} holds no pretrained_teacher_{size}.pth. "
            "Delete that directory to fetch it again."
        )
        raise AssetError(msg)
    return candidates[0]


def penalty_images(cache_dir: Path, *, allow_downloads: bool, reporter: Any) -> list[Path]:
    """Every usable penalty image, sorted so the order is a function of the tree alone.

    Readiness is "there is at least one image in here", found by walking the tree — not
    "there are class subdirectories", which is `ImageFolder`'s requirement and not ours. We
    never read a label off these files; they are natural images and nothing more, so
    inheriting a layout constraint from a loader we do not use would refuse valid trees.
    """
    root = ensure_asset(PENALTY, cache_dir, allow_downloads=allow_downloads, reporter=reporter)
    files = sorted(
        path
        for path in root.rglob("*")
        if path.suffix.lower() in _IMAGE_SUFFIXES and path.is_file()
    )
    if not files:
        msg = (
            f"the penalty set at {root} holds no images. Delete that directory so it can "
            "be fetched again."
        )
        raise AssetError(msg)
    return files


def load_penalty_array(
    path: Path, width: int, height: int, generator: np.random.Generator
) -> np.ndarray:
    """One penalty image as `(3, H, W)` float32 in `[0, 1]`, under the paper's transform.

    Resize to twice the model input and centre-crop back down, rather than resizing
    straight to it: the penalty images should look like *pieces* of natural scenes at a
    comparable scale to the dataset, and squashing a whole photograph into 256x256 gives the
    student a diet of miniatures. Colour is dropped 30% of the time so the penalty cannot be
    satisfied by learning that natural images are colourful.
    """
    image = decode.load(path)
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize((width * 2, height * 2), Image.Resampling.BILINEAR)
    if float(generator.random()) < GRAYSCALE_PROBABILITY:
        image = image.convert("L").convert("RGB")
    left = (image.width - width) // 2
    top = (image.height - height) // 2
    image = image.crop((left, top, left + width, top + height))
    array = np.asarray(image, dtype=np.float32) / 255.0
    return np.ascontiguousarray(array.transpose(2, 0, 1))


class PenaltyStream:
    """An endless, reproducible supply of penalty images.

    Reproducible is the point. The reference draws these from a shuffled torch `DataLoader`
    whose position is outside its checkpoint entirely, so continuing a run restarts the
    penalty sequence — handbook jobs.md records that as an accepted cost. Here the order is a
    permutation seeded from one integer, and `state()` is small enough to sit in a
    checkpoint, so a continued run picks up the sequence where it stopped.
    """

    def __init__(
        self,
        files: list[Path],
        size: tuple[int, int],
        generator: np.random.Generator,
    ) -> None:
        self._files = files
        self._size = size
        self._generator = generator
        self._epoch_seed = int(generator.integers(2**31))
        self._cursor = 0

    @property
    def _order(self) -> np.ndarray:
        return np.random.default_rng(self._epoch_seed).permutation(len(self._files))

    def __iter__(self) -> Iterator[np.ndarray]:
        while True:
            yield self.next()

    def next(self) -> np.ndarray:
        """The next penalty image, reshuffling when the set is exhausted."""
        order = self._order
        if self._cursor >= len(order):
            self._epoch_seed = int(self._generator.integers(2**31))
            self._cursor = 0
            order = self._order
        path = self._files[int(order[self._cursor])]
        self._cursor += 1
        width, height = self._size
        return load_penalty_array(path, width, height, self._generator)

    def state(self) -> dict[str, Any]:
        """What a checkpoint needs to resume this sequence.

        `files` is recorded as a count rather than a list: the paths are large and are
        derivable, and a *changed* count is the one thing worth reporting, because the
        permutation is over indices and a different tree makes the resumed order meaningless.
        """
        return {
            "epoch_seed": self._epoch_seed,
            "cursor": self._cursor,
            "files": len(self._files),
            "generator": self._generator.bit_generator.state,
        }

    def restore(self, state: dict[str, Any], reporter: Any) -> None:
        """Put the sequence back where a previous run left it, or say why it cannot be."""
        stored = int(state.get("files", 0))
        if stored != len(self._files):
            reporter.log(
                f"the penalty set now holds {len(self._files)} images where the checkpoint "
                f"was written against {stored}, so the penalty order restarts rather than "
                "resuming a permutation of a different set",
                "warning",
            )
            return
        self._epoch_seed = int(state["epoch_seed"])
        self._cursor = int(state["cursor"])
        self._generator.bit_generator.state = state["generator"]
