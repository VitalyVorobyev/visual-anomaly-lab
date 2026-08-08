"""Asset acquisition for `efficientad_custom` — refusals, atomicity, and the penalty order.

**Torch-free on purpose**, so this runs in the CI job that installs without the `dl` extra:
fetching the teacher and the penalty set is plain bytes-and-paths work, and the boundary is
worth keeping measurable.

**No test here touches the network.** The download path is exercised through `file://`
URLs, which is what lets the interesting parts — the checksum gate, the `.part` staging, the
atomic move — be tested at all rather than asserted in a docstring.
"""

from __future__ import annotations

import hashlib
import tarfile
import zipfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from anomaly_lab.models.efficientad_assets import (
    NELSON_TEACHER_SUBDIR,
    NELSON_TEACHERS,
    PENALTY_SUBDIR,
    TEACHER_SUBDIR,
    Asset,
    AssetError,
    PenaltyStream,
    ensure_asset,
    ensure_file,
    load_penalty_array,
    penalty_images,
    teacher_weights,
)


class Recorder:
    """A reporter that keeps what it was told, so a refusal can be read back."""

    def __init__(self) -> None:
        self.messages: list[str] = []

    def log(self, message: str, level: str = "info") -> None:
        self.messages.append(message)

    def progress(self, fraction: float, message: str | None = None) -> None:
        return

    def metric(self, name: str, value: float, step: int | None = None) -> None:
        return

    def should_cancel(self) -> bool:
        return False


def _write_image(path: Path, colour: tuple[int, int, int], size: int = 40) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (size, size), colour).save(path)


def _archive(tmp_path: Path, tree: Path, *, kind: str) -> tuple[Path, str]:
    """Pack `tree` and return the archive and its digest, ready to serve over `file://`."""
    archive = tmp_path / ("bundle.zip" if kind == "zip" else "bundle.tgz")
    if kind == "zip":
        with zipfile.ZipFile(archive, "w") as bundle:
            for path in sorted(tree.rglob("*")):
                if path.is_file():
                    bundle.write(path, path.relative_to(tree))
    else:
        with tarfile.open(archive, "w:gz") as bundle:
            for path in sorted(tree.rglob("*")):
                if path.is_file():
                    bundle.add(path, str(path.relative_to(tree)))
    return archive, hashlib.sha256(archive.read_bytes()).hexdigest()


def _asset(archive: Path, digest: str, subdir: str = "thing") -> Asset:
    return Asset(
        name=archive.name,
        url=archive.as_uri(),
        sha256=digest,
        subdir=subdir,
        purpose="a test asset",
    )


def test_a_missing_asset_with_downloads_off_is_refused_by_name(tmp_path: Path) -> None:
    """The refusal has to carry the URL and the path, or it is not actionable.

    Somebody who turned downloads off wants to place the file by hand. A message saying
    only "downloads are disabled" makes that a search through source code.
    """
    asset = _asset(tmp_path / "absent.tgz", "0" * 64)
    with pytest.raises(AssetError) as failure:
        ensure_asset(asset, tmp_path / "cache", allow_downloads=False, reporter=Recorder())

    message = str(failure.value)
    assert asset.url in message
    assert str(tmp_path / "cache" / "thing") in message


def test_an_already_present_asset_is_not_fetched_again(tmp_path: Path) -> None:
    """Present means present — and with downloads off, the check must come first."""
    cache = tmp_path / "cache"
    (cache / "thing").mkdir(parents=True)
    asset = _asset(tmp_path / "absent.tgz", "0" * 64)

    resolved = ensure_asset(asset, cache, allow_downloads=False, reporter=Recorder())
    assert resolved == cache / "thing"


@pytest.mark.parametrize("kind", ["zip", "tgz"])
def test_an_asset_is_downloaded_verified_and_extracted(tmp_path: Path, kind: str) -> None:
    """The whole path, over `file://`: fetch, checksum, unpack, move into place."""
    tree = tmp_path / "tree"
    _write_image(tree / "n01" / "a.jpeg", (200, 30, 30))
    _write_image(tree / "n02" / "b.jpeg", (30, 200, 30))
    archive, digest = _archive(tmp_path, tree, kind=kind)

    cache = tmp_path / "cache"
    resolved = ensure_asset(
        _asset(archive, digest), cache, allow_downloads=True, reporter=Recorder()
    )

    assert resolved == cache / "thing"
    assert sorted(path.name for path in resolved.rglob("*.jpeg")) == ["a.jpeg", "b.jpeg"]
    # The archive is not left behind to double the asset's footprint on disk.
    assert not (cache / archive.name).exists()


def test_a_checksum_mismatch_installs_nothing(tmp_path: Path) -> None:
    """A corrupt download must not become a corrupt cache.

    The failure mode this guards is the expensive one: a truncated 1.5 GB fetch that
    extracts far enough to look ready, and then fails inside training with an error about
    an image rather than about a download.
    """
    tree = tmp_path / "tree"
    _write_image(tree / "a.jpeg", (10, 10, 10))
    archive, _ = _archive(tmp_path, tree, kind="tgz")

    cache = tmp_path / "cache"
    with pytest.raises(AssetError, match="checksum"):
        ensure_asset(_asset(archive, "0" * 64), cache, allow_downloads=True, reporter=Recorder())

    assert not (cache / "thing").exists()
    assert list(cache.glob("*.part")) == []


def test_the_teacher_checkpoint_is_found_wherever_the_bundle_puts_it(tmp_path: Path) -> None:
    """The published zip nests its two checkpoints in a directory; ours walks for them."""
    tree = tmp_path / "tree"
    nested = tree / "efficientad_pretrained_weights"
    nested.mkdir(parents=True)
    (nested / "pretrained_teacher_small.pth").write_bytes(b"small")
    (nested / "pretrained_teacher_medium.pth").write_bytes(b"medium")
    archive, digest = _archive(tmp_path, tree, kind="zip")

    cache = tmp_path / "cache"
    asset = _asset(archive, digest, subdir="efficientad-teacher")
    ensure_asset(asset, cache, allow_downloads=True, reporter=Recorder())

    found = teacher_weights(cache, "medium", allow_downloads=False, reporter=Recorder())
    assert found.read_bytes() == b"medium"


def test_a_teacher_bundle_missing_the_size_is_refused_by_name(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    (cache / "efficientad-teacher").mkdir(parents=True)
    with pytest.raises(AssetError, match=r"pretrained_teacher_small\.pth"):
        teacher_weights(cache, "small", allow_downloads=False, reporter=Recorder())


def test_the_reproduction_teacher_is_fetched_as_a_bare_file(tmp_path: Path) -> None:
    """nelson1425 ships a `.pth`, not a bundle, so it takes the `ensure_file` path.

    The whole route over `file://`: refuse with downloads off, then fetch, checksum, and
    land under a *different* subdirectory from anomalib's — both teachers can be cached at
    once, because a run may be repeated against either without a refetch.
    """
    weights = tmp_path / "teacher_small.pth"
    weights.write_bytes(b"a different distillation")
    digest = hashlib.sha256(weights.read_bytes()).hexdigest()
    asset = Asset(
        name="teacher_small.pth",
        url=weights.as_uri(),
        sha256=digest,
        subdir=NELSON_TEACHER_SUBDIR,
        purpose="the reproduction's small teacher",
    )
    cache = tmp_path / "cache"

    with pytest.raises(AssetError) as refusal:
        ensure_file(asset, cache, allow_downloads=False, reporter=Recorder())
    assert asset.url in str(refusal.value)

    resolved = ensure_file(asset, cache, allow_downloads=True, reporter=Recorder())
    assert resolved == cache / NELSON_TEACHER_SUBDIR / "teacher_small.pth"
    assert resolved.read_bytes() == b"a different distillation"
    assert list(cache.rglob("*.part")) == []


def test_a_size_the_reproduction_does_not_publish_is_refused_by_name(tmp_path: Path) -> None:
    """It ships small and medium. A third size must name the alternative, not 404."""
    with pytest.raises(AssetError, match="teacher_source='anomalib'"):
        teacher_weights(
            tmp_path / "cache",
            "large",
            source="nelson1425",
            allow_downloads=False,
            reporter=Recorder(),
        )


def test_the_two_teacher_sources_do_not_share_a_cache_directory() -> None:
    """If they collided, switching source would silently reuse the other one's weights."""
    assert NELSON_TEACHER_SUBDIR != TEACHER_SUBDIR
    assert {asset.subdir for asset in NELSON_TEACHERS.values()} == {NELSON_TEACHER_SUBDIR}
    assert set(NELSON_TEACHERS) == {"small", "medium"}


def test_penalty_images_are_found_without_class_directories(tmp_path: Path) -> None:
    """We read no labels off these files, so we must not inherit `ImageFolder`'s layout rule.

    The reference loads the penalty set through `ImageFolder`, which requires one
    subdirectory per class and refuses a flat tree. A flat directory of natural images is a
    perfectly good penalty set, and refusing it would be inheriting a constraint from a
    loader we do not use.
    """
    cache = tmp_path / "cache"
    root = cache / PENALTY_SUBDIR
    _write_image(root / "flat-a.jpeg", (10, 20, 30))
    _write_image(root / "nested" / "deep" / "flat-b.png", (40, 50, 60))
    (root / "notes.txt").write_text("not an image", encoding="utf-8")

    found = penalty_images(cache, allow_downloads=False, reporter=Recorder())
    assert [path.name for path in found] == ["flat-a.jpeg", "flat-b.png"]


def test_a_penalty_directory_with_no_images_is_refused(tmp_path: Path) -> None:
    """Present-but-empty is the shape an interrupted extraction leaves behind."""
    cache = tmp_path / "cache"
    (cache / PENALTY_SUBDIR).mkdir(parents=True)
    with pytest.raises(AssetError, match="no images"):
        penalty_images(cache, allow_downloads=False, reporter=Recorder())


def test_a_penalty_image_is_resized_to_twice_the_input_then_centre_cropped(tmp_path: Path) -> None:
    """The transform is the paper's, and the crop is what makes it a *piece* of a scene.

    Checked by construction rather than by shape alone: a source whose left half is one
    colour and right half another, resized and centre-cropped, must still show both — a
    plain resize to the target would too, but a crop taken from a corner would not.
    """
    source = tmp_path / "wide.png"
    array = np.zeros((64, 128, 3), dtype=np.uint8)
    array[:, :64] = (255, 0, 0)
    array[:, 64:] = (0, 0, 255)
    Image.fromarray(array).save(source)

    result = load_penalty_array(source, 32, 32, np.random.default_rng(0))

    assert result.shape == (3, 32, 32)
    assert result.dtype == np.float32
    assert float(result.min()) >= 0.0 and float(result.max()) <= 1.0
    # Left column still red, right column still blue: the crop is centred.
    assert result[0, 16, 0] > result[2, 16, 0]
    assert result[2, 16, -1] > result[0, 16, -1]


def test_colour_is_dropped_on_roughly_three_in_ten_penalty_images(tmp_path: Path) -> None:
    """The paper's p=0.3 grayscale, pinned as behaviour rather than as a constant.

    A transform that silently stopped converting would leave the penalty term measuring
    "natural images are colourful", which is not what it is for — and nothing else in the
    system would notice.
    """
    source = tmp_path / "colour.png"
    _write_image(source, (200, 40, 40))

    def is_grayscale(array: np.ndarray) -> bool:
        return bool(np.allclose(array[0], array[1]) and np.allclose(array[1], array[2]))

    generator = np.random.default_rng(11)
    converted = [is_grayscale(load_penalty_array(source, 16, 16, generator)) for _ in range(200)]
    assert 40 <= sum(converted) <= 80


def test_the_penalty_stream_is_a_permutation_not_sampling_with_replacement(tmp_path: Path) -> None:
    """Every penalty image is seen once before any is seen twice.

    Sampling with replacement would work and would quietly weight some images 3x within an
    epoch — invisible in any loss curve, and exactly the sort of difference that makes two
    implementations disagree for a reason nobody can find.
    """
    cache = tmp_path / "cache"
    root = cache / PENALTY_SUBDIR
    for index in range(6):
        _write_image(root / f"{index}.png", (index * 40, 0, 0))
    files = penalty_images(cache, allow_downloads=False, reporter=Recorder())

    stream = PenaltyStream(files, (16, 16), np.random.default_rng(3))
    seen = [float(stream.next()[0].mean()) for _ in range(6)]
    assert len(set(seen)) == 6

    # And it keeps going past the end of the set rather than stopping.
    assert stream.next().shape == (3, 16, 16)


def test_the_penalty_order_survives_a_checkpoint(tmp_path: Path) -> None:
    """The improvement over the wrapper, whose penalty sequence restarts on every resume.

    ADR-0025 recorded that restart as an accepted cost. Recording the order costs four
    numbers, so it is not a cost worth accepting here.
    """
    cache = tmp_path / "cache"
    root = cache / PENALTY_SUBDIR
    for index in range(8):
        _write_image(root / f"{index}.png", (index * 30, 0, 0))
    files = penalty_images(cache, allow_downloads=False, reporter=Recorder())

    original = PenaltyStream(files, (16, 16), np.random.default_rng(5))
    for _ in range(3):
        original.next()
    state = original.state()
    expected = [float(original.next()[0].mean()) for _ in range(4)]

    restored = PenaltyStream(files, (16, 16), np.random.default_rng(99))
    restored.restore(state, Recorder())
    actual = [float(restored.next()[0].mean()) for _ in range(4)]

    assert actual == expected


def test_a_changed_penalty_set_restarts_the_order_and_says_so(tmp_path: Path) -> None:
    """A permutation of 8 indices means nothing over a set of 4 files.

    Resuming it anyway would silently train against the wrong images; the honest move is to
    restart and say why.
    """
    cache = tmp_path / "cache"
    root = cache / PENALTY_SUBDIR
    for index in range(4):
        _write_image(root / f"{index}.png", (index * 30, 0, 0))
    files = penalty_images(cache, allow_downloads=False, reporter=Recorder())

    stream = PenaltyStream(files, (16, 16), np.random.default_rng(5))
    reporter = Recorder()
    stream.restore({"epoch_seed": 1, "cursor": 2, "files": 8, "generator": None}, reporter)

    assert any("penalty order restarts" in message for message in reporter.messages)
