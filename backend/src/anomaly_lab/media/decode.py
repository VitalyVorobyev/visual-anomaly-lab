"""Reading pixels, and reading only what is needed.

Two operations, deliberately separate because their costs differ by three orders of
magnitude: `probe` reads a file's header, and `load` decodes it.

Bit depth is not special-cased at any call site (§9). The reference data mixes 24-bit
colour and 8-bit grayscale in the same dataset, and the tier renderer, the manifest and
the browser all treat that as ordinary.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, UnidentifiedImageError

# Pillow reports a mode, not a depth. This is the mapping back, and it is the only place
# that knows about modes at all.
_BITS_PER_MODE = {
    "1": 1,
    "L": 8,
    "P": 8,
    "LA": 16,
    "I;16": 16,
    "I": 32,
    "F": 32,
    "RGB": 24,
    "YCbCr": 24,
    "LAB": 24,
    "HSV": 24,
    "RGBA": 32,
    "CMYK": 32,
}
_DEFAULT_BITS = 8

# 3.9 MB per source image, so hashing is streamed rather than read whole.
_HASH_CHUNK_BYTES = 1024 * 1024


class UnreadableImageError(Exception):
    """The file is not an image this application can read. Reported, never fatal."""


@dataclass(frozen=True)
class ImageInfo:
    """Everything the catalog records about a file without decoding its pixels."""

    width: int
    height: int
    bit_depth: int
    file_size: int
    sha256: str


def bits_for_mode(mode: str) -> int:
    return _BITS_PER_MODE.get(mode, _DEFAULT_BITS)


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_HASH_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def probe(path: Path) -> ImageInfo:
    """Header plus hash, without decoding pixels.

    Pillow's `open` reads only enough to identify the file, so a scan over thousands of
    images costs a hash of each rather than a full decode of each.
    """
    try:
        with Image.open(path) as image:
            width, height = image.size
            bit_depth = bits_for_mode(image.mode)
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        msg = f"{path} could not be read as an image: {exc}"
        raise UnreadableImageError(msg) from exc

    return ImageInfo(
        width=width,
        height=height,
        bit_depth=bit_depth,
        file_size=path.stat().st_size,
        sha256=sha256_of(path),
    )


def load(path: Path) -> Image.Image:
    """Decode a file into an in-memory image.

    Palette images are expanded and 16-bit data is brought down to 8 bits, so that
    everything downstream sees either `L` or `RGB` and no renderer needs a mode switch.
    """
    try:
        with Image.open(path) as image:
            image.load()
            return _normalize(image)
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        msg = f"{path} could not be decoded: {exc}"
        raise UnreadableImageError(msg) from exc


def _normalize(image: Image.Image) -> Image.Image:
    if image.mode in {"L", "RGB"}:
        return image.copy()
    if image.mode in {"1", "I", "I;16", "F", "LA"}:
        return image.convert("L")
    return image.convert("RGB")
