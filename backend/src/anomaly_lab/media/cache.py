"""Rendered image tiers and their cache (§9).

Source images are large — the reference data is 1280x1024 uncompressed, about 3.9 MB
each — so serving them straight to a WebView would move hundreds of megabytes per screen
of thumbnails. Three tiers exist instead:

| tier      | size            | format          | used by                          |
| --------- | --------------- | --------------- | -------------------------------- |
| `thumb`   | 256 px long edge| WebP, q80       | the browser grid, ranked lists   |
| `preview` | 1024 px long edge| WebP, q85      | the sample viewer                |
| `full`    | native          | lossless PNG    | pixel-peeping, overlay alignment |

`full` is **lossless** because it is used to judge defects and to check that an anomaly
map lines up with what it claims to describe; JPEG-style ringing there would be mistaken
for a surface feature.

**`thumb` and `preview` are cached on disk; `full` is not.** A cached full tier would cost
roughly 1.2 MB per image — most of a gigabyte for one dataset — to save a render of an
image that is viewed once. It is rendered per request instead and kept out of the way by
an `ETag`, so a client that has seen it never asks for the bytes again.

**Keying by `image_id` alone is safe because imported files are immutable**: the path and
`sha256` are recorded at import, and `verify` is what detects drift. There is no cache
invalidation problem here, so none is built.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from PIL import Image

from anomaly_lab.config import Settings
from anomaly_lab.domain.entities import Image as ImageRow
from anomaly_lab.media.decode import load


class ImageTier(StrEnum):
    THUMB = "thumb"
    PREVIEW = "preview"
    FULL = "full"


@dataclass(frozen=True)
class TierSpec:
    long_edge: int | None
    """`None` means native resolution."""

    media_type: str
    cached: bool


TIERS: dict[ImageTier, TierSpec] = {
    ImageTier.THUMB: TierSpec(long_edge=256, media_type="image/webp", cached=True),
    ImageTier.PREVIEW: TierSpec(long_edge=1024, media_type="image/webp", cached=True),
    ImageTier.FULL: TierSpec(long_edge=None, media_type="image/png", cached=False),
}

_WEBP_QUALITY = {ImageTier.THUMB: 80, ImageTier.PREVIEW: 85}


def etag_for(image: ImageRow, tier: ImageTier) -> str:
    """A strong validator derived from the content, not the clock.

    The hash identifies the bytes and the tier identifies the rendering of them, so the
    pair changes if and only if what would be served changes.
    """
    return f'"{image.sha256}-{tier.value}"'


def cache_path(settings: Settings, image_id: int, tier: ImageTier) -> Path:
    return settings.thumbnails_dir / tier.value / f"{image_id}.webp"


def render(image: ImageRow, tier: ImageTier) -> bytes:
    """Render one tier into memory. Never touches the cache."""
    spec = TIERS[tier]
    source = load(Path(image.path))

    if spec.long_edge is None:
        buffer = io.BytesIO()
        source.save(buffer, "PNG", optimize=False)
        return buffer.getvalue()

    source.thumbnail((spec.long_edge, spec.long_edge), Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    source.save(buffer, "WEBP", quality=_WEBP_QUALITY[tier])
    return buffer.getvalue()


def ensure_cached(settings: Settings, image: ImageRow, tier: ImageTier) -> Path:
    """Return the cached file for a tier, rendering it on first use.

    Written to a temporary name and renamed into place, so a reader arriving mid-render
    sees either the previous file or the finished one, never a truncated image.
    """
    if not TIERS[tier].cached:
        msg = f"the {tier.value} tier is rendered on demand and never cached"
        raise ValueError(msg)

    path = cache_path(settings, image.id, tier)
    if path.is_file():
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = render(image, tier)
    scratch = path.with_name(f".{path.name}.{image.sha256[:8]}.partial")
    scratch.write_bytes(payload)
    scratch.replace(path)
    return path


def is_cached(settings: Settings, image: ImageRow, tier: ImageTier) -> bool:
    return TIERS[tier].cached and cache_path(settings, image.id, tier).is_file()
