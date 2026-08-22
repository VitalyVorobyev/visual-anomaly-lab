"""The numbers behind a picture, in a form a browser can index (handbook diagnostics.md).

A rendered anomaly map answers "is this region hotter than that one". It cannot answer
"how hot, in the units the metrics are computed in", and reading the answer back out of
the picture is not possible: the colormap clips at both ends, quantizes to 256 entries,
and is resampled to the source grid, so the inverse is multi-valued. It would also be
ADR-0007 run backwards — nothing about colormap or normalization is baked into stored
data, and inverting a colormap in the client is exactly that rule reversed.

So the values are served as values. **Deliberately not `.npy`**: that would mean a dtype
parser in TypeScript, which is what the diagnostics handbook refused when it ruled this
out for rendering.
This is a fixed 24-byte header and a float32 body, decodable with a `DataView` and a loop,
and it exists to be *read* — the colormap and the display range stay server-side.

    offset  size  field
    0       4     magic, ASCII "VAM1"
    4       4     width,    uint32 LE  (after decimation)
    8       4     height,   uint32 LE  (after decimation)
    12      4     stride,   uint32 LE  (1 = every source pixel)
    16      4     channels, uint32 LE
    20      4     reserved, zero

followed by `channels` planes of `width * height` little-endian float32, row-major, one
plane after another.

**`channels` is in the header so no caller has to know it in advance.** A preprocessed
source array is `(H, W, C)` where `C` depends on the experiment's colour mode, and a client
that guessed would either encode that schema in the UI — which the channel-count rule
forbids — or discover it by asking for planes until one 404s.
"""

from __future__ import annotations

import struct

import numpy as np

MAGIC = b"VAM1"
HEADER_SIZE = 24

MAX_VALUE_BYTES = 4 * 1024 * 1024
"""Roughly 1M floats across every plane. A 256x256 map — the default, and EfficientAD's
design point — is 0.25 MB and is never decimated; a 2048x2048 source becomes stride 2."""


class PlaneError(ValueError):
    """A caller asked to encode something this format cannot represent."""


def stride_for(width: int, height: int, channels: int = 1, *, limit: int = MAX_VALUE_BYTES) -> int:
    """The smallest integer decimation that fits `limit` bytes.

    An integer factor rather than a resize, so every value sent is a value the model
    actually produced. Interpolating would put numbers on the wire that appear nowhere in
    the array, which is the opposite of what a numeric readout is for.
    """
    stride = 1
    while (width + stride - 1) // stride * ((height + stride - 1) // stride) * channels * 4 > limit:
        stride += 1
    return stride


def encode_plane(array: np.ndarray, *, limit: int = MAX_VALUE_BYTES) -> bytes:
    """A 2-D plane or a `(H, W, C)` stack, as header plus float32 body.

    Both shapes go through one function because a caller should not have to care: an
    anomaly map is one plane, a preprocessed source is however many the experiment's
    colour mode produced, and the header carries the difference.
    """
    plane = np.asarray(array)
    if plane.ndim == 2:
        plane = plane[..., np.newaxis]
    if plane.ndim != 3:
        msg = f"a value plane must be 2-D or (H, W, C), got shape {plane.shape}"
        raise PlaneError(msg)

    height, width, channels = (int(size) for size in plane.shape)
    if channels < 1:
        msg = f"a value plane needs at least one channel, got shape {plane.shape}"
        raise PlaneError(msg)

    stride = stride_for(width, height, channels, limit=limit)
    sampled = plane[::stride, ::stride]

    header = MAGIC + struct.pack("<IIIII", sampled.shape[1], sampled.shape[0], stride, channels, 0)
    # Plane-major — channel 0 whole, then channel 1 — so a reader indexes one channel with
    # a single offset instead of a stride walk.
    body = np.ascontiguousarray(np.moveaxis(sampled, -1, 0), dtype="<f4")
    return header + body.tobytes()
