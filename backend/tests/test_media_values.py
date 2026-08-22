"""The float32 value plane (handbook diagnostics.md).

The format exists so a browser can read a number under the cursor without a numpy parser
in TypeScript and without inverting a colormap. Its whole contract is the 16-byte header,
so that is what these check — including the decimation, because a silently sampled plane
presented as exact values is exactly the failure this header exists to prevent.
"""

from __future__ import annotations

import struct

import numpy as np
import pytest

from anomaly_lab.media.values import HEADER_SIZE, MAGIC, PlaneError, encode_plane, stride_for


def decode(payload: bytes) -> tuple[np.ndarray, int]:
    """The reader the frontend implements, written once more here to check against.

    Returns `(H, W, C)`, so a single-plane payload comes back with a trailing 1 and the
    multi-channel case is not a separate code path here either.
    """
    assert payload[:4] == MAGIC
    width, height, stride, channels, reserved = struct.unpack("<IIIII", payload[4:HEADER_SIZE])
    assert reserved == 0
    values = np.frombuffer(payload[HEADER_SIZE:], dtype="<f4")
    assert values.size == width * height * channels
    return np.moveaxis(values.reshape(channels, height, width), 0, -1), stride


def test_a_plane_round_trips_exactly() -> None:
    array = np.arange(12, dtype=np.float32).reshape(3, 4) / 7.0
    plane, stride = decode(encode_plane(array))

    assert stride == 1
    assert plane.shape == (3, 4, 1)
    np.testing.assert_allclose(plane[..., 0], array, rtol=0, atol=0)


def test_the_header_reports_the_dimensions_actually_sent() -> None:
    """Not the source's — a reader indexing by the source shape would be off by the stride."""
    array = np.zeros((7, 5), dtype=np.float32)
    payload = encode_plane(array)
    width, height, _, channels, _ = struct.unpack("<IIIII", payload[4:HEADER_SIZE])

    assert (height, width, channels) == (7, 5, 1)
    assert len(payload) == HEADER_SIZE + 7 * 5 * 4


def test_the_default_map_size_is_never_decimated() -> None:
    """256x256 is the preprocessing default and EfficientAD's design point: 0.25 MB."""
    assert stride_for(256, 256) == 1
    _, stride = decode(encode_plane(np.zeros((256, 256), dtype=np.float32)))
    assert stride == 1


def test_a_large_plane_is_decimated_and_says_so() -> None:
    """A silent truncation would read as "this is the value", at the wrong resolution."""
    assert stride_for(2048, 2048) == 2

    plane, stride = decode(encode_plane(np.zeros((2048, 2048), dtype=np.float32)))
    assert stride == 2
    assert plane.shape == (1024, 1024, 1)


def test_decimation_sends_values_the_model_produced() -> None:
    """An integer stride rather than a resize: no interpolated number reaches the wire."""
    array = np.arange(2048 * 2048, dtype=np.float32).reshape(2048, 2048)
    plane, stride = decode(encode_plane(array))

    assert plane[0, 0, 0] == array[0, 0]
    assert plane[1, 1, 0] == array[stride, stride]
    assert plane[500, 300, 0] == array[500 * stride, 300 * stride]


def test_every_colour_plane_travels_in_one_response() -> None:
    """Channel count is data: a client that had to know C in advance would encode it."""
    array = np.random.default_rng(3).random((5, 4, 3)).astype(np.float32)
    payload = encode_plane(array)
    _, _, _, channels, _ = struct.unpack("<IIIII", payload[4:HEADER_SIZE])
    plane, _ = decode(payload)

    assert channels == 3
    assert plane.shape == (5, 4, 3)
    np.testing.assert_allclose(plane, array, rtol=0, atol=0)


def test_planes_are_stored_one_after_another_not_interleaved() -> None:
    """Plane-major, so a reader indexes one channel by an offset rather than a stride walk."""
    array = np.stack(
        [np.full((2, 3), 1.0), np.full((2, 3), 2.0), np.full((2, 3), 3.0)], axis=-1
    ).astype(np.float32)
    body = np.frombuffer(encode_plane(array)[HEADER_SIZE:], dtype="<f4")

    assert list(body[:6]) == [1.0] * 6
    assert list(body[6:12]) == [2.0] * 6
    assert list(body[12:]) == [3.0] * 6


def test_a_shape_the_format_cannot_represent_is_a_caller_error() -> None:
    with pytest.raises(PlaneError):
        encode_plane(np.zeros((2, 3, 4, 5), dtype=np.float32))
    with pytest.raises(PlaneError):
        encode_plane(np.zeros(6, dtype=np.float32))


def test_float64_is_narrowed_rather_than_refused() -> None:
    # `write_map` stores float32, but a diagnostic could be anything numeric.
    plane, _ = decode(encode_plane(np.array([[1.5, 2.25]], dtype=np.float64)))
    np.testing.assert_allclose(plane[..., 0], [[1.5, 2.25]])
