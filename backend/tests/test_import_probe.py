"""The measured half of a scan.

Everything here builds its own pixels with a known answer, because the point of the probe
is to replace a guess with a number, and a probe whose number is untested is a worse guess.
Torch-free by construction: this runs during an ordinary scan, in the API process, and in
the CI job that measures the torch-free boundary.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from anomaly_lab.datasets.manifest import (
    Manifest,
    ManifestImage,
    ManifestSample,
    WarningCode,
)
from anomaly_lab.datasets.probe import (
    MAX_COLOUR_SAMPLES,
    ProbeImage,
    measure,
    probe_alignment,
    probe_colour,
)


def _texture(size: tuple[int, int], seed: int) -> np.ndarray:
    """A smooth, non-repeating field — something phase correlation can actually lock onto."""
    rng = np.random.default_rng(seed)
    height, width = size
    coarse = rng.random((height // 8 + 2, width // 8 + 2))
    image = Image.fromarray((coarse * 255).astype(np.uint8)).resize(
        (width, height), Image.Resampling.BICUBIC
    )
    return np.asarray(image, dtype=np.uint8)


def _write(path: Path, array: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)
    return path


def _mono_as_rgb(path: Path, seed: int, *, gain: float = 1.0, offset: int = 0) -> Path:
    """One grey plane written into three colour planes, optionally white-balanced."""
    plane = _texture((64, 64), seed)
    scaled = np.clip(plane.astype(np.float64) * gain + offset, 0, 255).astype(np.uint8)
    return _write(path, np.dstack([plane, scaled, scaled]))


# ------------------------------------------------------------------------ colour


def test_a_grey_image_stored_in_three_planes_is_reported_as_redundant(tmp_path: Path) -> None:
    paths = [_mono_as_rgb(tmp_path / f"{i}.png", seed=i) for i in range(6)]

    probe = probe_colour(paths)

    assert probe is not None
    assert probe.colour_images == 6
    assert probe.identical_planes == 6
    assert probe.plane_r2 == 1.0
    assert probe.planes_are_redundant


def test_a_white_balance_gain_still_reads_as_redundant(tmp_path: Path) -> None:
    """The reason the verdict is a fit and not an equality test.

    A monochrome sensor whose planes differ by a gain and an offset stores one signal in
    three planes. `R == G == B` answers "no, this is colour" and sends the operator to the
    wrong conclusion; a regression answers "one plane predicts the others exactly".
    """
    paths = [_mono_as_rgb(tmp_path / f"{i}.png", seed=i, gain=0.8, offset=20) for i in range(6)]

    probe = probe_colour(paths)

    assert probe is not None
    assert probe.identical_planes == 0
    assert probe.plane_r2 is not None and probe.plane_r2 > 0.999
    assert probe.planes_are_redundant


def test_genuinely_different_planes_are_not_called_redundant(tmp_path: Path) -> None:
    paths = [
        _write(
            tmp_path / f"{i}.png",
            np.dstack(
                [_texture((64, 64), i), _texture((64, 64), i + 50), _texture((64, 64), i + 99)]
            ),
        )
        for i in range(6)
    ]

    probe = probe_colour(paths)

    assert probe is not None
    assert probe.plane_r2 is not None and probe.plane_r2 < 0.5
    assert not probe.planes_are_redundant


def test_one_flat_frame_does_not_decide_the_dataset(tmp_path: Path) -> None:
    """R^2 is a ratio against a plane's own variance, so a near-uniform frame scores badly
    while being perfectly redundant. The verdict therefore reads the median, and the worst
    single value stays visible as a diagnostic rather than as the answer."""
    paths = [_mono_as_rgb(tmp_path / f"{i}.png", seed=i) for i in range(9)]
    flat = np.full((64, 64), 7, dtype=np.uint8)
    paths.append(_write(tmp_path / "flat.png", np.dstack([flat, flat, flat + 1])))

    probe = probe_colour(paths)

    assert probe is not None
    assert probe.planes_are_redundant
    assert probe.plane_r2 is not None
    assert probe.plane_r2_min is not None
    assert probe.plane_r2_min <= probe.plane_r2


def test_the_colour_cap_is_respected_and_reported(tmp_path: Path) -> None:
    paths = [_mono_as_rgb(tmp_path / f"{i}.png", seed=i) for i in range(MAX_COLOUR_SAMPLES + 17)]

    probe = probe_colour(paths)

    assert probe is not None
    assert probe.images_read == MAX_COLOUR_SAMPLES
    assert probe.images_available == MAX_COLOUR_SAMPLES + 17


def test_mixed_modes_are_counted_rather_than_normalized_away(tmp_path: Path) -> None:
    paths = [_mono_as_rgb(tmp_path / f"c{i}.png", seed=i) for i in range(4)]
    paths.append(_write(tmp_path / "g.png", _texture((64, 64), 77)))

    probe = probe_colour(paths)

    assert probe is not None
    assert probe.modes == {"RGB": 4, "L": 1}


def test_an_unreadable_file_is_skipped_rather_than_fatal(tmp_path: Path) -> None:
    """The adapter already warns about unreadable files; the probe must not turn one into
    an exception that loses the whole measurement."""
    good = [_mono_as_rgb(tmp_path / f"{i}.png", seed=i) for i in range(3)]
    broken = tmp_path / "broken.png"
    broken.write_bytes(b"not a png")

    probe = probe_colour([*good, broken])

    assert probe is not None
    assert probe.images_read == 3
    assert probe.images_available == 4


# --------------------------------------------------------------------- alignment


def _shifted_pair(tmp_path: Path, index: int, dx: int, dy: int) -> list[ProbeImage]:
    base = _texture((256, 256), index)
    moved = np.roll(np.roll(base, dy, axis=0), dx, axis=1)
    return [
        ProbeImage(path=_write(tmp_path / f"{index}-a.png", base), channel="a"),
        ProbeImage(path=_write(tmp_path / f"{index}-b.png", moved), channel="b"),
    ]


def test_registered_channels_measure_as_zero_shift(tmp_path: Path) -> None:
    samples = [_shifted_pair(tmp_path, i, 0, 0) for i in range(4)]

    probe = probe_alignment(samples)

    assert probe is not None
    assert len(probe.offsets) == 1
    offset = probe.offsets[0]
    assert (offset.median_dx, offset.median_dy) == (0.0, 0.0)
    assert offset.confident
    assert probe.max_median_shift == 0.0


def test_a_known_translation_is_recovered(tmp_path: Path) -> None:
    """The claim the whole feature rests on: 'almost registered' becomes a number."""
    samples = [_shifted_pair(tmp_path, i, 12, -5) for i in range(4)]

    probe = probe_alignment(samples)

    assert probe is not None
    offset = probe.offsets[0]
    assert offset.median_dx == 12.0
    assert offset.median_dy == -5.0
    assert offset.confident


def test_channels_of_different_sizes_are_counted_not_resized(tmp_path: Path) -> None:
    """Resizing to compare would invent an answer for a sample that cannot have one."""
    samples = [
        [
            ProbeImage(path=_write(tmp_path / f"{i}-a.png", _texture((128, 128), i)), channel="a"),
            ProbeImage(path=_write(tmp_path / f"{i}-b.png", _texture((96, 160), i)), channel="b"),
        ]
        for i in range(3)
    ]

    probe = probe_alignment(samples)

    assert probe is not None
    assert probe.mismatched_dimensions == 3
    assert probe.offsets == []


def test_single_image_samples_produce_no_alignment_probe(tmp_path: Path) -> None:
    """A dataset with one view per sample has no registration question to answer."""
    samples = [
        [ProbeImage(path=_write(tmp_path / f"{i}.png", _texture((64, 64), i)), channel=None)]
        for i in range(5)
    ]

    assert probe_alignment(samples) is None


# ----------------------------------------------------------------- manifest level


def _manifest(samples: list[ManifestSample]) -> Manifest:
    return Manifest(adapter="test", dataset_name="d", root_path="/tmp", samples=samples)


def test_measure_warns_about_redundant_planes_and_offset_channels(tmp_path: Path) -> None:
    samples = []
    for index in range(4):
        base = _texture((256, 256), index)
        moved = np.roll(base, 9, axis=1)
        first = _write(tmp_path / f"{index}-a.png", np.dstack([base, base, base]))
        second = _write(tmp_path / f"{index}-b.png", np.dstack([moved, moved, moved]))
        samples.append(
            ManifestSample(
                group_key="g",
                external_id=str(index),
                images=[
                    ManifestImage(
                        path=str(first),
                        channel="a",
                        sha256="x",
                        width=256,
                        height=256,
                        bit_depth=24,
                        file_size=1,
                    ),
                    ManifestImage(
                        path=str(second),
                        channel="b",
                        sha256="y",
                        width=256,
                        height=256,
                        bit_depth=24,
                        file_size=1,
                    ),
                ],
            )
        )

    probe, warnings = measure(_manifest(samples))

    codes = {warning.code for warning in warnings}
    assert WarningCode.REDUNDANT_COLOUR_PLANES in codes
    assert WarningCode.CHANNEL_OFFSET in codes
    assert probe.alignment is not None
    assert probe.alignment.max_median_shift == 9.0

    offset_message = next(w.message for w in warnings if w.code is WarningCode.CHANNEL_OFFSET)
    assert "of 4" in offset_message  # the cap and the population are both stated


def test_measure_stays_quiet_when_there_is_nothing_to_act_on(tmp_path: Path) -> None:
    """Aligned channels carrying real colour are the ordinary case, not a finding."""
    samples = []
    for index in range(4):
        pixels = np.dstack(
            [
                _texture((128, 128), index),
                _texture((128, 128), index + 40),
                _texture((128, 128), index + 80),
            ]
        )
        first = _write(tmp_path / f"{index}-a.png", pixels)
        second = _write(tmp_path / f"{index}-b.png", pixels)
        samples.append(
            ManifestSample(
                group_key="g",
                external_id=str(index),
                images=[
                    ManifestImage(
                        path=str(first),
                        channel="a",
                        sha256="x",
                        width=128,
                        height=128,
                        bit_depth=24,
                        file_size=1,
                    ),
                    ManifestImage(
                        path=str(second),
                        channel="b",
                        sha256="y",
                        width=128,
                        height=128,
                        bit_depth=24,
                        file_size=1,
                    ),
                ],
            )
        )

    _, warnings = measure(_manifest(samples))

    assert warnings == []


def test_measure_survives_a_manifest_with_no_images() -> None:
    probe, warnings = measure(_manifest([]))

    assert probe.colour is None
    assert probe.alignment is None
    assert warnings == []
