"""What a scan measures about the pixels, as opposed to what the adapter reads from paths.

An adapter answers "what is this file", from its name and its position in a tree. Two
further questions are decided by the pixels, and both change what a user should do next:

  * **Is this dataset really in colour?** A monochrome camera writing three identical (or
    near-identical) planes is common, and the answer determines whether
    `PreprocessingOptions.color` is a free saving or a lossy choice.
  * **Are a sample's channels registered?** Grouped multi-view samples only support a
    shared annotation, or a per-channel comparison read as if the frames line up, when the
    frames actually do line up. On a moving line they may not.

Neither is discoverable by looking at the directory tree, and both are cheap enough to
answer on a bounded sample during a scan that already reads every byte to hash.

**The colour answer is a number, not a boolean.** The obvious test — are the planes
byte-identical — is the wrong one: it answers "no" for a sensor with per-plane read noise
or a white-balance gain, while the useful answer is "one plane predicts the others to
within a fraction of a grey level". Reporting the boolean would be actively misleading, so
this reports the coefficient of determination of a least-squares fit between planes, which
degrades gracefully from "identical" through "a gain and an offset" to "genuinely
different".

**Everything is capped and every cap is reported.** A probe that silently looked at the
first 32 files of an acquisition-ordered corpus would describe one production batch and
call it the dataset, so `evenly_spaced` picks across the whole range and the message says
how many of how many were read.

Deliberately numpy-and-Pillow only. This runs in the API process during an ordinary scan,
so it may not import torch, and it is exercised by the torch-free CI job.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from pydantic import BaseModel, Field

from anomaly_lab.datasets.manifest import Manifest, ManifestWarning, WarningCode
from anomaly_lab.models.base import evenly_spaced
from anomaly_lab.schemas import API_MODEL_CONFIG

MAX_COLOUR_SAMPLES = 32
"""Files decoded for the colour question. Plane redundancy is a property of the camera, not
of the part in front of it, so a few dozen frames settle it and a thousand would not settle
it better."""

MAX_ALIGNMENT_SAMPLES = 24
"""Samples decoded for the registration question. Each costs one FFT per extra channel."""

ALIGNMENT_LONG_EDGE = 512
"""Frames are downsampled to this long edge before correlating. Registration answers wanted
here are "0 or 12 pixels", not "0.3 pixels", and the cost is quadratic in the linear size."""

REDUNDANT_PLANES_R2 = 0.99
"""Above this, one plane predicts the others well enough that calling the dataset colour is
misleading. Chosen as a reporting threshold only — the measured value is always carried."""

SHARP_PEAK = 8.0
"""Peak-to-sidelobe ratio below which a correlation is not read as a translation at all.

A weak peak means "these two frames do not differ by a shift", which is *not* the same as
"they are aligned" — and the difference matters when the answer is about to justify a
shared annotation. Expressed in standard deviations above the correlation surface's own
mean, so it does not depend on how strongly two illuminations happen to correlate."""


class ColourProbe(BaseModel):
    """How much of this dataset's colour is information."""

    model_config = API_MODEL_CONFIG

    images_read: int
    images_available: int
    colour_images: int = Field(description="Of those read, how many decoded with 3+ planes.")
    identical_planes: int = Field(description="Of the colour images, how many had R==G==B.")
    plane_r2: float | None = Field(
        default=None,
        description=(
            "Median coefficient of determination of one colour plane regressed on another. "
            "1.0 means a plane is an exact affine function of another, so the colour "
            "carries no independent signal. None when nothing colour was read."
        ),
    )
    plane_r2_min: float | None = Field(
        default=None,
        description=(
            "The worst single image's value. Diagnostic only, and deliberately not what "
            "the verdict is taken from: R^2 is a ratio against the plane's own variance, "
            "so one nearly-uniform frame scores badly while being perfectly redundant."
        ),
    )
    modes: dict[str, int] = Field(default_factory=dict)
    """Pillow modes seen, with counts. More than one is worth knowing: a dataset that mixes
    `L` and `RGB` files is normalized by `load_array`, but not by a reader that assumes."""

    @property
    def planes_are_redundant(self) -> bool:
        return self.plane_r2 is not None and self.plane_r2 >= REDUNDANT_PLANES_R2


class ChannelOffset(BaseModel):
    """How far one channel sits from the sample's first channel, in source pixels."""

    model_config = API_MODEL_CONFIG

    channel: str
    reference: str
    samples_read: int
    median_dx: float
    median_dy: float
    max_shift: float
    median_peak: float = Field(
        description=(
            "Peak-to-sidelobe ratio of the correlation, in standard deviations above the "
            "surface mean. A diffuse peak means the two frames do not differ by a "
            "translation at all, which is not the same as aligned."
        ),
    )

    @property
    def confident(self) -> bool:
        return self.median_peak >= SHARP_PEAK


class AlignmentProbe(BaseModel):
    """Whether a grouped sample's channels share a coordinate frame."""

    model_config = API_MODEL_CONFIG

    samples_read: int
    samples_available: int
    offsets: list[ChannelOffset] = Field(default_factory=list)
    mismatched_dimensions: int = Field(
        default=0,
        description=(
            "Samples whose channels differ in width or height. These cannot be compared "
            "pixel-for-pixel at all, so they are counted and skipped rather than resized."
        ),
    )

    @property
    def max_median_shift(self) -> float:
        """The largest confident median offset, or 0.0 when nothing was measurable."""
        confident = [
            max(abs(offset.median_dx), abs(offset.median_dy))
            for offset in self.offsets
            if offset.confident
        ]
        return max(confident) if confident else 0.0


class ScanProbe(BaseModel):
    """The measured half of a scan."""

    model_config = API_MODEL_CONFIG

    colour: ColourProbe | None = None
    alignment: AlignmentProbe | None = None


@dataclass(frozen=True)
class ProbeImage:
    """One file the probe was asked to look at, with the channel it belongs to."""

    path: Path
    channel: str | None


def _plane_r2(array: np.ndarray) -> float:
    """Worst R^2 of one colour plane regressed on the first, over an (H, W, C) uint8 array.

    Least squares rather than a difference, because a monochrome sensor's planes routinely
    differ by a gain and an offset — a white balance — while carrying one signal. A raw
    difference calls that "colour"; a fit calls it what it is.
    """
    planes = array[:, :, :3].astype(np.float64).reshape(-1, 3)
    reference = planes[:, 0]
    design = np.vstack([reference, np.ones_like(reference)]).T
    worst = 1.0
    for index in (1, 2):
        target = planes[:, index]
        variance = target.var()
        if variance == 0.0:
            # A constant plane is perfectly predicted by anything.
            continue
        coefficients, *_ = np.linalg.lstsq(design, target, rcond=None)
        residual = target - design @ coefficients
        worst = min(worst, float(1.0 - residual.var() / variance))
    return worst


def probe_colour(paths: list[Path], *, limit: int = MAX_COLOUR_SAMPLES) -> ColourProbe | None:
    """Decode a bounded, evenly spaced sample and describe its colour content."""
    if not paths:
        return None

    chosen = evenly_spaced(len(paths), limit)
    modes: dict[str, int] = {}
    colour_images = identical = 0
    scores: list[float] = []
    read = 0

    for index in chosen:
        try:
            with Image.open(paths[index]) as handle:
                mode = handle.mode
                array = np.asarray(handle)
        except (OSError, ValueError):
            # Unreadable files are already the adapter's warning to raise; the probe
            # describes what it could read and never turns a skipped file into an error.
            continue
        read += 1
        modes[mode] = modes.get(mode, 0) + 1
        if array.ndim == 3 and array.shape[2] >= 3:
            colour_images += 1
            first = array[:, :, 0]
            if np.array_equal(first, array[:, :, 1]) and np.array_equal(first, array[:, :, 2]):
                identical += 1
            scores.append(_plane_r2(array))

    if read == 0:
        return None
    return ColourProbe(
        images_read=read,
        images_available=len(paths),
        colour_images=colour_images,
        identical_planes=identical,
        plane_r2=float(np.median(scores)) if scores else None,
        plane_r2_min=min(scores) if scores else None,
        modes=modes,
    )


def _grayscale_for_correlation(path: Path) -> np.ndarray | None:
    """One frame as a mean-removed float array, downsampled to a bounded long edge."""
    try:
        with Image.open(path) as handle:
            image = handle.convert("L")
            long_edge = max(image.size)
            if long_edge > ALIGNMENT_LONG_EDGE:
                scale = ALIGNMENT_LONG_EDGE / long_edge
                size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
                image = image.resize(size, Image.Resampling.BILINEAR)
            array = np.asarray(image, dtype=np.float64)
    except (OSError, ValueError):
        return None
    return array - array.mean()


def _phase_correlate(moving: np.ndarray, reference: np.ndarray) -> tuple[float, float, float]:
    """`(dx, dy, peak)` of the translation taking `reference` onto `moving`.

    Phase correlation rather than plain cross-correlation because the illuminations differ
    in brightness by design — that is the whole point of having them — and normalizing the
    magnitude away is what makes a bright-field and a dark-field frame comparable at all.
    """
    spectrum = np.fft.rfft2(moving) * np.conj(np.fft.rfft2(reference))
    spectrum /= np.maximum(np.abs(spectrum), 1e-9)
    surface = np.fft.irfft2(spectrum, s=moving.shape)
    peak_index = np.unravel_index(int(np.argmax(surface)), surface.shape)
    height, width = moving.shape
    dy = peak_index[0] - height if peak_index[0] > height // 2 else peak_index[0]
    dx = peak_index[1] - width if peak_index[1] > width // 2 else peak_index[1]

    # Peak-to-sidelobe ratio: how far the winning peak stands above the rest of the
    # surface, in standard deviations. The raw peak height is not usable on its own — a
    # bright-field and a dark-field frame of the same part correlate weakly by
    # construction, so an absolute threshold would reject a correct answer for looking
    # unimpressive, while a flat surface with no translation at all still has a maximum
    # somewhere.
    deviation = float(surface.std())
    peak = (float(surface[peak_index]) - float(surface.mean())) / deviation if deviation else 0.0
    return float(dx), float(dy), peak


def probe_alignment(
    samples: list[list[ProbeImage]],
    *,
    limit: int = MAX_ALIGNMENT_SAMPLES,
) -> AlignmentProbe | None:
    """Measure how far each channel sits from its sample's first channel.

    Answers in *source* pixels, upscaled back from the correlation resolution, because
    that is the frame an annotation lives in and therefore the frame the answer is needed
    in.
    """
    grouped = [images for images in samples if len(images) > 1]
    if not grouped:
        return None

    chosen = evenly_spaced(len(grouped), limit)
    measurements: dict[tuple[str, str], list[tuple[float, float, float]]] = {}
    mismatched = read = 0

    for index in chosen:
        images = grouped[index]
        reference_image = images[0]
        reference = _grayscale_for_correlation(reference_image.path)
        if reference is None:
            continue
        read += 1
        with Image.open(reference_image.path) as handle:
            source_long_edge = max(handle.size)
        scale = source_long_edge / max(reference.shape)

        for image in images[1:]:
            moving = _grayscale_for_correlation(image.path)
            if moving is None:
                continue
            if moving.shape != reference.shape:
                mismatched += 1
                continue
            dx, dy, peak = _phase_correlate(moving, reference)
            key = (image.channel or "unassigned", reference_image.channel or "unassigned")
            measurements.setdefault(key, []).append((dx * scale, dy * scale, peak))

    if read == 0:
        return None

    offsets = [
        ChannelOffset(
            channel=channel,
            reference=reference_name,
            samples_read=len(values),
            median_dx=float(np.median([dx for dx, _, _ in values])),
            median_dy=float(np.median([dy for _, dy, _ in values])),
            max_shift=float(max(max(abs(dx), abs(dy)) for dx, dy, _ in values)),
            median_peak=float(np.median([peak for _, _, peak in values])),
        )
        for (channel, reference_name), values in sorted(measurements.items())
    ]
    return AlignmentProbe(
        samples_read=read,
        samples_available=len(grouped),
        offsets=offsets,
        mismatched_dimensions=mismatched,
    )


def measure(manifest: Manifest) -> tuple[ScanProbe, list[ManifestWarning]]:
    """Probe a proposed manifest and describe anything worth a second look.

    Takes the manifest rather than the tree, so it is adapter-agnostic by construction:
    every adapter proposes samples and images, and none of them has to know this exists.

    A warning is emitted only when there is something to act on. "The planes are
    genuinely different" and "the channels are aligned" are the ordinary cases, recorded
    in the probe for the review screen and not shouted about.
    """
    paths = [Path(image.path) for sample in manifest.samples for image in sample.images]
    samples = [
        [ProbeImage(path=Path(image.path), channel=image.channel) for image in sample.images]
        for sample in manifest.samples
    ]

    colour = probe_colour(paths)
    alignment = probe_alignment(samples)
    warnings: list[ManifestWarning] = []

    if colour is not None and colour.planes_are_redundant and colour.plane_r2 is not None:
        exact = (
            f"{colour.identical_planes} of them byte-identical; " if colour.identical_planes else ""
        )
        warnings.append(
            ManifestWarning(
                code=WarningCode.REDUNDANT_COLOUR_PLANES,
                message=(
                    f"{colour.colour_images} of {colour.images_read} images sampled from "
                    f"{colour.images_available} decode as colour, but one plane predicts "
                    f"another with a median R^2 of {colour.plane_r2:.4f} ({exact}"
                    "so the colour carries almost no independent signal). The grayscale "
                    "model-input option would cost little and decode faster."
                ),
            )
        )
    if colour is not None and len(colour.modes) > 1:
        listed = ", ".join(f"{mode}x{count}" for mode, count in sorted(colour.modes.items()))
        warnings.append(
            ManifestWarning(
                code=WarningCode.MIXED_IMAGE_MODES,
                message=(
                    f"the {colour.images_read} images sampled from {colour.images_available} "
                    f"do not all share one colour mode ({listed}). Model input normalizes "
                    "this, but anything reading the files directly should not assume."
                ),
            )
        )
    if alignment is not None:
        shifted = [
            offset
            for offset in alignment.offsets
            if offset.confident and max(abs(offset.median_dx), abs(offset.median_dy)) >= 1.0
        ]
        if shifted:
            listed = "; ".join(
                f"{offset.channel} sits {offset.median_dx:+.1f},{offset.median_dy:+.1f} px "
                f"from {offset.reference}"
                for offset in shifted
            )
            warnings.append(
                ManifestWarning(
                    code=WarningCode.CHANNEL_OFFSET,
                    message=(
                        f"across {alignment.samples_read} of {alignment.samples_available} "
                        f"grouped samples, channels are not registered: {listed}. A shared "
                        "annotation and any pixel metric read across channels assume they "
                        "are."
                    ),
                )
            )
        if alignment.mismatched_dimensions:
            warnings.append(
                ManifestWarning(
                    code=WarningCode.CHANNEL_DIMENSIONS_DIFFER,
                    message=(
                        f"{alignment.mismatched_dimensions} channel pair(s) in the sampled "
                        f"{alignment.samples_read} samples differ in width or height, so "
                        "they were not compared. Such a sample cannot carry one annotation "
                        "across its channels."
                    ),
                )
            )

    return ScanProbe(colour=colour, alignment=alignment), warnings
