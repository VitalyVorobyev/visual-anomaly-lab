"""Rendering a map for display.

Everything visual is applied at view time (ADR-0007), so these are the decisions that can
change after an expensive run — and the ones that go wrong silently. A segmentation drawn
at the wrong threshold is still a picture of a region; it just is not a picture of the
region the reader thinks they are looking at.
"""

from __future__ import annotations

import io
from typing import ClassVar

import numpy as np
import pytest
from PIL import Image

from anomaly_lab.media.overlay import (
    CONTOUR_RGB,
    LABEL_FILL_ALPHA,
    LABEL_LINE_ALPHA,
    PREDICTION_RGB,
    boundary_of,
    fit_label_plane,
    parse_label_colours,
    render_anomaly_map,
    render_label_map,
    render_mask_contour,
    render_prediction_region,
)
from anomaly_lab.regions.transform import PixelBounds, SpatialTransform


def decode(payload: bytes) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(payload)).convert("RGBA"))


def a_map_with_a_hot_square() -> np.ndarray:
    """A cold field with one hot 4x4 block at rows 2-5, columns 3-6."""
    values = np.full((16, 16), 0.1, dtype=np.float32)
    values[2:6, 3:7] = 0.9
    return values


class TestPredictionRegion:
    def test_fills_exactly_where_the_map_crosses_the_threshold(self) -> None:
        rgba = decode(render_prediction_region(a_map_with_a_hot_square(), 0.5))

        opaque = rgba[..., 3] > 0
        expected = np.zeros((16, 16), dtype=bool)
        expected[2:6, 3:7] = True
        np.testing.assert_array_equal(opaque, expected)
        assert tuple(rgba[3, 4, :3]) == PREDICTION_RGB

    def test_the_threshold_is_inclusive_so_the_peak_is_never_dropped(self) -> None:
        values = np.array([[0.5, 0.4]], dtype=np.float32)
        rgba = decode(render_prediction_region(values, 0.5))
        assert rgba[0, 0, 3] > 0
        assert rgba[0, 1, 3] == 0

    def test_a_region_is_translucent_so_the_pixels_under_it_stay_judgeable(self) -> None:
        rgba = decode(render_prediction_region(a_map_with_a_hot_square(), 0.5))
        assert 0 < int(rgba[3, 4, 3]) < 255

    def test_an_outline_is_opaque_and_hollow(self) -> None:
        # Two experiments' predictions on one sample would hide each other if both filled.
        rgba = decode(render_prediction_region(a_map_with_a_hot_square(), 0.5, outline=True))
        assert rgba[2, 3, 3] == 255, "the corner of the region is on its boundary"
        assert rgba[8, 8, 3] == 0, "outside the region stays transparent"

    def test_a_map_below_the_threshold_everywhere_draws_nothing(self) -> None:
        # The honest rendering of "the model found nothing here" — not a blank failure.
        rgba = decode(render_prediction_region(a_map_with_a_hot_square(), 5.0))
        assert not (rgba[..., 3] > 0).any()

    def test_it_never_borrows_the_ground_truth_colour(self) -> None:
        # The whole point of drawing both is seeing where they disagree.
        assert PREDICTION_RGB != CONTOUR_RGB

    def test_resampling_to_the_source_grid_keeps_one_flat_colour(self) -> None:
        # NEAREST, not BILINEAR: a fading edge would read as a soft-boundary claim the
        # threshold never made.
        rgba = decode(render_prediction_region(a_map_with_a_hot_square(), 0.5, size=(64, 64)))
        painted = rgba[rgba[..., 3] > 0]
        assert {tuple(pixel[:3]) for pixel in painted} == {PREDICTION_RGB}


class TestBoundary:
    def test_a_solid_block_becomes_a_hollow_ring(self) -> None:
        region = np.zeros((9, 9), dtype=bool)
        region[2:7, 2:7] = True
        edge = boundary_of(region, width=1)

        assert edge[2, 2] and edge[2, 6] and edge[6, 6]
        assert not edge[4, 4], "the interior is eroded away"
        assert not edge[0, 0], "background stays background"

    def test_a_region_touching_the_frame_still_gets_an_outline(self) -> None:
        # A defect running off the edge of the image is common and must not vanish.
        region = np.zeros((8, 8), dtype=bool)
        region[0:3, 0:3] = True
        assert boundary_of(region, width=1).any()


class TestHeatmap:
    def test_alpha_follows_the_score_for_an_overlay(self) -> None:
        rgba = decode(render_anomaly_map(a_map_with_a_hot_square(), value_range=(0.0, 1.0)))
        assert rgba[3, 4, 3] > rgba[10, 10, 3], "the hot block is more opaque than the field"

    def test_a_standalone_panel_is_fully_opaque(self) -> None:
        # Outside an overlay, score-driven alpha dissolves the quiet regions into the page
        # and leaves "found nothing" indistinguishable from "rendered nothing".
        rgba = decode(
            render_anomaly_map(
                a_map_with_a_hot_square(), value_range=(0.0, 1.0), alpha_follows_score=False
            )
        )
        assert (rgba[..., 3] == 255).all()

    def test_the_run_range_is_used_rather_than_this_image_s_own(self) -> None:
        """A clean part must not look as alarming as a defective one.

        A map whose own peak is 0.2 is a *quiet* map when the run reaches 10.0. Stretched
        to its own extremes it saturates the colormap and renders exactly as loudly as the
        genuinely defective image next to it — which destroys the comparison the overlay
        exists for. Only the run's range makes two images of one experiment comparable.
        """
        quiet = np.linspace(0.0, 0.2, 64, dtype=np.float32).reshape(8, 8)
        own_scale = decode(render_anomaly_map(quiet, value_range=None))
        run_scale = decode(render_anomaly_map(quiet, value_range=(0.0, 10.0)))

        assert int(own_scale[7, 7, 3]) == 255, "its own peak saturates the scale"
        assert int(run_scale[7, 7, 3]) < 10, "against the run, the same peak is barely there"

    def test_uncovered_projected_pixels_are_transparent(self) -> None:
        values = np.ones((4, 4), dtype=np.float32)
        values[0, 0] = np.nan

        rgba = decode(render_anomaly_map(values, value_range=(0.0, 1.0), alpha_follows_score=False))

        assert rgba[0, 0, 3] == 0
        assert rgba[1, 1, 3] == 255


class TestMaskContour:
    def test_the_ground_truth_is_outlined_rather_than_filled(self) -> None:
        # Filling it would hide the pixels the reader is judging the map against.
        mask = np.zeros((9, 9), dtype=bool)
        mask[2:7, 2:7] = True
        rgba = decode(render_mask_contour(mask, width=1))

        assert tuple(rgba[2, 2, :3]) == CONTOUR_RGB
        assert rgba[4, 4, 3] == 0, "the interior is left visible"


class TestPreparedFrameMask:
    """The ground truth, drawn in the frame a diagnostic pane is actually in.

    A diagnostic is a prepared-frame quantity — nothing projects it, because a per-branch
    error map means what it means on the grid the branch computed it on — so the pane is
    drawn at the array's own size. The source-frame outline the sample overlay uses is
    therefore off by the pinned crop and letterbox when laid over one of those, which is
    invisible while a profile happens to be the identity and wrong as soon as it is not.

    This is the composition `GET /api/images/{id}/mask?frame=prepared` performs: project the
    mask through the pinned transform, *then* trace it.
    """

    def a_transform(self) -> SpatialTransform:
        """A 64x32 source into a 32x32 prepared frame: a real crop and a real letterbox.

        The numbers are chosen so the resolved geometry is a pure translation — the crop is
        exactly 32x16 and lands at scale 1.0 with eight rows of padding above and below —
        which makes every expectation below readable rather than a rounding argument.
        """
        transform = SpatialTransform.resolve(
            source_size=(64, 32),
            prepared_size=(32, 32),
            region=PixelBounds(left=16, top=4, right=48, bottom=20),
            padding_fraction=0.0,
        )
        assert (transform.scale_x, transform.scale_y) == (1.0, 1.0)
        assert (transform.pad_top, transform.pad_left) == (8, 0)
        return transform

    def a_source_mask(self) -> np.ndarray:
        """A block at source rows 6-13, columns 20-31 — comfortably inside the crop."""
        mask = np.zeros((32, 64), dtype=bool)
        mask[6:14, 20:32] = True
        return mask

    def test_the_png_is_the_prepared_size_rather_than_the_source_one(self) -> None:
        transform = self.a_transform()
        rgba = decode(
            render_mask_contour(
                transform.prepare_mask(self.a_source_mask()),
                size=(transform.prepared_width, transform.prepared_height),
            )
        )

        assert rgba.shape[:2] == (32, 32)

    def test_a_mask_pixel_lands_where_source_to_prepared_says(self) -> None:
        transform = self.a_transform()
        projected = transform.prepare_mask(self.a_source_mask())

        for source_point in [(20.0, 6.0), (31.0, 13.0), (25.0, 9.0)]:
            x, y = transform.source_to_prepared(source_point)
            assert projected[round(y), round(x)], f"{source_point} projected to {(x, y)}"

        # And the letterbox stays empty: padding is not part of the annotated region.
        assert not projected[: transform.pad_top].any()
        assert not projected[transform.pad_top + transform.resized_height :].any()

    def test_the_outline_is_traced_after_the_projection_and_is_still_hollow(self) -> None:
        # Tracing first and shrinking after would resample a two-pixel line down to less
        # than one and drop it out in places.
        transform = self.a_transform()
        rgba = decode(render_mask_contour(transform.prepare_mask(self.a_source_mask())))

        assert tuple(rgba[10, 4, :3]) == CONTOUR_RGB, "the region's corner is on its boundary"
        assert rgba[14, 10, 3] == 0, "the interior is left visible"

    def test_the_two_frames_are_not_the_same_picture(self) -> None:
        """The whole reason the parameter exists.

        Under the identity profile these two agree, which is exactly why the
        misregistration went unnoticed. Under any real crop they do not.
        """
        transform = self.a_transform()
        mask = self.a_source_mask()
        prepared = decode(render_mask_contour(transform.prepare_mask(mask)))
        source = decode(render_mask_contour(mask, size=(64, 32)))

        assert source.shape[:2] == (32, 64)
        assert prepared[10, 4, 3] > 0
        assert source[10, 4, 3] == 0, "nothing is annotated there in the source frame"


@pytest.mark.parametrize("threshold", [0.0, 0.5, 1.0])
def test_every_render_returns_a_png(threshold: float) -> None:
    payload = render_prediction_region(a_map_with_a_hot_square(), threshold)
    assert payload.startswith(b"\x89PNG")


class TestLabelMap:
    """The gallery's raster twin of the sample page's `LabelLayer` (`labelPaint.ts`)."""

    COLOURS: ClassVar[list[tuple[int, int, int]]] = [(59, 201, 219), (240, 136, 62)]

    @staticmethod
    def two_classes() -> np.ndarray:
        """Class 1 in rows 2-7 x columns 2-7, class 2 in rows 10-13 x columns 10-13, one
        pixel of an unknown class (NaN) at (0, 15)."""
        labels = np.zeros((16, 16), dtype=np.float32)
        labels[2:8, 2:8] = 1
        labels[10:14, 10:14] = 2
        labels[0, 15] = np.nan
        return labels

    def test_a_prediction_is_filled_in_its_class_colour_with_a_solid_border(self) -> None:
        pixels = decode(render_label_map(self.two_classes(), self.COLOURS, truth=False))
        assert tuple(pixels[4, 4]) == (*self.COLOURS[0], LABEL_FILL_ALPHA)
        assert tuple(pixels[11, 11]) == (*self.COLOURS[1], LABEL_FILL_ALPHA)
        # Every border pixel is drawn, none skipped.
        assert all(pixels[2, column, 3] == LABEL_LINE_ALPHA for column in range(2, 8))
        assert tuple(pixels[12, 13]) == (*self.COLOURS[1], LABEL_LINE_ALPHA)
        # Background and an unknown class stay clear.
        assert pixels[0, 0, 3] == 0 and pixels[0, 15, 3] == 0 and pixels[9, 9, 3] == 0

    def test_truth_is_a_dashed_border_and_nothing_inside(self) -> None:
        pixels = decode(render_label_map(self.two_classes(), self.COLOURS, truth=True))
        assert pixels[4, 4, 3] == 0 and pixels[11, 11, 3] == 0
        border = [pixels[2, column] for column in range(2, 8)]
        drawn = [pixel for pixel in border if pixel[3] == LABEL_LINE_ALPHA]
        # Broken, not solid: `(x + y) // 4` alternates along the top edge's (2..7, 2).
        assert 0 < len(drawn) < len(border)
        assert all(tuple(pixel[:3]) == self.COLOURS[0] for pixel in drawn)

    def test_colours_parse_from_the_query_form(self) -> None:
        assert parse_label_colours("3bc9db,#F0883E") == self.COLOURS
        with pytest.raises(ValueError, match="six-digit"):
            parse_label_colours("3bc9d")

    def test_a_large_plane_is_strided_to_the_edge_never_resampled(self) -> None:
        labels = np.zeros((1000, 600), dtype=np.float32)
        labels[::2, ::2] = 3
        fitted = fit_label_plane(labels, 256)
        assert max(fitted.shape) <= 256
        assert set(np.unique(fitted)) <= {0.0, 3.0}
        assert fit_label_plane(labels[:100, :100], 256).shape == (100, 100)
