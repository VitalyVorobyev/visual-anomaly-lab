"""A polygon owns the pixels whose centres it contains, by the even-odd rule.

The rule is a box's (`BoxShape.owned_pixels`), so a box and the polygon of its four corners
own the same pixels, and it is the editor's pixel readout (`pixelReadout.insidePolygon`),
which reads the same fixture file as this module. Every fixture is a synthetic document.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image, ImageDraw

from anomaly_lab.annotation_interchange import render_shapes
from anomaly_lab.annotation_render import polygon_coverage, rasterize, render_truth
from anomaly_lab.annotations.class_truth import (
    BoxTruth,
    ClassTruth,
    LabelTruth,
    load_class_mask,
)
from anomaly_lab.domain.annotations import AnnotationDocument, BoxShape, PolygonShape
from anomaly_lab.domain.entities import ClassPresence
from anomaly_lab.media.decode import sha256_of

FIXTURE = (
    Path(__file__).resolve().parents[2]
    / "frontend/src/components/annotation/polygonCentres.fixture.json"
)


def _owned(points: list[tuple[float, float]], size: tuple[int, int]) -> np.ndarray:
    width, height = size
    found = np.zeros((height, width), dtype=bool)
    coverage = polygon_coverage(points, size)
    if coverage is not None:
        rows, columns = coverage.mask.shape
        found[coverage.y : coverage.y + rows, coverage.x : coverage.x + columns] = coverage.mask
    return found


def _polygon(points: list[tuple[float, float]], *, op: str = "add") -> dict[str, Any]:
    return {
        "id": f"p{len(points)}-{op}",
        "label_key": "defect",
        "kind": "polygon",
        "operation": op,
        "points": [{"x": x, "y": y} for x, y in points],
    }


def _document(*shapes: dict[str, Any], size: tuple[int, int] = (16, 12)) -> AnnotationDocument:
    return AnnotationDocument.model_validate(
        {"image_width": size[0], "image_height": size[1], "shapes": list(shapes)}
    )


def _classes(document: AnnotationDocument) -> np.ndarray:
    return rasterize(document, ["defect"], source_mask_path=None, source_mask_sha256=None)


def _centres_inside(points: list[tuple[float, float]], size: tuple[int, int]) -> np.ndarray:
    """The readout's even-odd ray cast, one centre at a time — the reference."""
    width, height = size
    found = np.zeros((height, width), dtype=bool)
    for row in range(height):
        for column in range(width):
            x, y = column + 0.5, row + 0.5
            inside = False
            for index, (ax, ay) in enumerate(points):
                bx, by = points[index - 1]
                if (ay > y) != (by > y) and x < ((bx - ax) * (y - ay)) / (by - ay) + ax:
                    inside = not inside
            found[row, column] = inside
    return found


@pytest.mark.parametrize(
    "case", json.loads(FIXTURE.read_text())["polygons"], ids=lambda c: c["name"]
)
def test_the_shared_fixture_owns_what_the_readout_reads(case: dict[str, Any]) -> None:
    fixture = json.loads(FIXTURE.read_text())
    size = (fixture["width"], fixture["height"])
    points = [(float(x), float(y)) for x, y in case["points"]]
    expected = np.array([[pixel == "#" for pixel in row] for row in case["owned"]])
    assert np.array_equal(_owned(points, size), expected)


@pytest.mark.parametrize(
    "rect",
    [(1, 2, 3, 4), (0, 0, 16, 12), (1.5, 2.25, 9.5, 6.1), (0.5, 0.5, 3, 2), (3.6, 3, 0.8, 4)],
)
def test_a_box_and_the_polygon_of_its_corners_own_the_same_pixels(
    rect: tuple[float, float, float, float],
) -> None:
    x, y, width, height = rect
    box = {"id": "b", "label_key": "defect", "kind": "box", "x": x, "y": y}
    box |= {"width": width, "height": height}
    corners = [(x, y), (x + width, y), (x + width, y + height), (x, y + height)]
    from_box = _classes(_document(box))
    from_polygon = _classes(_document(_polygon(corners)))
    assert np.array_equal(from_box, from_polygon)
    x0, y0, x1, y1 = BoxShape.model_validate(box).owned_pixels()
    assert int((from_polygon > 0).sum()) == (x1 - x0) * (y1 - y0)


def test_a_triangle_owns_exactly_the_centres_inside_it() -> None:
    # The hypotenuse runs from (9, 1) to (1, 7): at a row centre y it is at x = 9 - (y - 1) * 4 / 3.
    # Row 1 (y = 1.5) reaches x = 8.33, so columns 1..7; row 3 (y = 3.5) reaches 5.67, so 1..5;
    # row 6 (y = 6.5) reaches 1.67, so column 1 alone. The top and left edges own their pixels.
    owned = _classes(_document(_polygon([(1, 1), (9, 1), (1, 7)]))) > 0
    expected = np.zeros((12, 16), dtype=bool)
    for row, last in {1: 7, 2: 6, 3: 5, 4: 3, 5: 2, 6: 1}.items():
        expected[row, 1 : last + 1] = True
    assert np.array_equal(owned, expected)


def test_holes_and_self_intersections_follow_the_even_odd_rule() -> None:
    size = (16, 12)
    # A frame and its hole as one ring, joined by a bridge that is traced twice and cancels.
    ring = [(1, 1), (15, 1), (15, 11), (1, 11), (1, 1), (5, 4), (5, 8), (11, 8), (11, 4), (5, 4)]
    owned = _owned([(float(x), float(y)) for x, y in ring], size)
    expected = np.zeros((12, 16), dtype=bool)
    expected[1:11, 1:15] = True
    expected[4:8, 5:11] = False
    assert np.array_equal(owned, expected)
    # A pentagram's inner pentagon winds twice: even-odd leaves it out, nonzero would fill it.
    star = [(8.0, 0.0), (12.7, 11.0), (1.0, 4.2), (15.0, 4.2), (3.3, 11.0)]
    stars = _owned(star, size)
    assert not stars[4:6, 6:10].any()
    assert stars[2, 8] and stars[4, 3] and stars[4, 12]
    assert np.array_equal(stars, _centres_inside(star, size))
    # And a concave notch is left out.
    notch = [(1.0, 1.0), (14.0, 1.0), (14.0, 10.0), (8.0, 4.0), (1.0, 10.0)]
    assert np.array_equal(_owned(notch, size), _centres_inside(notch, size))
    assert not _owned(notch, size)[8, 8]


def test_a_polygon_with_no_centre_inside_owns_nothing(tmp_path: Path) -> None:
    for points in (
        [(2.1, 2.1), (2.4, 2.1), (2.4, 2.4)],  # sub-pixel, between centres
        [(1.0, 1.0), (5.0, 5.0), (9.0, 9.0)],  # collinear: no area
        [(3.0, 3.0), (3.0, 3.0), (3.0, 3.0)],  # one point
    ):
        assert polygon_coverage(points, (16, 12)) is None
    rendered = render_truth(
        _document(_polygon([(2.1, 2.1), (2.4, 2.1), (2.4, 2.4)])),
        tmp_path / "revision-1.png",
        tmp_path / "revision-1.classes.png",
        classes=["defect"],
        source_mask_path=None,
        source_mask_sha256=None,
        instances_path=tmp_path / "revision-1.instances.json",
    )
    assert rendered.instances == ()
    assert rendered.class_table[0].pixels == 0


def test_a_polygon_past_the_frame_is_clipped() -> None:
    # Documents keep their points in the frame; the rasteriser clips whatever it is given.
    points = [(-3.0, -3.0), (5.0, -3.0), (5.0, 4.0), (-3.0, 4.0)]
    expected = np.zeros((12, 16), dtype=bool)
    expected[0:4, 0:5] = True
    assert np.array_equal(_owned(points, (16, 12)), expected)
    assert not _owned([(20.0, 1.0), (30.0, 1.0), (25.0, 9.0)], (16, 12)).any()


def test_a_polygon_cut_from_a_box_leaves_nothing_on_its_edges() -> None:
    box = {"id": "b", "label_key": "defect", "kind": "box", "x": 2, "y": 2, "width": 6}
    box["height"] = 5
    cut = _polygon([(2, 2), (8, 2), (8, 7), (2, 7)], op="subtract")
    assert not _classes(_document(box, cut)).any()


def test_interchange_renders_polygons_by_the_same_rule() -> None:
    shape = PolygonShape.model_validate(_polygon([(1, 1), (9, 1), (1, 7)]))
    rendered = render_shapes([shape], size=(16, 12))
    assert np.array_equal(rendered, _classes(_document(_polygon([(1, 1), (9, 1), (1, 7)]))) > 0)


def test_a_document_read_in_memory_names_its_polygon_rule() -> None:
    polygon = json.dumps({"shapes": [_polygon([(1, 1), (4, 1), (4, 4)])]})
    boxes = json.dumps({"shapes": [{"kind": "box"}]})
    for document, tagged in ((polygon, True), (boxes, False)):
        truth = ClassTruth(
            1, "defect", ClassPresence.PRESENT, "document", (4, 4), document=document
        )
        assert truth.identity.endswith(":polygon-centres") is tagged
    assert LabelTruth(1, "document", (4, 4), document=polygon).identity.endswith(":polygon-centres")
    assert BoxTruth(1, "document", (4, 4), document=polygon).identity.endswith(":polygon-centres")
    # A stored file is read as written: its identity is its digest, whatever drew it.
    stored = LabelTruth(1, "class_mask", (4, 4), path="x.png", sha256="abc")
    assert stored.identity == "class_mask:0:abc"


def test_a_revision_keeps_the_polygon_pixels_it_wrote(tmp_path: Path) -> None:
    # A class mask written while polygons were filled outline-inclusive holds their far-edge
    # pixels too. Completed revisions are immutable: it is read as written, against its own
    # digest, and only a new completion, or a document read in memory, uses the centre rule.
    corners = [(1, 1), (4, 1), (4, 5), (1, 5)]
    earlier = Image.new("L", (16, 12), 0)
    ImageDraw.Draw(earlier).polygon(corners, fill=1)
    path = tmp_path / "revision-1.classes.png"
    earlier.save(path)
    truth = ClassTruth(
        1,
        "defect",
        ClassPresence.PRESENT,
        "class_mask",
        (16, 12),
        path=str(path),
        sha256=sha256_of(path),
        index=1,
    )
    assert int(load_class_mask(truth).sum()) == 4 * 5
    assert (
        int((_classes(_document(_polygon([(float(x), float(y)) for x, y in corners]))) > 0).sum())
        == 3 * 4
    )
