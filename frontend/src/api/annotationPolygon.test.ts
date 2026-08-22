import { describe, expect, it } from "vitest";

import { polygonClick, snapTolerance } from "./annotationPolygon";

const square = [
  { x: 0, y: 0 },
  { x: 20, y: 0 },
  { x: 20, y: 20 },
];

describe("polygonClick", () => {
  it("adds an ordinary vertex", () => {
    expect(polygonClick([], { x: 5, y: 5 }, 4)).toBe("add");
    expect(polygonClick(square, { x: 0, y: 20 }, 4)).toBe("add");
  });

  it("closes on the first vertex once the ring is legal", () => {
    expect(polygonClick(square, { x: 1, y: 1 }, 4)).toBe("close");
    // Two points cannot close: three is the smallest ring the document schema accepts, so
    // here the first vertex is just a vertex and clicking it again is a duplicate.
    expect(polygonClick(square.slice(0, 2), { x: 1, y: 1 }, 4)).toBe("add");
  });

  it("drops a second click in the same place", () => {
    expect(polygonClick(square, { x: 20, y: 20 }, 4)).toBe("ignore");
    expect(polygonClick(square, { x: 22, y: 21 }, 4)).toBe("ignore");
    expect(polygonClick(square, { x: 30, y: 20 }, 4)).toBe("add");
  });

  it("walks a double-click through add, ignore, close", () => {
    // The sequence the scene actually sees: one click lands a vertex, the second click of
    // the double-click lands in the same place and is dropped, then `dblclick` closes.
    const points = [...square];
    const click = { x: 0, y: 20 };
    expect(polygonClick(points, click, 4)).toBe("add");
    points.push(click);
    expect(polygonClick(points, { x: 1, y: 20 }, 4)).toBe("ignore");
    expect(points).toHaveLength(4);
  });

  it("prefers closing over ignoring when the last vertex is the first", () => {
    // A one-point polygon: the same vertex is both first and last, and neither reading
    // closes anything, so the click has to be a duplicate rather than a ring.
    expect(polygonClick([{ x: 3, y: 3 }], { x: 3, y: 3 }, 4)).toBe("ignore");
  });
});

describe("snapTolerance", () => {
  it("shrinks in source pixels as the view zooms in", () => {
    // The target has to stay the same size under the hand, which is exactly why it cannot
    // be a constant in the document's own units.
    expect(snapTolerance(1, 9)).toBe(9);
    expect(snapTolerance(3, 9)).toBe(3);
    expect(snapTolerance(0.5, 9)).toBe(18);
  });

  it("survives a scale of zero rather than returning infinity", () => {
    expect(Number.isFinite(snapTolerance(0))).toBe(true);
  });
});
