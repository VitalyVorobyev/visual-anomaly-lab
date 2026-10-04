import { describe, expect, it } from "vitest";

import type { AnnotationDocument, BoxShape, PolygonShape } from "./client";
import {
  createHistory,
  historyReducer,
  replaceShape,
  shapeOutline,
  translateShape,
  withBoxRect,
  withPolygonPoints,
  withShape,
  withoutShape,
} from "./annotationState";

const empty: AnnotationDocument = {
  schema_version: 1,
  image_width: 20,
  image_height: 10,
  base: "empty",
  shapes: [],
};

const polygon: PolygonShape = {
  id: "p1",
  label_key: "defect",
  kind: "polygon",
  operation: "add",
  points: [
    { x: 1, y: 1 },
    { x: 5, y: 1 },
    { x: 3, y: 5 },
  ],
};

describe("annotation history", () => {
  it("undoes and redoes controlled document changes", () => {
    const initial = createHistory(empty);
    const changed = historyReducer(initial, {
      type: "commit",
      document: withShape(empty, polygon),
    });
    expect(changed.present.shapes).toHaveLength(1);

    const undone = historyReducer(changed, { type: "undo" });
    expect(undone.present.shapes).toHaveLength(0);
    expect(historyReducer(undone, { type: "redo" }).present.shapes).toHaveLength(1);
  });

  it("reshapes a polygon's ring and removes shapes without mutation", () => {
    const document = withShape(empty, polygon);
    const ring = [polygon.points[0]!, { x: 8, y: 2 }, polygon.points[2]!];
    const moved = withPolygonPoints(document, "p1", ring);
    expect((moved.shapes[0] as PolygonShape).points[1]).toEqual({ x: 8, y: 2 });
    expect((document.shapes[0] as PolygonShape).points[1]).toEqual({ x: 5, y: 1 });
    expect(withoutShape(moved, "p1").shapes).toHaveLength(0);
  });

  it("keeps a ring of three or more, and only on a polygon", () => {
    const document = withShape(empty, polygon);
    // A vertex inserted or removed is the same edit as one moved.
    const four = [...polygon.points, { x: 1, y: 4 }];
    expect((withPolygonPoints(document, "p1", four).shapes[0] as PolygonShape).points).toEqual(four);
    expect(withPolygonPoints(document, "p1", polygon.points.slice(0, 2))).toBe(document);
    expect(withPolygonPoints(document, "nope", four).shapes).toEqual(document.shapes);
  });

  it("replaces one raster region with its derived contours in place", () => {
    const second = { ...polygon, id: "p2" };
    const document = { ...empty, shapes: [polygon, second] };
    const replacements = [
      { ...polygon, id: "outer" },
      { ...polygon, id: "hole", operation: "subtract" as const },
    ];
    expect(replaceShape(document, "p1", replacements).shapes.map((shape) => shape.id)).toEqual([
      "outer",
      "hole",
      "p2",
    ]);
  });
});

describe("translateShape", () => {
  const bitmap = {
    id: "b1",
    label_key: "defect",
    kind: "bitmap" as const,
    operation: "add" as const,
    x: 4,
    y: 2,
    width: 6,
    height: 4,
    png_base64: "",
  };

  it("offsets every vertex of a polygon by the same amount", () => {
    const moved = translateShape(withShape(empty, polygon), "p1", 3, 2);
    expect((moved.shapes[0] as PolygonShape).points).toEqual([
      { x: 4, y: 3 },
      { x: 8, y: 3 },
      { x: 6, y: 7 },
    ]);
  });

  it("clamps the offset once, so a shape pushed at an edge keeps its shape", () => {
    // The polygon spans x 1..5 in a 20 px frame, so the most it can move left is 1 px.
    // Clamping each vertex on its own would flatten the left edge against x=0 and leave a
    // different triangle behind.
    const moved = translateShape(withShape(empty, polygon), "p1", -50, 0);
    expect((moved.shapes[0] as PolygonShape).points).toEqual([
      { x: 0, y: 1 },
      { x: 4, y: 1 },
      { x: 2, y: 5 },
    ]);
  });

  it("keeps a bitmap crop on integer source pixels", () => {
    const moved = translateShape({ ...empty, shapes: [bitmap] }, "b1", 2.6, -1.4);
    expect(moved.shapes[0]).toMatchObject({ x: 7, y: 1 });
  });

  it("stops a bitmap at the far edge of the frame", () => {
    const moved = translateShape({ ...empty, shapes: [bitmap] }, "b1", 100, 100);
    expect(moved.shapes[0]).toMatchObject({ x: 14, y: 6 });
  });

  it("returns the same document for an unknown shape or a zero move", () => {
    const document = withShape(empty, polygon);
    expect(translateShape(document, "nope", 5, 5)).toBe(document);
    expect(translateShape(document, "p1", 0, 0)).toBe(document);
  });

  it("refuses to move a shape that cannot fit rather than snapping it to an edge", () => {
    const wide = { ...bitmap, x: 0, y: 0, width: 40, height: 40 };
    const document = { ...empty, shapes: [wide] };
    expect(translateShape(document, "b1", 5, 5)).toBe(document);
  });
});

describe("boxes", () => {
  const box: BoxShape = {
    id: "r1",
    label_key: "defect",
    kind: "box",
    operation: "add",
    x: 2,
    y: 3,
    width: 4,
    height: 2,
  };

  it("are outlined by their four corners, in order", () => {
    expect(shapeOutline(box)).toEqual([
      { x: 2, y: 3 },
      { x: 6, y: 3 },
      { x: 6, y: 5 },
      { x: 2, y: 5 },
    ]);
  });

  it("move whole, clamped against their own extent", () => {
    const document = { ...empty, shapes: [box] };
    expect(translateShape(document, "r1", 1.5, 1).shapes[0]).toMatchObject({ x: 3.5, y: 4 });
    // Pushed far right, the box stops at the frame edge without shrinking.
    expect(translateShape(document, "r1", 100, 0).shapes[0]).toMatchObject({
      x: 16,
      y: 3,
      width: 4,
      height: 2,
    });
  });

  it("take a new extent, and keep their identity and class", () => {
    const document = { ...empty, shapes: [box] };
    const resized = withBoxRect(document, "r1", { x: 2, y: 1, width: 8, height: 4 });
    expect(resized.shapes[0]).toEqual({ ...box, x: 2, y: 1, width: 8, height: 4 });
    expect(document.shapes[0]).toBe(box);
  });

  it("refuse an extent without area, and leave every other shape alone", () => {
    const document = { ...empty, shapes: [box, polygon] };
    expect(withBoxRect(document, "r1", { x: 2, y: 3, width: 0, height: 2 })).toBe(document);
    expect(withBoxRect(document, "p1", { x: 0, y: 0, width: 1, height: 1 }).shapes).toEqual(
      document.shapes,
    );
  });
});
