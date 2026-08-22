import { describe, expect, it } from "vitest";

import type { AnnotationDocument, PolygonShape } from "./client";
import {
  createHistory,
  historyReducer,
  replaceShape,
  translateShape,
  withPolygonPoint,
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

  it("moves polygon vertices and removes shapes without mutation", () => {
    const document = withShape(empty, polygon);
    const moved = withPolygonPoint(document, "p1", 1, { x: 8, y: 2 });
    expect((moved.shapes[0] as PolygonShape).points[1]).toEqual({ x: 8, y: 2 });
    expect((document.shapes[0] as PolygonShape).points[1]).toEqual({ x: 5, y: 1 });
    expect(withoutShape(moved, "p1").shapes).toHaveLength(0);
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
