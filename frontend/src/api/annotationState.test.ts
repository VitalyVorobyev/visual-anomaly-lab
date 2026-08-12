import { describe, expect, it } from "vitest";

import type { AnnotationDocument, PolygonShape } from "./client";
import {
  createHistory,
  historyReducer,
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
});
