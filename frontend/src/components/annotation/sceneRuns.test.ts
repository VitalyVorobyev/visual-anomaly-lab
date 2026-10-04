import { describe, expect, it } from "vitest";

import type { AnnotationShape, BitmapShape, BoxShape, PolygonShape } from "../../api/client";
import { sceneRuns } from "./sceneRuns";

const polygon = (id: string): PolygonShape => ({
  id,
  label_key: "defect",
  kind: "polygon",
  operation: "add",
  points: [
    { x: 0, y: 0 },
    { x: 4, y: 0 },
    { x: 0, y: 4 },
  ],
});
const box = (id: string): BoxShape => ({
  id,
  label_key: "defect",
  kind: "box",
  operation: "subtract",
  x: 1,
  y: 1,
  width: 2,
  height: 2,
});
const bitmap = (id: string): BitmapShape => ({
  id,
  label_key: "defect",
  kind: "bitmap",
  operation: "add",
  x: 0,
  y: 0,
  width: 1,
  height: 1,
  png_base64: id,
});

function shape(runs: ReturnType<typeof sceneRuns>) {
  return runs.map((run) =>
    run.kind === "vector" ? run.shapes.map((item) => item.id) : `bitmap ${run.shape.id}`,
  );
}

describe("sceneRuns", () => {
  it("keeps document order, one vector layer per stretch and one raster per bitmap", () => {
    const shapes: AnnotationShape[] = [
      polygon("a"),
      box("b"),
      bitmap("c"),
      polygon("d"),
      bitmap("e"),
      bitmap("f"),
      box("g"),
    ];
    // A cut box between two adds must paint between them, so the runs are split at every
    // raster rather than gathered into one vector layer.
    expect(shape(sceneRuns(shapes))).toEqual([["a", "b"], "bitmap c", ["d"], "bitmap e", "bitmap f", ["g"]]);
  });

  it("keys a run by its first region, so the runs before an edit keep their layers", () => {
    const runs = sceneRuns([polygon("a"), box("b"), bitmap("c")]);
    expect(runs.map((run) => run.key)).toEqual(["a", "c"]);
  });

  it("is empty for an empty document", () => {
    expect(sceneRuns([])).toEqual([]);
  });
});
