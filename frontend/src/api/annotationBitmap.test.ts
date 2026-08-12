import { describe, expect, it } from "vitest";

import { strokeBounds, traceMask } from "./annotationBitmap";

describe("brush bitmap bounds", () => {
  it("crops a stroke with its antialias margin and clamps it to the source frame", () => {
    expect(
      strokeBounds(
        [
          { x: 2, y: 3 },
          { x: 18, y: 9 },
        ],
        4,
        20,
        12,
      ),
    ).toEqual({ minX: 0, minY: 0, maxX: 20, maxY: 12 });
  });

  it("returns an empty rectangle for an empty gesture", () => {
    expect(strokeBounds([], 8, 100, 100)).toEqual({ minX: 0, minY: 0, maxX: 0, maxY: 0 });
  });
});

describe("traceMask", () => {
  it("turns a filled region into a compact editable polygon", () => {
    const mask = new Uint8Array([
      0, 0, 0, 0,
      0, 1, 1, 0,
      0, 1, 1, 0,
      0, 0, 0, 0,
    ]);
    const polygons = traceMask(mask, 4, 4, {
      x: 10,
      y: 20,
      label_key: "scratch",
      operation: "add",
    });
    expect(polygons).toHaveLength(1);
    expect(polygons[0]).toMatchObject({
      label_key: "scratch",
      kind: "polygon",
      operation: "add",
      points: [
        { x: 11, y: 21 },
        { x: 13, y: 21 },
        { x: 13, y: 23 },
        { x: 11, y: 23 },
      ],
    });
  });

  it("preserves holes as inverse-operation polygons", () => {
    const mask = new Uint8Array(25).fill(1);
    mask[2 * 5 + 2] = 0;
    const polygons = traceMask(mask, 5, 5, {
      x: 0,
      y: 0,
      label_key: "defect",
      operation: "subtract",
    });
    expect(polygons.map((polygon) => polygon.operation).sort()).toEqual(["add", "subtract"]);
  });
});
