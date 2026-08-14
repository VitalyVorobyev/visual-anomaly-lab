import { describe, expect, it } from "vitest";

import { maskFromPixels, strokeBounds, traceMask } from "./annotationBitmap";

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

describe("maskFromPixels", () => {
  /** RGBA bytes for one row, from `[r, g, b, a]` tuples. */
  const rgba = (...pixels: [number, number, number, number][]) =>
    new Uint8ClampedArray(pixels.flat());

  it("reads luminance, so an opaque backend mask is not one solid region", () => {
    // This is exactly what `encode_png` writes: mode "L", fully opaque, black ground.
    // Reading the alpha byte called every one of these pixels filled, which is why tracing an
    // accepted MobileSAM candidate or an imported PNG produced its bounding box.
    const opaque = rgba([0, 0, 0, 255], [255, 255, 255, 255], [0, 0, 0, 255]);
    expect([...maskFromPixels(opaque, 3)]).toEqual([0, 1, 0]);
  });

  it("still reads a brush stroke, which is white on transparent black", () => {
    const stroke = rgba([0, 0, 0, 0], [255, 255, 255, 255], [255, 255, 255, 180]);
    expect([...maskFromPixels(stroke, 3)]).toEqual([0, 1, 1]);
  });

  it("never treats a fully transparent pixel as filled, whatever its colour says", () => {
    // A canvas clears to transparent black, but a cleared region of a *tinted* canvas can hold
    // stale colour bytes under a zero alpha. Those are not mask.
    expect([...maskFromPixels(rgba([255, 255, 255, 0]), 1)]).toEqual([0]);
  });
});
