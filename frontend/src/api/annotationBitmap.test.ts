import { describe, expect, it } from "vitest";

import type { BitmapShape, PolygonShape } from "./client";
import {
  maskBounds,
  maskFromPixels,
  paintRect,
  strokeBounds,
  strokeTargets,
  traceMask,
} from "./annotationBitmap";

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

describe("paintRect", () => {
  const region = { x: 10, y: 10, width: 20, height: 20 };

  it("grows to cover a stroke that leaves the region", () => {
    // The whole point of continuing a region: a defect drawn in three touches is one region,
    // and each touch may reach past what the last one covered.
    expect(paintRect(region, { minX: 4, minY: 12, maxX: 18, maxY: 40 }, false)).toEqual({
      minX: 4,
      minY: 10,
      width: 26,
      height: 30,
    });
  });

  it("never grows for an eraser, whichever way the stroke ran", () => {
    // Erasing cannot add a pixel, so a stroke that wanders far outside the region must not
    // drag its crop along — the crop is what the selection outline and the copy dimension
    // check both read.
    expect(paintRect(region, { minX: 0, minY: 0, maxX: 200, maxY: 200 }, true)).toEqual({
      minX: 10,
      minY: 10,
      width: 20,
      height: 20,
    });
  });
});

describe("maskBounds", () => {
  it("finds the tight box around what is painted", () => {
    const mask = new Uint8Array([
      0, 0, 0, 0,
      0, 1, 1, 0,
      0, 0, 1, 0,
      0, 0, 0, 0,
    ]);
    expect(maskBounds(mask, 4, 4)).toEqual({ minX: 1, minY: 1, width: 2, height: 2 });
  });

  it("reports nothing left, which is how the last of a region is erased", () => {
    expect(maskBounds(new Uint8Array(16), 4, 4)).toBeNull();
  });
});

describe("strokeTargets", () => {
  const bitmap = (id: string, x: number, y: number): BitmapShape => ({
    id,
    label_key: "defect",
    kind: "bitmap",
    operation: "add",
    x,
    y,
    width: 10,
    height: 10,
    png_base64: "",
  });
  const polygon: PolygonShape = {
    id: "ring",
    label_key: "defect",
    kind: "polygon",
    operation: "add",
    points: [
      { x: 0, y: 0 },
      { x: 30, y: 0 },
      { x: 0, y: 30 },
    ],
  };
  const shapes = [bitmap("under", 0, 0), polygon, bitmap("over", 5, 5), bitmap("far", 60, 60)];

  it("finds every painted region the stroke passes over, nearest first", () => {
    // What makes the eraser an eraser with nothing selected: it takes paint off what is under
    // the pointer instead of appending a `subtract` region, which is what it used to do.
    expect(strokeTargets(shapes, { minX: 6, minY: 6, maxX: 8, maxY: 8 }).map((s) => s.id)).toEqual([
      "over",
      "under",
    ]);
  });

  it("ignores regions the stroke misses and polygons it does not", () => {
    // A polygon has no pixels to take away; the eraser says so rather than inventing a cut.
    expect(strokeTargets(shapes, { minX: 20, minY: 20, maxX: 25, maxY: 25 })).toEqual([]);
  });

  it("treats a shared edge as a miss and one pixel inside as a hit", () => {
    expect(strokeTargets(shapes, { minX: 10, minY: 0, maxX: 12, maxY: 2 })).toEqual([]);
    expect(strokeTargets(shapes, { minX: 9, minY: 0, maxX: 12, maxY: 2 }).map((s) => s.id)).toEqual([
      "under",
    ]);
  });
});
