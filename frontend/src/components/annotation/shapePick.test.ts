import { describe, expect, it } from "vitest";

import type { AnnotationShape, BitmapShape } from "../../api/client";
import { pickBitmap, topmostShape } from "./shapePick";

/** A 4x3 crop at (10, 20) with its middle row painted: pixels (11..12, 21). */
const crop: BitmapShape = {
  id: "crop",
  label_key: "defect",
  kind: "bitmap",
  operation: "add",
  x: 10,
  y: 20,
  width: 4,
  height: 3,
  png_base64: "crop",
};
// prettier-ignore
const cropMask = new Uint8Array([
  0, 0, 0, 0,
  0, 1, 1, 0,
  0, 0, 0, 0,
]);
const over: BitmapShape = { ...crop, id: "over", png_base64: "over" };
const full = new Uint8Array(12).fill(1);

describe("pickBitmap", () => {
  const masks = new Map([["crop", cropMask]]);

  it("picks painted pixels, not the crop rectangle", () => {
    // Inside pixel (11, 21), area convention.
    expect(pickBitmap([crop], masks, { x: 11.5, y: 21.5 }, 0)).toEqual({ id: "crop", dist: 0 });
    // Inside the crop, on an unpainted pixel, with no tolerance: nothing.
    expect(pickBitmap([crop], masks, { x: 10.5, y: 20.5 }, 0)).toBeNull();
  });

  it("measures a near miss to the nearest painted pixel's square, within the radius", () => {
    // Half a pixel left of pixel (11, 21)'s left edge.
    expect(pickBitmap([crop], masks, { x: 10.5, y: 21.5 }, 1)).toEqual({ id: "crop", dist: 0.5 });
    expect(pickBitmap([crop], masks, { x: 10.5, y: 21.5 }, 0.4)).toBeNull();
    // Outside the crop altogether, still within reach of its painted row.
    expect(pickBitmap([crop], masks, { x: 12.5, y: 23.5 }, 2)).toEqual({ id: "crop", dist: 1.5 });
  });

  it("gives a tie to the later bitmap, which is painted over the earlier", () => {
    const both = new Map([
      ["crop", cropMask],
      ["over", full],
    ]);
    const shapes: AnnotationShape[] = [crop, over];
    expect(pickBitmap(shapes, both, { x: 11.5, y: 21.5 }, 1)?.id).toBe("over");
    expect(pickBitmap([over, crop], both, { x: 11.5, y: 21.5 }, 1)?.id).toBe("crop");
  });

  it("cannot pick a bitmap that has not been decoded yet", () => {
    expect(pickBitmap([crop], new Map(), { x: 11.5, y: 21.5 }, 1)).toBeNull();
  });
});

describe("topmostShape", () => {
  const order = new Map([
    ["a", 0],
    ["b", 1],
    ["c", 2],
  ]);

  it("prefers the nearest answer", () => {
    expect(topmostShape([{ id: "c", dist: 2 }, { id: "a", dist: 0.5 }], order)).toBe("a");
  });

  it("breaks a tie by document order: the region painted last is on top", () => {
    expect(topmostShape([{ id: "c", dist: 0 }, { id: "a", dist: 0 }, { id: "b", dist: 0 }], order)).toBe("c");
  });

  it("ignores answers that are not regions of the document", () => {
    expect(topmostShape([{ id: "assist-0", dist: 0 }, { id: "b", dist: 1 }], order)).toBe("b");
    expect(topmostShape([{ id: "suggestion", dist: 0 }], order)).toBeNull();
    expect(topmostShape([], order)).toBeNull();
  });
});
