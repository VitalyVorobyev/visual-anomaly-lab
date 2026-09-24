import { describe, expect, it } from "vitest";

import type { AnnotationDocument, BitmapShape, BoxShape, PolygonShape } from "../../api/client";
import { readPixel } from "./pixelReadout";

const square: PolygonShape = {
  id: "square",
  label_key: "defect",
  kind: "polygon",
  operation: "add",
  points: [
    { x: 2, y: 2 },
    { x: 8, y: 2 },
    { x: 8, y: 8 },
    { x: 2, y: 8 },
  ],
};

// A 2x2 crop at (4, 4) with only its top-left pixel set.
const cut: BitmapShape = {
  id: "cut",
  label_key: "defect",
  kind: "bitmap",
  operation: "subtract",
  x: 4,
  y: 4,
  width: 2,
  height: 2,
  png_base64: "cut-png",
};

function document(overrides: Partial<AnnotationDocument> = {}): AnnotationDocument {
  return {
    schema_version: 1,
    image_width: 10,
    image_height: 10,
    base: "empty",
    shapes: [square, cut],
    ...overrides,
  } as AnnotationDocument;
}

const masks = new Map([["cut-png", Uint8Array.from([1, 0, 0, 0])]]);

describe("readPixel", () => {
  it("names the pixel, the resolved value and the topmost region", () => {
    expect(readPixel(document(), { x: 3.7, y: 3.2 }, masks, null)).toEqual({
      x: 3,
      y: 3,
      value: 1,
      region: 1,
    });
  });

  it("folds the shapes in order, so a later cut clears an earlier add", () => {
    expect(readPixel(document(), { x: 4.5, y: 4.5 }, masks, null)).toEqual({
      x: 4,
      y: 4,
      value: 0,
      region: 2,
    });
    // Inside the cut's crop but not its mask: the square still holds.
    expect(readPixel(document(), { x: 5.5, y: 4.5 }, masks, null)?.value).toBe(1);
  });

  it("starts from the imported base only when the document is based on it", () => {
    const base = new Uint8Array(100);
    base[0] = 1;
    expect(readPixel(document({ shapes: [] }), { x: 0, y: 0 }, masks, base)?.value).toBe(0);
    expect(
      readPixel(document({ shapes: [], base: "source_mask" }), { x: 0, y: 0 }, masks, base),
    ).toEqual({ x: 0, y: 0, value: 1, region: null });
  });

  it("leaves an undecoded bitmap out rather than guessing, and reads nothing off the frame", () => {
    expect(readPixel(document(), { x: 4.5, y: 4.5 }, new Map(), null)?.region).toBe(1);
    expect(readPixel(document(), { x: -0.5, y: 3 }, masks, null)).toBeNull();
    expect(readPixel(document(), { x: 10, y: 3 }, masks, null)).toBeNull();
  });
});

describe("readPixel over a box", () => {
  const box: BoxShape = {
    id: "box",
    label_key: "defect",
    kind: "box",
    operation: "add",
    x: 1,
    y: 1,
    width: 3,
    height: 2,
  };

  it("reads a box as the polygon of its corners, tested at the pixel's centre", () => {
    const withBox = document({ shapes: [box] });
    expect(readPixel(withBox, { x: 1.2, y: 1.9 }, masks, null)).toEqual({
      x: 1,
      y: 1,
      value: 1,
      region: 1,
    });
    expect(readPixel(withBox, { x: 3.5, y: 2.5 }, masks, null)?.value).toBe(1);
    expect(readPixel(withBox, { x: 4.5, y: 2.5 }, masks, null)).toEqual({
      x: 4,
      y: 2,
      value: 0,
      region: null,
    });
  });
});
