/**
 * A label map is painted in the design system's palette, by the class's pinned position:
 * a prediction solid, truth dashed, background and ignored pixels clear.
 */

import { describe, expect, it } from "vitest";

import { SERIES_COLOURS } from "@vitavision/lab-ui";

import { classColour, paintLabels, rgbOf } from "./labelPaint";

function plane(rows: number[][]) {
  return {
    width: rows[0]?.length ?? 0,
    height: rows.length,
    values: Float32Array.from(rows.flat()),
  };
}

function alphaAt(pixels: Uint8ClampedArray, width: number, x: number, y: number): number {
  return pixels[(y * width + x) * 4 + 3] as number;
}

describe("label painting", () => {
  it("colours class i with the palette's i-th series, never a colour of the dataset's", () => {
    expect(classColour(1)).toBe(SERIES_COLOURS[0]);
    expect(classColour(2)).toBe(SERIES_COLOURS[1]);
    expect(rgbOf("#3bc9db")).toEqual([0x3b, 0xc9, 0xdb]);
  });

  it("fills a prediction faintly and outlines it, leaving background clear", () => {
    const square = plane([
      [0, 0, 0, 0, 0],
      [0, 1, 1, 1, 0],
      [0, 1, 1, 1, 0],
      [0, 1, 1, 1, 0],
      [0, 0, 0, 0, 0],
    ]);
    const pixels = paintLabels(square, "prediction");
    expect(alphaAt(pixels, 5, 0, 0)).toBe(0);
    const border = alphaAt(pixels, 5, 1, 1);
    const inside = alphaAt(pixels, 5, 2, 2);
    expect(border).toBeGreaterThan(inside);
    expect(inside).toBeGreaterThan(0);
    expect(Array.from(pixels.slice((1 * 5 + 1) * 4, (1 * 5 + 1) * 4 + 3))).toEqual(
      rgbOf(SERIES_COLOURS[0]),
    );
  });

  it("draws truth as a broken outline with nothing inside, and ignores NaN", () => {
    const size = 24;
    const rows = Array.from({ length: size }, (_, y) =>
      Array.from({ length: size }, (_, x) =>
        x === 0 ? Number.NaN : x > 2 && x < 21 && y > 2 && y < 21 ? 2 : 0,
      ),
    );
    const pixels = paintLabels(plane(rows), "truth");
    expect(alphaAt(pixels, size, 10, 10)).toBe(0);
    expect(alphaAt(pixels, size, 0, 5)).toBe(0);
    // Along the top edge some border pixels are drawn and some are the gaps between dashes.
    const top = Array.from({ length: 18 }, (_, index) => alphaAt(pixels, size, 3 + index, 3));
    expect(top.some((alpha) => alpha > 0)).toBe(true);
    expect(top.some((alpha) => alpha === 0)).toBe(true);
    expect(Array.from(pixels.slice((3 * size + 5) * 4, (3 * size + 5) * 4 + 3))).toEqual(
      rgbOf(SERIES_COLOURS[1]),
    );
  });
});
