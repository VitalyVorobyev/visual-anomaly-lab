/**
 * The half pixel between the document's area convention and the stage's centre convention
 * lives in one module. A point that crosses it and comes back must be the point it was, and a
 * pixel's centre must land on the stage's integer coordinate — at 8x a half-pixel error is four
 * screen pixels off the edge being traced.
 */

import { describe, expect, it } from "vitest";

import {
  flatArrayToStage,
  flatToStage,
  frameBounds,
  fromStage,
  rectFromStage,
  rectToStage,
  toStage,
} from "./stageFrame";

describe("stageFrame", () => {
  it("puts a pixel's centre on the stage's integer coordinate", () => {
    // Pixel (10, 5) covers [10, 11) x [5, 6); its centre is (10.5, 5.5), which the stage calls (10, 5).
    expect(toStage({ x: 10.5, y: 5.5 })).toEqual({ x: 10, y: 5 });
    expect(fromStage({ x: 10, y: 5 })).toEqual({ x: 10.5, y: 5.5 });
    // The frame's corners: 0 and W in the document are -0.5 and W - 0.5 on the stage.
    expect(toStage({ x: 0, y: 0 })).toEqual({ x: -0.5, y: -0.5 });
    expect(fromStage({ x: 255.5, y: 255.5 })).toEqual({ x: 256, y: 256 });
  });

  it("round-trips points, rects and rings exactly", () => {
    for (const point of [
      { x: 0, y: 0 },
      { x: 17.25, y: 3.75 },
      { x: 256, y: 128 },
    ]) {
      expect(fromStage(toStage(point))).toEqual(point);
      expect(toStage(fromStage(point))).toEqual(point);
    }
    const rect = { x: 80, y: 60, width: 120, height: 120 };
    expect(rectToStage(rect)).toEqual({ x: 79.5, y: 59.5, width: 120, height: 120 });
    expect(rectFromStage(rectToStage(rect))).toEqual(rect);
  });

  it("shifts flat rings and trails by the same half pixel", () => {
    expect(flatToStage([{ x: 1, y: 2 }, { x: 3.5, y: 4.5 }])).toEqual([0.5, 1.5, 3, 4]);
    expect(flatArrayToStage([1, 2, 3.5, 4.5])).toEqual([0.5, 1.5, 3, 4]);
    expect(flatToStage([])).toEqual([]);
  });

  it("bounds an edit by the whole frame, border included", () => {
    expect(frameBounds(256, 128)).toEqual({ x: -0.5, y: -0.5, width: 256, height: 128 });
    expect(rectFromStage(frameBounds(256, 128))).toEqual({ x: 0, y: 0, width: 256, height: 128 });
  });
});
