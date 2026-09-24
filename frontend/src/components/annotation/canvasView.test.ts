import { describe, expect, it } from "vitest";

import {
  INITIAL_CANVAS_VIEW,
  MAX_PIXEL_SCALE,
  actualPixelsView,
  fitScale,
  smoothAt,
  viewOrigin,
  zoomAbout,
  zoomRange,
} from "./canvasView";

const PANE = { width: 1000, height: 800 };

describe("the zoom ceiling", () => {
  it("is 32 screen pixels per source pixel, whatever the image's size", () => {
    for (const image of [
      { width: 4000, height: 3000 },
      { width: 1284, height: 1168 },
      { width: 200, height: 160 },
    ]) {
      const fit = fitScale(PANE, image);
      const [, max] = zoomRange(fit);
      expect(fit * max).toBeCloseTo(MAX_PIXEL_SCALE);
    }
    expect(MAX_PIXEL_SCALE).toBe(32);
  });

  it("never sits below fit, even for an image already magnified past it", () => {
    const tiny = { width: 10, height: 8 };
    const fit = fitScale(PANE, tiny);
    expect(fit).toBeGreaterThan(MAX_PIXEL_SCALE);
    expect(zoomRange(fit)[1]).toBe(1);
  });

  it("is where the wheel stops", () => {
    const image = { width: 4000, height: 3000 };
    let view = INITIAL_CANVAS_VIEW;
    for (let notch = 0; notch < 200; notch += 1) {
      view = zoomAbout(view, 1.1, { x: 500, y: 400 }, PANE, image);
    }
    expect(fitScale(PANE, image) * view.zoom).toBeCloseTo(MAX_PIXEL_SCALE);
  });
});

describe("zoomAbout", () => {
  it("holds the source point under the anchor where it is", () => {
    const image = { width: 1284, height: 1168 };
    const anchor = { x: 700, y: 300 };
    const fit = fitScale(PANE, image);
    const before = viewOrigin(PANE, image, fit, INITIAL_CANVAS_VIEW);
    const source = { x: (anchor.x - before.x) / fit, y: (anchor.y - before.y) / fit };

    const view = zoomAbout(INITIAL_CANVAS_VIEW, 2, anchor, PANE, image);
    const after = viewOrigin(PANE, image, fit, view);
    const scale = fit * view.zoom;
    expect(after.x + source.x * scale).toBeCloseTo(anchor.x);
    expect(after.y + source.y * scale).toBeCloseTo(anchor.y);
  });
});

describe("1:1 and smoothing", () => {
  it("puts one source pixel on one screen pixel, centred", () => {
    const image = { width: 1284, height: 1168 };
    const view = actualPixelsView(PANE, image);
    const fit = fitScale(PANE, image);
    expect(fit * view.zoom).toBeCloseTo(1);
    const origin = viewOrigin(PANE, image, fit, view);
    expect(origin.x).toBeCloseTo((PANE.width - image.width) / 2);
  });

  it("stops resampling smoothly once a source pixel is larger than a screen pixel", () => {
    expect(smoothAt(0.5)).toBe(true);
    expect(smoothAt(1)).toBe(true);
    expect(smoothAt(1.01)).toBe(false);
    expect(smoothAt(32)).toBe(false);
  });
});
