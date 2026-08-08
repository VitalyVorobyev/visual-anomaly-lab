import { describe, expect, it } from "vitest";

import {
  MAX_ZOOM,
  MIN_ZOOM,
  RESET_VIEW,
  nativeZoomFor,
  zoomAt,
} from "./ZoomPanCanvas";

/** A 400x300 box at the origin, so the centre is (200, 150). */
const RECT = { left: 0, top: 0, width: 400, height: 300 };

/**
 * Where a content point ends up on screen, under `translate(t) scale(z)` about the centre.
 * Coordinates are measured from the box's centre, which is what `transform-origin: center`
 * makes them.
 */
function project(view: { zoom: number; x: number; y: number }, content: { x: number; y: number }) {
  return { x: content.x * view.zoom + view.x, y: content.y * view.zoom + view.y };
}

/** The content point currently under a client-space pointer. */
function contentUnder(view: { zoom: number; x: number; y: number }, pointer: { x: number; y: number }) {
  const px = pointer.x - RECT.left - RECT.width / 2;
  const py = pointer.y - RECT.top - RECT.height / 2;
  return { x: (px - view.x) / view.zoom, y: (py - view.y) / view.zoom };
}

describe("zoomAt", () => {
  it("keeps whatever is under the cursor under the cursor", () => {
    /**
     * The property the previous centre-anchored zoom lacked. Without it, magnifying a
     * defect in a corner means every wheel notch has to be undone with a drag — the
     * gesture fights the thing it is for.
     */
    const start = { zoom: 2, x: 30, y: -12 };
    const pointer = { x: 320, y: 70 };
    const before = contentUnder(start, pointer);

    const after = zoomAt(start, pointer, RECT, 1.6);
    const moved = project(after, before);

    expect(moved.x).toBeCloseTo(pointer.x - RECT.width / 2, 6);
    expect(moved.y).toBeCloseTo(pointer.y - RECT.height / 2, 6);
  });

  it("holds the anchor across a zoom in and back out", () => {
    const start = { zoom: 1.5, x: -20, y: 8 };
    const pointer = { x: 120, y: 240 };
    const before = contentUnder(start, pointer);

    const zoomed = zoomAt(zoomAt(start, pointer, RECT, 3), pointer, RECT, 1 / 3);
    const moved = project(zoomed, before);

    expect(zoomed.zoom).toBeCloseTo(start.zoom, 6);
    expect(moved.x).toBeCloseTo(pointer.x - RECT.width / 2, 6);
    expect(moved.y).toBeCloseTo(pointer.y - RECT.height / 2, 6);
  });

  it("resets rather than leaving an offset that cannot be seen", () => {
    // At 1x the content fills the frame, so a pan offset there is a way to be lost with no
    // visible handle to get back.
    expect(zoomAt({ zoom: 1.2, x: 200, y: 90 }, { x: 10, y: 10 }, RECT, 0.1)).toEqual(RESET_VIEW);
  });

  it("clamps at both ends", () => {
    expect(zoomAt({ zoom: 8, x: 0, y: 0 }, { x: 200, y: 150 }, RECT, 100).zoom).toBe(MAX_ZOOM);
    expect(zoomAt({ zoom: 1, x: 0, y: 0 }, { x: 200, y: 150 }, RECT, 0.001)).toEqual(RESET_VIEW);
  });
});

describe("nativeZoomFor", () => {
  it("is the ratio of source pixels to frame pixels", () => {
    expect(nativeZoomFor(1600, 400)).toBe(4);
    expect(nativeZoomFor(800, 400)).toBe(2);
  });

  it("reports 1 for an image smaller than its frame", () => {
    // There are no further pixels to reveal; magnifying past this shows the resampler
    // rather than the sensor.
    expect(nativeZoomFor(200, 400)).toBe(MIN_ZOOM);
  });

  it("clamps a huge source to the zoom ceiling", () => {
    expect(nativeZoomFor(40_000, 400)).toBe(MAX_ZOOM);
  });

  it("falls back to the tier boundary when the width is unknown", () => {
    // `nativeWidth` is optional; a missing or zero width must not produce Infinity or NaN.
    expect(Number.isFinite(nativeZoomFor(0, 400))).toBe(true);
    expect(Number.isFinite(nativeZoomFor(400, 0))).toBe(true);
  });
});
