/**
 * The one boundary between the document's coordinates and the stage's.
 *
 * Two conventions name the same pixel. The annotation document, the tools, the pixel readout
 * and the keyboard cursor use the **area** convention: pixel `(i, j)` covers
 * `[i, i + 1) × [j, j + 1)`, so the frame spans `[0, W] × [0, H]` and a pixel's centre is at
 * `i + 0.5`. That is what the backend rasterises and what completion evaluates. The stage's
 * vector layers use the **centre** convention: pixel `i`'s centre is at `i`, so the frame spans
 * `[-0.5, W - 0.5]` (`PIXEL_CENTRE` in `@vitavision/stage2d`).
 *
 * Every vector item handed to the stage goes through `toStage`, and every point that comes back
 * from it — a press, a hover, a drag, an editor's vertices — through `fromStage`. Nothing else
 * converts, so a half-pixel shift has exactly one place to live. A drag's *delta* needs no
 * conversion, and neither does a raster: an element positioned in CSS pixels inside the stage
 * box is already in the area convention.
 */

import { PIXEL_CENTRE, type Point, type Rect } from "@vitavision/stage2d";

/** A document point as the stage draws it. */
export function toStage(point: Point): Point {
  return { x: point.x - PIXEL_CENTRE, y: point.y - PIXEL_CENTRE };
}

/** A stage point as the document records it. */
export function fromStage(point: Point): Point {
  return { x: point.x + PIXEL_CENTRE, y: point.y + PIXEL_CENTRE };
}

export function rectToStage(rect: Rect): Rect {
  return { ...rect, x: rect.x - PIXEL_CENTRE, y: rect.y - PIXEL_CENTRE };
}

export function rectFromStage(rect: Rect): Rect {
  return { ...rect, x: rect.x + PIXEL_CENTRE, y: rect.y + PIXEL_CENTRE };
}

/** Document points as the flat `[x0, y0, x1, y1, …]` ring or trail a stage layer takes. */
export function flatToStage(points: readonly Point[]): number[] {
  const flat = new Array<number>(points.length * 2);
  points.forEach((point, index) => {
    flat[2 * index] = point.x - PIXEL_CENTRE;
    flat[2 * index + 1] = point.y - PIXEL_CENTRE;
  });
  return flat;
}

/** A flat document trail (a brush gesture's `flat`) in stage coordinates. */
export function flatArrayToStage(flat: ArrayLike<number>): number[] {
  return Array.from(flat, (value) => value - PIXEL_CENTRE);
}

/**
 * The whole frame in stage coordinates: what a vertex or a box corner may reach, so a region can
 * lie exactly on the image border, as the area convention allows.
 */
export function frameBounds(width: number, height: number): Rect {
  return { x: -PIXEL_CENTRE, y: -PIXEL_CENTRE, width, height };
}
