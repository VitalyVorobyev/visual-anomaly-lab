/**
 * What a click means while a polygon is being drawn.
 *
 * Pulled out of the scene because it is the whole interaction: the editor used to need an
 * explicit "Close" button in a side panel, which is neither where the hand is nor what anyone
 * expects from a polygon tool. Closing is now a click back on the first vertex, or a
 * double-click anywhere.
 *
 * The `ignore` rule is what makes the double-click work, and it is also worth having on its
 * own. A double-click is two clicks: the first adds a vertex, the second lands in the same
 * place and is dropped, and only then does `dblclick` fire and close the ring. Without the
 * rule a double-click would leave a duplicate vertex behind — invisible on screen, awkward in
 * every algorithm that reads the polygon afterwards, and impossible to select and delete.
 */

import type { AnnotationPoint } from "./client";

export type PolygonClick = "add" | "close" | "ignore";

export function polygonClick(
  points: readonly AnnotationPoint[],
  point: AnnotationPoint,
  tolerance: number,
): PolygonClick {
  const first = points[0];
  const last = points.at(-1);
  // Three points is the smallest ring the document schema accepts, so below that the first
  // vertex is an ordinary vertex and clicking it again is a duplicate, not a close.
  if (first !== undefined && points.length >= 3 && within(point, first, tolerance)) return "close";
  if (last !== undefined && within(point, last, tolerance)) return "ignore";
  return "add";
}

/**
 * How close a click has to be to count as the same vertex, in source pixels.
 *
 * Derived from a screen radius so that the target stays the same size under the hand however
 * far the view is zoomed in — the reason it cannot be a constant in the document's own units.
 */
export function snapTolerance(scale: number, screenRadius = 9): number {
  return screenRadius / Math.max(scale, 0.0001);
}

function within(a: AnnotationPoint, b: AnnotationPoint, tolerance: number): boolean {
  return Math.hypot(a.x - b.x, a.y - b.y) <= tolerance;
}
