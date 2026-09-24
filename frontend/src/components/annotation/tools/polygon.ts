/**
 * Polygon: a click adds a vertex, a click back on the first vertex closes the ring, and so
 * does a double-click anywhere.
 *
 * `polygonClick` holds the whole click rule, including the `ignore` that lets a double-click
 * close a ring without leaving the duplicate vertex its second click would otherwise add.
 */

import { polygonClick, snapTolerance } from "../../../api/annotationPolygon";
import type { ToolContext, ToolEffect, ToolModule } from "./types";

function closable(context: ToolContext): boolean {
  return context.pendingPoints.length >= 3;
}

export const polygonTool: ToolModule = {
  pansWithPrimary: false,
  cursor: "crosshair",
  down: (context, point) => {
    // Source pixels are themselves Konva Image nodes, so a useful canvas click almost never
    // targets the Stage object. Shape handlers stop propagation; anything reaching a tool is
    // the image or the background, and therefore an empty-scene gesture.
    const effects: ToolEffect[] = [{ type: "deselect" }];
    const decision = polygonClick(context.pendingPoints, point, snapTolerance(context.scale));
    if (decision === "close") effects.push({ type: "closePolygon" });
    else if (decision === "add") effects.push({ type: "vertex", point });
    return { gesture: null, effects };
  },
  move: (_context, gesture) => ({ gesture, effects: [] }),
  up: () => [],
  key: (context, point, input) =>
    input.key === "Enter" && closable(context)
      ? [{ type: "closePolygon" }]
      : [{ type: "vertex", point }],
  // The second click of the pair was dropped as a duplicate, so this closes the ring the first
  // click completed — a polygon tool's ordinary gesture, and the reason the "Close" button in
  // the inspector is gone.
  doubleClick: (context) => (closable(context) ? [{ type: "closePolygon" }] : []),
  snaps: (context, point) =>
    closable(context) &&
    polygonClick(context.pendingPoints, point, snapTolerance(context.scale)) === "close",
};
