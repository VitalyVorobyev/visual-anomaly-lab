/**
 * Box: a drag from one corner to the opposite one draws an axis-aligned box region.
 *
 * The box is normalised, so it is the same box whichever corner the drag began from. A drag
 * that spans no area — a click — draws nothing: a document refuses a box without one, and a
 * click that silently minted a sliver would be a region nobody meant to draw.
 */

import { boxBetween } from "../../../api/annotationState";
import type { ToolEffect, ToolModule } from "./types";

export const boxTool: ToolModule = {
  pansWithPrimary: false,
  cursor: "crosshair",
  // A click on the empty scene still clears the selection, as it does under Polygon.
  down: (_context, point) => ({
    gesture: { kind: "box", start: point, end: point },
    effects: [{ type: "deselect" }],
  }),
  move: (_context, gesture, point) =>
    gesture.kind === "box"
      ? { gesture: { ...gesture, end: point }, effects: [] }
      : { gesture, effects: [] },
  up: (_context, gesture): ToolEffect[] => {
    if (gesture.kind !== "box") return [];
    const rect = boxBetween(gesture.start, gesture.end ?? gesture.start);
    return rect ? [{ type: "box", rect }] : [];
  },
  // A box is a drag; there is no keyboard equivalent to hand it, so the key is the page's.
  key: () => null,
  doubleClick: () => [],
  snaps: () => false,
};
