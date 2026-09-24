/**
 * Brush and eraser: one module, because on the canvas they are one gesture.
 *
 * Both collect a trail of source-pixel samples and hand the whole trail over on release; which
 * of them *adds* and which *removes* is decided by `useDocumentCommands`, from the tool the
 * gesture was made with. Enter or Space on the focused canvas is a one-sample stroke at the
 * keyboard cursor.
 */

import type { ToolModule } from "./types";

export const brushTool: ToolModule = {
  pansWithPrimary: false,
  cursor: "crosshair",
  down: (_context, point) => ({
    gesture: { kind: "stroke", points: [point], flat: [point.x, point.y] },
    effects: [],
  }),
  move: (_context, gesture, point) => {
    if (gesture.kind !== "stroke") return { gesture, effects: [] };
    gesture.points.push(point);
    gesture.flat.push(point.x, point.y);
    return { gesture, effects: [] };
  },
  up: (_context, gesture) =>
    gesture.kind === "stroke" && gesture.points.length > 0
      ? [{ type: "stroke", points: gesture.points }]
      : [],
  key: (_context, point) => [{ type: "stroke", points: [point] }],
  doubleClick: () => [],
  snaps: () => false,
};
