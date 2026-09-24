/**
 * Contour assist: MobileSAM prompts in source pixels.
 *
 * In point mode a click is a positive prompt and a Shift-click a negative one. In box mode a
 * drag spans a box, normalised so it is the same box whichever corner the drag began from.
 */

import type { AnnotationPoint, AssistBox } from "../../../api/client";
import type { ToolModule } from "./types";

export function spanBox(start: AnnotationPoint, point: AnnotationPoint): AssistBox {
  return {
    x0: Math.min(start.x, point.x),
    y0: Math.min(start.y, point.y),
    x1: Math.max(start.x, point.x),
    y1: Math.max(start.y, point.y),
  };
}

export const assistTool: ToolModule = {
  pansWithPrimary: false,
  cursor: "crosshair",
  down: (context, point, input) =>
    context.assistMode === "point"
      ? {
          gesture: null,
          effects: [
            { type: "assistPoint", point: { ...point, kind: input.shiftKey ? "negative" : "positive" } },
          ],
        }
      : {
          gesture: { kind: "box", start: point },
          effects: [{ type: "assistBox", box: spanBox(point, point) }],
        },
  move: (_context, gesture, point) =>
    gesture.kind === "box"
      ? { gesture, effects: [{ type: "assistBox", box: spanBox(gesture.start, point) }] }
      : { gesture, effects: [] },
  up: () => [],
  // A box is a drag; there is no keyboard equivalent to hand it, so the key is the page's.
  key: (context, point, input) =>
    context.assistMode === "box"
      ? null
      : [{ type: "assistPoint", point: { ...point, kind: input.shiftKey ? "negative" : "positive" } }],
  doubleClick: () => [],
  snaps: () => false,
};
