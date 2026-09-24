/**
 * Select: regions answer the pointer themselves (click selects, drag moves, a vertex reshapes),
 * and anything that reaches the tool is the empty scene — which pans, and on a click without
 * movement clears the selection. A double-click toggles Fit and the previous view.
 */

import type { ToolModule } from "./types";

export const selectTool: ToolModule = {
  pansWithPrimary: true,
  cursor: "grab",
  // Unreachable while `pansWithPrimary` holds: the canvas pans instead.
  down: () => ({ gesture: null, effects: [] }),
  move: (_context, gesture) => ({ gesture, effects: [] }),
  up: () => [],
  // The arrows nudge the selected region under Select; that is the canvas's, not a tool key.
  key: () => null,
  doubleClick: () => [{ type: "toggleFit" }],
  snaps: () => false,
};
