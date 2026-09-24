/**
 * What an editing tool is, to the canvas: a set of pure functions from pointer and key input
 * to *effects*.
 *
 * A tool never touches Konva, React state or the document. It is handed a source-pixel point
 * and the little context it needs (the open polygon, the view scale, the assist mode), and it
 * answers with the gesture it is now tracking and the effects the canvas should emit through
 * its callbacks. That is what makes each tool testable as arithmetic, and what keeps the
 * per-tool `if` chains out of the scene.
 */

import type { AnnotationPoint, AssistBox, AssistPoint } from "../../../api/client";

export type EditorTool = "select" | "polygon" | "brush" | "eraser" | "assist";

export interface ToolContext {
  /** The polygon being drawn, in source pixels. */
  pendingPoints: readonly AnnotationPoint[];
  /** Screen pixels per source pixel, for tolerances that must stay a constant size on screen. */
  scale: number;
  assistMode: "point" | "box";
}

/**
 * A drag in progress, owned by the canvas between `down` and `up`.
 *
 * A stroke's arrays are **appended in place**. The scene used to copy the whole trail into a
 * new array on every `mousemove` — quadratic in the stroke's length — and re-render every
 * shape in the document to draw one more segment.
 */
export type Gesture =
  | { kind: "stroke"; points: AnnotationPoint[]; flat: number[] }
  | { kind: "box"; start: AnnotationPoint };

export type ToolEffect =
  | { type: "deselect" }
  | { type: "stroke"; points: AnnotationPoint[] }
  | { type: "vertex"; point: AnnotationPoint }
  | { type: "closePolygon" }
  | { type: "assistPoint"; point: AssistPoint }
  | { type: "assistBox"; box: AssistBox }
  | { type: "toggleFit" };

export interface ToolStep {
  gesture: Gesture | null;
  effects: ToolEffect[];
}

export interface KeyInput {
  key: string;
  shiftKey: boolean;
}

export interface ToolModule {
  /**
   * Whether a primary-button drag pans rather than draws. Right-drag pans under every tool;
   * only Select gives the left button to the view as well.
   */
  pansWithPrimary: boolean;
  /** The pointer over the stage while the tool is in hand. */
  cursor: "grab" | "crosshair";
  down: (context: ToolContext, point: AnnotationPoint, input: { shiftKey: boolean }) => ToolStep;
  move: (context: ToolContext, gesture: Gesture, point: AnnotationPoint) => ToolStep;
  up: (context: ToolContext, gesture: Gesture) => ToolEffect[];
  /**
   * Enter or Space on the focused canvas, at the keyboard cursor. `null` means the tool does
   * not use the key, and it is left to the page.
   */
  key: (context: ToolContext, point: AnnotationPoint, input: KeyInput) => ToolEffect[] | null;
  doubleClick: (context: ToolContext) => ToolEffect[];
  /** Whether a click here would close the open ring — the first vertex lights up when so. */
  snaps: (context: ToolContext, point: AnnotationPoint) => boolean;
}

