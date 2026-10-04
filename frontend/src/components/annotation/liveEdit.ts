/**
 * An edit in progress — a region being moved, a vertex dragged, a box resized — and the document
 * as it looks meanwhile.
 *
 * The committed document changes once, when the gesture ends, so one gesture is one undo step.
 * Until then the scene draws `applyEdit(document, edit)`: the *same* pure edit the commit will
 * make, so what is shown while dragging is exactly what is saved on release — a move clamped at
 * the frame's edge is clamped in the preview too, and a bitmap moves by whole pixels in both.
 */

import type { StageDrag } from "@vitavision/stage2d";

import { translateShape, withBoxRect, withPolygonPoints } from "../../api/annotationState";
import type { AnnotationDocument, AnnotationPoint } from "../../api/client";
import type { BoxRect } from "./tools";

export type LiveEdit =
  | { kind: "move"; id: string; dx: number; dy: number }
  | { kind: "polygon"; id: string; points: AnnotationPoint[] }
  | { kind: "box"; id: string; rect: BoxRect };

export function applyEdit(document: AnnotationDocument, edit: LiveEdit | null): AnnotationDocument {
  if (edit === null) return document;
  if (edit.kind === "move") return translateShape(document, edit.id, edit.dx, edit.dy);
  if (edit.kind === "polygon") return withPolygonPoints(document, edit.id, edit.points);
  return withBoxRect(document, edit.id, edit.rect);
}

/**
 * How far a press may travel, in screen pixels, and still be a click that selects without
 * moving — the stage's own click slop. A region has to survive a shaking hand, or selecting it
 * would drag it a pixel and land in undo history.
 */
const CLICK_SLOP = 3;

/**
 * The drag that moves a region pressed at `start` (stage coordinates) from `client`: previewed
 * through `onEdit` once it leaves the slop, committed through `onMove` on release. Deltas are
 * the same in either coordinate convention, so nothing here converts.
 */
export function moveDrag(
  id: string,
  start: AnnotationPoint,
  client: AnnotationPoint,
  onEdit: (edit: LiveEdit | null) => void,
  onMove: (shapeId: string, dx: number, dy: number) => void,
): StageDrag {
  let moving = false;
  return {
    // A finger on a region moves it, as the mouse does; on bare image it pans.
    claimsTouch: true,
    onMove: (point, event) => {
      if (!moving && Math.hypot(event.clientX - client.x, event.clientY - client.y) <= CLICK_SLOP) {
        return;
      }
      moving = true;
      onEdit({ kind: "move", id, dx: point.x - start.x, dy: point.y - start.y });
    },
    onEnd: (point) => {
      onEdit(null);
      if (moving) onMove(id, point.x - start.x, point.y - start.y);
    },
    onCancel: () => onEdit(null),
  };
}
