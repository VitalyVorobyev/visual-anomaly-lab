/**
 * Which region a press lands on, as arithmetic.
 *
 * The stage answers "what is under the pointer" layer by layer: each vector run's `AreaSet`
 * from its outlines, and the bitmaps from `pickBitmap`, which reads the masks the readout has
 * already decoded — the painted pixels, not the crop rectangle around them. `topmostShape`
 * then chooses among those answers the way the document is read: the nearest wins, and among
 * regions the pointer is inside, the one painted last — the one on top.
 */

import type { AnnotationPoint, AnnotationShape } from "../../api/client";

/**
 * The layer-id prefix of every stage layer whose hits are regions of the document — the
 * vector runs, the bitmaps, the crop outlines — so a press can tell a region from an assist
 * prompt or the suggestion's outline.
 */
export const REGION_LAYER = "annotation-region";

export interface ShapeHit {
  id: string | number;
  dist: number;
}

/**
 * The bitmap region under `point` (area convention), or the nearest painted pixel within
 * `radius` image pixels. Distance is to the pixel's square, so a point on a painted pixel is at
 * zero. Ties go to the later region, which is painted over the earlier one. A bitmap whose
 * mask has not been decoded yet cannot be picked, as the readout does not read it either.
 */
export function pickBitmap(
  shapes: readonly AnnotationShape[],
  masks: ReadonlyMap<string, Uint8Array>,
  point: AnnotationPoint,
  radius: number,
): { id: string; dist: number } | null {
  let best: { id: string; dist: number } | null = null;
  for (const shape of shapes) {
    if (shape.kind !== "bitmap") continue;
    const mask = masks.get(shape.png_base64);
    if (!mask) continue;
    const dist = maskDistance(shape, mask, point, radius);
    if (dist === null) continue;
    if (best === null || dist <= best.dist) best = { id: shape.id, dist };
  }
  return best;
}

/** Distance from `point` to the nearest set pixel of the crop, or `null` past `radius`. */
function maskDistance(
  crop: { x: number; y: number; width: number; height: number },
  mask: Uint8Array,
  point: AnnotationPoint,
  radius: number,
): number | null {
  const reach = Math.max(0, radius);
  const left = Math.max(crop.x, Math.floor(point.x - reach));
  const right = Math.min(crop.x + crop.width - 1, Math.floor(point.x + reach));
  const top = Math.max(crop.y, Math.floor(point.y - reach));
  const bottom = Math.min(crop.y + crop.height - 1, Math.floor(point.y + reach));
  let nearest: number | null = null;
  for (let y = top; y <= bottom; y += 1) {
    const dy = Math.max(y - point.y, 0, point.y - (y + 1));
    for (let x = left; x <= right; x += 1) {
      if (mask[(y - crop.y) * crop.width + (x - crop.x)] !== 1) continue;
      const dx = Math.max(x - point.x, 0, point.x - (x + 1));
      const dist = Math.hypot(dx, dy);
      if (dist <= reach && (nearest === null || dist < nearest)) nearest = dist;
    }
  }
  return nearest;
}

/**
 * The region a press selects, among the stage's answers: the nearest, and among equally near
 * ones the latest in document order. Answers that are not a region of `order` (an assist
 * prompt, the suggestion's outline) are ignored.
 */
export function topmostShape(
  hits: readonly ShapeHit[],
  order: ReadonlyMap<string, number>,
): string | null {
  let best: { id: string; dist: number; index: number } | null = null;
  for (const hit of hits) {
    const id = String(hit.id);
    const index = order.get(id);
    if (index === undefined) continue;
    if (
      best === null ||
      hit.dist < best.dist ||
      (hit.dist === best.dist && index > best.index)
    ) {
      best = { id, dist: hit.dist, index };
    }
  }
  return best?.id ?? null;
}
