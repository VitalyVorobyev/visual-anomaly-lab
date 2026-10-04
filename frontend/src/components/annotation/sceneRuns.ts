/**
 * The document's regions as the stage paints them: in document order, as runs.
 *
 * Order is meaning here, not decoration. A `subtract` region clears what the regions before it
 * set, so a cut has to be painted *over* the region it cuts, and an `add` after it over the cut.
 * Polygons and boxes are vector items and bitmaps are rasters, which are different layers; a
 * single vector layer under every raster would paint a cut polygon beneath the brushed region
 * it removes. So the shapes are split into runs that keep the order: each stretch of
 * consecutive polygons and boxes is one vector layer (painted in item order inside it), and
 * each bitmap is a raster of its own between them.
 */

import type { AnnotationShape, BitmapShape, BoxShape, PolygonShape } from "../../api/client";

export type VectorShape = PolygonShape | BoxShape;

export type SceneRun =
  | { kind: "vector"; key: string; shapes: VectorShape[] }
  | { kind: "bitmap"; key: string; shape: BitmapShape };

export function sceneRuns(shapes: readonly AnnotationShape[]): SceneRun[] {
  const runs: SceneRun[] = [];
  let vector: VectorShape[] | null = null;
  for (const shape of shapes) {
    if (shape.kind === "bitmap") {
      vector = null;
      runs.push({ kind: "bitmap", key: shape.id, shape });
      continue;
    }
    if (vector === null) {
      vector = [];
      runs.push({ kind: "vector", key: shape.id, shapes: vector });
    }
    vector.push(shape);
  }
  return runs;
}
