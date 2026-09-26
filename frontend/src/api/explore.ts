/**
 * The pure half of Explore on the client: which grid cell a click lands in, the overlay URL
 * for one reading of a map, and the candidate a mask becomes when it is carried to the
 * editor.
 *
 * A cluster is picked here rather than by asking the server, because the answer is already
 * on the client: a clusters response carries every cell's cluster and the transform that
 * placed the image in the encoder's frame. `cellAt` is `SpatialTransform.source_to_prepared`
 * followed by a floor division — the same arithmetic `explore/grid.source_cell` does.
 */

import { classColour, rgbOf } from "../components/viewer/labelPaint";
import { apiBaseUrl, type BitmapShape, type SpatialTransform } from "./client";

export interface ImagePoint {
  x: number;
  y: number;
}

/** The `(row, col)` of the grid cell a source pixel lies in, clamped to the grid. */
export function cellAt(
  transform: SpatialTransform,
  patch: number,
  rows: number,
  cols: number,
  point: ImagePoint,
): { row: number; col: number } {
  const scaleX = transform.resized_width / (transform.crop_right - transform.crop_left);
  const scaleY = transform.resized_height / (transform.crop_bottom - transform.crop_top);
  const x = (point.x - transform.crop_left + 0.5) * scaleX - 0.5 + transform.pad_left;
  const y = (point.y - transform.crop_top + 0.5) * scaleY - 0.5 + transform.pad_top;
  const clamp = (value: number, size: number) => Math.min(Math.max(value, 0), size - 1);
  return { row: clamp(Math.floor(y / patch), rows), col: clamp(Math.floor(x / patch), cols) };
}

/** The cluster under a click, or `null` off the image (cell value 0) or without clusters. */
export function clusterAt(
  answer: {
    transform: SpatialTransform;
    patch_size: number;
    grid_rows: number;
    grid_cols: number;
    cells?: number[] | null;
  },
  point: ImagePoint,
): number | null {
  if (!answer.cells) return null;
  const { row, col } = cellAt(
    answer.transform,
    answer.patch_size,
    answer.grid_rows,
    answer.grid_cols,
    point,
  );
  const value = answer.cells[row * answer.grid_cols + col] ?? 0;
  return value > 0 ? value : null;
}

/** `#rrggbb` as the six digits the server's `colours` parameter takes. */
export function hexDigits(colour: string): string {
  return rgbOf(colour)
    .map((byte) => byte.toString(16).padStart(2, "0"))
    .join("");
}

/** Cluster `i` (1-based) is drawn in the design system's `i`th series colour. */
export function clusterColour(cluster: number): string {
  return classColour(cluster);
}

/**
 * One reading of a stored map. Similarity alone is a heatmap on [0, 1]; with a threshold it
 * is the filled mask at that cut. Clusters name their colours, and `cluster` keeps one.
 * Instances are coloured the same way, one series colour each, and `instance` draws that
 * one's whole mask alone.
 */
export function exploreMapUrl(
  mapUrl: string,
  options: {
    threshold?: number;
    clusters?: number;
    cluster?: number | null;
    instances?: number;
    instance?: number | null;
  } = {},
): string {
  const query = new URLSearchParams();
  if (options.threshold !== undefined) {
    query.set("threshold", options.threshold.toFixed(3));
    query.set("colours", hexDigits(clusterColour(1)));
  }
  const labelled = options.clusters ?? options.instances;
  if (labelled !== undefined) {
    query.set(
      "colours",
      Array.from({ length: labelled }, (_, index) => hexDigits(clusterColour(index + 1))).join(","),
    );
    if (options.cluster) query.set("cluster", String(options.cluster));
    if (options.instance) query.set("instance", String(options.instance));
  }
  const suffix = query.toString();
  return `${apiBaseUrl}${mapUrl}${suffix ? `?${suffix}` : ""}`;
}

/**
 * A mask carried from Explore into the annotation editor, where it waits as an assist
 * candidate the person accepts or discards. It travels in the navigation's state — nothing
 * is written anywhere until it is accepted there.
 */
export interface CarriedCandidate {
  imageId: number;
  shape: BitmapShape;
  area: number;
  /** What produced it, in words: "similarity ≥ 0.60 · DINOv2 ViT-B/14". */
  source: string;
}

/** The candidate in a navigation's state, if one was carried for this image. */
export function carriedFrom(state: unknown, imageId: number): CarriedCandidate | null {
  if (typeof state !== "object" || state === null) return null;
  const carried = (state as { exploreCandidate?: unknown }).exploreCandidate;
  if (typeof carried !== "object" || carried === null) return null;
  const candidate = carried as Partial<CarriedCandidate>;
  if (candidate.imageId !== imageId || !candidate.shape || candidate.shape.kind !== "bitmap") {
    return null;
  }
  return {
    imageId,
    shape: candidate.shape,
    area: typeof candidate.area === "number" ? candidate.area : 0,
    source: typeof candidate.source === "string" ? candidate.source : "Explore",
  };
}
