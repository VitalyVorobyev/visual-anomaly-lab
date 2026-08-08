/**
 * URLs for image tiers.
 *
 * These are built by hand rather than fetched through the typed client, because they are
 * consumed by `<img src>`: the browser does the request, the caching and the decoding,
 * and a `fetch` wrapper would only get in the way of all three. The backend serves these
 * with a content-derived `ETag` and `Cache-Control: immutable`, so a grid that scrolls
 * back over itself re-renders from cache without touching the network.
 */

import { apiBaseUrl } from "./client";
import type { ImageTier } from "./client";

export function imageUrl(imageId: number, tier: ImageTier): string {
  return `${apiBaseUrl}/api/images/${imageId}/${tier}`;
}

/**
 * One experiment's anomaly map for an image, colormapped and sized to the source.
 *
 * `alpha` is the overlay decision: opacity follows the score, so the source image shows
 * through wherever the model found nothing. Pass `false` for a standalone panel — there
 * the same map is the whole picture, and a clean image would otherwise render blank.
 */
export function anomalyMapUrl(imageId: number, experimentId: number, alpha = true): string {
  const query = new URLSearchParams({ experiment_id: String(experimentId) });
  if (!alpha) query.set("alpha", "false");
  return `${apiBaseUrl}/api/images/${imageId}/anomaly-map?${query.toString()}`;
}

/**
 * The model's own segmentation: where its map crosses `threshold`, and nothing else.
 *
 * The heatmap says how much, everywhere; this says *where*, with an edge. Only something
 * with an edge can be laid against the ground-truth outline and read as agreement or
 * disagreement, which is the comparison the sample page exists to support.
 *
 * `threshold` is in the map's own units and must come from the **run-wide** range, not
 * from this image's extremes — a per-image cut is a different cut on every image, so two
 * samples' regions would not be comparable.
 */
export function predictionUrl(
  imageId: number,
  experimentId: number,
  threshold: number,
  outline = false,
): string {
  const query = new URLSearchParams({
    experiment_id: String(experimentId),
    render: outline ? "contour" : "region",
    threshold: String(threshold),
  });
  return `${apiBaseUrl}/api/images/${imageId}/anomaly-map?${query.toString()}`;
}

/** The ground-truth outline, transparent everywhere else, ready to stack on the source. */
export function maskUrl(imageId: number): string {
  return `${apiBaseUrl}/api/images/${imageId}/mask`;
}
