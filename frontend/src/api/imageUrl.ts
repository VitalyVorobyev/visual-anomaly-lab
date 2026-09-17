/**
 * URLs for image tiers.
 *
 * These are built by hand rather than fetched through the typed client, because they are
 * consumed by `<img src>`: the browser does the request, the caching and the decoding,
 * and a `fetch` wrapper would only get in the way of all three. The backend serves these
 * with a content-derived `ETag` and `Cache-Control: immutable`, so a grid that scrolls
 * back over itself re-renders from cache without touching the network.
 */

import type { StageView } from "@vitavision/lab-ui";

import { apiBaseUrl } from "./client";
import type { ImageTier } from "./client";

export function imageUrl(imageId: number, tier: ImageTier): string {
  return `${apiBaseUrl}/api/images/${imageId}/${tier}`;
}

/**
 * Which tier a stage's current magnification deserves.
 *
 * `StageView.scale` is **CSS pixels per image pixel**, so this is a statement about the
 * picture rather than about the frame it happens to be in: past a half screen pixel per
 * image pixel the `preview` tier's 1024 px long edge is being upsampled, and the lossless
 * `full` tier is what the reader is actually asking for. Below it, `full` would move a
 * megabyte to draw fewer pixels than `preview` already holds.
 *
 * Its predecessor compared a *frame-relative* zoom against a constant, which meant the same
 * number said different things in a narrow column and a wide one.
 */
export function tierFor(view: StageView | null): "full" | "preview" {
  return (view?.scale ?? 0) > 0.5 ? "full" : "preview";
}

/** One immutable profile revision's materialised, lossless model input. */
export function preparedImageUrl(profileId: number, imageId: number): string {
  return `${apiBaseUrl}/api/region-profiles/${profileId}/prepared/${imageId}`;
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

/**
 * The ground-truth outline, transparent everywhere else, ready to stack on the source.
 *
 * `experimentId` asks for it in that experiment's **prepared** frame instead. That is not a
 * convenience: a diagnostic pane is drawn at the array's own prepared or grid size, because
 * a diagnostic *is* a prepared-frame quantity and projecting it would be this layer
 * inventing an interpretation. A source-frame outline laid over one is therefore
 * misregistered by exactly the pinned crop and letterbox. The mask meets the pane rather
 * than the other way round.
 *
 * The overlay stack on the sample page stays source-frame, because every layer in it
 * already is.
 */
export function maskUrl(imageId: number, prepared?: { experimentId: number }): string {
  if (prepared === undefined) return `${apiBaseUrl}/api/images/${imageId}/mask`;
  const query = new URLSearchParams({
    frame: "prepared",
    experiment_id: String(prepared.experimentId),
  });
  return `${apiBaseUrl}/api/images/${imageId}/mask?${query.toString()}`;
}
