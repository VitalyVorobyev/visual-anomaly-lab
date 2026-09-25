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

import { classColour, rgbOf } from "../components/viewer/labelPaint";
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
 * A supervised run's label map drawn for a gallery tile: the method's (filled, solid border)
 * or the truth's (dashed border only), at the thumbnail's size.
 *
 * The colours travel in the URL, one per pinned class, from `classColour` — the same
 * function `LabelLayer` and the legend paint with — so the palette has exactly one home,
 * the design system, and the server keeps no copy of it to drift.
 */
export function labelMapUrl(
  imageId: number,
  experimentId: number,
  classes: readonly string[],
  truth: boolean,
): string {
  const colours = classes
    .map((_, index) =>
      rgbOf(classColour(index + 1))
        .map((byte) => byte.toString(16).padStart(2, "0"))
        .join(""),
    )
    .join(",");
  const query = new URLSearchParams({ colours });
  if (truth) query.set("truth", "true");
  return `${apiBaseUrl}/api/experiments/${experimentId}/images/${imageId}/label-map?${query.toString()}`;
}

/**
 * One image's kept detections and true boxes, drawn by the server as an SVG in the source
 * frame for a gallery tile: `colours` are the match, false-positive and missed tones as
 * `#rrggbb`, so the tones stay the design system's (`boxTones.ts`).
 */
export function boxMapUrl(
  imageId: number,
  experimentId: number,
  colours: readonly [string, string, string],
  layers: { predictions: boolean; truth: boolean },
): string {
  const query = new URLSearchParams({
    colours: colours.map((colour) => colour.replace(/^#/, "")).join(","),
  });
  if (!layers.predictions) query.set("predictions", "false");
  if (!layers.truth) query.set("truth", "false");
  return `${apiBaseUrl}/api/experiments/${experimentId}/images/${imageId}/box-map?${query.toString()}`;
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
/** One annotation class's outline, from that class's own truth (ADR-0040). */
export function classMaskUrl(imageId: number, classKey: string): string {
  const query = new URLSearchParams({ class_key: classKey });
  return `${apiBaseUrl}/api/images/${imageId}/mask?${query.toString()}`;
}

export function maskUrl(imageId: number, prepared?: { experimentId: number }): string {
  if (prepared === undefined) return `${apiBaseUrl}/api/images/${imageId}/mask`;
  const query = new URLSearchParams({
    frame: "prepared",
    experiment_id: String(prepared.experimentId),
  });
  return `${apiBaseUrl}/api/images/${imageId}/mask?${query.toString()}`;
}

/**
 * The imported mask a `base="source_mask"` annotation document starts from.
 *
 * Not `maskUrl`: that is the image's *current* truth, newest completed revision first, and
 * drawn as the editor's base it showed a revision — edits included — as though it were the
 * layer under them. This is the pinned import, binary and source-sized, for the editor to
 * tint in the label's colour.
 */
export function sourceMaskUrl(imageId: number): string {
  return `${apiBaseUrl}/api/images/${imageId}/annotations/source-mask`;
}
