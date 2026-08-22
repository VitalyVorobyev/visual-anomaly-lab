/**
 * The value-plane endpoints this app fetches from — the addressing half of handbook diagnostics.md.
 *
 * `@vitavision/lab-ui`'s `api/mapValues` owns the wire format (decode, index); it knows
 * nothing about this backend's routes. Those routes are application-specific, so they stay
 * here rather than in the shared package.
 */

import { apiBaseUrl } from "./client";
import type { DiagnosticEntry } from "./client";

export function anomalyMapValuesUrl(imageId: number, experimentId: number): string {
  const query = new URLSearchParams({ experiment_id: String(experimentId) });
  return `${apiBaseUrl}/api/images/${imageId}/anomaly-map/values?${query.toString()}`;
}

/** Every colour plane of what the *model* read — preprocessed, not the display tier. */
export function sourceValuesUrl(experimentId: number, imageId: number): string {
  return `${apiBaseUrl}/api/experiments/${experimentId}/images/${imageId}/source-values`;
}

/** The same `(key, image_id)` addressing the rendered pane uses, asking for the numbers. */
export function diagnosticValuesUrl(experimentId: number, entry: DiagnosticEntry): string {
  const query = new URLSearchParams({ key: entry.key, format: "raw" });
  if (entry.image_id !== null && entry.image_id !== undefined) {
    query.set("image_id", String(entry.image_id));
  }
  return `${apiBaseUrl}/api/experiments/${experimentId}/diagnostics/payload?${query.toString()}`;
}
