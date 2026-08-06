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
