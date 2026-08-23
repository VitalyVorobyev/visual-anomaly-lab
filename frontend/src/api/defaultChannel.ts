/**
 * Which of a part's photographs stands for the part.
 *
 * A sample is a physical object photographed under several illuminations, and most screens
 * have room for one of them: a grid tile, a queue card, the tab the sample viewer opens on.
 * Until a dataset could say otherwise this was always `images[0]`, and "first" means the
 * lowest `channel.position` — the order the channel folders happened to be scanned in at
 * import. That is a fact about the source tree, not about which view the part is judged
 * under, and on a dataset whose useful illumination is the third one it costs two keystrokes
 * on every single sample.
 *
 * `Dataset.default_channel` is the dataset's answer, stored as a channel *name* so that a
 * re-import which renumbers the dictionary does not silently repoint it (handbook
 * domain-model.md).
 */

import type { ImageSummary } from "./client";

/**
 * The index of the image a part should be shown as.
 *
 * Never `-1`. A dataset naming a channel that a later import renamed away resolves to the
 * first image rather than to nothing: the stored name is validated when it is written, and
 * forgiving when it is read, because the alternative is a blank screen for a preference
 * nobody remembers setting.
 */
export function preferredImageIndex(
  images: readonly Pick<ImageSummary, "channel">[],
  defaultChannel: string | null | undefined,
): number {
  if (!defaultChannel) return 0;
  const found = images.findIndex((image) => image.channel === defaultChannel);
  return found === -1 ? 0 : found;
}
