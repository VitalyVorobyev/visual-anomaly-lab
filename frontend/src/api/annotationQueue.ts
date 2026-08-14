/**
 * What one unit of annotation work is.
 *
 * A dataset annotates either each image or each whole sample (ADR-0036), and that decides
 * the shape of the queue, not just the shape of a save: under sample scope one completion
 * writes the same mask to every image of the part, so listing the part's three channels as
 * three cards would offer the same job three times and make "next" advance through a
 * document that is already done.
 *
 * Shared by the queue grid and the editor's traversal rather than written twice, because
 * the two must agree on what "item 4 of 92" counts — an editor that indexed images into a
 * queue of samples would jump to the wrong part at the first multi-channel sample.
 */

import type { ImageSummary, SampleSummary } from "./client";

export interface QueueUnit {
  sample: SampleSummary;
  /** The image the card previews and the editor opens on. */
  image: ImageSummary;
}

export function queueUnits(items: SampleSummary[], perSample: boolean): QueueUnit[] {
  return items.flatMap((sample) =>
    // A sample with no images contributes nothing under either scope, which is why this is
    // a slice rather than an index: there is no unit of work without something to look at.
    (perSample ? sample.images.slice(0, 1) : sample.images).map((image) => ({ sample, image })),
  );
}
