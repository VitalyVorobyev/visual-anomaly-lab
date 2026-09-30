/**
 * A sample's channels as the items of a `Tabs` strip (`@vitavision/ui`).
 *
 * Rendered from the sample's own image list. There is no constant here for how many
 * channels there should be, no padding of a short list and no special case for a long
 * one — a two-channel capture group goes through exactly this code, which is the UI half
 * of "channel count is data, never schema" (ADR-0041, §12).
 *
 * An image with no channel is labelled `unassigned` rather than hidden: a dataset whose
 * directory names the matcher did not recognize still has to be browsable.
 *
 * The position is the identity, not the channel name: two images of one sample can
 * legitimately carry the same name, and an unassigned one carries none at all. So a tab's
 * id is its index as a string, and a strip reads it back with `Number(id)`.
 *
 * Nothing is disabled for being *busy*. The annotation editor used to lock the other
 * channels while its draft was dirty, which read as a broken control: the work was one
 * keystroke away from being safe and the editor knew it. It saves and then navigates.
 * `unavailable` is a different claim — the editor's second strip cannot choose the channel
 * already in the first pane, because that would show the same photograph twice — and it
 * carries its reason so the tab explains itself rather than merely refusing.
 *
 * A strip of these sits in a horizontal scroller, so it is rendered with
 * `className="flex-nowrap"`: wrapping is how a strip that was supposed to half-scroll
 * instead grew to three lines inside a 44 px row and was clipped by it. A channel strip
 * gives way by scrolling, and only scrolling.
 */

import type { TabItem } from "@vitavision/ui";

import type { ImageSummary } from "./client";

export function channelTabItems(
  images: readonly Pick<ImageSummary, "channel">[],
  /** One index the strip may not choose, and why. */
  unavailable?: { index: number; reason: string },
): TabItem<string>[] {
  return images.map((image, index) => ({
    id: String(index),
    label: image.channel ?? "unassigned",
    ...(unavailable?.index === index ? { disabled: true, title: unavailable.reason } : {}),
  }));
}
