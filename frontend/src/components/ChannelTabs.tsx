/**
 * Tabs for a sample's channels.
 *
 * Rendered from the sample's own image list. There is no constant here for how many
 * channels there should be, no padding of a short list and no special case for a long
 * one — a two-channel capture group renders through exactly this code, which is the UI
 * half of "channel count is data, never schema" (ADR-0005, §12).
 *
 * An image with no channel is labelled `unassigned` rather than hidden: a dataset whose
 * directory names the matcher did not recognize still has to be browsable.
 */

import type { ImageSummary } from "../api/client";
import { Tabs } from "@vitavision/lab-ui";

export function ChannelTabs({
  images,
  active,
  onSelect,
  label = "Channels",
  unavailable,
}: {
  images: ImageSummary[];
  active: number;
  onSelect: (index: number) => void;
  /** What this strip is choosing, for a reader who cannot see two strips at once. */
  label?: string;
  /** One index this strip may not choose, and why. */
  unavailable?: { index: number; reason: string };
}) {
  // The position is the identity here, not the channel name: two images of one sample can
  // legitimately carry the same name, and an unassigned one carries none at all.
  //
  // Nothing is disabled for being *busy*. The annotation editor used to lock the other
  // channels while its draft was dirty, which read as a broken control: the work was one
  // keystroke away from being safe and the editor knew it. It saves and then navigates.
  // `unavailable` is a different claim — the editor's second strip cannot choose the channel
  // already in the first pane, because that would show the same photograph twice — and it
  // carries its reason so the tab explains itself rather than merely refusing.
  //
  // `flex-nowrap` overrides the primitive's default, which is right for a `Tabs` that owns
  // its width and wrong for one inside a horizontal scroller: wrapping is how a strip that
  // was supposed to half-scroll instead grew to three lines inside a 44 px row and was
  // clipped by it. A channel strip gives way by scrolling, and only scrolling.
  return (
    <Tabs
      className="flex-nowrap"
      label={label}
      active={String(active)}
      onSelect={(id) => onSelect(Number(id))}
      items={images.map((image, index) => ({
        id: String(index),
        label: image.channel ?? "unassigned",
        ...(unavailable?.index === index
          ? { disabled: true, title: unavailable.reason }
          : {}),
      }))}
    />
  );
}
