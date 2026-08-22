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
  disabled = false,
  disabledReason = "Save your changes before switching channel",
}: {
  images: ImageSummary[];
  active: number;
  onSelect: (index: number) => void;
  /** For the annotation editor, where switching image-scoped truth would drop an edit. */
  disabled?: boolean;
  disabledReason?: string;
}) {
  // The position is the identity here, not the channel name: two images of one sample can
  // legitimately carry the same name, and an unassigned one carries none at all.
  return (
    <Tabs
      label="Channels"
      active={String(active)}
      onSelect={(id) => onSelect(Number(id))}
      items={images.map((image, index) => ({
        id: String(index),
        label: image.channel ?? "unassigned",
        disabled: disabled && index !== active,
        ...(disabled && index !== active ? { title: disabledReason } : {}),
      }))}
    />
  );
}
