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

export function ChannelTabs({
  images,
  active,
  onSelect,
}: {
  images: ImageSummary[];
  active: number;
  onSelect: (index: number) => void;
}) {
  return (
    <div role="tablist" className="flex flex-wrap gap-1">
      {images.map((image, index) => (
        <button
          key={image.id}
          type="button"
          role="tab"
          aria-selected={index === active}
          onClick={() => onSelect(index)}
          className={`rounded px-2 py-1 text-xs ${
            index === active
              ? "bg-slate-900 text-white dark:bg-slate-100 dark:text-slate-900"
              : "bg-slate-100 text-slate-600 dark:bg-slate-800 dark:text-slate-300"
          }`}
        >
          {image.channel ?? "unassigned"}
        </button>
      ))}
    </div>
  );
}
