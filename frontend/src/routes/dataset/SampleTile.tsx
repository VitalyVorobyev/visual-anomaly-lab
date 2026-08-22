/**
 * One sample in the browse grid: a thumbnail you can open, and a box you can tick.
 *
 * The checkbox lives *beside* the link rather than inside it, which is the whole point of
 * the wrapper. Nested in the anchor it had to cancel the click to stop the navigation, and
 * `preventDefault()` on a checkbox is not inert: it cancels the activation behaviour, so
 * the browser restores the pre-click `checked` value once dispatch ends. React had already
 * flushed the state update and written `checked = true` *during* dispatch, so the browser
 * won by a moment and the box stayed empty until that input next rendered — a tick that
 * appeared one interaction late. Outside the anchor there is nothing to cancel.
 *
 * Selection is driven entirely from `onClick` rather than `onCheckedChange`, because the
 * grid extends ranges with shift and toggles with the platform modifier (`select` in
 * `DatasetRoute`), and a boolean callback cannot carry them. Keyboard activation
 * synthesises a click with its modifiers intact, so both input paths land in one place.
 */

import { Link } from "react-router";

import type { SampleSummary } from "../../api/client";
import { imageUrl } from "../../api/imageUrl";
import { Badge, Checkbox, cn, focusRing } from "@vitavision/lab-ui";
import type { Label } from "../../api/client";

const LABEL_TONE: Record<Label, "normal" | "defect" | "unlabeled"> = {
  normal: "normal",
  defect: "defect",
  unlabeled: "unlabeled",
};

/** The modifiers the grid's selection rules read, from a mouse or a keyboard. */
export type SelectModifiers = { shiftKey: boolean; metaKey: boolean; ctrlKey: boolean };

export function SampleTile({
  datasetId,
  sample,
  search,
  channelId,
  selected,
  onSelect,
}: {
  datasetId: number;
  sample: SampleSummary;
  search: string;
  /** The channel the grid is filtered to, if any. */
  channelId?: number | undefined;
  selected: boolean;
  onSelect: (event: SelectModifiers) => void;
}) {
  // Filtering by channel means "samples having an image in this channel", so the grid keeps
  // showing whole samples — but it must show *that* channel. Previewing the first image
  // instead made the filter look inert: the same bright-field thumbnails came back however
  // the rail was set. Falling back to the first image keeps a sample whose channel dropped
  // out of a re-import visible rather than blank.
  const cover = sample.images.find((image) => image.channel_id === channelId) ?? sample.images[0];

  return (
    <div className="relative">
      <Link
        // The browse filters travel with the link so the viewer can page through the same
        // set the grid is showing.
        to={`/datasets/${datasetId}/samples/${sample.id}${search ? `?${search}` : ""}`}
        onClick={(event) => {
          // A modified click selects instead of navigating; a plain click still opens.
          if (event.shiftKey || event.metaKey || event.ctrlKey) {
            event.preventDefault();
            onSelect(event);
          }
        }}
        className={cn(
          "flex h-full flex-col overflow-hidden rounded-control border transition-colors",
          selected ? "border-signal ring-2 ring-signal" : "border-line hover:border-line-strong",
          focusRing,
        )}
        title={`${sample.group_key}/${sample.external_id}`}
      >
        <div className="flex h-20 items-center justify-center bg-raised">
          {cover ? (
            <img
              // The grid never asks for a full-resolution decode.
              src={imageUrl(cover.id, "thumb")}
              alt=""
              loading="lazy"
              className="h-full w-full object-contain"
            />
          ) : (
            <span className="text-xs text-fg-subtle">no image</span>
          )}
        </div>
        <div className="flex items-center justify-between gap-1 px-1.5 py-1">
          <span className="truncate font-mono text-xs">{sample.external_id}</span>
          <span className="flex items-center gap-1">
            {/* Which channel is on screen when the grid is filtered to one; otherwise the
                channel count, which is whatever this sample has. */}
            <span className="truncate font-mono text-[10px] text-fg-subtle">
              {channelId === undefined ? `${sample.images.length}ch` : (cover?.channel ?? "—")}
            </span>
            <Badge tone={LABEL_TONE[sample.label]}>{sample.label.slice(0, 3)}</Badge>
          </span>
        </div>
      </Link>

      {/* The discoverable path to selection; the modifier keys are the fast one. Both end
          up in the same place. */}
      <span className="absolute top-1 left-1 z-10">
        <Checkbox
          checked={selected}
          aria-label={`Select ${sample.group_key}/${sample.external_id}`}
          onCheckedChange={() => undefined}
          onClick={onSelect}
        />
      </span>
    </div>
  );
}
