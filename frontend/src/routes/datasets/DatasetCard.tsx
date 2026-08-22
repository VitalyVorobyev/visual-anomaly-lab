/**
 * One dataset, as a picture and a sentence.
 *
 * The card used to carry three lines of numbers under its name — the label run, then
 * `1100 samples · 1100 images · csv_table`, then an absolute path — and none of them told
 * you what the dataset *was*. `candle` is a VisA class; nothing on the card said whether
 * that meant tea lights, wax pellets or a production line. A thumbnail answers that in the
 * time it takes to look, and the counts are one click away in the band, where they are
 * needed.
 *
 * So the card states what a catalogue entry is for: recognise it, and open it. The delete
 * and edit actions live in a corner that appears on hover or keyboard focus, so the resting
 * grid is a wall of pictures rather than a wall of buttons.
 */

import { Pencil, Trash2 } from "lucide-react";
import { ImageOff } from "lucide-react";
import { Link } from "react-router";

import type { DatasetSummary } from "../../api/client";
import { imageUrl } from "../../api/imageUrl";
import { Button, cn, focusRing } from "@vitavision/lab-ui";

export function DatasetCard({
  dataset,
  onEdit,
  onDelete,
}: {
  dataset: DatasetSummary;
  onEdit: (dataset: DatasetSummary) => void;
  onDelete: (dataset: DatasetSummary) => void;
}) {
  return (
    <li className="group relative">
      <Link
        to={`/datasets/${dataset.id}`}
        className={cn(
          "flex h-full flex-col overflow-hidden rounded-panel border border-line bg-surface",
          "transition-colors hover:border-line-strong",
          focusRing,
        )}
      >
        <Cover dataset={dataset} />
        <div className="flex min-h-0 flex-col gap-1 px-3.5 py-3">
          <h3 className="truncate text-sm font-medium tracking-tight text-fg">{dataset.name}</h3>
          {dataset.description ? (
            // Two lines is the budget: enough for a sentence, and a fixed ceiling so one
            // wordy entry cannot make its whole row taller than the rest.
            <p className="line-clamp-2 text-xs leading-relaxed text-fg-muted">
              {dataset.description}
            </p>
          ) : (
            <p className="text-xs text-fg-subtle italic">No description yet</p>
          )}
        </div>
      </Link>

      {/* Outside the link, so a click on either does not also open the dataset. Revealed on
          hover, and on focus so it is reachable without a pointer. */}
      <div
        className={cn(
          "absolute top-2 right-2 flex items-center gap-1 rounded-control bg-overlay/90 p-0.5 backdrop-blur-sm",
          "opacity-0 transition-opacity group-hover:opacity-100 focus-within:opacity-100",
        )}
      >
        <Button
          variant="ghost"
          size="sm"
          aria-label={`Edit ${dataset.name}`}
          icon={<Pencil />}
          onClick={() => onEdit(dataset)}
        />
        <Button
          variant="ghost"
          size="sm"
          aria-label={`Delete ${dataset.name}`}
          icon={<Trash2 />}
          onClick={() => onDelete(dataset)}
        />
      </div>
    </li>
  );
}

/**
 * The dataset's own first normal image, at thumbnail tier.
 *
 * `aspect-[4/3]` with `object-cover` rather than letting the image size the card: a
 * catalogue of mixed aspect ratios has to align on a grid, and a row whose height is set
 * by whichever dataset happens to be portrait is not a grid. `loading="lazy"` because a
 * hundred datasets is a hundred requests otherwise, and only the top rows are seen.
 */
function Cover({ dataset }: { dataset: DatasetSummary }) {
  if (dataset.cover_image_id === null) {
    return (
      <div className="grid aspect-[4/3] w-full place-items-center border-b border-line bg-raised">
        <ImageOff className="size-5 text-fg-subtle" aria-hidden />
      </div>
    );
  }

  return (
    <img
      src={imageUrl(dataset.cover_image_id, "thumb")}
      alt=""
      loading="lazy"
      draggable={false}
      className="aspect-[4/3] w-full border-b border-line bg-canvas object-cover"
    />
  );
}
