/**
 * "Check 24": the sampled crops of one recipe, failures first, each opening on the live stage.
 *
 * The live stage answers "what does this image become"; this answers "where does it fail",
 * which one image cannot. A card is one button — the whole card is the target, and there is
 * no control inside it to nest.
 */

import { Badge, cn, focusRing } from "@vitavision/lab-ui";

import type { RegionPreparationEntry } from "../../api/client";
import { imageUrl } from "../../api/imageUrl";
import { failuresFirst } from "./recipe";

export function CheckGrid({
  entries,
  activeImageId,
  onOpen,
}: {
  entries: RegionPreparationEntry[];
  activeImageId: number | undefined;
  onOpen: (entry: RegionPreparationEntry) => void;
}) {
  return (
    <div className="grid grid-cols-2 gap-3 md:grid-cols-3 2xl:grid-cols-4">
      {failuresFirst(entries).map((entry) => (
        <CheckCard
          key={entry.image_id}
          entry={entry}
          active={entry.image_id === activeImageId}
          onOpen={() => onOpen(entry)}
        />
      ))}
    </div>
  );
}

function CheckCard({
  entry,
  active,
  onOpen,
}: {
  entry: RegionPreparationEntry;
  active: boolean;
  onOpen: () => void;
}) {
  const transform = entry.transform;
  const failed = entry.status !== "succeeded" || !transform;
  return (
    <button
      type="button"
      onClick={onOpen}
      aria-label={`Open image ${entry.image_id} on the stage`}
      aria-current={active ? "true" : undefined}
      className={cn(
        "overflow-hidden rounded-panel border bg-surface text-left transition-colors hover:border-signal",
        active ? "border-signal" : failed ? "border-defect" : "border-line",
        focusRing,
      )}
    >
      <div className="relative aspect-[4/3] bg-raised">
        {!failed ? (
          <>
            <img
              src={imageUrl(entry.image_id, "preview")}
              alt=""
              loading="lazy"
              className="absolute inset-0 h-full w-full object-contain"
            />
            <svg
              viewBox={`0 0 ${transform.source_width} ${transform.source_height}`}
              preserveAspectRatio="xMidYMid meet"
              className="pointer-events-none absolute inset-0 h-full w-full text-signal"
              aria-hidden
            >
              <rect
                x={transform.crop_left}
                y={transform.crop_top}
                width={transform.crop_right - transform.crop_left}
                height={transform.crop_bottom - transform.crop_top}
                fill="none"
                stroke="currentColor"
                strokeWidth={1.5}
                vectorEffect="non-scaling-stroke"
              />
            </svg>
          </>
        ) : (
          <div className="grid h-full place-items-center px-4 text-center text-xs text-defect">
            {entry.error ?? "Preparation failed"}
          </div>
        )}
      </div>
      <div className="flex items-center justify-between gap-2 px-2.5 py-2">
        <span className="font-mono text-xs text-fg">image {entry.image_id}</span>
        <span className="flex items-center gap-1.5">
          {entry.extractor_confidence !== null && entry.extractor_confidence !== undefined && (
            <span className="font-mono text-[10px] text-fg-subtle">
              {entry.extractor_confidence.toFixed(2)}
            </span>
          )}
          <Badge tone={failed ? "defect" : "normal"}>{failed ? "failed" : "ok"}</Badge>
        </span>
      </div>
    </button>
  );
}
