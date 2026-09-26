/**
 * Small pictures of samples, for a choice made in front of images.
 *
 * Figures, not links: a guided step is a decision, and a strip of tab stops that each leave
 * the run would bury its one control. Where truth has a shape it is drawn over the image —
 * the defect's or the class's own outline from `GET /api/images/{id}/mask`, source-sized, so
 * it registers with the thumbnail under `object-contain` — and an image without one simply
 * shows none.
 */

import { cn, Skeleton } from "@vitavision/lab-ui";
import { Children, useState, type ReactNode } from "react";

import type { SampleSummary } from "../../api/client";
import { preferredImageIndex } from "../../api/defaultChannel";
import { classMaskUrl, imageUrl, maskUrl } from "../../api/imageUrl";
import { useSample } from "../../hooks/useCatalog";

/** Which outline to lay over a thumbnail: the anomaly truth, one class's, or none. */
export type Outline = { kind: "defect" } | { kind: "class"; key: string } | undefined;

export function SampleThumb({
  sample,
  defaultChannel,
  outline,
  caption,
  size = "md",
}: {
  sample: SampleSummary;
  defaultChannel?: string | null | undefined;
  outline?: Outline;
  /** Under the picture; the sample's own id when omitted. */
  caption?: string;
  size?: "sm" | "md" | undefined;
}) {
  const image = sample.images[preferredImageIndex(sample.images, defaultChannel)];
  const [outlineMissing, setOutlineMissing] = useState(false);
  const overlay =
    image === undefined || outline === undefined || outlineMissing
      ? undefined
      : outline.kind === "defect"
        ? maskUrl(image.id)
        : classMaskUrl(image.id, outline.key);
  return (
    <figure className="flex min-w-0 flex-col gap-1">
      <div
        className={cn(
          "relative overflow-hidden rounded-control border border-line bg-raised",
          size === "sm" ? "h-16 w-20" : "h-24 w-28",
        )}
      >
        {image ? (
          <>
            <img
              src={imageUrl(image.id, "thumb")}
              alt={`${sample.group_key}/${sample.external_id}`}
              loading="lazy"
              draggable={false}
              className="absolute inset-0 h-full w-full object-contain"
            />
            {overlay && (
              // The outline is drawn at the source's size; the thumbnail is its reduction,
              // so the same `object-contain` box lays one exactly over the other.
              <img
                src={overlay}
                alt=""
                aria-hidden
                loading="lazy"
                draggable={false}
                onError={() => setOutlineMissing(true)}
                className="absolute inset-0 h-full w-full object-contain"
              />
            )}
          </>
        ) : (
          <span className="grid h-full place-items-center text-[10px] text-fg-subtle">no image</span>
        )}
      </div>
      <figcaption
        className={cn(
          "truncate font-mono text-[10px] text-fg-subtle",
          size === "sm" ? "w-20" : "w-28",
        )}
        title={`${sample.group_key}/${sample.external_id}`}
      >
        {caption ?? sample.external_id}
      </figcaption>
    </figure>
  );
}

/** A sample known only by its id — a split's example — read, then drawn as above. */
export function SampleThumbById({
  datasetId,
  sampleId,
  defaultChannel,
  outline,
  size,
}: {
  datasetId: number;
  sampleId: number;
  defaultChannel?: string | null | undefined;
  outline?: Outline;
  size?: "sm" | "md" | undefined;
}) {
  const sample = useSample(datasetId, sampleId);
  if (!sample.data) {
    return <Skeleton className={size === "sm" ? "h-16 w-20" : "h-24 w-28"} />;
  }
  return (
    <SampleThumb
      sample={sample.data}
      defaultChannel={defaultChannel}
      outline={outline}
      size={size}
    />
  );
}

/** A row of thumbnails, or its skeleton, or a sentence saying there is nothing to show. */
export function ThumbStrip({
  pending,
  empty,
  label,
  children,
}: {
  pending: boolean;
  empty: string | null;
  label: string;
  children: ReactNode;
}) {
  if (pending) {
    return (
      <div className="flex gap-2" aria-busy>
        {[0, 1, 2, 3].map((index) => (
          <Skeleton key={index} className="h-24 w-28" />
        ))}
      </div>
    );
  }
  if (empty !== null) return <p className="text-xs text-fg-subtle">{empty}</p>;
  return (
    <ul aria-label={label} className="flex flex-wrap gap-2">
      {Children.map(children, (child) => (
        <li>{child}</li>
      ))}
    </ul>
  );
}
