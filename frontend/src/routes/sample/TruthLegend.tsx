/**
 * The classes this sample's truth shows, each with the colour it is drawn in, and one line
 * saying where that truth comes from — or why there is nothing to draw.
 *
 * The colours are the dataset's taxonomy (the editor paints with the same ones), so they
 * arrive as data and are applied as a style, never as a design token.
 */

import { ErrorBox, Skeleton } from "@vitavision/lab-ui";

import type { ImageTruth } from "../../api/client";
import { legendClasses, truthSummary } from "./truthLayers";

export function TruthLegend({
  truths,
  pending,
  error,
}: {
  truths: readonly (ImageTruth | undefined)[];
  pending: boolean;
  error: Error | null;
}) {
  if (error) return <ErrorBox>{error.message}</ErrorBox>;
  if (pending) return <Skeleton className="h-4 w-40" />;
  const classes = legendClasses(truths);
  const boxes = truths.reduce((total, truth) => total + (truth?.boxes.length ?? 0), 0);
  return (
    <div className="flex flex-col gap-1.5" aria-label="Truth legend">
      {classes.length > 0 && (
        <ul className="flex flex-col gap-1">
          {classes.map((entry) => (
            <li key={entry.key} className="flex items-center gap-2 text-xs text-fg">
              <span
                aria-hidden
                className="size-2.5 shrink-0 rounded-[2px] ring-1 ring-line"
                style={{ backgroundColor: entry.color }}
              />
              <span className="min-w-0 truncate">{entry.name}</span>
            </li>
          ))}
        </ul>
      )}
      <p className="text-[11px] leading-4 text-fg-subtle">
        {truthSummary(truths)}
        {boxes > 0 && ` · ${boxes} ${boxes === 1 ? "box" : "boxes"}`}
      </p>
    </div>
  );
}
