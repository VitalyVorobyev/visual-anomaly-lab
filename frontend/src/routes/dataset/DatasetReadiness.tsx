/**
 * One line beside "New experiment" saying whether pressing it can lead anywhere yet.
 *
 * A run needs a built region profile and a split. Before this, the reader learned that only
 * inside the create form, and the form's links out to fix it threw the form away. Now the
 * band says it first, in the order the steps have to be done, and each step is the link that
 * does it. Held to one line with no wrap, because the band's height is fixed by construction.
 */

import { Check, CircleDashed } from "lucide-react";
import { Link } from "react-router";

import { cn, focusRing } from "@vitavision/lab-ui";

import { useDatasetReadiness, type ReadinessStep } from "../../hooks/useDatasetReadiness";

const STEP: Record<ReadinessStep, { label: string; path: string }> = {
  prepare: { label: "Build a region profile", path: "prepare" },
  split: { label: "Make a split", path: "splits" },
};

const LINK = cn("rounded-sm transition-colors hover:text-signal", focusRing);

export function DatasetReadiness({ datasetId }: { datasetId: number }) {
  const readiness = useDatasetReadiness(datasetId);
  if (!readiness.known) return null;

  if (readiness.missing.length === 0) {
    return (
      <p
        aria-label="Readiness"
        className="flex items-center gap-1.5 whitespace-nowrap text-xs text-fg-muted"
      >
        <Check aria-hidden className="size-3.5 text-normal" />
        Ready to train
        <span aria-hidden className="text-fg-subtle">·</span>
        <Link to={`/datasets/${datasetId}/experiments`} className={LINK}>
          {readiness.runs === 1 ? "1 run" : `${readiness.runs} runs`}
        </Link>
      </p>
    );
  }

  return (
    <nav
      aria-label="Before training"
      className="flex items-center gap-2 whitespace-nowrap text-xs text-fg-muted"
    >
      <span>Before training:</span>
      {readiness.missing.map((step, index) => (
        <Link
          key={step}
          to={`/datasets/${datasetId}/${STEP[step].path}`}
          className={cn(LINK, "flex items-center gap-1 font-medium text-warn")}
        >
          <CircleDashed aria-hidden className="size-3.5" />
          {index + 1}. {STEP[step].label}
        </Link>
      ))}
    </nav>
  );
}
