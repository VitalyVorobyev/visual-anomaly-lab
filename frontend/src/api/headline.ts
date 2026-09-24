/**
 * A run's one-line metric, whichever task it is (ADR-0039): the evaluator names it, and this
 * gives it a short label. Two runs of different tasks never share a column's meaning, so the
 * label always travels with the number.
 */

import type { ExperimentSummary } from "./client";

const SHORT: Record<string, string> = {
  sample_roc_auc: "AUROC",
  foreground_iou: "IoU",
  mean_iou: "mIoU",
};

export function headlineLabel(metric: string): string {
  return SHORT[metric] ?? metric;
}

/** `0.912 AUROC`, or `null` when the metric could not be computed — a dash, never a zero. */
export function formatHeadline(
  run: Pick<ExperimentSummary, "headline_metric" | "headline_value">,
): string | null {
  const value = run.headline_value;
  if (value === null || value === undefined) return null;
  return `${value.toFixed(3)} ${headlineLabel(run.headline_metric ?? "sample_roc_auc")}`;
}
