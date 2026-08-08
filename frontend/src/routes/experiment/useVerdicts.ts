/**
 * The ordered, filtered set of samples the results view is showing.
 *
 * One hook, used by the gallery and by the open sample, so "what comes next" means the
 * same thing in both. The sample page rebuilds it from the URL rather than being handed
 * it, which makes stepping through a filtered set a cache hit instead of a second fetch.
 *
 * **The threshold is a server round trip and the classification comes back with it**
 * (§12 rule 3). Nothing here re-applies `score >= threshold`; the server tags every row
 * with its outcome, and this only sorts and filters what it was given. Holding that rule
 * in TypeScript as well as Python is how the two quietly drift apart.
 */

import { useMemo } from "react";

import type { SampleVerdict } from "../../api/client";
import type { ResultsState } from "../../api/resultsState";
import { useResults, useThreshold } from "../../hooks/useExperiments";

export interface Verdicts {
  /** Ordered and filtered exactly as the gallery draws them. */
  shown: SampleVerdict[];
  /** Every classified row, before the outcome filter. */
  all: SampleVerdict[];
  /** The threshold actually in force, suggested or chosen. */
  threshold: number;
  /** Why that threshold, when it is the server's suggestion rather than a choice. */
  rationale: string | undefined;
  /** A short description of the current filter, for a "12 of 270 · false negatives" line. */
  label: string;
  isPending: boolean;
  error: Error | null;
}

const OUTCOME_PLURAL: Record<string, string> = {
  tp: "true positives",
  tn: "true negatives",
  fp: "false positives",
  fn: "false negatives",
  unlabeled: "unlabeled",
};

export function useVerdicts(
  experimentId: number | undefined,
  state: ResultsState,
): Verdicts {
  // The suggested threshold and its rationale, which is also where the score range comes
  // from. Requested even when a threshold is chosen: the rationale is worth showing beside
  // a choice that overrode it.
  const results = useResults(experimentId, state.subset);
  const threshold = state.threshold ?? results.data?.suggested_threshold ?? 0;
  const report = useThreshold(experimentId, state.subset, threshold);

  const all = useMemo(() => report.data?.samples ?? [], [report.data]);

  const shown = useMemo(() => {
    const filtered =
      state.outcome === undefined ? all : all.filter((entry) => entry.outcome === state.outcome);
    const direction = state.sort === "score-asc" ? 1 : -1;
    return [...filtered].sort((left, right) => direction * (right.score - left.score));
  }, [all, state.outcome, state.sort]);

  return {
    shown,
    all,
    threshold,
    rationale: state.threshold === undefined ? results.data?.threshold_rationale : undefined,
    label:
      state.outcome === undefined
        ? `${all.length} samples`
        : (OUTCOME_PLURAL[state.outcome] ?? state.outcome),
    isPending: results.isPending || report.isPending,
    error: results.error ?? report.error,
  };
}
