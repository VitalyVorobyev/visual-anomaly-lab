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
import type { Outcome, ResultsState } from "../../api/resultsState";
import { MISTAKE_OUTCOMES } from "../../api/resultsState";
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

  // Filtering and ordering happen once, here, so the gallery's grid and the sample page's
  // prev/next cannot disagree about what "the next one" means.
  const shown = useMemo(() => {
    const filtered = all.filter((entry) => matches(entry.outcome, state));
    // `right - left` is descending; negate it for ascending. Getting this backwards puts
    // the *least* anomalous samples under a control that says "most anomalous", which is
    // a wrong answer wearing a correct label.
    const descending = state.sort !== "score-asc";
    return [...filtered].sort((left, right) =>
      descending ? right.score - left.score : left.score - right.score,
    );
  }, [all, state.outcome, state.mistakesOnly, state.sort]);

  return {
    shown,
    all,
    threshold,
    rationale: state.threshold === undefined ? results.data?.threshold_rationale : undefined,
    label: describeFilter(state, all.length),
    isPending: results.isPending || report.isPending,
    error: results.error ?? report.error,
  };
}

/** The one place an outcome is tested against the current filter. */
export function matches(outcome: string, state: ResultsState): boolean {
  if (state.mistakesOnly) return MISTAKE_OUTCOMES.includes(outcome as Outcome);
  return state.outcome === undefined || outcome === state.outcome;
}

function describeFilter(state: ResultsState, total: number): string {
  if (state.mistakesOnly) return "mistakes";
  if (state.outcome === undefined) return `${total} samples`;
  return OUTCOME_PLURAL[state.outcome] ?? state.outcome;
}
