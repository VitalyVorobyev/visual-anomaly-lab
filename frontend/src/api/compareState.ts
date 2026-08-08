/**
 * What the comparison screen is showing, carried in the query string.
 *
 * Same reasoning as `resultsState.ts`: a comparison is a *finding*, and a finding that
 * cannot be linked to is one that has to be reconstructed by hand every time somebody wants
 * to look at it again. Which runs, which subset, which operating point and which filter all
 * ride in the URL, so the screen survives a reload and the link survives the session.
 *
 * The operating point is state rather than a preference. It decides every confusion matrix
 * on the screen, and two readers looking at the same link must be looking at the same
 * numbers (ADR-0028).
 */

import type { MapScale, OperatingPoint, Subset } from "./client";
import type { Outcome } from "./resultsState";
import { DEFAULT_CUT, OUTCOMES } from "./resultsState";

export const COMPARE_VIEWS = ["metrics", "curves", "config", "samples"] as const;
export type CompareView = (typeof COMPARE_VIEWS)[number];

const SUBSETS: readonly Subset[] = ["train", "val", "test"];
const OPERATING_POINTS: readonly OperatingPoint[] = ["f1", "recall"];

/** Six columns of numbers is already a wide table; the backend refuses a seventh. */
export const MAX_RUNS = 6;

export const DEFAULT_RECALL_TARGET = 0.95;

export interface CompareState {
  /** The runs, in the order their columns appear. Order is meaningful and is preserved. */
  ids: number[];
  /** `undefined` lets the server choose the most test-like subset every run has scored. */
  subset: Subset | undefined;
  at: OperatingPoint;
  recallTarget: number;
  view: CompareView;
  /** Only the rows where the runs did not predict the same thing. */
  disagreeOnly: boolean;
  /**
   * Rows where **at least one** run tagged the sample this way.
   *
   * Per-run rather than per-row is the only thing it can be here: an outcome belongs to a
   * run's verdict, and the whole point of the table is that the verdicts differ. "Show me
   * the samples somebody called a false positive" is the question this answers.
   */
  outcome: Outcome | undefined;
  /* The layers of the side-by-side sample view, carried for the same reason the results
     screen carries its own: a reader who turned the segmentation on wants it still on for
     the next sample, and the view is worth linking to. */
  heatmap: boolean;
  region: boolean;
  truth: boolean;
  /**
   * Where every run's segmentation cuts, as a fraction of **that run's** range.
   *
   * The one number on this screen that is allowed to be shared, and only because it is not
   * in score units. A fraction resolves to a different value in every run — which is
   * exactly what makes it the same operating point in all of them (ADR-0028).
   */
  cut: number;
}

export const EMPTY_COMPARE: CompareState = {
  ids: [],
  subset: undefined,
  at: "f1",
  recallTarget: DEFAULT_RECALL_TARGET,
  view: "metrics",
  disagreeOnly: false,
  outcome: undefined,
  heatmap: true,
  region: false,
  truth: true,
  cut: DEFAULT_CUT,
};

/**
 * One run's cut in its own units, from the shared fraction.
 *
 * `null` when that run recorded no range — a method with no anomaly map, or one that has
 * not been scored. Deliberately not falling back to another run's range: that would draw
 * one method's segmentation at a cut derived from a different method's score distribution,
 * which is the fabricated comparability ADR-0028 exists to refuse.
 */
export function cutFor(state: CompareState, range: MapScale | null | undefined): number | null {
  if (!range) return null;
  return range.low + (range.high - range.low) * state.cut;
}

export function readCompareState(params: URLSearchParams): CompareState {
  return {
    ids: readIds(params.get("ids")),
    subset: readOneOf(params.get("subset"), SUBSETS),
    at: readOneOf(params.get("at"), OPERATING_POINTS) ?? EMPTY_COMPARE.at,
    recallTarget: readFraction(params.get("rt")) ?? DEFAULT_RECALL_TARGET,
    view: readOneOf(params.get("view"), COMPARE_VIEWS) ?? EMPTY_COMPARE.view,
    disagreeOnly: params.get("d") === "1",
    outcome: readOneOf(params.get("outcome"), OUTCOMES),
    heatmap: readFlag(params.get("map"), EMPTY_COMPARE.heatmap),
    region: readFlag(params.get("seg"), EMPTY_COMPARE.region),
    truth: readFlag(params.get("gt"), EMPTY_COMPARE.truth),
    cut: readFraction(params.get("cut")) ?? DEFAULT_CUT,
  };
}

/** Only non-default values are written, so an untouched comparison has a readable URL. */
export function writeCompareState(state: CompareState): URLSearchParams {
  const params = new URLSearchParams();
  if (state.ids.length > 0) params.set("ids", state.ids.join(","));
  if (state.subset !== undefined) params.set("subset", state.subset);
  if (state.at !== EMPTY_COMPARE.at) params.set("at", state.at);
  if (state.recallTarget !== DEFAULT_RECALL_TARGET) params.set("rt", String(state.recallTarget));
  if (state.view !== EMPTY_COMPARE.view) params.set("view", state.view);
  if (state.disagreeOnly) params.set("d", "1");
  if (state.outcome !== undefined) params.set("outcome", state.outcome);
  if (state.heatmap !== EMPTY_COMPARE.heatmap) params.set("map", state.heatmap ? "1" : "0");
  if (state.region !== EMPTY_COMPARE.region) params.set("seg", state.region ? "1" : "0");
  if (state.truth !== EMPTY_COMPARE.truth) params.set("gt", state.truth ? "1" : "0");
  if (state.cut !== DEFAULT_CUT) params.set("cut", String(state.cut));
  return params;
}

/**
 * Add or remove one run, keeping the order the reader chose.
 *
 * Appending rather than sorting: the columns read left to right in the order they were
 * picked, and a table that silently reorders itself when a column is added is one whose
 * comparisons have to be re-found after every click.
 */
export function toggleRun(ids: number[], id: number): number[] {
  return ids.includes(id) ? ids.filter((found) => found !== id) : [...ids, id];
}

/**
 * Why this run cannot join the current selection, or `null` when it can.
 *
 * A hard constraint, matching the server's: runs on a different dataset or a different
 * split are computed over different samples, and putting them in adjacent columns is
 * exactly the error a column layout invites (ADR-0028). Differing *preprocessing* is not
 * checked here — that one is legitimate, and the report warns about it loudly instead.
 */
export function refusalReason(
  candidate: { id: number; dataset_id: number; split_id: number },
  anchor: { dataset_id: number; split_id: number } | undefined,
  selected: number[],
): string | null {
  if (selected.includes(candidate.id)) return null;
  if (selected.length >= MAX_RUNS) {
    return `At most ${MAX_RUNS} runs at once — a wider table stops being read.`;
  }
  if (anchor === undefined) return null;
  if (candidate.dataset_id !== anchor.dataset_id) {
    return "A different dataset. Runs on different data are not comparable.";
  }
  if (candidate.split_id !== anchor.split_id) {
    return "A different split of this dataset, so the numbers cover different samples.";
  }
  return null;
}

function readIds(raw: string | null): number[] {
  if (raw === null) return [];
  const seen = new Set<number>();
  for (const part of raw.split(",")) {
    const value = Number(part);
    // A hand-edited URL cannot put a duplicate or a non-number into a request the server
    // would answer with a 422 the reader did not cause.
    if (Number.isInteger(value) && value > 0) seen.add(value);
  }
  return [...seen].slice(0, MAX_RUNS);
}

function readFraction(raw: string | null): number | undefined {
  if (raw === null) return undefined;
  const value = Number(raw);
  return Number.isFinite(value) && value > 0 && value <= 1 ? value : undefined;
}

function readFlag(raw: string | null, fallback: boolean): boolean {
  if (raw === "1") return true;
  if (raw === "0") return false;
  return fallback;
}

function readOneOf<T extends string>(raw: string | null, allowed: readonly T[]): T | undefined {
  return allowed.find((value) => value === raw);
}
