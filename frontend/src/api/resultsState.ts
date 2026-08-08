/**
 * What the results view is showing, shared between the gallery and the open sample.
 *
 * The same reasoning as `browseState.ts`, for the same reason: the sample page has to step
 * through **the set the user is looking at**, and it is reached by a link from the gallery.
 * Carrying subset, filter and threshold in the query string is what lets the sample page
 * rebuild the *identical* `useThreshold` query — a cache hit rather than a second fetch —
 * and so know which sample comes next under the filter that is actually on screen.
 *
 * The overlay layers ride along for a different reason. They are a display preference, not
 * a filter, but a reader who has turned the segmentation on wants it still on for the next
 * sample and after a reload — the same argument that put the experiment's active tab in
 * the URL. They cost three characters each and make the view linkable.
 *
 * Every value is validated on the way in, so a hand-edited URL cannot put an outcome the
 * server never emits into a filter, or an out-of-range cut into a render request.
 */

import type { Subset } from "./client";

/** The verdict buckets the server tags rows with. `all` is the absence of a filter. */
export const OUTCOMES = ["tp", "fp", "tn", "fn", "unlabeled"] as const;
export type Outcome = (typeof OUTCOMES)[number];

const SUBSETS: readonly Subset[] = ["train", "val", "test"];

/** The mistakes, which is the filter anyone actually reaches for first. */
export const MISTAKE_OUTCOMES: readonly Outcome[] = ["fp", "fn"];

export type SortOrder = "score-desc" | "score-asc";

export interface ResultsState {
  subset: Subset | undefined;
  /** `undefined` means every outcome. */
  outcome: Outcome | undefined;
  /** `undefined` means "whatever the server suggested", which is the honest default. */
  threshold: number | undefined;
  sort: SortOrder;
  /** The colormapped heatmap: how anomalous, everywhere. */
  heatmap: boolean;
  /** The model's own segmentation: where, with an edge. */
  region: boolean;
  /** The ground-truth outline, where a mask exists. */
  truth: boolean;
  /**
   * Where the segmentation cuts, as a fraction of the **run-wide** map range.
   *
   * A fraction rather than a raw value so it means the same thing on every image of the
   * run, and survives being carried to a sample whose own peak is somewhere else entirely.
   */
  cut: number;
}

/**
 * Where the segmentation starts, as a fraction of the run's range.
 *
 * Below half deliberately. The run-wide high end is the maximum over every map's 99.9th
 * percentile (ADR-0019), so it is set by the single hottest image in the run — usually the
 * worst false positive. Every other image's peak sits below it, often far below, and a cut
 * at half the range is therefore a much stronger claim on a typical image than it sounds.
 * Measured on the reference run: the defective sample's region is 148 px at 0.3 and 18 px
 * at 0.5, which is a speck rather than a shape.
 *
 * A clean image still draws nothing at any cut, which is the property that matters — this
 * moves where a *hot* image's region starts, and never invents one on a cold image.
 */
export const DEFAULT_CUT = 0.3;

export const EMPTY_RESULTS: ResultsState = {
  subset: undefined,
  outcome: undefined,
  threshold: undefined,
  sort: "score-desc",
  heatmap: true,
  region: false,
  truth: true,
  cut: DEFAULT_CUT,
};

export function readResultsState(params: URLSearchParams): ResultsState {
  return {
    subset: readOneOf(params.get("subset"), SUBSETS),
    outcome: readOneOf(params.get("outcome"), OUTCOMES),
    threshold: readFloat(params.get("t")),
    sort: params.get("sort") === "score-asc" ? "score-asc" : "score-desc",
    heatmap: readFlag(params.get("map"), EMPTY_RESULTS.heatmap),
    region: readFlag(params.get("seg"), EMPTY_RESULTS.region),
    truth: readFlag(params.get("gt"), EMPTY_RESULTS.truth),
    cut: readFraction(params.get("cut")) ?? DEFAULT_CUT,
  };
}

/** Only non-default values are written, so an untouched view has a clean URL. */
export function writeResultsState(state: ResultsState): URLSearchParams {
  const params = new URLSearchParams();
  if (state.subset !== undefined) params.set("subset", state.subset);
  if (state.outcome !== undefined) params.set("outcome", state.outcome);
  if (state.threshold !== undefined) params.set("t", String(state.threshold));
  if (state.sort !== EMPTY_RESULTS.sort) params.set("sort", state.sort);
  if (state.heatmap !== EMPTY_RESULTS.heatmap) params.set("map", state.heatmap ? "1" : "0");
  if (state.region !== EMPTY_RESULTS.region) params.set("seg", state.region ? "1" : "0");
  if (state.truth !== EMPTY_RESULTS.truth) params.set("gt", state.truth ? "1" : "0");
  if (state.cut !== DEFAULT_CUT) params.set("cut", String(state.cut));
  return params;
}

/**
 * The segmentation cut in the map's own units.
 *
 * `null` when the run recorded no range — before anything is scored, and for a method that
 * emits no map. A cut has to come from the run-wide range rather than the image on screen:
 * derived per image it would be a different cut on every image, and two samples' regions
 * would not be comparable. That is the same mistake the run-wide range exists to prevent
 * for the heatmap (ADR-0019).
 */
export function cutValue(
  state: ResultsState,
  range: { low: number; high: number } | null | undefined,
): number | null {
  if (!range) return null;
  return range.low + (range.high - range.low) * state.cut;
}

function readFloat(raw: string | null): number | undefined {
  if (raw === null) return undefined;
  const value = Number(raw);
  return Number.isFinite(value) ? value : undefined;
}

function readFraction(raw: string | null): number | undefined {
  const value = readFloat(raw);
  return value !== undefined && value >= 0 && value <= 1 ? value : undefined;
}

function readFlag(raw: string | null, fallback: boolean): boolean {
  if (raw === "1") return true;
  if (raw === "0") return false;
  return fallback;
}

function readOneOf<T extends string>(raw: string | null, allowed: readonly T[]): T | undefined {
  return allowed.find((value) => value === raw);
}
