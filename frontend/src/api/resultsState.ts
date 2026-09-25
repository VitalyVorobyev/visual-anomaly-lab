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
 * The active tab rides along for the same reason the layers do, and it fixes a specific
 * bug: the gallery's tile link, the sample page's "← Results" link and its prev/next were
 * all built from `writeResultsState`, which knew nothing about tabs — so every one of them
 * dropped `tab=samples` and landed the reader on Overview. Making the tab part of this
 * state fixes all three at once, and any future link built the same way is correct by
 * construction rather than by remembering.
 *
 * What counts as a default is the task's (`resultsDefaults`). An anomaly run opens on its
 * heatmap; a supervised segmentation run's answer is its label map, and a heatmap over it hides
 * the thing the run exists to show. Reading and writing both take the task, so an untouched
 * view's URL stays clean under either set of defaults and a URL written under one reads back
 * as the same state under the same task.
 *
 * Every value is validated on the way in, so a hand-edited URL cannot put an outcome the
 * server never emits into a filter, or an out-of-range cut into a render request.
 */

import type { Subset, Task } from "./client";
import type { TabId } from "./experimentTabs";
import { parseTab } from "./experimentTabs";

/**
 * The verdict buckets the server tags rows with. `all` is the absence of a filter.
 *
 * Two vocabularies, disjoint: an anomaly run is `tp`/`fp`/`tn`/`fn` at a threshold (the
 * threshold report); a segmentation run is `hit`/`low_iou`/`miss`/`false_presence`/
 * `correct_absence` against its class truth (ADR-0040), and a supervised one adds
 * `false_class`, a class predicted on a sample that does not show it (ADR-0039). A detection
 * run reuses `hit`/`miss`/`false_presence`/`correct_absence` for its boxes at the run's cut
 * and adds `mixed`, a sample with both a missed box and a box on nothing. Disjoint is what
 * lets one URL parameter and one "mistakes" set serve every task.
 */
export const OUTCOMES = [
  "tp",
  "fp",
  "tn",
  "fn",
  "hit",
  "low_iou",
  "miss",
  "false_class",
  "false_presence",
  "mixed",
  "correct_absence",
  "unlabeled",
] as const;
export type Outcome = (typeof OUTCOMES)[number];

const SUBSETS: readonly Subset[] = ["train", "val", "test"];

/** The mistakes, which is the filter anyone actually reaches for first. */
export const MISTAKE_OUTCOMES: readonly Outcome[] = [
  "fp",
  "fn",
  "miss",
  "false_class",
  "false_presence",
  "low_iou",
  "mixed",
];

export type SortOrder = "score-desc" | "score-asc";

export interface ResultsState {
  /**
   * The experiment view this state belongs to, so a link built from it comes back to the
   * tab it left. Not a filter and not a display preference — a piece of *where you were*,
   * which is exactly what the round trip to a sample and back was losing.
   */
  tab: TabId;
  subset: Subset | undefined;
  /** `undefined` means every outcome. */
  outcome: Outcome | undefined;
  /**
   * False positives and false negatives together — the filter anyone reaches for first,
   * and the one a ranked list makes hardest to assemble by hand.
   *
   * A separate flag rather than letting `outcome` hold a set: every other filter is one
   * bucket, and widening the stored shape to express the single case that is two of them
   * would make every reader handle a list where a value will do. When set it wins.
   */
  mistakesOnly: boolean;
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
   * The map's peak, with the window the localization verdict was decided in.
   *
   * Off by default, unlike the heatmap and the outline. It answers a narrower question —
   * *why* this image was judged localized or off target — and a marker drawn on every
   * sample would compete with the layers that answer the first question.
   */
  peak: boolean;
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
 * percentile (handbook diagnostics.md), so it is set by the single hottest image in the run — usually the
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
  tab: "overview",
  subset: undefined,
  outcome: undefined,
  mistakesOnly: false,
  threshold: undefined,
  sort: "score-desc",
  heatmap: true,
  region: false,
  truth: true,
  peak: false,
  cut: DEFAULT_CUT,
};

/** The layers a view opens with — the only part of the state whose default is the task's. */
type OverlayDefaults = Pick<ResultsState, "heatmap" | "region" | "truth" | "peak">;

const MAP_FIRST: OverlayDefaults = { heatmap: true, region: false, truth: true, peak: false };

/**
 * Which layers each task opens on.
 *
 * A map-producing task opens on its map: the anomaly heatmap, and a few-shot run's foreground
 * probability, of which its prediction is only a cut. A supervised segmentation run opens on
 * its label map with the truth beside it; its foreground map is a secondary question and,
 * drawn first, covers the answer. A detection run opens the same way on its boxes: the
 * predictions and the truth they are matched against, with no heatmap over them.
 */
const ANSWER_FIRST: OverlayDefaults = { heatmap: false, region: true, truth: true, peak: false };

const OVERLAY_DEFAULTS: Record<Task, OverlayDefaults> = {
  anomaly: MAP_FIRST,
  few_shot_segmentation: MAP_FIRST,
  semantic_segmentation: ANSWER_FIRST,
  object_detection: ANSWER_FIRST,
};

/**
 * The untouched view of a run of this task. `undefined` — the experiment not loaded yet — is
 * the anomaly one; the state is re-read from the URL once the task is known, so nothing
 * written before then outlives it.
 */
export function resultsDefaults(task: Task | undefined): ResultsState {
  return { ...EMPTY_RESULTS, ...OVERLAY_DEFAULTS[task ?? "anomaly"] };
}

export function readResultsState(params: URLSearchParams, task: Task | undefined): ResultsState {
  const defaults = resultsDefaults(task);
  return {
    tab: parseTab(params.get("tab")),
    subset: readOneOf(params.get("subset"), SUBSETS),
    outcome: readOneOf(params.get("outcome"), OUTCOMES),
    mistakesOnly: params.get("outcome") === "mistakes",
    threshold: readFloat(params.get("t")),
    sort: params.get("sort") === "score-asc" ? "score-asc" : "score-desc",
    heatmap: readFlag(params.get("map"), defaults.heatmap),
    region: readFlag(params.get("seg"), defaults.region),
    truth: readFlag(params.get("gt"), defaults.truth),
    peak: readFlag(params.get("pk"), defaults.peak),
    cut: readFraction(params.get("cut")) ?? DEFAULT_CUT,
  };
}

/**
 * Only values that differ from the task's defaults are written, so an untouched view has a
 * clean URL. Written and read back under the same task, a state comes back unchanged.
 */
export function writeResultsState(state: ResultsState, task: Task | undefined): URLSearchParams {
  const defaults = resultsDefaults(task);
  const params = new URLSearchParams();
  if (state.tab !== defaults.tab) params.set("tab", state.tab);
  if (state.subset !== undefined) params.set("subset", state.subset);
  if (state.mistakesOnly) params.set("outcome", "mistakes");
  else if (state.outcome !== undefined) params.set("outcome", state.outcome);
  if (state.threshold !== undefined) params.set("t", String(state.threshold));
  if (state.sort !== defaults.sort) params.set("sort", state.sort);
  if (state.heatmap !== defaults.heatmap) params.set("map", state.heatmap ? "1" : "0");
  if (state.region !== defaults.region) params.set("seg", state.region ? "1" : "0");
  if (state.truth !== defaults.truth) params.set("gt", state.truth ? "1" : "0");
  if (state.peak !== defaults.peak) params.set("pk", state.peak ? "1" : "0");
  if (state.cut !== DEFAULT_CUT) params.set("cut", String(state.cut));
  return params;
}

/**
 * The state with its subset made explicit: the one the URL names, else the last one scored.
 *
 * An absent subset is not a neutral default on the server — `GET …/results` with no subset
 * ranks **every** scored subset together, with a threshold suggested over the union. The
 * Samples tab used to send exactly that while Overview and Benchmark showed `test`, so the
 * gallery was ranking a different population from the confusion matrix beside it. Resolving
 * once, before any tab reads the state, gives the three tabs and every link built from them
 * one subset, and puts it in the links so the sample page asks the identical question.
 */
export function resolveSubset(state: ResultsState, scored: readonly Subset[]): ResultsState {
  if (state.subset !== undefined && scored.includes(state.subset)) return state;
  return { ...state, subset: scored.at(-1) };
}

/**
 * The segmentation cut in the map's own units.
 *
 * `null` when the run recorded no range — before anything is scored, and for a method that
 * emits no map. A cut has to come from the run-wide range rather than the image on screen:
 * derived per image it would be a different cut on every image, and two samples' regions
 * would not be comparable. That is the same mistake the run-wide range exists to prevent
 * for the heatmap (handbook diagnostics.md).
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
