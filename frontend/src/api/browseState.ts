/**
 * The browse filters, shared between the grid and the open sample.
 *
 * They live in the URL rather than in component state for one concrete reason: the sample
 * viewer has to page through **the set the user is looking at**, and it is reached by a
 * link from the grid. Carrying the filters in the query string is what lets the viewer
 * rebuild the *same* `useSamples` query — a cache hit rather than a second fetch — and so
 * know which sample comes next. Back/forward and a shareable URL come along for free.
 *
 * Every value is validated on the way in, so a hand-edited URL cannot put a label the API
 * does not know into a request.
 */

import type { BulkLabelFilter, Label, Subset } from "./client";
import type { SampleQuery } from "./queryKeys";

/**
 * One page holds the whole of most datasets, so the page boundary is rarely reached —
 * which matters because crossing it while paging through samples costs a fetch.
 */
export const PAGE_SIZE = 200;

const LABELS: readonly Label[] = ["normal", "defect", "unlabeled"];
const SUBSETS: readonly Subset[] = ["train", "val", "test"];

/**
 * Fields are required-but-nullable rather than optional: every reader has to consider the
 * "no filter" case anyway, and a uniform shape keeps the two routes from drifting.
 */
export interface BrowseState {
  label: Label | undefined;
  channelId: number | undefined;
  splitId: number | undefined;
  subset: Subset | undefined;
  offset: number;
}

export const EMPTY_BROWSE: BrowseState = {
  label: undefined,
  channelId: undefined,
  splitId: undefined,
  subset: undefined,
  offset: 0,
};

export function readBrowseState(params: URLSearchParams): BrowseState {
  const splitId = readNumber(params.get("split"));
  return {
    label: readOneOf(params.get("label"), LABELS),
    channelId: readNumber(params.get("channel")),
    splitId,
    // A subset without a split is meaningless — the API says so, and dropping it here
    // stops a stale `subset=train` in the URL from silently narrowing an unsplit view.
    subset: splitId === undefined ? undefined : readOneOf(params.get("subset"), SUBSETS),
    offset: readNumber(params.get("offset")) ?? 0,
  };
}

/** Only non-default values are written, so an unfiltered browse has a clean URL. */
export function writeBrowseState(state: BrowseState): URLSearchParams {
  const params = new URLSearchParams();
  if (state.label !== undefined) params.set("label", state.label);
  if (state.channelId !== undefined) params.set("channel", String(state.channelId));
  if (state.splitId !== undefined) params.set("split", String(state.splitId));
  if (state.subset !== undefined && state.splitId !== undefined) {
    params.set("subset", state.subset);
  }
  if (state.offset > 0) params.set("offset", String(state.offset));
  return params;
}

/** The `useSamples` query these filters describe — identical on both routes, by construction. */
export function toSampleQuery(state: BrowseState): SampleQuery {
  return {
    label: state.label,
    channelId: state.channelId,
    splitId: state.splitId,
    subset: state.subset,
    limit: PAGE_SIZE,
    offset: state.offset,
  };
}

/**
 * The same filters as the bulk-label endpoint wants them.
 *
 * Paging is deliberately dropped: "label everything matching these filters" means the
 * whole matching set, not the page that happens to be on screen.
 */
export function toBulkFilters(state: BrowseState): BulkLabelFilter {
  return {
    label: state.label ?? null,
    channel_id: state.channelId ?? null,
    split_id: state.splitId ?? null,
    subset: state.subset ?? null,
  };
}

function readNumber(raw: string | null): number | undefined {
  if (raw === null) return undefined;
  const value = Number(raw);
  return Number.isInteger(value) && value >= 0 ? value : undefined;
}

function readOneOf<T extends string>(raw: string | null, allowed: readonly T[]): T | undefined {
  return allowed.find((value) => value === raw);
}
