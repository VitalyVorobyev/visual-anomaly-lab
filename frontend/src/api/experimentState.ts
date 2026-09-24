/** URL-backed state for the global and dataset-scoped experiment catalogues. */

import type { ExperimentStatus } from "./client";

export type ExperimentSort = "newest" | "oldest" | "name" | "method" | "status";

const STATUSES: readonly ExperimentStatus[] = ["draft", "training", "trained", "failed"];
const SORTS: readonly ExperimentSort[] = ["newest", "oldest", "name", "method", "status"];
const DAY = /^\d{4}-\d{2}-\d{2}$/;

export interface ExperimentListQuery {
  datasetId?: number | undefined;
  /** Any of these methods; empty or absent is every method. */
  modelTypes?: string[] | undefined;
  status?: ExperimentStatus | undefined;
  query?: string | undefined;
  /** Inclusive `YYYY-MM-DD` days, in UTC. */
  createdFrom?: string | undefined;
  createdTo?: string | undefined;
  sort?: ExperimentSort | undefined;
}

export interface ExperimentCatalogState {
  modelTypes: string[];
  status: ExperimentStatus | undefined;
  query: string;
  createdFrom: string | undefined;
  createdTo: string | undefined;
  sort: ExperimentSort;
}

export const EMPTY_EXPERIMENT_CATALOG: ExperimentCatalogState = {
  modelTypes: [],
  status: undefined,
  query: "",
  createdFrom: undefined,
  createdTo: undefined,
  sort: "newest",
};

export function readExperimentCatalogState(params: URLSearchParams): ExperimentCatalogState {
  return {
    modelTypes: [...new Set(params.getAll("method").map((value) => value.trim()).filter(Boolean))],
    status: readOneOf(params.get("status"), STATUSES),
    query: params.get("q")?.slice(0, 200) ?? "",
    createdFrom: readDay(params.get("from")),
    createdTo: readDay(params.get("to")),
    sort: readOneOf(params.get("sort"), SORTS) ?? "newest",
  };
}

export function writeExperimentCatalogState(state: ExperimentCatalogState): URLSearchParams {
  const params = new URLSearchParams();
  const query = state.query.trim();
  if (query) params.set("q", query);
  for (const method of state.modelTypes) params.append("method", method);
  if (state.status !== undefined) params.set("status", state.status);
  if (state.createdFrom !== undefined) params.set("from", state.createdFrom);
  if (state.createdTo !== undefined) params.set("to", state.createdTo);
  if (state.sort !== "newest") params.set("sort", state.sort);
  return params;
}

/** How many filters narrow the list — what "Clear N" counts. */
export function activeFilterCount(state: ExperimentCatalogState): number {
  return [
    state.query.trim() || undefined,
    state.modelTypes.length > 0 ? "methods" : undefined,
    state.status,
    state.createdFrom,
    state.createdTo,
  ].filter(Boolean).length;
}

export function toExperimentListQuery(
  state: ExperimentCatalogState,
  datasetId?: number,
): ExperimentListQuery {
  return {
    datasetId,
    modelTypes: state.modelTypes,
    status: state.status,
    query: state.query.trim() || undefined,
    createdFrom: state.createdFrom,
    createdTo: state.createdTo,
    sort: state.sort,
  };
}

function readDay(raw: string | null): string | undefined {
  return raw !== null && DAY.test(raw) ? raw : undefined;
}

function readOneOf<T extends string>(raw: string | null, allowed: readonly T[]): T | undefined {
  return allowed.find((value) => value === raw);
}
