/** URL-backed state for the global and dataset-scoped experiment catalogues. */

import type { ExperimentStatus } from "./client";

export type ExperimentSort = "newest" | "oldest" | "name";

const STATUSES: readonly ExperimentStatus[] = ["draft", "training", "trained", "failed"];
const SORTS: readonly ExperimentSort[] = ["newest", "oldest", "name"];

export interface ExperimentListQuery {
  datasetId?: number | undefined;
  modelType?: string | undefined;
  status?: ExperimentStatus | undefined;
  query?: string | undefined;
  sort?: ExperimentSort | undefined;
}

export interface ExperimentCatalogState {
  modelType: string | undefined;
  status: ExperimentStatus | undefined;
  query: string;
  sort: ExperimentSort;
}

export const EMPTY_EXPERIMENT_CATALOG: ExperimentCatalogState = {
  modelType: undefined,
  status: undefined,
  query: "",
  sort: "newest",
};

export function readExperimentCatalogState(params: URLSearchParams): ExperimentCatalogState {
  return {
    modelType: clean(params.get("method")),
    status: readOneOf(params.get("status"), STATUSES),
    query: params.get("q")?.slice(0, 200) ?? "",
    sort: readOneOf(params.get("sort"), SORTS) ?? "newest",
  };
}

export function writeExperimentCatalogState(state: ExperimentCatalogState): URLSearchParams {
  const params = new URLSearchParams();
  const query = state.query.trim();
  if (query) params.set("q", query);
  if (state.modelType !== undefined) params.set("method", state.modelType);
  if (state.status !== undefined) params.set("status", state.status);
  if (state.sort !== "newest") params.set("sort", state.sort);
  return params;
}

export function toExperimentListQuery(
  state: ExperimentCatalogState,
  datasetId?: number,
): ExperimentListQuery {
  return {
    datasetId,
    modelType: state.modelType,
    status: state.status,
    query: state.query.trim() || undefined,
    sort: state.sort,
  };
}

function clean(value: string | null): string | undefined {
  const trimmed = value?.trim();
  return trimmed ? trimmed : undefined;
}

function readOneOf<T extends string>(raw: string | null, allowed: readonly T[]): T | undefined {
  return allowed.find((value) => value === raw);
}
