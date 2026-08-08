/**
 * One place that decides what a query is called.
 *
 * M1 had a single `["health"]` literal, which needed no ceremony. M2 introduces eight
 * families with parameters, and the thing that actually goes wrong without a factory is
 * invalidation: a mutation invalidates `["samples", datasetId]` while a screen subscribed
 * to `["sample", datasetId, id]`, and the stale row stays on screen with no error
 * anywhere. Keys are built here so that mismatch is a type error instead.
 *
 * The convention is hierarchical: `datasets()` is a prefix of `dataset(id)`, which is a
 * prefix of `samples(id, …)`, so invalidating a dataset invalidates everything under it.
 */

import type { Label, Subset } from "./client";

export interface SampleQuery {
  label?: Label | undefined;
  channelId?: number | undefined;
  splitId?: number | undefined;
  subset?: Subset | undefined;
  limit?: number | undefined;
  offset?: number | undefined;
}

export const queryKeys = {
  health: () => ["health"] as const,

  adapters: () => ["import", "adapters"] as const,
  manifest: (manifestId: string) => ["import", "manifest", manifestId] as const,

  datasets: () => ["datasets"] as const,
  dataset: (datasetId: number) => ["datasets", datasetId] as const,
  samples: (datasetId: number, query: SampleQuery = {}) =>
    ["datasets", datasetId, "samples", query] as const,
  sample: (datasetId: number, sampleId: number) =>
    ["datasets", datasetId, "sample", sampleId] as const,

  splits: (datasetId: number) => ["datasets", datasetId, "splits"] as const,

  jobs: () => ["jobs"] as const,
  job: (jobId: number) => ["jobs", jobId] as const,
  jobMetrics: (jobId: number) => ["jobs", jobId, "metrics"] as const,

  modelTypes: () => ["experiments", "model-types"] as const,
  experiments: (datasetId?: number) => ["experiments", "list", datasetId ?? null] as const,
  /**
   * Every experiment list, whichever dataset it was filtered to.
   *
   * `experiments(datasetId)` is a *complete* key, not a prefix, so invalidating one list
   * leaves the others stale. A finished run changes the headline metric on all of them.
   */
  experimentLists: () => ["experiments", "list"] as const,
  experiment: (experimentId: number) => ["experiments", experimentId] as const,
  results: (experimentId: number, subset?: Subset) =>
    ["experiments", experimentId, "results", subset ?? null] as const,
  threshold: (experimentId: number, subset: Subset | undefined, value: number) =>
    ["experiments", experimentId, "threshold", subset ?? null, value] as const,
  sampleImages: (experimentId: number, sampleId: number) =>
    ["experiments", experimentId, "sample-images", sampleId] as const,
  previews: (experimentId: number, subset?: Subset) =>
    ["experiments", experimentId, "previews", subset ?? null] as const,
  artifacts: (experimentId: number) => ["experiments", experimentId, "artifacts"] as const,
  diagnostics: (experimentId: number) => ["experiments", experimentId, "diagnostics"] as const,
  curves: (experimentId: number, subset?: Subset) =>
    ["experiments", experimentId, "curves", subset ?? null] as const,
} as const;
