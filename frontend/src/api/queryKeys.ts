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
} as const;
