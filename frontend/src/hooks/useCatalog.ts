/**
 * Reading and editing the catalog.
 *
 * Every hook here funnels through `unwrap`, so an error response becomes a thrown
 * `Error` and TanStack Query's error state means what it says.
 */

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

import { api, unwrap } from "../api/client";
import type {
  BulkLabelRequest,
  DatasetDeletionPreview,
  DatasetDeletionResult,
  DatasetDetail,
  DatasetSummary,
  DatasetUpdate,
  Label,
  JobSummary,
  ReferencePackCatalog,
  RegisterReferencePacksParams,
  SamplePage,
  SampleSummary,
  SplitDetail,
} from "../api/client";
import type { SampleQuery } from "../api/queryKeys";
import { queryKeys } from "../api/queryKeys";

export function useDatasets() {
  return useQuery<DatasetSummary[]>({
    queryKey: queryKeys.datasets(),
    queryFn: async () => unwrap(await api.GET("/api/datasets"), "the dataset list"),
  });
}

export function useReferencePacks() {
  return useQuery<ReferencePackCatalog>({
    queryKey: queryKeys.referencePacks(),
    queryFn: async () =>
      unwrap(await api.GET("/api/reference-packs"), "the reference dataset catalog"),
  });
}

export function useRegisterReferencePacks() {
  return useMutation<JobSummary, Error, RegisterReferencePacksParams>({
    mutationFn: async (body) =>
      unwrap(
        await api.POST("/api/reference-packs/register", { body }),
        "the reference dataset registration job",
      ),
  });
}

export function useDataset(datasetId: number | undefined) {
  return useQuery<DatasetDetail>({
    queryKey: queryKeys.dataset(datasetId ?? -1),
    queryFn: async () =>
      unwrap(
        await api.GET("/api/datasets/{dataset_id}", {
          params: { path: { dataset_id: datasetId as number } },
        }),
        "the dataset",
      ),
    enabled: datasetId !== undefined,
  });
}

export function useSamples(datasetId: number | undefined, query: SampleQuery) {
  return useQuery<SamplePage>({
    queryKey: queryKeys.samples(datasetId ?? -1, query),
    queryFn: async () =>
      unwrap(
        await api.GET("/api/datasets/{dataset_id}/samples", {
          params: {
            path: { dataset_id: datasetId as number },
            query: {
              label: query.label,
              channel_id: query.channelId,
              split_id: query.splitId,
              subset: query.subset,
              annotated: query.annotated,
              limit: query.limit,
              offset: query.offset,
            },
          },
        }),
        "the sample page",
      ),
    enabled: datasetId !== undefined,
    // Keeps the previous page on screen while the next one loads, so paging through the
    // grid does not flash empty.
    placeholderData: (previous) => previous,
  });
}

export function useSample(datasetId: number | undefined, sampleId: number | undefined) {
  return useQuery<SampleSummary>({
    queryKey: queryKeys.sample(datasetId ?? -1, sampleId ?? -1),
    queryFn: async () =>
      unwrap(
        await api.GET("/api/datasets/{dataset_id}/samples/{sample_id}", {
          params: {
            path: { dataset_id: datasetId as number, sample_id: sampleId as number },
          },
        }),
        "the sample",
      ),
    enabled: datasetId !== undefined && sampleId !== undefined,
  });
}

export function useSetLabel(datasetId: number) {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: async ({ sampleId, label }: { sampleId: number; label: Label }) =>
      unwrap(
        await api.PATCH("/api/datasets/{dataset_id}/samples/{sample_id}", {
          params: { path: { dataset_id: datasetId, sample_id: sampleId } },
          body: { label },
        }),
        "the updated sample",
      ),
    onSuccess: () => {
      // The dataset key is a prefix of every sample and page key under it, so one
      // invalidation refreshes the grid, the open sample and the label counts together.
      void queryClient.invalidateQueries({ queryKey: queryKeys.dataset(datasetId) });
    },
  });
}

/**
 * Label many samples at once — a selection, or everything matching the browse filters.
 *
 * The filter form deliberately sends the *filters* rather than the ids they resolve to:
 * the server re-evaluates them against the same clause the grid pages with, so the set
 * that gets labelled is the set whose count was shown, even though only one page of it
 * was ever loaded.
 */
export function useSetLabels(datasetId: number) {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: async (body: BulkLabelRequest) =>
      unwrap(
        await api.PATCH("/api/datasets/{dataset_id}/samples", {
          params: { path: { dataset_id: datasetId } },
          body,
        }),
        "the number of samples labelled",
      ),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: queryKeys.dataset(datasetId) });
    },
  });
}

/**
 * Edit what a dataset says about itself: its description, and what it is filed under.
 *
 * The body carries only the keys the caller sets. That is the contract the endpoint is
 * built on -- an absent key leaves its column alone, and an explicit `null` clears the
 * override so the reference pack's own text comes back -- so never spread a whole dataset
 * object in here to "keep the other field", which would rewrite it with what was on screen.
 */
export function useUpdateDataset() {
  const queryClient = useQueryClient();
  return useMutation<DatasetDetail, Error, { datasetId: number; changes: DatasetUpdate }>({
    mutationFn: async ({ datasetId, changes }) =>
      unwrap(
        await api.PATCH("/api/datasets/{dataset_id}", {
          params: { path: { dataset_id: datasetId } },
          body: changes,
        }),
        "the dataset edit",
      ),
    onSuccess: (_result, { datasetId }) => {
      void queryClient.invalidateQueries({ queryKey: queryKeys.datasets() });
      void queryClient.invalidateQueries({ queryKey: queryKeys.dataset(datasetId) });
    },
  });
}

/** One dataset's new filing, as the collection dialog computes it. */
export type CollectionMove = { datasetId: number; collection: string };

/**
 * File several datasets at once, which is what naming a collection actually is.
 *
 * Sequential rather than fanned out: SQLite has one writer, the list is a dozen entries at
 * most, and a fan-out buys nothing while making a partial failure harder to describe. If
 * one PATCH fails the rest are abandoned and the error says how many landed — the caller
 * keeps its dialog open, and the invalidation below still runs, so the catalogue shows what
 * actually happened rather than what was asked for.
 */
export function useMoveDatasets() {
  const queryClient = useQueryClient();
  return useMutation<void, Error, CollectionMove[]>({
    mutationFn: async (moves) => {
      let done = 0;
      try {
        for (const { datasetId, collection } of moves) {
          unwrap(
            await api.PATCH("/api/datasets/{dataset_id}", {
              params: { path: { dataset_id: datasetId } },
              body: { collection },
            }),
            "the dataset edit",
          );
          done += 1;
        }
      } catch (error) {
        const reason = error instanceof Error ? error.message : String(error);
        throw new Error(`${done} of ${moves.length} datasets were filed. ${reason}`);
      }
    },
    onSettled: (_result, _error, moves) => {
      void queryClient.invalidateQueries({ queryKey: queryKeys.datasets() });
      for (const { datasetId } of moves) {
        void queryClient.invalidateQueries({ queryKey: queryKeys.dataset(datasetId) });
      }
    },
  });
}

export function useDeleteDataset() {
  const queryClient = useQueryClient();
  return useMutation<DatasetDeletionResult, Error, number>({
    mutationFn: async (datasetId) =>
      unwrap(
        await api.DELETE("/api/datasets/{dataset_id}", {
          params: { path: { dataset_id: datasetId } },
        }),
        "the dataset deletion",
      ),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: queryKeys.datasets() });
      void queryClient.invalidateQueries({ queryKey: queryKeys.experimentLists() });
      void queryClient.invalidateQueries({ queryKey: queryKeys.comparisons() });
    },
  });
}

export function useDatasetDeletionPreview(datasetId: number | undefined) {
  return useQuery<DatasetDeletionPreview>({
    queryKey: queryKeys.datasetDeletion(datasetId ?? -1),
    queryFn: async () =>
      unwrap(
        await api.GET("/api/datasets/{dataset_id}/deletion-preview", {
          params: { path: { dataset_id: datasetId as number } },
        }),
        "the dataset deletion preview",
      ),
    enabled: datasetId !== undefined,
  });
}

export function useSplits(datasetId: number | undefined) {
  return useQuery<SplitDetail[]>({
    queryKey: queryKeys.splits(datasetId ?? -1),
    queryFn: async () =>
      unwrap(
        await api.GET("/api/splits", {
          params: { query: { dataset_id: datasetId as number } },
        }),
        "the split list",
      ),
    enabled: datasetId !== undefined,
  });
}
