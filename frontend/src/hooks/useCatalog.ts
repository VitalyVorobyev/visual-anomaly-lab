/**
 * Reading and editing the catalog.
 *
 * Every hook here funnels through `unwrap`, so an error response becomes a thrown
 * `Error` and TanStack Query's error state means what it says.
 */

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

import { api, unwrap } from "../api/client";
import type {
  DatasetDetail,
  DatasetSummary,
  Label,
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

export function useDeleteDataset() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: async (datasetId: number) => {
      const { error } = await api.DELETE("/api/datasets/{dataset_id}", {
        params: { path: { dataset_id: datasetId } },
      });
      if (error !== undefined) {
        throw new Error("The dataset could not be deleted.");
      }
    },
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: queryKeys.datasets() });
    },
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
