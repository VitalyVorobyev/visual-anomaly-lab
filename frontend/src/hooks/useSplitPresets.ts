/**
 * Split presets, dry runs, creation and deletion.
 *
 * A split cannot be edited, so what it will contain is read before it exists: a preset
 * carries the composition Create would produce, and the custom form previews its params
 * through the same server-side planner. Every key sits under `queryKeys.splits`, so
 * creating or deleting a split refreshes the presets' next seed and derived name with it.
 */

import { keepPreviousData, useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

import { api, unwrap } from "../api/client";
import type {
  SplitDeletionPreview,
  SplitDeletionResult,
  SplitDetail,
  SplitParamsInput,
  SplitPreset,
  SplitPreview,
} from "../api/client";
import { queryKeys } from "../api/queryKeys";

/**
 * Every field the wire requires, at the server's defaults. The fractions mean nothing to
 * a strategy that does not read them; they are sent only because the schema requires them.
 */
export const DEFAULT_SPLIT_PARAMS: SplitParamsInput = {
  strategy: "normal_only_train",
  train_normal_fraction: 0.6,
  val_normal_fraction: 0.2,
  val_defect_fraction: 0.3,
  holdout_from_train: 0,
  train_fraction: 0.7,
  // Assigned rather than left out: unlabelled samples are excluded from every metric
  // later, but they have to be scored to appear in the ranked lists.
  unlabeled_subset: "test",
};

export interface SplitRequest {
  params: SplitParamsInput;
  /** Omitted: the first seed no split of the same params has used. */
  seed?: number;
  /** Omitted: `<label> · seed <n>`, derived by the server. */
  name?: string;
}

export function useSplitPresets(datasetId: number | undefined) {
  return useQuery<SplitPreset[]>({
    queryKey: queryKeys.splitPresets(datasetId ?? -1),
    queryFn: async () =>
      unwrap(
        await api.GET("/api/datasets/{dataset_id}/split-presets", {
          params: { path: { dataset_id: datasetId as number } },
        }),
        "the split presets",
      ),
    enabled: datasetId !== undefined,
  });
}

/** The dry run of a request; `undefined` asks nothing. The last answer stays while typing. */
export function useSplitPreview(datasetId: number, request: SplitRequest | undefined) {
  return useQuery<SplitPreview>({
    queryKey: queryKeys.splitPreview(datasetId, request ?? null),
    queryFn: async () =>
      unwrap(
        await api.POST("/api/datasets/{dataset_id}/splits/preview", {
          params: { path: { dataset_id: datasetId } },
          body: { params: request?.params ?? DEFAULT_SPLIT_PARAMS, seed: request?.seed ?? null },
        }),
        "the split preview",
      ),
    enabled: request !== undefined,
    placeholderData: keepPreviousData,
  });
}

export function useCreateSplit(datasetId: number) {
  const queryClient = useQueryClient();
  return useMutation<SplitDetail, Error, SplitRequest>({
    mutationFn: async (request) =>
      unwrap(
        await api.POST("/api/splits", {
          body: {
            dataset_id: datasetId,
            name: request.name?.trim() || null,
            seed: request.seed ?? null,
            params: request.params,
          },
        }),
        "the new split",
      ),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: queryKeys.dataset(datasetId) });
    },
  });
}

export function useSplitDeletionPreview(splitId: number | undefined) {
  return useQuery<SplitDeletionPreview>({
    queryKey: queryKeys.splitDeletion(splitId ?? -1),
    queryFn: async () =>
      unwrap(
        await api.GET("/api/splits/{split_id}/deletion-preview", {
          params: { path: { split_id: splitId as number } },
        }),
        "the split deletion preview",
      ),
    enabled: splitId !== undefined,
  });
}

export function useDeleteSplit(datasetId: number) {
  const queryClient = useQueryClient();
  return useMutation<SplitDeletionResult, Error, number>({
    mutationFn: async (splitId) =>
      unwrap(
        await api.DELETE("/api/splits/{split_id}", {
          params: { path: { split_id: splitId } },
        }),
        "the split deletion",
      ),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: queryKeys.dataset(datasetId) });
    },
  });
}
