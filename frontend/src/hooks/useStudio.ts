/**
 * Freezing a reference studio session into an experiment (ADR-0040).
 *
 * The studio's references become what every few-shot run trains on — a `manual` split whose
 * `train` subset is exactly those samples — and the run is created and started the way the
 * create screen and the run bar do it, through the same routes. Nothing here is a second
 * path to a result: the session ends on the experiment page.
 */

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

import { api, unwrap } from "../api/client";
import { queryKeys } from "../api/queryKeys";

export interface FreezeRequest {
  datasetId: number;
  classKey: string;
  references: number[];
  regionProfileId: number;
  methodKey: string;
  methodTitle: string;
}

/** A split name that is new every time — splits are immutable, and a name is unique per dataset. */
function splitName(classKey: string, references: number[]): string {
  const stamp = new Date().toISOString().slice(0, 19).replace("T", " ");
  return `${classKey} · ${references.length} references · ${stamp}`;
}

export function useFreezeReferences() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: async (request: FreezeRequest) => {
      const split = unwrap(
        await api.POST("/api/splits", {
          body: {
            dataset_id: request.datasetId,
            name: splitName(request.classKey, request.references),
            seed: 0,
            params: {
              strategy: "manual",
              sample_ids: request.references,
              // Required on the wire and meaningless for `manual`, like `imported`'s.
              train_normal_fraction: 0.6,
              val_normal_fraction: 0.2,
              val_defect_fraction: 0.3,
              holdout_from_train: 0,
              train_fraction: 0.7,
              unlabeled_subset: "test",
            },
          },
        }),
        "the reference split",
      );
      const experiment = unwrap(
        await api.POST("/api/experiments", {
          body: {
            name: `${request.methodTitle} · ${request.classKey} · ${request.references.length} refs`,
            dataset_id: request.datasetId,
            split_id: split.id,
            region_profile_id: request.regionProfileId,
            model_type: request.methodKey,
            task: "few_shot_segmentation",
            target_label: request.classKey,
            config: {},
            preprocessing: {},
            evaluation: {},
            channels: [],
          },
        }),
        "the experiment",
      );
      unwrap(
        await api.POST("/api/experiments/{experiment_id}/train", {
          params: { path: { experiment_id: experiment.id } },
          body: { experiment_id: experiment.id, diagnostics: true, then_score: true },
        }),
        "the queued run",
      );
      return experiment;
    },
    onSuccess: (_experiment, request) => {
      void queryClient.invalidateQueries({ queryKey: queryKeys.splits(request.datasetId) });
      void queryClient.invalidateQueries({ queryKey: ["experiments"] });
    },
  });
}

export interface PreviewInput {
  datasetId: number;
  classKey: string;
  methodKey: string | undefined;
  profileId: number | undefined;
  references: number[];
  imageId: number | undefined;
}

/**
 * The focused image segmented by the current references, through the resident worker
 * (ADR-0026). Keyed by everything the fit depends on, so a changed reference is a new answer
 * rather than an old one; the first request of a set of references pays the fit.
 */
export function useStudioPreview(input: PreviewInput, enabled: boolean) {
  const references = [...input.references].sort((a, b) => a - b);
  return useQuery({
    queryKey: [
      "studio-preview",
      input.datasetId,
      input.classKey,
      input.methodKey,
      input.profileId,
      references,
      input.imageId,
    ] as const,
    queryFn: async () =>
      unwrap(
        await api.POST("/api/datasets/{dataset_id}/studio/preview", {
          params: { path: { dataset_id: input.datasetId } },
          body: {
            class_key: input.classKey,
            method: input.methodKey as string,
            profile_id: input.profileId as number,
            references,
            image_id: input.imageId as number,
          },
        }),
        "the preview",
      ),
    enabled:
      enabled &&
      references.length > 0 &&
      input.methodKey !== undefined &&
      input.profileId !== undefined &&
      input.imageId !== undefined,
    // A refused preview (a job is running, a reference lost its truth) is an answer to
    // show, not a transient to retry into.
    retry: false,
    staleTime: Infinity,
  });
}

export type RegionAction = "accept" | "fix" | "absent";

/**
 * Turn what the studio shows into truth (ADR-0040): the preview's region accepted as a
 * completed revision, opened as a draft for the editor, or the class confirmed absent. The
 * region is read from the preview's own map on the server, never sent from here.
 */
export function useSetStudioRegion(datasetId: number) {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: async (input: {
      imageId: number;
      classKey: string;
      action: RegionAction;
      generation?: string | undefined;
    }) =>
      unwrap(
        await api.POST("/api/images/{image_id}/studio/region", {
          params: { path: { image_id: input.imageId } },
          body: {
            class_key: input.classKey,
            action: input.action,
            generation: input.generation ?? null,
          },
        }),
        "the studio's annotation",
      ),
    onSuccess: (_outcome, input) => {
      void queryClient.invalidateQueries({ queryKey: queryKeys.classCoverageAll() });
      void queryClient.invalidateQueries({ queryKey: queryKeys.dataset(datasetId) });
      void queryClient.invalidateQueries({ queryKey: queryKeys.annotationRevisions(input.imageId) });
      void queryClient.invalidateQueries({ queryKey: queryKeys.annotationDraft(input.imageId) });
    },
  });
}

/**
 * One rail page segmented by the current references, least certain first — where the
 * references are weakest, and so where the reader's attention is worth most.
 */
export function useStudioBatch(input: Omit<PreviewInput, "imageId"> & { imageIds: number[] }, enabled: boolean) {
  const references = [...input.references].sort((a, b) => a - b);
  return useQuery({
    queryKey: [
      "studio-batch",
      input.datasetId,
      input.classKey,
      input.methodKey,
      input.profileId,
      references,
      input.imageIds,
    ] as const,
    queryFn: async () =>
      unwrap(
        await api.POST("/api/datasets/{dataset_id}/studio/preview-batch", {
          params: { path: { dataset_id: input.datasetId } },
          body: {
            class_key: input.classKey,
            method: input.methodKey as string,
            profile_id: input.profileId as number,
            references,
            image_ids: input.imageIds,
          },
        }),
        "the ordered page",
      ),
    enabled:
      enabled &&
      references.length > 0 &&
      input.imageIds.length > 0 &&
      input.methodKey !== undefined &&
      input.profileId !== undefined,
    retry: false,
    staleTime: Infinity,
  });
}
