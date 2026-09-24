/**
 * Freezing a reference studio session into an experiment (ADR-0040).
 *
 * The studio's references become what every few-shot run trains on — a `manual` split whose
 * `train` subset is exactly those samples — and the run is created and started the way the
 * create screen and the run bar do it, through the same routes. Nothing here is a second
 * path to a result: the session ends on the experiment page.
 */

import { useMutation, useQueryClient } from "@tanstack/react-query";

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
