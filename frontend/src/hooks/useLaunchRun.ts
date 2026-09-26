/**
 * One press from a set of choices to a queued run: create the split if it is still a preset,
 * create the experiment, and queue Train & score — the run bar's first primary action — so
 * the reader lands on a run that is already working.
 *
 * The guided run and the classic form's "Create & run" both end here, through the same
 * routes the Splits tab, the create form and the run bar use; nothing is a second path to a
 * result. Each stage reports which one failed, and a split created before a later stage
 * failed is handed back (`onSplit`) so a retry reuses it rather than drawing another.
 */

import { useMutation, useQueryClient } from "@tanstack/react-query";

import { api, unwrap } from "../api/client";
import type { ExperimentDetail, SplitParamsInput } from "../api/client";
import { queryKeys } from "../api/queryKeys";
import type { CreateExperimentInput } from "./useExperiments";

export interface LaunchRequest {
  /** An existing split, or the params of a preset still to be created. */
  split: { id: number } | { params: SplitParamsInput };
  experiment: Omit<CreateExperimentInput, "split_id">;
  /** Queue Train & score once the experiment exists; false stops at a draft. */
  run: boolean;
  /** Told the new split's id the moment it exists, before any later stage can fail. */
  onSplit?: (splitId: number) => void;
}

export function useLaunchRun() {
  const queryClient = useQueryClient();
  return useMutation<ExperimentDetail, Error, LaunchRequest>({
    mutationFn: async (request) => {
      const datasetId = request.experiment.dataset_id;
      let splitId: number;
      if ("id" in request.split) {
        splitId = request.split.id;
      } else {
        const created = unwrap(
          await api.POST("/api/splits", {
            body: { dataset_id: datasetId, name: null, seed: null, params: request.split.params },
          }),
          "the new split",
        );
        splitId = created.id;
        request.onSplit?.(splitId);
      }
      const experiment = unwrap(
        await api.POST("/api/experiments", { body: { ...request.experiment, split_id: splitId } }),
        "the created experiment",
      );
      if (request.run) {
        unwrap(
          await api.POST("/api/experiments/{experiment_id}/train", {
            params: { path: { experiment_id: experiment.id } },
            body: { experiment_id: experiment.id, diagnostics: true, then_score: true },
          }),
          "the queued run",
        );
      }
      return experiment;
    },
    onSettled: (_experiment, _error, request) => {
      void queryClient.invalidateQueries({ queryKey: queryKeys.dataset(request.experiment.dataset_id) });
      void queryClient.invalidateQueries({ queryKey: ["experiments"] });
    },
  });
}
