/**
 * Reading and driving experiments.
 *
 * The same shape as `useCatalog`: every hook funnels through `unwrap`, so an error
 * response becomes a thrown `Error` and TanStack Query's error state means what it says.
 *
 * Notice what is *not* here: nothing knows a method's name. The create screen builds its
 * form from `useModelTypes`, and every other hook takes an experiment id. That is what
 * ADR-0007 buys — a new method appears in the picker without a line changing here.
 */

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

import { api, unwrap } from "../api/client";
import type { ExperimentListQuery } from "../api/experimentState";
import type {
  ArtifactListing,
  CurveSet,
  DiagnoseResponse,
  ExperimentDetail,
  ExperimentDeletionPreview,
  ExperimentDeletionResult,
  ExperimentSummary,
  DiagnosticIndex,
  ImageScore,
  MethodCatalog,
  MetricSummary,
  PruneResult,
  PruneScope,
  ResultsPage,
  SamplePreview,
  Subset,
  ThresholdReport,
} from "../api/client";
import { queryKeys } from "../api/queryKeys";

export function useModelTypes() {
  return useQuery<MethodCatalog>({
    queryKey: queryKeys.modelTypes(),
    queryFn: async () =>
      unwrap(await api.GET("/api/experiments/model-types"), "the method catalog"),
    // The registry cannot change while the app is running, and the picker is opened often.
    staleTime: Infinity,
  });
}

export function useExperiments(query: ExperimentListQuery = {}) {
  return useQuery<ExperimentSummary[]>({
    queryKey: queryKeys.experiments(query),
    queryFn: async () =>
      unwrap(
        await api.GET("/api/experiments", {
          params: {
            query: {
              ...(query.datasetId === undefined ? {} : { dataset_id: query.datasetId }),
              ...(query.modelType === undefined ? {} : { model_type: query.modelType }),
              ...(query.status === undefined ? {} : { status: query.status }),
              ...(query.query === undefined ? {} : { q: query.query }),
              ...(query.sort === undefined ? {} : { sort: query.sort }),
            },
          },
        }),
        "the experiment list",
      ),
  });
}

export function useExperiment(experimentId: number | undefined) {
  return useQuery<ExperimentDetail>({
    queryKey: queryKeys.experiment(experimentId ?? -1),
    queryFn: async () =>
      unwrap(
        await api.GET("/api/experiments/{experiment_id}", {
          params: { path: { experiment_id: experimentId as number } },
        }),
        "the experiment",
      ),
    enabled: experimentId !== undefined,
  });
}

export function useExperimentDeletionPreview(experimentId: number | undefined) {
  return useQuery<ExperimentDeletionPreview>({
    queryKey: queryKeys.experimentDeletion(experimentId ?? -1),
    queryFn: async () =>
      unwrap(
        await api.GET("/api/experiments/{experiment_id}/deletion-preview", {
          params: { path: { experiment_id: experimentId as number } },
        }),
        "the deletion preview",
      ),
    enabled: experimentId !== undefined,
  });
}

export interface CreateExperimentInput {
  name: string;
  dataset_id: number;
  split_id: number;
  region_profile_id: number;
  model_type: string;
  config: Record<string, unknown>;
  preprocessing: Record<string, unknown>;
  evaluation: Record<string, unknown>;
}

export function useCreateExperiment() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: async (body: CreateExperimentInput) =>
      unwrap(await api.POST("/api/experiments", { body }), "the created experiment"),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: ["experiments"] });
    },
  });
}

/**
 * Start a training or an inference run.
 *
 * Both return a `JobSummary`, and the caller follows it with the same `useJob` the import
 * screen uses. There is deliberately no training-specific progress mechanism.
 */
export function useStartRun(experimentId: number) {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: async ({
      kind,
      subsets,
      additionalSteps,
    }: {
      kind: "train" | "infer";
      subsets?: Subset[];
      /** Continue the stored model instead of retraining it (ADR-0025). */
      additionalSteps?: number;
    }) => {
      const result =
        kind === "train"
          ? await api.POST("/api/experiments/{experiment_id}/train", {
              params: { path: { experiment_id: experimentId } },
              body: {
                experiment_id: experimentId,
                diagnostics: true,
                // Omitted entirely for a fresh run: an empty control means unset, and the
                // default lives in Python alone.
                ...(additionalSteps === undefined ? {} : { additional_steps: additionalSteps }),
              },
            })
          : await api.POST("/api/experiments/{experiment_id}/infer", {
              params: { path: { experiment_id: experimentId } },
              body: {
                experiment_id: experimentId,
                subsets: subsets ?? ["val", "test"],
                diagnostics: true,
                // Deliberately not sent. An untouched control contributes nothing, so the
                // budget is defined in Python alone — a number pinned here would silently
                // override every later change to it (`api/schemaForm.ts`).
              },
            });
      return unwrap(result, "the queued job");
    },
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: queryKeys.experiment(experimentId) });
    },
  });
}

export function useResults(experimentId: number | undefined, subset: Subset | undefined) {
  return useQuery<ResultsPage>({
    queryKey: queryKeys.results(experimentId ?? -1, subset),
    queryFn: async () =>
      unwrap(
        await api.GET("/api/experiments/{experiment_id}/results", {
          params: {
            path: { experiment_id: experimentId as number },
            query: subset === undefined ? {} : { subset },
          },
        }),
        "the results",
      ),
    enabled: experimentId !== undefined,
  });
}

/**
 * The confusion matrix at one threshold.
 *
 * Nothing is persisted per threshold (ADR-0011), so this is a read over a few hundred
 * stored floats. `placeholderData` keeps the previous numbers on screen while the next
 * request lands, which is what makes dragging the slider feel like a filter rather than
 * a series of round trips.
 */
export function useThreshold(
  experimentId: number | undefined,
  subset: Subset | undefined,
  value: number,
) {
  return useQuery<ThresholdReport>({
    queryKey: queryKeys.threshold(experimentId ?? -1, subset, value),
    queryFn: async () =>
      unwrap(
        await api.GET("/api/experiments/{experiment_id}/threshold", {
          params: {
            path: { experiment_id: experimentId as number },
            query: subset === undefined ? { value } : { value, subset },
          },
        }),
        "the threshold report",
      ),
    enabled: experimentId !== undefined,
    placeholderData: (previous) => previous,
  });
}

export function useSampleImages(experimentId: number | undefined, sampleId: number | undefined) {
  return useQuery<ImageScore[]>({
    queryKey: queryKeys.sampleImages(experimentId ?? -1, sampleId ?? -1),
    queryFn: async () =>
      unwrap(
        await api.GET("/api/experiments/{experiment_id}/samples/{sample_id}/images", {
          params: {
            path: { experiment_id: experimentId as number, sample_id: sampleId as number },
          },
        }),
        "the per-image scores",
      ),
    enabled: experimentId !== undefined && sampleId !== undefined,
  });
}

/**
 * One image per scored sample, so a gallery can draw a tile per row.
 *
 * Kept out of the threshold report on purpose: that response is recomputed on every slider
 * tick and none of this changes when the threshold moves. One request per subset, held for
 * as long as the run's results stand.
 */
export function useSamplePreviews(experimentId: number | undefined, subset: Subset | undefined) {
  return useQuery<SamplePreview[]>({
    queryKey: queryKeys.previews(experimentId ?? -1, subset),
    queryFn: async () =>
      unwrap(
        await api.GET("/api/experiments/{experiment_id}/previews", {
          params: {
            path: { experiment_id: experimentId as number },
            query: subset === undefined ? {} : { subset },
          },
        }),
        "the sample previews",
      ),
    enabled: experimentId !== undefined,
    staleTime: Infinity,
  });
}

/** What the run left on disk, and where. A listing — nothing here downloads anything. */
export function useArtifacts(experimentId: number | undefined) {
  return useQuery<ArtifactListing>({
    queryKey: queryKeys.artifacts(experimentId ?? -1),
    queryFn: async () =>
      unwrap(
        await api.GET("/api/experiments/{experiment_id}/artifacts", {
          params: { path: { experiment_id: experimentId as number } },
        }),
        "the artifact listing",
      ),
    enabled: experimentId !== undefined,
  });
}

export function useDiagnostics(experimentId: number | undefined) {
  return useQuery<DiagnosticIndex>({
    queryKey: queryKeys.diagnostics(experimentId ?? -1),
    queryFn: async () =>
      unwrap(
        await api.GET("/api/experiments/{experiment_id}/diagnostics", {
          params: { path: { experiment_id: experimentId as number } },
        }),
        "the diagnostics index",
      ),
    enabled: experimentId !== undefined,
  });
}

/**
 * The ROC and PR arrays behind one subset's headline numbers.
 *
 * Recomputed per request from the stored scores, like the threshold report — nothing here
 * is persisted, so a curve can never disagree with the metric it is drawn beside.
 */
export function useCurves(
  experimentId: number | undefined,
  subset: Subset | undefined,
  enabled = true,
) {
  return useQuery<CurveSet>({
    queryKey: queryKeys.curves(experimentId ?? -1, subset),
    queryFn: async () =>
      unwrap(
        await api.GET("/api/experiments/{experiment_id}/curves", {
          params: {
            path: { experiment_id: experimentId as number },
            query: subset === undefined ? {} : { subset },
          },
        }),
        "the curves",
      ),
    enabled: experimentId !== undefined && enabled,
  });
}

export function useReevaluate(experimentId: number) {
  const queryClient = useQueryClient();
  return useMutation<MetricSummary[]>({
    mutationFn: async () =>
      unwrap(
        await api.POST("/api/experiments/{experiment_id}/reevaluate", {
          params: { path: { experiment_id: experimentId } },
        }),
        "the recomputed metrics",
      ),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: queryKeys.experiment(experimentId) });
    },
  });
}

/**
 * Cancel a running job.
 *
 * Lifted out of `JobProgress` so the run bar can offer the same action without the console
 * being on screen — the reason the bar exists is that starting and stopping a run should
 * not depend on which tab you are looking at.
 */
export function useCancelJob(experimentId?: number) {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: async (jobId: number) => {
      const { error } = await api.POST("/api/jobs/{job_id}/cancel", {
        params: { path: { job_id: jobId } },
      });
      if (error !== undefined) throw new Error("The job could not be cancelled.");
      return jobId;
    },
    onSuccess: (jobId) => {
      void queryClient.invalidateQueries({ queryKey: queryKeys.job(jobId) });
      if (experimentId !== undefined) {
        void queryClient.invalidateQueries({ queryKey: queryKeys.experiment(experimentId) });
      }
    },
  });
}

export function useDeleteExperiment() {
  const queryClient = useQueryClient();
  return useMutation<ExperimentDeletionResult, Error, number>({
    mutationFn: async (experimentId: number) => {
      return unwrap(
        await api.DELETE("/api/experiments/{experiment_id}", {
          params: { path: { experiment_id: experimentId } },
        }),
        "the deletion result",
      );
    },
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: ["experiments"] });
    },
  });
}

/**
 * Ask the method what it saw in one image, now (ADR-0026).
 *
 * Deliberately without an optimistic update: the honest thing on screen while this runs is
 * "the first request loads the model", not a pane that will appear. The whole index is
 * refetched on success because an on-demand emission can widen a key's display range
 * (ADR-0027), which changes how every *other* pane of that key is drawn.
 */
export function useDiagnoseImage(experimentId: number) {
  const queryClient = useQueryClient();
  return useMutation<DiagnoseResponse, Error, number>({
    mutationFn: async (imageId: number) =>
      unwrap(
        await api.POST("/api/experiments/{experiment_id}/diagnose", {
          params: { path: { experiment_id: experimentId } },
          body: { image_id: imageId },
        }),
        "the diagnostics for this image",
      ),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: queryKeys.diagnostics(experimentId) });
      void queryClient.invalidateQueries({ queryKey: queryKeys.artifacts(experimentId) });
    },
  });
}

/** Delete stored diagnostics to reclaim disk, and report what that freed (ADR-0027). */
export function useClearDiagnostics(experimentId: number) {
  const queryClient = useQueryClient();
  return useMutation<PruneResult, Error, PruneScope>({
    mutationFn: async (scope: PruneScope) =>
      unwrap(
        await api.DELETE("/api/experiments/{experiment_id}/diagnostics", {
          params: { path: { experiment_id: experimentId }, query: { scope } },
        }),
        "what the clear reclaimed",
      ),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: queryKeys.diagnostics(experimentId) });
      void queryClient.invalidateQueries({ queryKey: queryKeys.artifacts(experimentId) });
    },
  });
}
