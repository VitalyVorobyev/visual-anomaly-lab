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
import type {
  CurveSet,
  ExperimentDetail,
  ExperimentSummary,
  DiagnosticIndex,
  ImageScore,
  MethodCatalog,
  MetricSummary,
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

export function useExperiments(datasetId?: number) {
  return useQuery<ExperimentSummary[]>({
    queryKey: queryKeys.experiments(datasetId),
    queryFn: async () =>
      unwrap(
        await api.GET("/api/experiments", {
          params: { query: datasetId === undefined ? {} : { dataset_id: datasetId } },
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

export interface CreateExperimentInput {
  name: string;
  dataset_id: number;
  split_id: number;
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
    mutationFn: async ({ kind, subsets }: { kind: "train" | "infer"; subsets?: Subset[] }) => {
      const result =
        kind === "train"
          ? await api.POST("/api/experiments/{experiment_id}/train", {
              params: { path: { experiment_id: experimentId } },
              body: { experiment_id: experimentId, diagnostics: true },
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
export function useCurves(experimentId: number | undefined, subset: Subset | undefined) {
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
    enabled: experimentId !== undefined,
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

export function useDeleteExperiment() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: async (experimentId: number) => {
      const { error } = await api.DELETE("/api/experiments/{experiment_id}", {
        params: { path: { experiment_id: experimentId } },
      });
      if (error !== undefined) throw new Error("The experiment could not be deleted.");
    },
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: ["experiments"] });
    },
  });
}
