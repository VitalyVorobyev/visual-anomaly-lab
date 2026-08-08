/**
 * Several runs, read against each other in one round trip.
 *
 * One request rather than N: the metrics could be assembled from N experiment details, but
 * the operating point could not. Every threshold on the screen is resolved per run by one
 * rule, and that rule lives in Python beside `suggest_threshold` — re-deriving it here for
 * N runs would be the same rule in two languages, free to drift (ADR-0028).
 */

import { useQueries, useQuery } from "@tanstack/react-query";

import { api, unwrap } from "../api/client";
import type {
  ComparisonReport,
  CurveSet,
  ImageScore,
  OperatingPoint,
  Subset,
} from "../api/client";
import { queryKeys } from "../api/queryKeys";

export function useComparison({
  ids,
  subset,
  at,
  recallTarget,
}: {
  ids: number[];
  subset: Subset | undefined;
  at: OperatingPoint;
  recallTarget: number;
}) {
  return useQuery<ComparisonReport>({
    queryKey: queryKeys.comparison(ids, subset, at, recallTarget),
    queryFn: async () =>
      unwrap(
        await api.GET("/api/compare", {
          params: {
            query: {
              ids,
              at,
              recall_target: recallTarget,
              ...(subset === undefined ? {} : { subset }),
            },
          },
        }),
        "the comparison",
      ),
    // Fewer than two runs is not an error state, it is the empty state of the picker. The
    // server would answer 422 and the screen would show a failure the reader caused by
    // arriving.
    enabled: ids.length >= 2,
  });
}

/**
 * Every selected run's curves, as one hook over a list that changes length.
 *
 * `useQueries` rather than a `useCurves` per run in a `map`: the number of hooks a
 * component calls may not change between renders, and the whole premise of this screen is
 * that a column can be added. Each entry keeps its own cache slot under the *same* key the
 * single-run Benchmark tab uses, so a run already looked at there is drawn without a
 * request.
 */
export function useCurveSets(ids: number[], subset: Subset | undefined) {
  return useQueries({
    queries: ids.map((id) => ({
      queryKey: queryKeys.curves(id, subset),
      queryFn: async (): Promise<CurveSet> =>
        unwrap(
          await api.GET("/api/experiments/{experiment_id}/curves", {
            params: {
              path: { experiment_id: id },
              query: subset === undefined ? {} : { subset },
            },
          }),
          "the curves",
        ),
    })),
  });
}

/**
 * One sample's per-image scores, from every selected run.
 *
 * The image ids are the same across runs — the comparison is constrained to one dataset,
 * so a sample is the same sample in every column — but the scores, the maps and their
 * scales are each the run's own. Keyed identically to `useSampleImages`, so opening a
 * sample here and then on its own experiment screen costs one request between them.
 */
export function useSampleImageSets(ids: number[], sampleId: number | undefined) {
  return useQueries({
    queries: ids.map((id) => ({
      queryKey: queryKeys.sampleImages(id, sampleId ?? -1),
      queryFn: async (): Promise<ImageScore[]> =>
        unwrap(
          await api.GET("/api/experiments/{experiment_id}/samples/{sample_id}/images", {
            params: { path: { experiment_id: id, sample_id: sampleId as number } },
          }),
          "the per-image scores",
        ),
      enabled: sampleId !== undefined,
    })),
  });
}
