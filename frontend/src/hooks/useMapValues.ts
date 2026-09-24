/**
 * The value planes behind one image, fetched once and then indexed locally.
 *
 * One fetch per plane per image, `staleTime: Infinity`, because the alternative — a point
 * query per `pointermove` — gives a readout that lags the cursor and shows nothing at all
 * while panning (handbook diagnostics.md). At the default 256x256 a plane is 256 KB, which on a loopback
 * interface to a local sidecar is cheaper than the round trips it replaces.
 *
 * `enabled` is the whole cost control: nothing is fetched until a pointer is actually over
 * a canvas, so a reader who never hovers pays nothing.
 */

import { useQuery } from "@tanstack/react-query";

import { fetchPlane, type ValuePlane } from "@vitavision/lab-ui";
import { anomalyMapValuesUrl, labelPlaneUrl, sourceValuesUrl } from "../api/valuesUrl";

const FOREVER = {
  staleTime: Number.POSITIVE_INFINITY,
  gcTime: 10 * 60 * 1000,
  retry: false,
} as const;

export function useAnomalyValues(
  experimentId: number | undefined,
  imageId: number | undefined,
  enabled: boolean,
) {
  return useQuery<ValuePlane>({
    queryKey: ["values", "map", experimentId, imageId],
    queryFn: () => fetchPlane(anomalyMapValuesUrl(imageId as number, experimentId as number)),
    enabled: enabled && experimentId !== undefined && imageId !== undefined,
    ...FOREVER,
  });
}

/**
 * Every colour plane the experiment preprocesses to, in one query.
 *
 * The channel count is data, never schema: it arrives in the response header, so a mono
 * experiment and a three-channel one are the same code path here and neither is encoded
 * anywhere in the UI.
 */
export function useSourceValues(
  experimentId: number | undefined,
  imageId: number | undefined,
  enabled: boolean,
) {
  return useQuery<ValuePlane>({
    queryKey: ["values", "source", experimentId, imageId],
    queryFn: () => fetchPlane(sourceValuesUrl(experimentId as number, imageId as number)),
    enabled: enabled && experimentId !== undefined && imageId !== undefined,
    ...FOREVER,
  });
}

/**
 * A supervised run's label map for one image — `truth` for the annotation over the run's
 * pinned classes. Fetched when its layer is on rather than on hover: it is the picture, not a
 * readout. Unlike the planes above it can change under an open run — inference again, or an
 * annotation completed — so it is refetched when the sample is opened again rather than kept
 * for the session. A 404 is an answer (no label map, or truth that does not answer for every
 * class), so it is not retried.
 */
export function useLabelPlane(
  experimentId: number | undefined,
  imageId: number | undefined,
  truth: boolean,
  enabled: boolean,
) {
  return useQuery<ValuePlane>({
    queryKey: ["values", "labels", experimentId, imageId, truth],
    queryFn: () => fetchPlane(labelPlaneUrl(experimentId as number, imageId as number, truth)),
    enabled: enabled && experimentId !== undefined && imageId !== undefined,
    retry: false,
  });
}
