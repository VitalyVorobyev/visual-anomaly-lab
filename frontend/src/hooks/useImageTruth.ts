/**
 * An image's truth as the sample view draws it (`GET /api/images/{id}/truth`): its classes,
 * its drawn boxes, and where its region overlay is. One small read per image, shared by every
 * pane and the legend.
 */

import { useQueries, useQuery } from "@tanstack/react-query";

import { api, unwrap, type ImageTruth } from "../api/client";
import { queryKeys } from "../api/queryKeys";

async function fetchTruth(imageId: number): Promise<ImageTruth> {
  return unwrap(
    await api.GET("/api/images/{image_id}/truth", { params: { path: { image_id: imageId } } }),
    "the image's truth",
  );
}

export function useImageTruth(imageId: number | undefined, enabled = true) {
  return useQuery<ImageTruth>({
    queryKey: queryKeys.imageTruth(imageId ?? -1),
    queryFn: () => fetchTruth(imageId as number),
    enabled: enabled && imageId !== undefined,
  });
}

/** Every image of a sample: the legend lists what any channel shows. */
export function useSampleTruth(imageIds: readonly number[]) {
  return useQueries({
    queries: imageIds.map((imageId) => ({
      queryKey: queryKeys.imageTruth(imageId),
      queryFn: () => fetchTruth(imageId),
    })),
  });
}
