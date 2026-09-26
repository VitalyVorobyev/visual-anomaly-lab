/**
 * Explore's server state: whether it can run, one request per question, and a mask turned
 * into an annotation candidate. The encoder lives in the resident worker (ADR-0026); every
 * request here is a browse click, not a job.
 */

import { useMutation, useQuery } from "@tanstack/react-query";

import { api, unwrap } from "../api/client";
import type {
  ExploreCapability,
  ExploreRequest,
  ExploreResponse,
  ExploreShape,
  ExploreShapeRequest,
} from "../api/client";
import { queryKeys } from "../api/queryKeys";

export function useExploreCapability() {
  return useQuery<ExploreCapability>({
    queryKey: queryKeys.exploreCapability(),
    queryFn: async () =>
      unwrap(await api.GET("/api/explore/capability"), "the Explore capability"),
  });
}

export function useExploreRequest() {
  return useMutation<ExploreResponse, Error, { imageId: number; body: ExploreRequest }>({
    mutationFn: async ({ imageId, body }) =>
      unwrap(
        await api.POST("/api/images/{image_id}/explore", {
          params: { path: { image_id: imageId } },
          body,
        }),
        "what the encoder sees",
      ),
  });
}

export function useExploreShape() {
  return useMutation<ExploreShape, Error, { mapId: string; body: ExploreShapeRequest }>({
    mutationFn: async ({ mapId, body }) =>
      unwrap(
        await api.POST("/api/explore/maps/{map_id}/shape", {
          params: { path: { map_id: mapId } },
          body,
        }),
        "the candidate mask",
      ),
  });
}
