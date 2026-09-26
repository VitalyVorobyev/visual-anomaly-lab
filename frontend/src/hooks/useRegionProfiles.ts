/**
 * Dataset-owned spatial-input profiles and their bounded preparation jobs.
 *
 * A profile says where to look and carries no size: a build is one profile at one size, so
 * every preparation call names the size it prepares at.
 */

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

import { api, unwrap } from "../api/client";
import type {
  JobSummary,
  RegionBuildSummary,
  RegionExtractorDescription,
  RegionProfileCreate,
  RegionProfileDeletionPreview,
  RegionProfileDeletionResult,
  RegionProfileRevision,
} from "../api/client";
import { queryKeys } from "../api/queryKeys";

export function useRegionExtractors() {
  return useQuery<RegionExtractorDescription[]>({
    queryKey: queryKeys.regionExtractors(),
    queryFn: async () =>
      unwrap(await api.GET("/api/region-extractors"), "the region extractor catalogue"),
  });
}

export function useRegionProfiles(datasetId: number | undefined) {
  return useQuery<RegionProfileRevision[]>({
    queryKey: queryKeys.regionProfiles(datasetId ?? -1),
    queryFn: async () =>
      unwrap(
        await api.GET("/api/datasets/{dataset_id}/region-profiles", {
          params: { path: { dataset_id: datasetId as number } },
        }),
        "the region profiles",
      ),
    enabled: datasetId !== undefined,
  });
}

export function useCreateRegionProfile(datasetId: number) {
  const queryClient = useQueryClient();
  return useMutation<RegionProfileRevision, Error, RegionProfileCreate>({
    mutationFn: async (body) =>
      unwrap(
        await api.POST("/api/datasets/{dataset_id}/region-profiles", {
          params: { path: { dataset_id: datasetId } },
          body,
        }),
        "the region profile revision",
      ),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: queryKeys.regionProfiles(datasetId) });
    },
  });
}

export function useRegionProfileDeletionPreview(profileId: number | undefined) {
  return useQuery<RegionProfileDeletionPreview>({
    queryKey: queryKeys.regionProfileDeletion(profileId ?? -1),
    queryFn: async () =>
      unwrap(
        await api.GET("/api/region-profiles/{profile_id}/deletion-preview", {
          params: { path: { profile_id: profileId as number } },
        }),
        "the deletion preview",
      ),
    enabled: profileId !== undefined,
  });
}

export function useDeleteRegionProfile(datasetId: number) {
  const queryClient = useQueryClient();
  return useMutation<RegionProfileDeletionResult, Error, number>({
    mutationFn: async (profileId) =>
      unwrap(
        await api.DELETE("/api/region-profiles/{profile_id}", {
          params: { path: { profile_id: profileId } },
        }),
        "the deletion result",
      ),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: queryKeys.regionProfiles(datasetId) });
    },
  });
}

/** A prepared frame's `width` × `height`, in pixels. */
export interface FrameSize {
  width: number;
  height: number;
}

export interface PreparationRequest {
  profileId: number;
  size: FrameSize;
}

export function useStartRegionPreview() {
  return useMutation<JobSummary, Error, PreparationRequest>({
    mutationFn: async ({ profileId, size }) =>
      unwrap(
        await api.POST("/api/region-profiles/{profile_id}/preview", {
          params: { path: { profile_id: profileId } },
          body: size,
        }),
        "the region preview job",
      ),
  });
}

export function useStartRegionBuild() {
  return useMutation<JobSummary, Error, PreparationRequest>({
    mutationFn: async ({ profileId, size }) =>
      unwrap(
        await api.POST("/api/region-profiles/{profile_id}/build", {
          params: { path: { profile_id: profileId } },
          body: size,
        }),
        "the region build job",
      ),
  });
}

/** The completed build of one profile at one size; a 404 means "not built at this size". */
export function useRegionBuild(
  profileId: number | undefined,
  size: FrameSize | undefined,
  enabled = true,
) {
  return useQuery<RegionBuildSummary>({
    queryKey: queryKeys.regionBuild(profileId ?? -1, size?.width ?? -1, size?.height ?? -1),
    queryFn: async () =>
      unwrap(
        await api.GET("/api/region-profiles/{profile_id}/build", {
          params: {
            path: { profile_id: profileId as number },
            query: { width: size?.width as number, height: size?.height as number },
          },
        }),
        "the region build report",
      ),
    enabled: profileId !== undefined && size !== undefined && enabled,
    retry: false,
  });
}

/** Every completed build of a profile, one per size. */
export function useRegionBuilds(profileId: number | undefined) {
  return useQuery<RegionBuildSummary[]>({
    queryKey: queryKeys.regionBuilds(profileId ?? -1),
    queryFn: async () =>
      unwrap(
        await api.GET("/api/region-profiles/{profile_id}/builds", {
          params: { path: { profile_id: profileId as number } },
        }),
        "the region builds",
      ),
    enabled: profileId !== undefined,
  });
}
