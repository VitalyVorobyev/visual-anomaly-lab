/** Dataset-owned spatial-input profiles and their bounded preparation jobs. */

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

export function useStartRegionPreview() {
  return useMutation<JobSummary, Error, number>({
    mutationFn: async (profileId) =>
      unwrap(
        await api.POST("/api/region-profiles/{profile_id}/preview", {
          params: { path: { profile_id: profileId } },
        }),
        "the region preview job",
      ),
  });
}

export function useStartRegionBuild() {
  return useMutation<JobSummary, Error, number>({
    mutationFn: async (profileId) =>
      unwrap(
        await api.POST("/api/region-profiles/{profile_id}/build", {
          params: { path: { profile_id: profileId } },
        }),
        "the region build job",
      ),
  });
}

export function useRegionBuild(profileId: number | undefined, enabled = true) {
  return useQuery<RegionBuildSummary>({
    queryKey: queryKeys.regionBuild(profileId ?? -1),
    queryFn: async () =>
      unwrap(
        await api.GET("/api/region-profiles/{profile_id}/build", {
          params: { path: { profile_id: profileId as number } },
        }),
        "the region build report",
      ),
    enabled: profileId !== undefined && enabled,
    retry: false,
  });
}
