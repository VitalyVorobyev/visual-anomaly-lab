/** Licensed model assets are server state; acquisition is an ordinary job. */

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

import { api, unwrap } from "../api/client";
import type { JobSummary, ModelAssetCatalog } from "../api/client";
import { queryKeys } from "../api/queryKeys";

export function useModelAssets() {
  return useQuery<ModelAssetCatalog>({
    queryKey: queryKeys.modelAssets(),
    queryFn: async () =>
      unwrap(await api.GET("/api/model-assets"), "the model asset catalogue"),
  });
}

export function useInstallModelAsset() {
  const queryClient = useQueryClient();
  return useMutation<JobSummary, Error, string>({
    mutationFn: async (assetKey) =>
      unwrap(
        await api.POST("/api/model-assets/{asset_key}/install", {
          params: { path: { asset_key: assetKey } },
          body: { license_accepted: true },
        }),
        "the model asset download job",
      ),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: queryKeys.modelAssets() });
    },
  });
}
