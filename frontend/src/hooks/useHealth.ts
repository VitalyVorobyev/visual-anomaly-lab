import { useQuery } from "@tanstack/react-query";

import { api, type HealthResponse } from "../api/client";

export const HEALTH_POLL_INTERVAL_MS = 5000;

/**
 * Live backend health.
 *
 * Polled rather than fetched once: the whole point of the screen is to show whether the
 * sidecar is up *right now*, including after it has died.
 */
export function useHealth() {
  return useQuery<HealthResponse>({
    queryKey: ["health"],
    queryFn: async () => {
      const { data, error } = await api.GET("/api/health");
      if (error !== undefined || data === undefined) {
        throw new Error("The backend did not return a health report.");
      }
      return data;
    },
    refetchInterval: HEALTH_POLL_INTERVAL_MS,
    retry: false,
  });
}
