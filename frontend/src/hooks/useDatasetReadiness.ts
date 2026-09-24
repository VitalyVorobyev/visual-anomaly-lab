/**
 * What a dataset still needs before a run can train on it.
 *
 * Training needs a *built* region profile and a split, and the only place that used to say
 * so was the create form — after the reader had already opened it. This reads the same two
 * facts the form checks, with the same rule for "built" (a build report exists and no
 * image failed), so the band and the form cannot disagree about whether a dataset is ready.
 */

import { useQueries } from "@tanstack/react-query";

import { api, unwrap } from "../api/client";
import type { RegionBuildSummary } from "../api/client";
import { queryKeys } from "../api/queryKeys";
import { useSplits } from "./useCatalog";
import { useExperiments } from "./useExperiments";
import { useRegionProfiles } from "./useRegionProfiles";

export interface DatasetReadiness {
  /** False until every read it depends on has answered. */
  known: boolean;
  profiles: number;
  /** Profiles whose build finished with no failed image. */
  builtProfiles: number;
  splits: number;
  runs: number;
  /** What is still missing, in the order it has to be done. */
  missing: ReadinessStep[];
}

export type ReadinessStep = "prepare" | "split";

/** A build is usable when it exists and nothing in it failed — the create form's rule. */
export function isUsableBuild(build: RegionBuildSummary | undefined): boolean {
  return build !== undefined && build.failed === 0;
}

export function useDatasetReadiness(datasetId: number | undefined): DatasetReadiness {
  const profiles = useRegionProfiles(datasetId);
  const splits = useSplits(datasetId);
  const runs = useExperiments(datasetId === undefined ? {} : { datasetId });
  const builds = useQueries({
    queries: (profiles.data ?? []).map((profile) => ({
      queryKey: queryKeys.regionBuild(profile.id),
      queryFn: async () =>
        unwrap(
          await api.GET("/api/region-profiles/{profile_id}/build", {
            params: { path: { profile_id: profile.id } },
          }),
          "the region build report",
        ),
      // An unbuilt profile answers 404, which is a fact here, not a failure to retry.
      retry: false,
    })),
  });

  const builtProfiles = builds.filter((build) => isUsableBuild(build.data)).length;
  const missing: ReadinessStep[] = [];
  if (builtProfiles === 0) missing.push("prepare");
  if ((splits.data?.length ?? 0) === 0) missing.push("split");

  return {
    known:
      profiles.data !== undefined &&
      splits.data !== undefined &&
      builds.every((build) => !build.isPending),
    profiles: profiles.data?.length ?? 0,
    builtProfiles,
    splits: splits.data?.length ?? 0,
    runs: runs.data?.length ?? 0,
    missing,
  };
}
