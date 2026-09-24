/**
 * What a dataset still needs before a run of each task can train on it.
 *
 * Every run needs a *built* region profile. An anomaly run needs a split drawn or adopted for
 * it; a few-shot run needs a class with a reference and something to test on, and a split of
 * references (ADR-0040). This reads the facts the create form checks, with the same rule for
 * "built" (a build report exists and no image failed), so the band and the form cannot
 * disagree about whether a dataset is ready.
 */

import { useQueries } from "@tanstack/react-query";

import { api, unwrap } from "../api/client";
import type { RegionBuildSummary, SplitDetail, Task } from "../api/client";
import { queryKeys } from "../api/queryKeys";
import { useClassCoverage } from "./useAnnotations";
import { useSplits } from "./useCatalog";
import { useExperiments, useModelTypes } from "./useExperiments";
import { useRegionProfiles } from "./useRegionProfiles";

export interface DatasetReadiness {
  /** False until every read it depends on has answered. */
  known: boolean;
  profiles: number;
  /** Profiles whose build finished with no failed image. */
  builtProfiles: number;
  splits: number;
  runs: number;
  /**
   * Each task some method can run (ADR-0039), with what it still needs, in the order it has to
   * be done. A task whose list is empty can be trained now.
   */
  tasks: TaskReadiness[];
}

export interface TaskReadiness {
  task: Task;
  missing: ReadinessStep[];
}

export type ReadinessStep = "prepare" | "split" | "annotate" | "references";

/** The tasks the band speaks for, in the order it names them. */
export const READINESS_TASKS: Task[] = ["anomaly", "few_shot_segmentation"];

/**
 * Which split strategies a task trains on: an anomaly run on a drawn or adopted partition
 * whose train is normals, a few-shot run on a split of references (ADR-0040), and a
 * supervised run on annotated samples — drawn by class (`class_stratified`) or listed by hand
 * (`manual`) (ADR-0039).
 */
export function splitServesTask(split: Pick<SplitDetail, "strategy">, task: Task): boolean {
  if (task === "few_shot_segmentation") {
    return split.strategy === "manual" || split.strategy === "few_shot";
  }
  if (task === "semantic_segmentation") {
    return split.strategy === "class_stratified" || split.strategy === "manual";
  }
  return split.strategy === "normal_only_train" || split.strategy === "imported";
}

/** A build is usable when it exists and nothing in it failed — the create form's rule. */
export function isUsableBuild(build: RegionBuildSummary | undefined): boolean {
  return build !== undefined && build.failed === 0;
}

export function useDatasetReadiness(datasetId: number | undefined): DatasetReadiness {
  const profiles = useRegionProfiles(datasetId);
  const splits = useSplits(datasetId);
  const runs = useExperiments(datasetId === undefined ? {} : { datasetId });
  const catalog = useModelTypes();
  const coverage = useClassCoverage(datasetId);
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
  const offered = READINESS_TASKS.filter((task) =>
    (catalog.data?.methods ?? []).some((method) => method.capabilities.tasks.includes(task)),
  );
  // A class a few-shot run can use has a reference to learn from and something else to test on.
  const usableClass = (coverage.data ?? []).some(
    (entry) => entry.present >= 1 && entry.present + entry.absent >= 2,
  );
  const tasks = offered.map((task) => {
    const missing: ReadinessStep[] = [];
    if (builtProfiles === 0) missing.push("prepare");
    if (task === "few_shot_segmentation" && !usableClass) missing.push("annotate");
    if (!(splits.data ?? []).some((split) => splitServesTask(split, task))) {
      missing.push(task === "few_shot_segmentation" ? "references" : "split");
    }
    return { task, missing };
  });

  return {
    known:
      profiles.data !== undefined &&
      splits.data !== undefined &&
      catalog.data !== undefined &&
      (!offered.includes("few_shot_segmentation") || coverage.data !== undefined) &&
      builds.every((build) => !build.isPending),
    profiles: profiles.data?.length ?? 0,
    builtProfiles,
    splits: splits.data?.length ?? 0,
    runs: runs.data?.total ?? 0,
    tasks,
  };
}
