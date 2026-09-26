/**
 * What a dataset still needs before a run of each task can train on it.
 *
 * Every run needs a *built* region profile. An anomaly run needs a split drawn or adopted for
 * it; a few-shot run needs a class with a reference and something to test on, and a split of
 * references (ADR-0040); a supervised run — segmentation or detection — needs the same annotated
 * class and a split of annotated samples, drawn by class or listed by hand (ADR-0039). This reads
 * the facts the create form checks, with the same rule for
 * "built" (a build report exists and no image failed), so the band and the form cannot
 * disagree about whether a dataset is ready.
 */

import { useQueries } from "@tanstack/react-query";

import { api, unwrap } from "../api/client";
import type { RegionBuildSummary, SplitDetail, Task } from "../api/client";
import { queryKeys } from "../api/queryKeys";
import { useClassCoverage } from "./useAnnotations";
import { truthServesTask } from "../api/truth";
import { useDataset, useSplits } from "./useCatalog";
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
export const READINESS_TASKS: Task[] = [
  "anomaly",
  "few_shot_segmentation",
  "semantic_segmentation",
  "object_detection",
];

/** The tasks that learn from annotation, and so wait on a class that has some. */
const ANNOTATED_TASKS: Task[] = [
  "few_shot_segmentation",
  "semantic_segmentation",
  "object_detection",
];

/**
 * The tasks that fit on annotated samples of every pinned class (ADR-0039). Segmentation and
 * detection share one presence rule, so one split serves both.
 */
export const SUPERVISED_TASKS: Task[] = ["semantic_segmentation", "object_detection"];

/**
 * Which split strategies a task trains on: an anomaly run on a drawn or adopted partition
 * whose train is normals, a few-shot run on a split of references (ADR-0040), and a
 * supervised run — segmentation or detection — on annotated samples, drawn by class
 * (`class_stratified`) or listed by hand (`manual`) (ADR-0039).
 */
export function splitServesTask(split: Pick<SplitDetail, "strategy">, task: Task): boolean {
  if (task === "few_shot_segmentation") {
    return split.strategy === "manual" || split.strategy === "few_shot";
  }
  if (SUPERVISED_TASKS.includes(task)) {
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
  const dataset = useDataset(datasetId);
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
  // A task some method runs, that the dataset's truth can serve: a dataset of classes alone
  // is not offered anomaly detection until a sample carries a verdict (ADR-0041). Not waited
  // on: the band above reads the same dataset, so it is in cache, and an unknown truth
  // offers every task rather than none.
  const offered = READINESS_TASKS.filter(
    (task) =>
      (catalog.data?.methods ?? []).some((method) => method.capabilities.tasks.includes(task)) &&
      truthServesTask(dataset.data?.truth, task),
  );
  // A class a segmentation run can use has an annotated sample to learn from and another to
  // test on — for few-shot a reference, for a supervised run a training image.
  const usableClass = (coverage.data ?? []).some(
    (entry) => entry.present >= 1 && entry.present + entry.absent >= 2,
  );
  const tasks = offered.map((task) => {
    const missing: ReadinessStep[] = [];
    if (builtProfiles === 0) missing.push("prepare");
    if (ANNOTATED_TASKS.includes(task) && !usableClass) missing.push("annotate");
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
      (!offered.some((task) => ANNOTATED_TASKS.includes(task)) || coverage.data !== undefined) &&
      builds.every((build) => !build.isPending),
    profiles: profiles.data?.length ?? 0,
    builtProfiles,
    splits: splits.data?.length ?? 0,
    runs: runs.data?.total ?? 0,
    tasks,
  };
}
