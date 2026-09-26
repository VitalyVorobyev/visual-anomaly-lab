/**
 * What a dataset's truth is for (ADR-0041).
 *
 * A sample's normal/defect label is **anomaly** truth; a class drawn in a completed
 * annotation is **class** truth. The server derives which of the two a dataset holds
 * (`DatasetSummary.truth`), and the screens follow it: a dataset of classes alone shows its
 * classes, not a column of zero defects, and offers the tasks that read classes.
 */

import type { Task, TruthKind } from "./client";

type Truth = readonly TruthKind[] | undefined;

export function hasLabels(truth: Truth): boolean {
  return truth?.includes("labels") ?? false;
}

export function hasClasses(truth: Truth): boolean {
  return truth?.includes("classes") ?? false;
}

/**
 * Whether the anomaly-labelling surfaces belong on screen: the label rail, its hotkeys, bulk
 * labelling. Yes for a dataset with labels, and for one with no truth yet — labelling is how
 * an unlabelled import becomes an anomaly dataset. No for a dataset of classes alone, which
 * opts in explicitly. Unknown truth (still loading) keeps them, so an anomaly dataset never
 * loses its rail for a frame.
 */
export function labelsApply(truth: Truth): boolean {
  return truth === undefined || hasLabels(truth) || !hasClasses(truth);
}

/**
 * Whether a dataset's truth can serve a task. An anomaly run needs anomaly verdicts, so a
 * dataset of classes alone is not offered one; every other task reads classes, and a dataset
 * with none yet can still be annotated for it.
 */
export function truthServesTask(truth: Truth, task: Task): boolean {
  return task !== "anomaly" || labelsApply(truth);
}
