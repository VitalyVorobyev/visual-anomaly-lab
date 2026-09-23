/**
 * The tone each lifecycle status is drawn in, in one place.
 *
 * The experiment header coloured only `trained` and drew a failed run grey, while the
 * catalogue drew the same run red — one fact, two colours, a screen apart. `defect` is
 * the one tone that means "this went wrong", so a failure gets it everywhere or nowhere.
 */

import type { Tone } from "@vitavision/lab-ui";

import type { ExperimentStatus, JobStatus } from "./client";

const EXPERIMENT: Record<ExperimentStatus, Tone> = {
  draft: "unlabeled",
  training: "info",
  trained: "normal",
  failed: "defect",
};

const JOB: Record<JobStatus, Tone> = {
  queued: "neutral",
  running: "info",
  succeeded: "normal",
  failed: "defect",
  cancelled: "unlabeled",
};

export function experimentStatusTone(status: string): Tone {
  return EXPERIMENT[status as ExperimentStatus] ?? "neutral";
}

export function jobStatusTone(status: string): Tone {
  return JOB[status as JobStatus] ?? "neutral";
}
