/**
 * The unsent create-experiment form, kept for the length of a browser session.
 *
 * The form's prerequisite links — "Prepare the dataset", "Create a split" — leave the page,
 * and the form used to be component state, so following the advice it gave discarded
 * everything typed so far. The draft is a convenience of one reader in one tab, which is
 * exactly `sessionStorage`; it is cleared when the experiment is created. Every access is
 * guarded, because storage can be absent or refuse (a private window, a locked-down
 * profile) and the form must work the same without it.
 */

import type { RawValues } from "@vitavision/lab-ui";

import type { Task } from "./client";

const TASKS: Task[] = [
  "anomaly",
  "few_shot_segmentation",
  "semantic_segmentation",
  "object_detection",
];

export interface ExperimentDraft {
  name: string;
  /** The task decides everything after it, so a draft without it restores the wrong form. */
  task?: Task | undefined;
  /** The class a few-shot run segments, when the reader chose one. */
  targetLabel?: string | undefined;
  datasetId?: number | undefined;
  splitId?: number | undefined;
  regionProfileId?: number | undefined;
  methodKey?: string | undefined;
  /** Configuration values belong to the method they were typed for. */
  configValues: RawValues;
  preprocessingValues: RawValues;
  evaluationValues: RawValues;
  channels: string[];
  /** The typed input size; empty means the method's own. */
  width?: string;
  height?: string;
}

/** One draft per mount: the dataset-scoped form and the cross-dataset one do not share. */
export function draftKey(datasetId: number | undefined): string {
  return `anomaly-lab:experiment-draft:${datasetId ?? "any"}`;
}

export function readDraft(key: string, storage = safeSession()): ExperimentDraft | null {
  try {
    const raw = storage?.getItem(key);
    if (!raw) return null;
    const parsed = JSON.parse(raw) as Partial<ExperimentDraft>;
    if (typeof parsed !== "object" || parsed === null) return null;
    return {
      name: typeof parsed.name === "string" ? parsed.name : "",
      task: TASKS.find((task) => task === parsed.task),
      targetLabel: typeof parsed.targetLabel === "string" ? parsed.targetLabel : "",
      datasetId: numberOrUndefined(parsed.datasetId),
      splitId: numberOrUndefined(parsed.splitId),
      regionProfileId: numberOrUndefined(parsed.regionProfileId),
      methodKey: typeof parsed.methodKey === "string" ? parsed.methodKey : undefined,
      configValues: recordOrEmpty(parsed.configValues),
      preprocessingValues: recordOrEmpty(parsed.preprocessingValues),
      evaluationValues: recordOrEmpty(parsed.evaluationValues),
      channels: Array.isArray(parsed.channels)
        ? parsed.channels.filter((entry): entry is string => typeof entry === "string")
        : [],
      width: typeof parsed.width === "string" ? parsed.width : "",
      height: typeof parsed.height === "string" ? parsed.height : "",
    };
  } catch {
    return null;
  }
}

export function writeDraft(key: string, draft: ExperimentDraft, storage = safeSession()): void {
  try {
    storage?.setItem(key, JSON.stringify(draft));
  } catch {
    // A full or refused store costs the convenience, never the form.
  }
}

export function clearDraft(key: string, storage = safeSession()): void {
  try {
    storage?.removeItem(key);
  } catch {
    // As above.
  }
}

function safeSession(): Storage | undefined {
  try {
    return globalThis.sessionStorage;
  } catch {
    return undefined;
  }
}

function numberOrUndefined(value: unknown): number | undefined {
  return typeof value === "number" && Number.isFinite(value) ? value : undefined;
}

function recordOrEmpty(value: unknown): RawValues {
  return typeof value === "object" && value !== null && !Array.isArray(value)
    ? (value as RawValues)
    : {};
}
