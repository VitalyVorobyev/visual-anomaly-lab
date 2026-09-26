/**
 * The guided run's choices, and the defaults they start from.
 *
 * A guided run is five decisions — goal, look, split, method, run — each with a default, so
 * pressing Next through all of them gives a run that makes sense. A choice left alone is
 * `undefined` here and resolved on screen from what the dataset and the registry say; only a
 * choice the reader made is stored, so a default that changes underneath (a new preset, a
 * method made the recommendation) is followed rather than frozen.
 *
 * The step is in the URL; the choices are in `sessionStorage`, per dataset, so a detour — to
 * Prepare to adjust a region profile, to the sample viewer — comes back to the same run.
 * Every access is guarded: storage can be absent or refuse, and the run must work without it.
 */

import type { RawValues } from "@vitavision/lab-ui";

import type { ClassGeometry, Task, TruthKind } from "./client";
import { hasClasses, labelsApply } from "./truth";

export const STEPS = ["goal", "look", "split", "method", "run"] as const;
export type Step = (typeof STEPS)[number];

export const STEP_TITLE: Record<Step, string> = {
  goal: "Goal",
  look: "Look",
  split: "Split",
  method: "Method",
  run: "Run",
};

/** Which split: a preset still to be created, or one the dataset already has. */
export type SplitChoice = { kind: "preset"; key: string } | { kind: "existing"; id: number };

export interface GuidedRunState {
  task?: Task;
  /** The class a few-shot run segments; unset is the most frequent. */
  targetClass?: string;
  /** A saved region profile revision; unset is the dataset's "Full frame". */
  profileId?: number;
  split?: SplitChoice;
  methodKey?: string;
  /** Configuration typed for `methodKey`; cleared when the method changes. */
  configValues: RawValues;
  name: string;
  /** The furthest step reached, so the rail can offer every step up to it. */
  reached: Step;
  /**
   * A split this run already created before a later stage failed, keyed by what it was
   * drawn from, so pressing Start again reuses it rather than drawing another.
   */
  created?: { key: string; splitId: number };
}

export const EMPTY_GUIDED_RUN: GuidedRunState = { configValues: {}, name: "", reached: "goal" };

export function stepIndex(step: Step): number {
  return STEPS.indexOf(step);
}

export function readStep(value: string | null): Step {
  return STEPS.find((step) => step === value) ?? "goal";
}

/** The later of two steps: how far the reader has been. */
export function furthest(a: Step, b: Step): Step {
  return stepIndex(a) >= stepIndex(b) ? a : b;
}

/**
 * The task a dataset's truth answers most directly (ADR-0041): verdicts ask anomaly
 * detection; classes drawn as boxes ask detection; classes drawn as regions ask few-shot
 * segmentation, which needs the least of them. Only a task in `offered` is suggested.
 */
export function suggestedTask(
  truth: readonly TruthKind[] | undefined,
  geometry: ClassGeometry | null | undefined,
  offered: readonly Task[],
): Task | undefined {
  const order: Task[] = [];
  if (truth?.includes("labels")) order.push("anomaly");
  if (hasClasses(truth)) {
    order.push(
      ...(geometry === "boxes"
        ? (["object_detection", "few_shot_segmentation"] as Task[])
        : (["few_shot_segmentation", "semantic_segmentation", "object_detection"] as Task[])),
    );
  }
  // No truth yet: labelling is how an import becomes an anomaly dataset.
  if (labelsApply(truth)) order.push("anomaly");
  return order.find((task) => offered.includes(task)) ?? offered[0];
}

export function guidedRunKey(datasetId: number): string {
  return `anomaly-lab:guided-run:${datasetId}`;
}

export function readGuidedRun(key: string, storage = safeSession()): GuidedRunState {
  try {
    const raw = storage?.getItem(key);
    if (!raw) return EMPTY_GUIDED_RUN;
    const parsed = JSON.parse(raw) as Partial<GuidedRunState> | null;
    if (typeof parsed !== "object" || parsed === null) return EMPTY_GUIDED_RUN;
    return {
      task: TASKS.find((task) => task === parsed.task),
      targetClass: typeof parsed.targetClass === "string" ? parsed.targetClass : undefined,
      profileId: finite(parsed.profileId),
      split: splitChoice(parsed.split),
      methodKey: typeof parsed.methodKey === "string" ? parsed.methodKey : undefined,
      configValues:
        typeof parsed.configValues === "object" &&
        parsed.configValues !== null &&
        !Array.isArray(parsed.configValues)
          ? (parsed.configValues as RawValues)
          : {},
      name: typeof parsed.name === "string" ? parsed.name : "",
      reached: readStep(typeof parsed.reached === "string" ? parsed.reached : null),
      created:
        typeof parsed.created?.key === "string" && finite(parsed.created.splitId) !== undefined
          ? { key: parsed.created.key, splitId: parsed.created.splitId }
          : undefined,
    };
  } catch {
    return EMPTY_GUIDED_RUN;
  }
}

export function writeGuidedRun(key: string, state: GuidedRunState, storage = safeSession()): void {
  try {
    storage?.setItem(key, JSON.stringify(state));
  } catch {
    // A full or refused store costs the convenience, never the run.
  }
}

export function clearGuidedRun(key: string, storage = safeSession()): void {
  try {
    storage?.removeItem(key);
  } catch {
    // As above.
  }
}

const TASKS: Task[] = [
  "anomaly",
  "few_shot_segmentation",
  "semantic_segmentation",
  "object_detection",
];

function splitChoice(value: unknown): SplitChoice | undefined {
  if (typeof value !== "object" || value === null) return undefined;
  const choice = value as Record<string, unknown>;
  if (choice["kind"] === "preset" && typeof choice["key"] === "string") {
    return { kind: "preset", key: choice["key"] };
  }
  const id = finite(choice["id"]);
  if (choice["kind"] === "existing" && id !== undefined) return { kind: "existing", id };
  return undefined;
}

function finite(value: unknown): number | undefined {
  return typeof value === "number" && Number.isFinite(value) ? value : undefined;
}

function safeSession(): Storage | undefined {
  try {
    return globalThis.sessionStorage;
  } catch {
    return undefined;
  }
}
