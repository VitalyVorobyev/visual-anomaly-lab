/**
 * A run's input size on the create form: empty means the method's own.
 *
 * The size belongs to the run, not to the region profile — a profile says only where to look.
 * Left empty, the backend resolves the method's native size for its configuration, which the
 * form shows beside the fields; typed, both dimensions snap to the multiple the method reads
 * (a DINO backbone's patch), and the caption says why.
 */

import type { InputSizeAnswer, RegionProfileRevision } from "./client";

export const MIN_INPUT = 8;
export const MAX_INPUT = 2048;

/** The name and extractor the backend gives every dataset's implicit profile. */
export const FULL_FRAME_NAME = "Full frame";

/** The dataset's implicit "Full frame" profile — its newest revision — if it is listed. */
export function fullFrameProfile(
  profiles: RegionProfileRevision[],
): RegionProfileRevision | undefined {
  return profiles
    .filter((profile) => profile.name === FULL_FRAME_NAME && profile.extractor_type === "identity")
    .sort((a, b) => b.revision_no - a.revision_no)[0];
}

export interface InputSizeState {
  /** What to send, or undefined for "the method's own". */
  named?: { width: number; height: number };
  /** Why the typed size cannot be sent, said beside the fields. */
  error?: string;
  /** The line under the fields: the resolved default, or the rule a typed size follows. */
  caption: string;
  /** The multiple both dimensions snap to; 1 when any size reads. */
  multiple: number;
}

function patchReason(multiple: number): string {
  return `the method reads ${multiple}-pixel patches, so both sides are multiples of ${multiple}`;
}

export function inputSizeState(
  width: string,
  height: string,
  answer: InputSizeAnswer | undefined,
): InputSizeState {
  const multiple = Math.max(1, answer?.multiple ?? 1);
  const native = answer
    ? `${answer.width} × ${answer.height} · from ${answer.model_type}`
    : "The method's own size.";
  const rule = multiple > 1 ? ` Snaps to ${multiple}: ${patchReason(multiple)}.` : "";
  const blankWidth = width.trim() === "";
  const blankHeight = height.trim() === "";
  if (blankWidth && blankHeight) {
    return { caption: native, multiple };
  }
  const caption = `Default ${native}.${rule}`;
  if (blankWidth || blankHeight) {
    return { caption, multiple, error: "Give both sides, or neither for the method's own." };
  }
  const w = Number(width);
  const h = Number(height);
  const inRange = (value: number) =>
    Number.isInteger(value) && value >= MIN_INPUT && value <= MAX_INPUT;
  if (!inRange(w) || !inRange(h)) {
    return { caption, multiple, error: `Whole pixels between ${MIN_INPUT} and ${MAX_INPUT}.` };
  }
  if (w % multiple !== 0 || h % multiple !== 0) {
    return { caption, multiple, error: `Not a multiple of ${multiple}: ${patchReason(multiple)}.` };
  }
  return { named: { width: w, height: h }, caption, multiple };
}

/** The nearest multiple of `multiple` within bounds; an empty or unreadable value is kept. */
export function snapToMultiple(value: string, multiple: number): string {
  const number = Number(value);
  if (value.trim() === "" || !Number.isFinite(number)) return value;
  const step = Math.max(1, multiple);
  const lowest = Math.ceil(MIN_INPUT / step) * step;
  const highest = Math.floor(MAX_INPUT / step) * step;
  const snapped = Math.round(number / step) * step;
  return String(Math.min(highest, Math.max(lowest, snapped)));
}
