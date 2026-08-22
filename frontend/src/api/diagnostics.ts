/**
 * Reading the diagnostics index (ADR-0018).
 *
 * The one rule this file exists to enforce: **nothing selects a renderer by `key`**. A
 * key is a caption and a file name; `kind` is the instruction. `pixel_reference` proves
 * why it matters — it emits `reference_median` as an `image` on a three-channel
 * preprocessing config and as a `map` on a greyscale one, from the same line of Python.
 * Anything keyed on the name would draw one of those two wrong.
 *
 * That is also what makes M6 free: `efficientad_custom` emits the same kinds and every
 * view here works on it with no new code.
 */

import { apiBaseUrl } from "./client";
import type { DiagnosticEntry, DiagnosticIndex, DiagnosticOrigin } from "./client";

/** Kinds whose payload is an array behind the payload route, rather than inline JSON. */
const ARRAY_KINDS = new Set(["map", "image", "grid"]);

export function isArrayKind(entry: DiagnosticEntry): boolean {
  return ARRAY_KINDS.has(entry.kind);
}

/**
 * Where to fetch one diagnostic's rendered PNG.
 *
 * Built by hand rather than through the typed client because it is consumed by `<img
 * src>`, the same reasoning as `imageUrl` — the browser does the request, the caching and
 * the decoding, and a `fetch` wrapper would get in the way of all three.
 */
export function diagnosticPayloadUrl(
  experimentId: number,
  entry: DiagnosticEntry,
  frame = 0,
): string {
  const query = new URLSearchParams({ key: entry.key });
  if (entry.image_id !== null && entry.image_id !== undefined) {
    query.set("image_id", String(entry.image_id));
  }
  if (frame > 0) query.set("frame", String(frame));
  return `${apiBaseUrl}/api/experiments/${experimentId}/diagnostics/payload?${query.toString()}`;
}

/** Run-scoped entries — the architecture graph, what the teacher sees. */
export function modelScoped(index: DiagnosticIndex | undefined): DiagnosticEntry[] {
  return (index?.entries ?? []).filter((entry) => entry.scope === "model");
}

/** Entries recorded about one particular image. */
export function imageScoped(
  index: DiagnosticIndex | undefined,
  imageId: number,
): DiagnosticEntry[] {
  return (index?.entries ?? []).filter(
    (entry) => entry.scope === "image" && entry.image_id === imageId,
  );
}

/**
 * Every image the index holds anything about, in ascending order.
 *
 * `origin` narrows it to one producer. The distinction is load-bearing rather than
 * cosmetic: the run's images are a *sample* whose size the budget explains, and the
 * on-demand ones are questions somebody asked (handbook diagnostics.md). Counting them together produces
 * a number larger than the stated budget, which reads as a cap that does not work.
 */
export function diagnosedImageIds(
  index: DiagnosticIndex | undefined,
  origin?: DiagnosticOrigin,
): number[] {
  const ids = new Set<number>();
  for (const entry of index?.entries ?? []) {
    if (entry.scope !== "image" || typeof entry.image_id !== "number") continue;
    if (origin !== undefined && entry.origin !== origin) continue;
    ids.add(entry.image_id);
  }
  return [...ids].sort((left, right) => left - right);
}

/** Whether this entry is an answer to a question, rather than part of a run's sample. */
export function isOnDemand(entry: DiagnosticEntry): boolean {
  return entry.origin === "on_demand";
}

/** Narrow a set of entries to the kinds a view can draw. */
export function ofKinds(entries: DiagnosticEntry[], kinds: string[]): DiagnosticEntry[] {
  const wanted = new Set(kinds);
  return entries.filter((entry) => wanted.has(entry.kind));
}

/** How many small multiples a `grid` payload holds, from the shape the index recorded. */
export function gridFrameCount(entry: DiagnosticEntry): number {
  const first = entry.shape?.[0];
  return typeof first === "number" && first > 0 ? first : 0;
}

/**
 * What to say when an image has no diagnostics, given the run's budget.
 *
 * `null` when there is nothing worth saying. The budget is recorded in the index for
 * exactly this: "that image was not one of the first twelve" is a different fact from
 * "this model records nothing", and a blank panel says neither.
 */
export function budgetNote(index: DiagnosticIndex | undefined): string | null {
  if (!index) return null;
  const budget = index.image_budget;
  if (budget === null || budget === undefined) return null;
  // Run-origin only. `image_budget` and `truncated_images` describe a run, so counting an
  // image somebody diagnosed afterwards against them would report a sample the run never
  // took — and, once a few images have been asked about, a count above the stated cap.
  const kept = diagnosedImageIds(index, "run").length;
  if (index.truncated_images === 0) return null;
  return (
    `Per-image diagnostics were recorded for ${kept} image(s); ` +
    `${index.truncated_images} more were dropped at the run's budget of ${budget}.`
  );
}

/** How many images were diagnosed after the fact, or `null` when none were. */
export function onDemandNote(index: DiagnosticIndex | undefined): string | null {
  const asked = diagnosedImageIds(index, "on_demand").length;
  if (asked === 0) return null;
  return `${asked} image(s) diagnosed on demand, kept until you clear them.`;
}

/**
 * Why *this* image has no panes — the best explanation the index supports.
 *
 * Three genuinely different facts, and the panel used to say only the last one. Beside a
 * button offering to record diagnostics, "this method records nothing" is not merely
 * imprecise, it contradicts what is on screen: the common case is a run that recorded
 * plenty, for other images.
 */
export function missingNote(index: DiagnosticIndex | undefined): string {
  const budget = budgetNote(index);
  if (budget !== null) return budget;

  const elsewhere = diagnosedImageIds(index).length;
  if (elsewhere > 0) {
    return (
      `This run recorded per-image diagnostics for ${elsewhere} other image(s), ` +
      "chosen across the run rather than by what you opened."
    );
  }
  return (
    "This run recorded no per-image diagnostics. A method reports them by calling " +
    "ctx.emit_diagnostic during predict."
  );
}
