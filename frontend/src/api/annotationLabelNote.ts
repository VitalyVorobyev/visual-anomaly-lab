/**
 * What to say when a sample's label and the document open above it disagree.
 *
 * The editor can now set the label, which makes the contradiction reachable: somebody marks a
 * part normal because they cannot find the defect the import claimed, while three regions are
 * still drawn on it. Neither side is authoritative — the label describes the part and the
 * document describes the pixels — so this never blocks anything. It states the *measured*
 * consequence and leaves the judgement where it belongs.
 *
 * The consequences are read off the evaluation protocol rather than invented. `eval/runner.py`
 * gives a maskless image an all-zero ground truth **only** when its label is `normal`, counts a
 * maskless `defect` image into `skipped_unannotated`, and drops `unlabeled` samples from
 * image-level ROC and average precision entirely.
 *
 * The rule lives here, apart from the route, so it is assertable without a Konva harness — the
 * same reason `annotationPanes.ts` and `annotationQueue.ts` exist.
 */

import type { AnnotationDocument, Label } from "./client";

/**
 * Whether the document asserts any defect at all.
 *
 * A `source_mask` base counts even with no shapes on top of it: the imported mask *is* the
 * truth until something subtracts it. This deliberately does not try to be exact — a base that
 * has been entirely erased still reads as truth here — because the alternative is rasterising
 * the document on every keystroke to decide whether to show a hint. Over-reporting truth is
 * the safe direction: it never accuses somebody of leaving a defect sample undrawn when they
 * have drawn one.
 */
function carriesTruth(document: AnnotationDocument): boolean {
  if (document.base === "source_mask") return true;
  return document.shapes.some((shape) => shape.operation === "add");
}

/** How many regions are being claimed, for a message that can count them. */
function addedRegions(document: AnnotationDocument): number {
  return document.shapes.filter((shape) => shape.operation === "add").length;
}

/**
 * The note beside the label control, or `null` when the two halves of truth agree.
 *
 * `normal` with nothing drawn and `defect` with something drawn are the agreeing cases and say
 * nothing at all — which is most of the time, and is what keeps the mark meaningful when it
 * does appear.
 */
export function labelNote(label: Label, document: AnnotationDocument): string | null {
  const truth = carriesTruth(document);

  if (label === "normal" && truth) {
    const count = addedRegions(document);
    const regions =
      document.base === "source_mask" && count === 0
        ? "an imported mask is"
        : `${count} defect region${count === 1 ? "" : "s"} ${count === 1 ? "is" : "are"}`;
    return `Labelled normal, but ${regions} still drawn. A normal sample's mask is read as ground truth, so completing this records a defect on a part called good.`;
  }

  if (label === "defect" && !truth) {
    return "Labelled defect with nothing drawn. Pixel metrics skip a defect image that has no ground truth, and report how many they skipped.";
  }

  if (label === "unlabeled" && truth) {
    return "Unlabeled samples are excluded from every metric, the regions drawn here included. Label the part normal or defect for this work to count.";
  }

  return null;
}
