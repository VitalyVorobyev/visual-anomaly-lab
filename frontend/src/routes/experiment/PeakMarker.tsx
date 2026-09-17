/**
 * Where the map peaked, and the window that verdict was decided in.
 *
 * The localization verdict is one bit — `localized`, `off target`, or nothing — and a bit is
 * not reviewable. This draws the two things that produced it, in the image's own pixel
 * coordinates: the peak the backend recorded as the map's argmax, and the tolerance window
 * around it that was tested against the annotation. Laid over the ground-truth outline, the
 * verdict becomes something a reader can check by eye rather than take on trust.
 *
 * **The window is a square, not a disc, because the rule is.** `eval/localization.hits`
 * tests `mask[y - r : y + r + 1, x - r : x + r + 1].any()` — a Chebyshev neighbourhood — and
 * drawing a circle of radius `r` would show a shape the verdict was never decided against:
 * a peak in the corner of the box counts as a hit while sitting visibly outside the circle,
 * which is a picture that contradicts the badge beside it. `caliper` with its direction
 * arrow suppressed is the box primitive `MeasureOverlay` offers.
 *
 * A layer rather than a component of either viewer: the results screen and the comparison
 * screen both draw it, and the coordinates come from `ImageScore`, which both already hold.
 * It must be rendered **inside** an `ImageStage` — it reads that stage's scale so a stroke
 * declared at one screen pixel stays one screen pixel at 8x.
 */

import { MeasureOverlay, useStage, type MeasurePrimitive, type MeasureTone } from "@vitavision/lab-ui";

import type { ImageScore } from "../../api/client";

/**
 * The verdict's colour, in the overlay's own tone vocabulary.
 *
 * `signal` for `null` deliberately: the marker still shows *where the map peaked*, which is
 * worth seeing on a normal part or an unannotated defect, but it must not borrow the green
 * of a passed check for an image nothing was checked against.
 */
export function peakTone(localized: boolean | null | undefined): MeasureTone {
  if (localized === true) return "normal";
  if (localized === false) return "warn";
  return "signal";
}

export function PeakMarker({ image }: { image: ImageScore }) {
  const stage = useStage();
  const peak = image.peak;
  if (!peak) return null;

  const tone = peakTone(image.localized);
  const primitives: MeasurePrimitive[] = [
    { kind: "point", x: peak.x, y: peak.y, cross: true, tone },
  ];

  // No window where no tolerance was resolved — an unscored run, or a run whose stored
  // configuration predates the verdict. A box drawn at a guessed radius would be a claim
  // about a rule that was never applied.
  const radius = image.tolerance_px;
  if (radius !== null && radius > 0) {
    const side = radius * 2 + 1;
    primitives.push({
      kind: "caliper",
      cx: peak.x,
      cy: peak.y,
      width: side,
      height: side,
      angle: 0,
      showDirection: false,
      tone,
    });
  }

  return (
    <MeasureOverlay
      nativeWidth={image.width}
      nativeHeight={image.height}
      primitives={primitives}
      // CSS pixels per image pixel, which is what keeps the cross and the box outline the
      // same weight at every zoom instead of growing into a blob.
      strokeScale={stage.view.scale}
    />
  );
}
