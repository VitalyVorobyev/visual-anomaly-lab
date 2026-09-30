/**
 * A detection run's boxes, as the viewer draws them (ADR-0039).
 *
 * One convention on every surface: **truth is dashed and a prediction solid**, and each is
 * toned by what the run's confidence cut made of it — `normal` for a match, `defect` for a
 * kept detection that matched nothing, `warn` for a true box no kept detection found. The
 * class is not the outline's colour: it is the tag's, in the pinned-class palette
 * (`classColour`), so a box says both what it claims and whether that held.
 *
 * Nothing here decides a verdict. The server matched every box at IoU 0.5 and the resolved
 * cut, and this only reads `kept`, `matched` and `found` (ADR-0028).
 */

import { toneColor, type MeasureTone } from "@vitavision/ui";

import type { ImageBoxes } from "../../api/client";
import { classColour, colourHex } from "./labelPaint";
import type { VectorShape } from "./VectorLayer";

export const BOX_TONE = {
  match: "normal",
  false_positive: "defect",
  missed: "warn",
} as const satisfies Record<string, MeasureTone>;

/** The shapes of one image's boxes, truth under predictions; below-cut detections are left out. */
export function boxShapes(
  boxes: ImageBoxes,
  layers: { predictions: boolean; truth: boolean },
): VectorShape[] {
  const shapes: VectorShape[] = [];
  if (layers.truth) {
    (boxes.truth ?? []).forEach((item, index) => {
      const [x0 = 0, y0 = 0, x1 = 0, y1 = 0] = item.box;
      shapes.push({
        id: `truth:${index}`,
        kind: "box",
        x: x0,
        y: y0,
        width: x1 - x0,
        height: y1 - y0,
        dashed: true,
        // A class the run does not pin is neither found nor missed: drawn, and muted.
        tone:
          item.class_index === null ? "muted" : item.found ? BOX_TONE.match : BOX_TONE.missed,
      });
    });
  }
  if (layers.predictions) {
    (boxes.predictions ?? []).forEach((item, index) => {
      if (!item.kept) return;
      const [x0 = 0, y0 = 0, x1 = 0, y1 = 0] = item.box;
      shapes.push({
        id: `prediction:${index}`,
        kind: "box",
        x: x0,
        y: y0,
        width: x1 - x0,
        height: y1 - y0,
        tone: item.matched ? BOX_TONE.match : BOX_TONE.false_positive,
        label: `${item.label_key} ${item.confidence.toFixed(2)}`,
        labelColour: classColour(item.class_index + 1),
      });
    });
  }
  return shapes;
}

/**
 * The three tones as `#rrggbb`, for the server-drawn tile, which cannot read a CSS variable.
 *
 * Read from the theme's own custom properties (`toneColor` names them), so the tile follows
 * the theme it is drawn in and matches the viewer's own outlines.
 */
export function boxToneColours(): [string, string, string] {
  return [
    colourHex(toneColor(BOX_TONE.match)),
    colourHex(toneColor(BOX_TONE.false_positive)),
    colourHex(toneColor(BOX_TONE.missed)),
  ];
}
