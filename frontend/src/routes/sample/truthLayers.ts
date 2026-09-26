/**
 * What the sample view draws over one image, in order: its truth, then Explore.
 *
 * The truth is the image's own (`GET /api/images/{id}/truth`): a raster of every region but a
 * drawn box — filled in the class's colour, or outlined for an imported anomaly mask — and
 * each box as a rectangle tagged with its class. Explore's answer is a question asked of an
 * encoder, not truth, so it always draws above; while Explore is on, truth steps back to a
 * fraction of its weight so the two can be read together, and the switch still hides it.
 *
 * Pure, so the selection is tested without a stage.
 */

import { apiBaseUrl, type ImageTruth, type TruthClass } from "../../api/client";
import type { RasterLayer } from "../../components/viewer/SampleStage";
import type { VectorShape } from "../../components/viewer/VectorLayer";

/** The fraction of its weight truth keeps while Explore is on. */
export const EXPLORE_DIM = 0.35;

export interface TruthView {
  on: boolean;
  /** 0–1, the reader's setting. */
  opacity: number;
  /** Explore is on: truth dims beneath it. */
  exploring: boolean;
}

export function truthOpacity(view: TruthView): number {
  return view.opacity * (view.exploring ? EXPLORE_DIM : 1);
}

export function truthLayers(
  truth: ImageTruth | undefined,
  view: TruthView,
): { layers: RasterLayer[]; shapes: VectorShape[] } {
  if (!truth || !view.on) return { layers: [], shapes: [] };
  const opacity = truthOpacity(view);
  const classes = new Map(truth.classes.map((entry) => [entry.key, entry]));
  const layers: RasterLayer[] = [];
  if (truth.regions_url) {
    // The colours are in the URL as well as in the picture, so a recoloured class is a new
    // `src` and not a cached image of the old palette.
    const palette = truth.classes.map((entry) => entry.color.replace(/^#/, "")).join(",");
    layers.push({
      key: "truth",
      src: `${apiBaseUrl}${truth.regions_url}&c=${palette}`,
      opacity,
    });
  }
  const shapes: VectorShape[] = truth.boxes.map((box, index) => {
    const entry = classes.get(box.label_key);
    return {
      kind: "box",
      id: `truth-${index}`,
      x: box.x,
      y: box.y,
      width: box.width,
      height: box.height,
      label: entry?.name ?? box.label_key,
      colour: entry?.color,
      labelColour: entry?.color,
      opacity,
    };
  });
  return { layers, shapes };
}

/** Truth first, Explore over it: a stage stacks its layers in order. */
export function stackLayers(
  truth: readonly RasterLayer[],
  explore: readonly RasterLayer[],
): RasterLayer[] {
  return [...truth, ...explore];
}

/** The classes any of a sample's images shows, once each, in the order they first appear. */
export function legendClasses(truths: readonly (ImageTruth | undefined)[]): TruthClass[] {
  const seen = new Map<string, TruthClass>();
  for (const truth of truths) {
    for (const entry of truth?.classes ?? []) {
      if (!seen.has(entry.key)) seen.set(entry.key, entry);
    }
  }
  return [...seen.values()];
}

/** One line saying what the truth is, or why there is none to draw. */
export function truthSummary(truths: readonly (ImageTruth | undefined)[]): string {
  const known = truths.filter((truth): truth is ImageTruth => truth !== undefined);
  if (known.length === 0) return "";
  if (known.some((truth) => truth.source === "revision")) {
    if (legendClasses(known).length === 0) return "Completed annotation: no class present";
    return known.every((truth) => truth.source === "revision")
      ? "Completed annotation"
      : "Completed annotation on some channels";
  }
  if (known.some((truth) => truth.source === "imported_mask")) return "Imported defect mask, outlined";
  if (known.every((truth) => truth.source === "normal")) return "Labelled normal: nothing to draw";
  return "No truth for this sample yet";
}
