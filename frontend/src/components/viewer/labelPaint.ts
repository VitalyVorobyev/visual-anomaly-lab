/**
 * A label map, painted: one colour per class, from the design system's palette.
 *
 * The backend serves a supervised run's label maps as class indices (a value plane), never as
 * a picture, because a class's colour belongs to the interface — the lab-ui chart palette by
 * the class's pinned position, so the same run reads the same on every screen and no dataset
 * decides it. Painting happens here, once per plane, into RGBA at the plane's own size; the
 * stage stretches that over the photograph, which is exact because the plane is in the
 * source frame and any decimation is an integer stride.
 *
 * Two styles, after the viewer's one convention: **a prediction is solid** (a faint fill and a
 * full outline) and **truth is dashed** (an outline only, broken along the diagonal). Pixels
 * that are background (0) or ignored (NaN — a class the run does not know) are left clear.
 */

import { seriesColour } from "@vitavision/lab-ui";

export type LabelStyle = "prediction" | "truth";

const FILL_ALPHA = 80;
const LINE_ALPHA = 235;

/** The colour of label index `index` — `classes[index - 1]` — as a CSS colour. */
export function classColour(index: number): string {
  return seriesColour(Math.max(0, index - 1));
}

/** `#rrggbb` as three bytes; anything else reads as a neutral grey. */
export function rgbOf(colour: string): [number, number, number] {
  const match = /^#([0-9a-f]{2})([0-9a-f]{2})([0-9a-f]{2})$/i.exec(colour);
  if (!match) return [139, 148, 155];
  return [
    Number.parseInt(match[1] as string, 16),
    Number.parseInt(match[2] as string, 16),
    Number.parseInt(match[3] as string, 16),
  ];
}

/**
 * How many plane pixels either side of a border the line spreads: none below 800 px, so a
 * small map keeps one-pixel lines, and one more per 800 px after that, so a large frame drawn
 * to fit still shows a line rather than a hairline.
 */
export function lineRadius(width: number, height: number): number {
  return Math.floor(Math.max(width, height) / 800);
}

/**
 * The RGBA bytes for one label plane (the first channel of `values`).
 *
 * A border pixel is a labelled one with a 4-neighbour of another label, grown by
 * `lineRadius`; a dashed line keeps the runs of it where `(x + y)` falls in an even period.
 */
export function paintLabels(
  plane: { width: number; height: number; values: Float32Array },
  style: LabelStyle,
): Uint8ClampedArray {
  const { width, height, values } = plane;
  const size = width * height;
  const labels = new Int16Array(size);
  for (let index = 0; index < size; index += 1) {
    const value = values[index] as number;
    labels[index] = Number.isFinite(value) ? Math.round(value) : -1;
  }

  const edge = new Uint8Array(size);
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const index = y * width + x;
      const label = labels[index] as number;
      if (label <= 0) continue;
      if (
        (x > 0 && labels[index - 1] !== label) ||
        (x < width - 1 && labels[index + 1] !== label) ||
        (y > 0 && labels[index - width] !== label) ||
        (y < height - 1 && labels[index + width] !== label)
      ) {
        edge[index] = 1;
      }
    }
  }
  const line = grow(edge, width, height, lineRadius(width, height));
  const period = 4 * (2 * lineRadius(width, height) + 1);

  const palette = new Map<number, [number, number, number]>();
  const out = new Uint8ClampedArray(size * 4);
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const index = y * width + x;
      const label = labels[index] as number;
      if (label <= 0) continue;
      const onLine = line[index] === 1;
      const dashOn = Math.floor((x + y) / period) % 2 === 0;
      let alpha = 0;
      if (style === "prediction") alpha = onLine ? LINE_ALPHA : FILL_ALPHA;
      else if (onLine && dashOn) alpha = LINE_ALPHA;
      if (alpha === 0) continue;
      let rgb = palette.get(label);
      if (rgb === undefined) {
        rgb = rgbOf(classColour(label));
        palette.set(label, rgb);
      }
      out[index * 4] = rgb[0];
      out[index * 4 + 1] = rgb[1];
      out[index * 4 + 2] = rgb[2];
      out[index * 4 + 3] = alpha;
    }
  }
  return out;
}

/** A binary mask dilated by a (2r+1)-square, separably. */
function grow(mask: Uint8Array, width: number, height: number, radius: number): Uint8Array {
  if (radius <= 0) return mask;
  const across = new Uint8Array(mask.length);
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      if (mask[y * width + x] !== 1) continue;
      for (let dx = Math.max(0, x - radius); dx <= Math.min(width - 1, x + radius); dx += 1) {
        across[y * width + dx] = 1;
      }
    }
  }
  const out = new Uint8Array(mask.length);
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      if (across[y * width + x] !== 1) continue;
      for (let dy = Math.max(0, y - radius); dy <= Math.min(height - 1, y + radius); dy += 1) {
        out[dy * width + x] = 1;
      }
    }
  }
  return out;
}
