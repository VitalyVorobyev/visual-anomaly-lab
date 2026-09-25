/**
 * What the document says about one source pixel: which region is on top of it, and whether
 * the mask the document resolves to is set there.
 *
 * The value is folded the way the document is read — the base, then each shape in order, an
 * `add` setting the pixel and a `subtract` clearing it. A polygon or a box is tested at the pixel's
 * centre, which is what completion rasterises — a box half-open, `x <= centre < x + width`, the
 * pixels it covers and no more, and a polygon by the even-odd rule with the same half-open edges,
 * so the polygon of a box's corners reads as the box; a bitmap is looked up in its decoded crop. A
 * bitmap whose crop has not been decoded yet is left out rather than guessed.
 *
 * A readout, never truth: the backend's renderer is what evaluation reads, and this exists so
 * that "is that pixel in or out" can be answered by pointing at it.
 */

import type { AnnotationDocument, AnnotationPoint, PolygonShape } from "../../api/client";

export interface PixelReading {
  x: number;
  y: number;
  /** The resolved mask at this pixel. */
  value: 0 | 1;
  /** 1-based position of the topmost region covering the pixel, as the region list numbers it. */
  region: number | null;
}

/** The pixel under a source-coordinate point, or `null` outside the frame. */
export function pixelAt(
  point: AnnotationPoint,
  width: number,
  height: number,
): { x: number; y: number } | null {
  const x = Math.floor(point.x);
  const y = Math.floor(point.y);
  if (x < 0 || y < 0 || x >= width || y >= height) return null;
  return { x, y };
}

export function readPixel(
  document: AnnotationDocument,
  point: AnnotationPoint,
  masks: ReadonlyMap<string, Uint8Array>,
  base: Uint8Array | null,
): PixelReading | null {
  const pixel = pixelAt(point, document.image_width, document.image_height);
  if (!pixel) return null;
  let value: 0 | 1 =
    document.base === "source_mask" && base?.[pixel.y * document.image_width + pixel.x] ? 1 : 0;
  let region: number | null = null;
  document.shapes.forEach((shape, index) => {
    const covered =
      shape.kind === "bitmap"
        ? insideBitmap(shape, pixel.x, pixel.y, masks.get(shape.png_base64))
        : shape.kind === "box"
          ? // Exactly the pixels the box covers, which is what completion owns.
            insideBox(shape, pixel.x + 0.5, pixel.y + 0.5)
          : insidePolygon(shape, pixel.x + 0.5, pixel.y + 0.5);
    if (!covered) return;
    region = index + 1;
    value = shape.operation === "add" ? 1 : 0;
  });
  return { ...pixel, value, region };
}

function insideBitmap(
  shape: { x: number; y: number; width: number; height: number },
  x: number,
  y: number,
  mask: Uint8Array | undefined,
): boolean {
  if (!mask) return false;
  const column = x - shape.x;
  const row = y - shape.y;
  if (column < 0 || row < 0 || column >= shape.width || row >= shape.height) return false;
  return mask[row * shape.width + column] === 1;
}

/** Half-open, as completion rasterises a box: `x <= px < x + width`, likewise in y. */
function insideBox(
  shape: { x: number; y: number; width: number; height: number },
  x: number,
  y: number,
): boolean {
  return x >= shape.x && x < shape.x + shape.width && y >= shape.y && y < shape.y + shape.height;
}

/**
 * Even-odd ray casting, which completion evaluates at every pixel centre
 * (`annotation_render.polygon_coverage`): a centre on a left or top edge is inside, one on a right or
 * bottom edge is outside. `polygonCentres.fixture.json` pins the two against each other.
 */
export function insidePolygon(shape: Pick<PolygonShape, "points">, x: number, y: number): boolean {
  const { points } = shape;
  let inside = false;
  for (let i = 0, j = points.length - 1; i < points.length; j = i, i += 1) {
    const a = points[i];
    const b = points[j];
    if (!a || !b) continue;
    if (a.y > y !== b.y > y && x < ((b.x - a.x) * (y - a.y)) / (b.y - a.y) + a.x) {
      inside = !inside;
    }
  }
  return inside;
}
