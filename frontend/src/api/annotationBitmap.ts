/** Turn one source-coordinate brush stroke into the annotation model's bitmap primitive.
 *
 * **A bitmap mask's channel is luminance, everywhere.** The backend defines it that way —
 * `annotation_bitmap.decode_png` is `convert("L") > 0` and `encode_png` writes mode `L` on an
 * opaque black ground — so this side has to agree, and for a while it did not: rendering relied
 * on transparency and tracing read the alpha byte. Those agreed with the backend only by luck,
 * because a brush stroke happens to be white on transparent *black*. Anything the backend
 * produced — a MobileSAM candidate, an imported PNG/LabelMe/COCO mask — arrives fully opaque, so
 * it drew as a grey rectangle over its whole crop and traced as its bounding box.
 *
 * Everything here therefore reads luminance and converts to alpha only at the moment of
 * painting, which works for both shapes of PNG.
 *
 * **Strokes are rasterised here, not by Canvas2D.** See `rasterizeStroke`: a brush size is
 * a diameter in source pixels, `1` marks exactly one pixel, and the brush and the eraser
 * cover the same footprint at the same setting. Everything this module emits is the opaque
 * black-and-white the backend itself writes.
 */

import type { AnnotationPoint, AnnotationShape, BitmapShape, PolygonShape } from "./client";
import { nextShapeId } from "./annotationState";

/** Above this luminance a pixel is part of the mask. The backend's rule is `> 0`; this keeps a
 * little headroom for a resampled or antialiased edge without changing what "filled" means. */
export const MASK_LUMINANCE_THRESHOLD = 64;

function context2d(width: number, height: number): CanvasRenderingContext2D {
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const context = canvas.getContext("2d", { willReadFrequently: true });
  if (!context) throw new Error("This browser cannot process an annotation mask.");
  return context;
}

/** A decoded mask as one byte per pixel, read from luminance rather than alpha. */
export function maskFromPixels(pixels: Uint8ClampedArray, length: number): Uint8Array {
  const mask = new Uint8Array(length);
  for (let index = 0; index < length; index += 1) {
    const offset = index * 4;
    // Fully transparent is not part of the mask whatever its colour channels say: a canvas
    // clears to transparent *black*, and an unpainted pixel must not read as filled.
    if ((pixels[offset + 3] ?? 0) === 0) continue;
    const luminance = Math.max(
      pixels[offset] ?? 0,
      pixels[offset + 1] ?? 0,
      pixels[offset + 2] ?? 0,
    );
    mask[index] = luminance >= MASK_LUMINANCE_THRESHOLD ? 1 : 0;
  }
  return mask;
}

/**
 * The mask painted in one colour, ready for a Konva `Image`.
 *
 * Alpha comes from the mask, never from the source PNG, which is what lets an opaque
 * backend-produced mask render as an overlay instead of a filled rectangle.
 */
export function tintedMask(
  image: CanvasImageSource,
  width: number,
  height: number,
  color: string,
): HTMLCanvasElement {
  const context = context2d(width, height);
  context.drawImage(image, 0, 0, width, height);
  const data = context.getImageData(0, 0, width, height);
  const mask = maskFromPixels(data.data, width * height);

  context.clearRect(0, 0, width, height);
  context.fillStyle = color;
  context.fillRect(0, 0, width, height);
  const tint = context.getImageData(0, 0, width, height);
  for (let index = 0; index < mask.length; index += 1) {
    tint.data[index * 4 + 3] = mask[index] ? 255 : 0;
  }
  context.putImageData(tint, 0, 0);
  return context.canvas;
}

/**
 * The pixels one stroke covers, as one byte per pixel over `box`.
 *
 * **The brush is exact, and that is the whole point of doing this by hand.** Strokes used
 * to be Canvas2D paths: fractional coordinates, an antialiased edge, and then a luminance
 * threshold applied to the result. One radius therefore produced *three* different
 * footprints — a new region painted white on transparent kept every touched pixel, a
 * stroke continuing an existing region needed a quarter coverage to count, and the eraser
 * needed three quarters — so brush and eraser at the same setting did not undo each other,
 * and no setting marked a single pixel. `size` is now a diameter in source pixels, `1`
 * means one pixel, and every path here rasterises identically.
 *
 * Two halves, each doing one job:
 *
 * - **the spine**, the pixels under the pointer, joined by an 8-connected Bresenham walk.
 *   Canvas's `lineTo` used to close the gaps between `mousemove` samples; at one pixel
 *   wide that has to be done in integers or a fast drag comes out dotted.
 * - **the stamp**, a disc of the given diameter placed on every spine pixel, written as
 *   one row span per offset rather than pixel by pixel. That keeps the cost linear in the
 *   diameter instead of quadratic, which matters for a 128-wide brush over a long drag.
 */
export function rasterizeStroke(
  points: AnnotationPoint[],
  size: number,
  box: { minX: number; minY: number; width: number; height: number },
): Uint8Array {
  const mask = new Uint8Array(box.width * box.height);
  if (points.length === 0 || box.width <= 0 || box.height <= 0) return mask;

  const radius = size / 2;
  const reach = Math.max(0, Math.ceil(radius - 0.5));
  // Half-width of the disc on each row, precomputed once for the whole stroke. A pixel is
  // in when its centre is within `radius` of the spine's centre, which for size 1 admits
  // the centre pixel alone — 0 <= 0.25 — and nothing at dy = ±1.
  const halfWidth: number[] = [];
  for (let dy = -reach; dy <= reach; dy += 1) {
    const span = radius * radius - dy * dy;
    halfWidth.push(span < 0 ? -1 : Math.floor(Math.sqrt(span)));
  }

  const stamp = (x: number, y: number) => {
    for (let dy = -reach; dy <= reach; dy += 1) {
      const half = halfWidth[dy + reach]!;
      if (half < 0) continue;
      const row = y - box.minY + dy;
      if (row < 0 || row >= box.height) continue;
      const from = Math.max(0, x - box.minX - half);
      const to = Math.min(box.width, x - box.minX + half + 1);
      if (from < to) mask.fill(1, row * box.width + from, row * box.width + to);
    }
  };

  let previous: { x: number; y: number } | null = null;
  for (const point of points) {
    const current = { x: Math.floor(point.x), y: Math.floor(point.y) };
    if (previous === null) {
      stamp(current.x, current.y);
    } else {
      for (const pixel of walk(previous, current)) stamp(pixel.x, pixel.y);
    }
    previous = current;
  }
  return mask;
}

/** Every pixel from `from` to `to` inclusive, 8-connected (Bresenham). */
function* walk(
  from: { x: number; y: number },
  to: { x: number; y: number },
): Generator<{ x: number; y: number }> {
  const stepX = from.x < to.x ? 1 : -1;
  const stepY = from.y < to.y ? 1 : -1;
  const spanX = Math.abs(to.x - from.x);
  const spanY = -Math.abs(to.y - from.y);
  let error = spanX + spanY;
  let { x, y } = from;
  for (;;) {
    yield { x, y };
    if (x === to.x && y === to.y) return;
    const doubled = 2 * error;
    if (doubled >= spanY) {
      error += spanY;
      x += stepX;
    }
    if (doubled <= spanX) {
      error += spanX;
      y += stepY;
    }
  }
}

/** A mask as the opaque black-and-white PNG the backend itself writes. */
function encodeMask(mask: Uint8Array, width: number, height: number): string {
  const context = context2d(width, height);
  const image = context.createImageData(width, height);
  for (let index = 0; index < mask.length; index += 1) {
    const value = mask[index] ? 255 : 0;
    image.data[index * 4] = value;
    image.data[index * 4 + 1] = value;
    image.data[index * 4 + 2] = value;
    image.data[index * 4 + 3] = 255;
  }
  context.putImageData(image, 0, 0);
  return context.canvas.toDataURL("image/png").split(",", 2)[1] ?? "";
}

export function bitmapStroke({
  points,
  size,
  imageWidth,
  imageHeight,
  labelKey,
  operation,
}: {
  points: AnnotationPoint[];
  size: number;
  imageWidth: number;
  imageHeight: number;
  labelKey: string;
  operation: "add" | "subtract";
}): BitmapShape | null {
  if (points.length === 0) return null;
  const { minX, minY, maxX, maxY } = strokeBounds(points, size, imageWidth, imageHeight);
  const width = maxX - minX;
  const height = maxY - minY;
  if (width <= 0 || height <= 0) return null;

  const mask = rasterizeStroke(points, size, { minX, minY, width, height });
  // Cropped to what was painted rather than to the padded stroke box: the crop is what the
  // selection outline draws and what the copy-to-channel dimension check reads.
  const bounds = maskBounds(mask, width, height);
  if (!bounds) return null;

  const cropped = new Uint8Array(bounds.width * bounds.height);
  for (let row = 0; row < bounds.height; row += 1) {
    const source = (bounds.minY + row) * width + bounds.minX;
    cropped.set(mask.subarray(source, source + bounds.width), row * bounds.width);
  }

  return {
    id: nextShapeId(),
    label_key: labelKey,
    kind: "bitmap",
    operation,
    x: minX + bounds.minX,
    y: minY + bounds.minY,
    width: bounds.width,
    height: bounds.height,
    png_base64: encodeMask(cropped, bounds.width, bounds.height),
  };
}

/**
 * Extend or erase an existing region with one more stroke.
 *
 * This is what makes a defect drawable in as many touches as it takes, and what makes the
 * eraser an eraser: until now every gesture minted a new shape, so a region painted in three
 * strokes was three regions, and the "eraser" was a brush that happened to append a
 * `subtract` layer — it could not take paint back off the thing under the cursor.
 *
 * Both PNG shapes the contract allows are normalised on the way in by compositing over an
 * opaque black ground: a white-on-transparent brush layer and an opaque backend mask land on
 * the same luminance. Erasing is then painting *black*, not a compositing mode, and the output
 * is opaque black-and-white — the format the backend itself writes.
 *
 * Returns `null` when the last of the region has been erased; the caller removes the shape.
 */
export async function paintStroke(
  target: BitmapShape,
  {
    points,
    size,
    imageWidth,
    imageHeight,
    erase,
  }: {
    points: AnnotationPoint[];
    size: number;
    imageWidth: number;
    imageHeight: number;
    erase: boolean;
  },
): Promise<BitmapShape | null> {
  if (points.length === 0) return target;
  const { minX, minY, width, height } = paintRect(
    target,
    strokeBounds(points, size, imageWidth, imageHeight),
    erase,
  );
  if (width <= 0 || height <= 0) return target;

  // The region as it stands, read through the one luminance rule both PNG shapes obey:
  // a white-on-transparent brush layer and an opaque backend mask land the same way.
  const existing = await loadBitmap(target.png_base64);
  const context = context2d(width, height);
  context.fillStyle = "#000000";
  context.fillRect(0, 0, width, height);
  context.drawImage(existing, target.x - minX, target.y - minY, target.width, target.height);
  const mask = maskFromPixels(context.getImageData(0, 0, width, height).data, width * height);

  // Combined on the mask itself rather than by painting, so the eraser removes exactly the
  // pixels the brush at the same size would have added. Painting black over white made
  // that untrue: a pixel had to lose three quarters of its coverage to clear but only gain
  // a quarter to fill, so retracing a stroke with the eraser left a fringe behind.
  const stroke = rasterizeStroke(points, size, { minX, minY, width, height });
  for (let index = 0; index < mask.length; index += 1) {
    if (!stroke[index]) continue;
    mask[index] = erase ? 0 : 1;
  }

  const bounds = maskBounds(mask, width, height);
  if (!bounds) return null;

  // Re-cropped to what is actually painted. Without this an erased region keeps the rectangle
  // it was largest at, and the crop is what the selection outline and the copy dimension check
  // both read.
  const cropped = new Uint8Array(bounds.width * bounds.height);
  for (let row = 0; row < bounds.height; row += 1) {
    const source = (bounds.minY + row) * width + bounds.minX;
    cropped.set(mask.subarray(source, source + bounds.width), row * bounds.width);
  }
  return {
    ...target,
    x: minX + bounds.minX,
    y: minY + bounds.minY,
    width: bounds.width,
    height: bounds.height,
    png_base64: encodeMask(cropped, bounds.width, bounds.height),
  };
}

/**
 * The rectangle a stroke is rasterised into.
 *
 * Adding grows the region to cover the stroke; erasing cannot add a pixel, so its canvas never
 * has to grow past the region it is taking paint off. Getting that wrong is not cosmetic — the
 * crop is what the selection outline draws and what the copy-to-channel dimension check reads.
 */
export function paintRect(
  target: Pick<BitmapShape, "x" | "y" | "width" | "height">,
  stroke: { minX: number; minY: number; maxX: number; maxY: number },
  erase: boolean,
): { minX: number; minY: number; width: number; height: number } {
  const minX = erase ? target.x : Math.min(target.x, stroke.minX);
  const minY = erase ? target.y : Math.min(target.y, stroke.minY);
  const maxX = erase ? target.x + target.width : Math.max(target.x + target.width, stroke.maxX);
  const maxY = erase ? target.y + target.height : Math.max(target.y + target.height, stroke.maxY);
  return { minX, minY, width: maxX - minX, height: maxY - minY };
}

/** The tight box around everything filled, or `null` when nothing is. */
export function maskBounds(
  mask: Uint8Array,
  width: number,
  height: number,
): { minX: number; minY: number; width: number; height: number } | null {
  let minX = width;
  let minY = height;
  let maxX = -1;
  let maxY = -1;
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      if (!mask[y * width + x]) continue;
      if (x < minX) minX = x;
      if (x > maxX) maxX = x;
      if (y < minY) minY = y;
      if (y > maxY) maxY = y;
    }
  }
  if (maxX < 0) return null;
  return { minX, minY, width: maxX - minX + 1, height: maxY - minY + 1 };
}

/**
 * The painted regions a stroke passes over, nearest the front first.
 *
 * This is what makes the eraser an eraser with nothing selected. It used to append a
 * `subtract` layer instead, which is a *new region* — so the tool whose whole job is to take
 * something away added something, and the region list grew by one every time it was used.
 * A cut is still available, but as a deliberate choice in the New region panel rather than as
 * the eraser's silent side effect.
 *
 * Later shapes are painted over earlier ones, so the reverse of document order is what the
 * pointer is nearest.
 */
export function strokeTargets(
  shapes: readonly AnnotationShape[],
  stroke: { minX: number; minY: number; maxX: number; maxY: number },
): BitmapShape[] {
  return shapes
    .filter(
      (shape): shape is BitmapShape =>
        shape.kind === "bitmap" &&
        shape.x < stroke.maxX &&
        shape.x + shape.width > stroke.minX &&
        shape.y < stroke.maxY &&
        shape.y + shape.height > stroke.minY,
    )
    .reverse();
}

/** The rectangle a stroke of this diameter can reach, clamped to the image. */
export function strokeBounds(
  points: AnnotationPoint[],
  size: number,
  imageWidth: number,
  imageHeight: number,
) {
  if (points.length === 0) return { minX: 0, minY: 0, maxX: 0, maxY: 0 };
  const reach = Math.ceil(size / 2) + 1;
  return {
    minX: Math.max(0, Math.floor(Math.min(...points.map((point) => point.x)) - reach)),
    minY: Math.max(0, Math.floor(Math.min(...points.map((point) => point.y)) - reach)),
    maxX: Math.min(imageWidth, Math.ceil(Math.max(...points.map((point) => point.x)) + reach)),
    maxY: Math.min(imageHeight, Math.ceil(Math.max(...points.map((point) => point.y)) + reach)),
  };
}

/** Convert an editable bitmap region into editable vector contours.
 *
 * This deliberately traces the annotation mask rather than pretending to infer an image
 * boundary. Image-aware refinement belongs to the spatial-localisation plugin; this local,
 * deterministic operation is the useful bridge from a quick brush mark to movable vertices.
 */
export async function traceBitmapShape(shape: BitmapShape): Promise<PolygonShape[]> {
  const image = await loadBitmap(shape.png_base64);
  const context = context2d(shape.width, shape.height);
  context.drawImage(image, 0, 0, shape.width, shape.height);
  const pixels = context.getImageData(0, 0, shape.width, shape.height).data;
  // Luminance, not alpha. Reading alpha traced an opaque backend-produced mask — an accepted
  // MobileSAM candidate, an imported PNG — as its own bounding box, because every one of its
  // pixels is opaque.
  const mask = maskFromPixels(pixels, shape.width * shape.height);
  return traceMask(mask, shape.width, shape.height, shape);
}

export function traceMask(
  mask: Uint8Array,
  width: number,
  height: number,
  source: Pick<BitmapShape, "label_key" | "operation" | "x" | "y">,
): PolygonShape[] {
  if (mask.length !== width * height) throw new Error("Mask dimensions do not match its pixels.");

  type Edge = { start: AnnotationPoint; end: AnnotationPoint };
  const edges: Edge[] = [];
  const filled = (x: number, y: number) =>
    x >= 0 && y >= 0 && x < width && y < height && mask[y * width + x] === 1;
  const add = (x1: number, y1: number, x2: number, y2: number) =>
    edges.push({ start: { x: x1, y: y1 }, end: { x: x2, y: y2 } });

  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      if (!filled(x, y)) continue;
      if (!filled(x, y - 1)) add(x, y, x + 1, y);
      if (!filled(x + 1, y)) add(x + 1, y, x + 1, y + 1);
      if (!filled(x, y + 1)) add(x + 1, y + 1, x, y + 1);
      if (!filled(x - 1, y)) add(x, y + 1, x, y);
    }
  }

  const byStart = new Map<string, Edge[]>();
  for (const edge of edges) {
    const key = pointKey(edge.start);
    byStart.set(key, [...(byStart.get(key) ?? []), edge]);
  }
  const remaining = new Set(edges);
  const contours: AnnotationPoint[][] = [];
  while (remaining.size > 0) {
    const first = remaining.values().next().value as Edge | undefined;
    if (!first) break;
    const contour = [first.start];
    let edge = first;
    remaining.delete(edge);
    while (pointKey(edge.end) !== pointKey(first.start)) {
      contour.push(edge.end);
      const next = (byStart.get(pointKey(edge.end)) ?? []).find((candidate) =>
        remaining.has(candidate),
      );
      if (!next) break;
      edge = next;
      remaining.delete(edge);
    }
    if (contour.length >= 3) contours.push(contour);
  }

  return contours
    .map((contour) => {
      const xs = contour.map((point) => point.x);
      const ys = contour.map((point) => point.y);
      const extent = Math.min(
        Math.max(...xs) - Math.min(...xs),
        Math.max(...ys) - Math.min(...ys),
      );
      const tolerance = extent <= 2
        ? 0.25
        : Math.min(3, Math.max(1.25, Math.min(width, height) * 0.01));
      return simplifyClosed(contour, tolerance);
    })
    .filter((contour) => contour.length >= 3 && Math.abs(signedArea(contour)) >= 1)
    .map((contour) => {
      const isOuter = signedArea(contour) > 0;
      return {
        id: nextShapeId(),
        label_key: source.label_key,
        kind: "polygon" as const,
        operation: isOuter ? source.operation : invertOperation(source.operation),
        points: contour.map((point) => ({ x: point.x + source.x, y: point.y + source.y })),
      };
    });
}

function loadBitmap(pngBase64: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const image = new globalThis.Image();
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error("The bitmap region could not be decoded."));
    image.src = `data:image/png;base64,${pngBase64}`;
  });
}

function pointKey(point: AnnotationPoint): string {
  return `${point.x},${point.y}`;
}

function signedArea(points: AnnotationPoint[]): number {
  return points.reduce((area, point, index) => {
    const next = points[(index + 1) % points.length]!;
    return area + point.x * next.y - next.x * point.y;
  }, 0) / 2;
}

function invertOperation(operation: "add" | "subtract") {
  return operation === "add" ? "subtract" : "add";
}

function simplifyClosed(points: AnnotationPoint[], tolerance: number): AnnotationPoint[] {
  const withoutCollinear = points.filter((point, index) => {
    const previous = points[(index - 1 + points.length) % points.length]!;
    const next = points[(index + 1) % points.length]!;
    return (point.x - previous.x) * (next.y - point.y) !==
      (point.y - previous.y) * (next.x - point.x);
  });
  if (withoutCollinear.length <= 3) return withoutCollinear;

  // Split the ring at the point farthest from its first point, then simplify both open
  // chains. This avoids the degenerate equal-endpoint case in ordinary RDP.
  const anchor = withoutCollinear[0]!;
  let split = 1;
  let distance = -1;
  for (let index = 1; index < withoutCollinear.length; index += 1) {
    const candidate = withoutCollinear[index]!;
    const nextDistance = (candidate.x - anchor.x) ** 2 + (candidate.y - anchor.y) ** 2;
    if (nextDistance > distance) {
      distance = nextDistance;
      split = index;
    }
  }
  const first = rdp(withoutCollinear.slice(0, split + 1), tolerance);
  const second = rdp([...withoutCollinear.slice(split), anchor], tolerance);
  return [...first.slice(0, -1), ...second.slice(0, -1)];
}

function rdp(points: AnnotationPoint[], tolerance: number): AnnotationPoint[] {
  if (points.length <= 2) return points;
  const first = points[0]!;
  const last = points.at(-1)!;
  let maximum = 0;
  let split = 0;
  for (let index = 1; index < points.length - 1; index += 1) {
    const distance = segmentDistance(points[index]!, first, last);
    if (distance > maximum) {
      maximum = distance;
      split = index;
    }
  }
  if (maximum <= tolerance) return [first, last];
  const left = rdp(points.slice(0, split + 1), tolerance);
  const right = rdp(points.slice(split), tolerance);
  return [...left.slice(0, -1), ...right];
}

function segmentDistance(point: AnnotationPoint, start: AnnotationPoint, end: AnnotationPoint) {
  const lengthSquared = (end.x - start.x) ** 2 + (end.y - start.y) ** 2;
  if (lengthSquared === 0) return Math.hypot(point.x - start.x, point.y - start.y);
  const position = Math.max(
    0,
    Math.min(
      1,
      ((point.x - start.x) * (end.x - start.x) +
        (point.y - start.y) * (end.y - start.y)) /
        lengthSquared,
    ),
  );
  return Math.hypot(
    point.x - (start.x + position * (end.x - start.x)),
    point.y - (start.y + position * (end.y - start.y)),
  );
}
