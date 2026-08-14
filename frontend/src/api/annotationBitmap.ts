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
 */

import type { AnnotationPoint, BitmapShape, PolygonShape } from "./client";
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

export function bitmapStroke({
  points,
  radius,
  imageWidth,
  imageHeight,
  labelKey,
  operation,
}: {
  points: AnnotationPoint[];
  radius: number;
  imageWidth: number;
  imageHeight: number;
  labelKey: string;
  operation: "add" | "subtract";
}): BitmapShape | null {
  if (points.length === 0) return null;
  const { minX, minY, maxX, maxY } = strokeBounds(
    points,
    radius,
    imageWidth,
    imageHeight,
  );
  const width = maxX - minX;
  const height = maxY - minY;
  if (width <= 0 || height <= 0) return null;

  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const context = canvas.getContext("2d");
  if (!context) throw new Error("This browser cannot create a brush layer.");
  context.strokeStyle = "#ffffff";
  context.fillStyle = "#ffffff";
  context.lineWidth = radius * 2;
  context.lineCap = "round";
  context.lineJoin = "round";
  context.beginPath();
  context.moveTo(points[0]!.x - minX, points[0]!.y - minY);
  for (const point of points.slice(1)) context.lineTo(point.x - minX, point.y - minY);
  context.stroke();
  // A click without movement is a useful one-dot brush stroke.
  if (points.length === 1) {
    context.beginPath();
    context.arc(points[0]!.x - minX, points[0]!.y - minY, radius, 0, Math.PI * 2);
    context.fill();
  }

  return {
    id: nextShapeId(),
    label_key: labelKey,
    kind: "bitmap",
    operation,
    x: minX,
    y: minY,
    width,
    height,
    png_base64: canvas.toDataURL("image/png").split(",", 2)[1] ?? "",
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
    radius,
    imageWidth,
    imageHeight,
    erase,
  }: {
    points: AnnotationPoint[];
    radius: number;
    imageWidth: number;
    imageHeight: number;
    erase: boolean;
  },
): Promise<BitmapShape | null> {
  if (points.length === 0) return target;
  const { minX, minY, width, height } = paintRect(
    target,
    strokeBounds(points, radius, imageWidth, imageHeight),
    erase,
  );
  if (width <= 0 || height <= 0) return target;

  const existing = await loadBitmap(target.png_base64);
  const context = context2d(width, height);
  context.fillStyle = "#000000";
  context.fillRect(0, 0, width, height);
  context.drawImage(existing, target.x - minX, target.y - minY, target.width, target.height);

  const paint = erase ? "#000000" : "#ffffff";
  context.strokeStyle = paint;
  context.fillStyle = paint;
  context.lineWidth = radius * 2;
  context.lineCap = "round";
  context.lineJoin = "round";
  context.beginPath();
  context.moveTo(points[0]!.x - minX, points[0]!.y - minY);
  for (const point of points.slice(1)) context.lineTo(point.x - minX, point.y - minY);
  context.stroke();
  if (points.length === 1) {
    context.beginPath();
    context.arc(points[0]!.x - minX, points[0]!.y - minY, radius, 0, Math.PI * 2);
    context.fill();
  }

  const mask = maskFromPixels(context.getImageData(0, 0, width, height).data, width * height);
  const bounds = maskBounds(mask, width, height);
  if (!bounds) return null;

  // Re-cropped to what is actually painted. Without this an erased region keeps the rectangle
  // it was largest at, and the crop is what the selection outline and the copy dimension check
  // both read.
  const cropped = context2d(bounds.width, bounds.height);
  cropped.drawImage(context.canvas, -bounds.minX, -bounds.minY);
  return {
    ...target,
    x: minX + bounds.minX,
    y: minY + bounds.minY,
    width: bounds.width,
    height: bounds.height,
    png_base64: cropped.canvas.toDataURL("image/png").split(",", 2)[1] ?? "",
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

export function strokeBounds(
  points: AnnotationPoint[],
  radius: number,
  imageWidth: number,
  imageHeight: number,
) {
  if (points.length === 0) return { minX: 0, minY: 0, maxX: 0, maxY: 0 };
  return {
    minX: Math.max(0, Math.floor(Math.min(...points.map((point) => point.x)) - radius - 1)),
    minY: Math.max(0, Math.floor(Math.min(...points.map((point) => point.y)) - radius - 1)),
    maxX: Math.min(
      imageWidth,
      Math.ceil(Math.max(...points.map((point) => point.x)) + radius + 1),
    ),
    maxY: Math.min(
      imageHeight,
      Math.ceil(Math.max(...points.map((point) => point.y)) + radius + 1),
    ),
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
