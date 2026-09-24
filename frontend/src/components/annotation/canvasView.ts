/**
 * The annotation scene's view, as pure arithmetic.
 *
 * `CanvasView.zoom` is a multiple of *fit* and the pan is an offset from the fitted origin, so
 * a view means the same framing whatever size the pane is — which is what lets one view be
 * shared by the editor and a reference pane beside it, and survive a channel switch.
 *
 * The screen transform is `screen = origin + source · scale`, with `scale = fit · zoom` in
 * CSS pixels per source pixel.
 */

import type { AnnotationPoint } from "../../api/client";

export interface CanvasView {
  zoom: number;
  panX: number;
  panY: number;
}

export interface Size {
  width: number;
  height: number;
}

export const INITIAL_CANVAS_VIEW: CanvasView = { zoom: 1, panX: 0, panY: 0 };

/** Fit leaves a margin so the frame's edge and its shadow are visible. */
const FIT_MARGIN = 0.94;
/** How far below fit zooming out may go: room around the part, not a lost image. */
export const MIN_ZOOM_VS_FIT = 0.25;
export const MAX_ZOOM_VS_FIT = 12;
/** One wheel notch. */
export const WHEEL_STEP = 1.1;

export function fitScale(size: Size, image: Size): number {
  return Math.min(size.width / image.width, size.height / image.height) * FIT_MARGIN;
}

/** Where source pixel (0, 0) lands on screen for `view`. */
export function viewOrigin(size: Size, image: Size, fit: number, view: CanvasView) {
  return {
    x: (size.width - image.width * fit) / 2 + view.panX,
    y: (size.height - image.height * fit) / 2 + view.panY,
  };
}

/** The `zoom` range this pane allows, as `[min, max]` multiples of fit. */
export function zoomRange(_fit: number): [number, number] {
  return [MIN_ZOOM_VS_FIT, MAX_ZOOM_VS_FIT];
}

export function clampZoom(zoom: number, fit: number): number {
  const [min, max] = zoomRange(fit);
  return clamp(zoom, min, max);
}

/** The source pixel under a screen point, clamped to the frame. */
export function toSource(
  pointer: AnnotationPoint,
  origin: AnnotationPoint,
  scale: number,
  image: Size,
): AnnotationPoint {
  return {
    x: clamp((pointer.x - origin.x) / scale, 0, image.width),
    y: clamp((pointer.y - origin.y) / scale, 0, image.height),
  };
}

/** Change the zoom while holding whatever is under `anchor` (a screen point) where it is. */
export function zoomAbout(
  view: CanvasView,
  factor: number,
  anchor: AnnotationPoint,
  size: Size,
  image: Size,
): CanvasView {
  const fit = fitScale(size, image);
  const origin = viewOrigin(size, image, fit, view);
  const scale = fit * view.zoom;
  const before = { x: (anchor.x - origin.x) / scale, y: (anchor.y - origin.y) / scale };
  const zoom = clampZoom(view.zoom * factor, fit);
  const nextScale = fit * zoom;
  const fitOrigin = viewOrigin(size, image, fit, INITIAL_CANVAS_VIEW);
  return {
    zoom,
    panX: anchor.x - before.x * nextScale - fitOrigin.x,
    panY: anchor.y - before.y * nextScale - fitOrigin.y,
  };
}

/** One source pixel to one screen pixel, centred. */
export function actualPixelsView(size: Size, image: Size): CanvasView {
  const fit = fitScale(size, image);
  const actualOrigin = { x: (size.width - image.width) / 2, y: (size.height - image.height) / 2 };
  const fitOrigin = viewOrigin(size, image, fit, INITIAL_CANVAS_VIEW);
  return {
    zoom: 1 / fit,
    panX: actualOrigin.x - fitOrigin.x,
    panY: actualOrigin.y - fitOrigin.y,
  };
}

export function isFitView(view: CanvasView): boolean {
  return Math.abs(view.zoom - 1) < 0.0001 && Math.abs(view.panX) < 0.5 && Math.abs(view.panY) < 0.5;
}

export function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}
