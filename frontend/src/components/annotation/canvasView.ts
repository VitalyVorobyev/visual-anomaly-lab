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

import { MAX_SCALE } from "@vitavision/lab-ui";

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
/**
 * The most a view may magnify, in screen pixels per *source* pixel — lab-ui's `MAX_SCALE`,
 * the one ceiling every viewer in the lab family shares.
 *
 * It used to be twelve times *fit*, which is a different ceiling for every image: a frame
 * three times the size of the pane stopped at four screen pixels per source pixel, too coarse
 * to place a one-pixel correction, while a small crop could be blown up past any use.
 * Expressed against true 1:1, the same pixel is equally reachable on every dataset.
 */
export const MAX_PIXEL_SCALE = MAX_SCALE;

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

/**
 * The `zoom` range this pane allows, as `[min, max]` multiples of fit. The ceiling is
 * `MAX_PIXEL_SCALE` source-to-screen, never below fit itself.
 */
export function zoomRange(fit: number): [number, number] {
  return [MIN_ZOOM_VS_FIT, Math.max(1, MAX_PIXEL_SCALE / fit)];
}

export function clampZoom(zoom: number, fit: number): number {
  const [min, max] = zoomRange(fit);
  return clamp(zoom, min, max);
}

/** The source coordinate under a screen point, unclamped: the pointer may be off the frame. */
export function toSourceUnclamped(
  pointer: AnnotationPoint,
  origin: AnnotationPoint,
  scale: number,
): AnnotationPoint {
  return { x: (pointer.x - origin.x) / scale, y: (pointer.y - origin.y) / scale };
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

/** `zoomAbout` the centre of the pane: the rail's zoom buttons, which have no pointer. */
export function zoomBy(view: CanvasView, factor: number, size: Size, image: Size): CanvasView {
  return zoomAbout(view, factor, { x: size.width / 2, y: size.height / 2 }, size, image);
}

/**
 * Whether the photograph should be resampled smoothly at this scale. Above 1:1 a source pixel
 * covers several screen pixels, and smoothing would blur exactly the edge being traced — so it
 * is drawn as the square it is.
 */
export function smoothAt(scale: number): boolean {
  return scale <= 1;
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
