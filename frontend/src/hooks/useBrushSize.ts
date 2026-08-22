/**
 * How wide the brush and the eraser are, in source pixels, remembered between visits.
 *
 * A preference about the reader rather than the page, so `localStorage` rather than the URL —
 * the same reasoning as `useMaskOpacity`, and the same need to survive channel and queue
 * navigation. It matters more here than for a display setting: somebody correcting a mask a
 * pixel at a time sets this once and would otherwise re-set it after every reload.
 *
 * **The value is a diameter.** It used to be a radius, while the control was labelled "Brush
 * size" and read out in `px` — so a setting of 18 painted 36 pixels wide and a floor of 2
 * painted 4. Diameter is what the label always claimed and what makes `1` mean one pixel.
 */

import { useCallback, useState } from "react";

const KEY = "anomaly-lab:brush-size";

/** One pixel. The rasteriser marks exactly that; see `rasterizeStroke`. */
export const MIN_BRUSH_SIZE = 1;
export const MAX_BRUSH_SIZE = 128;
/** The previous default, which was a radius of 18 — the same footprint, honestly named. */
export const DEFAULT_BRUSH_SIZE = 36;

export function clampBrushSize(value: number): number {
  if (!Number.isFinite(value)) return DEFAULT_BRUSH_SIZE;
  return Math.min(MAX_BRUSH_SIZE, Math.max(MIN_BRUSH_SIZE, Math.round(value)));
}

function read(): number {
  try {
    const raw = globalThis.localStorage?.getItem(KEY);
    if (raw === null || raw === undefined) return DEFAULT_BRUSH_SIZE;
    const parsed = Number(raw);
    return Number.isFinite(parsed) ? clampBrushSize(parsed) : DEFAULT_BRUSH_SIZE;
  } catch {
    // Private browsing, or a WebView with storage disabled. The default is a fine answer.
    return DEFAULT_BRUSH_SIZE;
  }
}

export function useBrushSize(): [number, (value: number) => void] {
  const [size, setSize] = useState<number>(read);

  const update = useCallback((value: number) => {
    const clamped = clampBrushSize(value);
    setSize(clamped);
    try {
      globalThis.localStorage?.setItem(KEY, String(clamped));
    } catch {
      // Not worth failing the drag over; it still applies for this session.
    }
  }, []);

  return [size, update];
}
