/**
 * How strongly annotated regions are painted over the photograph, remembered between visits.
 *
 * A preference about the reader rather than the page, so `localStorage` rather than the URL —
 * the same reasoning as `useCollapsedGroups`. It also has to survive channel and queue
 * navigation, which is a second reason not to hold it in the editor's per-target state.
 *
 * Why it is a control at all: the right weight depends on the imagery. A defect mask over a
 * bright specular surface needs to be heavy to be seen; over a dark field, the same weight hides
 * the texture the operator is judging. Neither answer is correct for both, and picking one for
 * everybody is how the mask ends up invisible on exactly the dataset in front of you.
 */

import { useCallback, useState } from "react";

const KEY = "anomaly-lab:mask-opacity";

export const DEFAULT_MASK_OPACITY = 0.45;

function read(): number {
  try {
    const raw = globalThis.localStorage?.getItem(KEY);
    const parsed = raw === null || raw === undefined ? Number.NaN : Number(raw);
    if (!Number.isFinite(parsed) || parsed < 0 || parsed > 1) return DEFAULT_MASK_OPACITY;
    return parsed;
  } catch {
    // Private browsing, or a WebView with storage disabled. The default is a fine answer.
    return DEFAULT_MASK_OPACITY;
  }
}

export function useMaskOpacity(): [number, (value: number) => void] {
  const [opacity, setOpacity] = useState<number>(read);

  const update = useCallback((value: number) => {
    const clamped = Math.min(1, Math.max(0, value));
    setOpacity(clamped);
    try {
      globalThis.localStorage?.setItem(KEY, String(clamped));
    } catch {
      // Not worth failing the drag over; it still applies for this session.
    }
  }, []);

  return [opacity, update];
}
