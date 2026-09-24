/**
 * Presentation state: how the reader is looking, as opposed to what they are looking at.
 *
 * It lives above the keyed editor because it belongs to the person, not to the document.
 * Everything used to sit inside `EditorReady`, which is keyed by the draft's target, so
 * changing the second channel of a side-by-side comparison remounted the editor and reset
 * the pane mode, the zoom and the active tool along with it — the reported "changing the
 * left channel jumps back to a single channel". Per-target state (history, selection, the
 * pending polygon) keeps the key and keeps resetting, which is the point of the key.
 */

import type { Dispatch, SetStateAction } from "react";
import { useCallback, useMemo, useState } from "react";

import type { PaneMode } from "../../api/annotationPanes";
import {
  INITIAL_CANVAS_VIEW,
  type CanvasView,
  type EditorTool,
} from "../../components/annotation/AnnotationCanvas";
import { useBrushSize } from "../../hooks/useBrushSize";
import { useMaskOpacity } from "../../hooks/useMaskOpacity";

export interface Workspace {
  paneMode: PaneMode;
  setPaneMode: (mode: PaneMode) => void;
  /** A channel position, not an image id — see `resolveReference`. */
  referenceIndex: number | null;
  setReferenceIndex: (index: number | null) => void;
  overlayOpacity: number;
  setOverlayOpacity: (value: number) => void;
  maskOpacity: number;
  setMaskOpacity: (value: number) => void;
  /**
   * Whether the drawn regions are on screen at all.
   *
   * Beside `maskOpacity` rather than inside it, and unlike it *not* remembered in
   * `localStorage`: opacity is a preference about the imagery, while hiding is a moment of
   * looking. An editor that opened with every annotation invisible, because of a keystroke
   * from a previous session, reads as work that has been lost.
   */
  regionsHidden: boolean;
  /** The raw dispatch, so `H` can flip it without reading a value out of a stale closure. */
  setRegionsHidden: Dispatch<SetStateAction<boolean>>;
  tool: EditorTool;
  setTool: (tool: EditorTool) => void;
  brushSize: number;
  setBrushSize: (radius: number) => void;
  view: CanvasView;
  setView: (view: CanvasView) => void;
}

export function useWorkspace(frame: string): Workspace {
  const [paneMode, setPaneMode] = useState<PaneMode>("single");
  const [referenceIndex, setReferenceIndex] = useState<number | null>(null);
  const [overlayOpacity, setOverlayOpacity] = useState(0.5);
  const [maskOpacity, setMaskOpacity] = useMaskOpacity();
  const [regionsHidden, setRegionsHidden] = useState(false);
  const [tool, setTool] = useState<EditorTool>("select");
  const [brushSize, setBrushSize] = useBrushSize();
  // The view is stamped with the frame it was expressed on and derived back out, so moving
  // to another part resets it during render rather than in an effect that would first paint
  // the previous part's zoom over the new photograph.
  const [viewMemo, setViewMemo] = useState({ frame, view: INITIAL_CANVAS_VIEW });
  const view = viewMemo.frame === frame ? viewMemo.view : INITIAL_CANVAS_VIEW;
  const setView = useCallback((next: CanvasView) => setViewMemo({ frame, view: next }), [frame]);

  return useMemo(
    () => ({
      paneMode,
      setPaneMode,
      referenceIndex,
      setReferenceIndex,
      overlayOpacity,
      setOverlayOpacity,
      maskOpacity,
      setMaskOpacity,
      regionsHidden,
      setRegionsHidden,
      tool,
      setTool,
      brushSize,
      setBrushSize,
      view,
      setView,
    }),
    [
      brushSize,
      maskOpacity,
      overlayOpacity,
      paneMode,
      referenceIndex,
      regionsHidden,
      setMaskOpacity,
      setView,
      tool,
      view,
    ],
  );
}
