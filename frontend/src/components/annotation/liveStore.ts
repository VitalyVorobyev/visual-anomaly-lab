/**
 * The scene's fast-changing state, outside React's render of the scene.
 *
 * What changes at pointer rate — the gesture being drawn, the pointer, the keyboard cursor —
 * lives here, and only the components that draw it (the drafts, the readouts) subscribe. A
 * `setState` on the stage component instead would re-render every region and editor to draw
 * one more segment of a brush trail. The committed scene re-renders when the document, the
 * view or the selection changes, and not otherwise.
 *
 * Still no second store of annotation truth: everything here is a gesture in progress or a
 * cursor, and none of it is committed until the gesture ends.
 */

import { useSyncExternalStore } from "react";

import type { AnnotationPoint } from "../../api/client";
import type { Gesture } from "./tools";

export interface LiveState {
  /** The drag in progress. Its arrays are appended in place; `revision` says when. */
  gesture: Gesture | null;
  revision: number;
  /**
   * The source coordinate under the mouse, or `null` when it is off the frame. The brush
   * footprint is drawn here, the open polygon reaches to it, and the pixel readout reads it.
   */
  pointer: AnnotationPoint | null;
  /** The keyboard cursor, in source pixels. */
  keyboardPoint: AnnotationPoint;
  keyboardFocused: boolean;
}

export interface LiveStore {
  get: () => LiveState;
  set: (patch: Partial<LiveState>) => void;
  subscribe: (listener: () => void) => () => void;
}

export function createLiveStore(initial: LiveState): LiveStore {
  let state = initial;
  const listeners = new Set<() => void>();
  return {
    get: () => state,
    set: (patch) => {
      state = { ...state, ...patch };
      for (const listener of listeners) listener();
    },
    subscribe: (listener) => {
      listeners.add(listener);
      return () => listeners.delete(listener);
    },
  };
}

export function useLive<T>(store: LiveStore, select: (state: LiveState) => T): T {
  return useSyncExternalStore(store.subscribe, () => select(store.get()));
}
