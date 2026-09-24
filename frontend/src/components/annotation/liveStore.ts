/**
 * The scene's fast-changing state, outside React's render of the scene.
 *
 * A pointer move used to be a `setState` on the canvas component, which re-rendered the
 * whole Konva tree — every region, every vertex handle, the photograph — to draw one more
 * segment of a brush trail. What changes at pointer rate now lives here, and only the
 * components that draw it (the live layer, the readout) subscribe. The static layer re-renders
 * when the document, the view or the selection changes, and not otherwise.
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
  /** Whether the pointer is over the open ring's first vertex. */
  snapReady: boolean;
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
