/**
 * The draft session: one document's history, its concurrency token and its persistence.
 *
 * Everything about *keeping* the work lives here — undo history, the `ETag` that owns the
 * persisted row, save, idle autosave, completion, discard and the unload guard. What the
 * reader does to the document (`useDocumentCommands`) and where they go next
 * (`useQueueNavigation`) are separate seams that call in.
 */

import { useCallback, useEffect, useReducer, useRef, useState } from "react";
import { useBeforeUnload } from "react-router";

import {
  type HistoryAction,
  canonical,
  createHistory,
  historyReducer,
} from "../../api/annotationState";
import { ApiError } from "../../api/client";
import {
  type DraftEnvelope,
  type DraftTarget,
  useCompleteDraft,
  useDiscardDraft,
  useSaveDraft,
} from "../../hooks/useAnnotations";
import type { Flash } from "./useFlashMessage";

/** 412 rather than a substring of the detail: the status is the contract, the prose is not. */
export function isConflict(error: Error | null | undefined): boolean {
  return error instanceof ApiError && error.status === 412;
}

/** How long a dirty document sits idle before it is saved on its own. */
export const AUTOSAVE_MS = 1200;

export function useDraftSession({
  target,
  initial,
  imageIds,
  flash,
}: {
  target: DraftTarget;
  initial: DraftEnvelope;
  /** Every image the completion writes, so their caches are refreshed with it. */
  imageIds: number[];
  flash: Flash;
}) {
  const [history, dispatchHistory] = useReducer(historyReducer, initial.document, createHistory);

  /**
   * The history as it will be once every dispatched action has rendered.
   *
   * `history` in a closure is the history *of the render that made the closure*, and an async
   * edit that commits a document built from it discards everything dispatched in between — see
   * `useDocumentCommands.applyStroke`. The reducer is pure, so running it here as well as in
   * React gives every caller the same answer React is about to render; a render adopts React's
   * own object, which is the same value.
   */
  const latestHistory = useRef(history);
  const renderedHistory = useRef(history);
  if (renderedHistory.current !== history) {
    renderedHistory.current = history;
    latestHistory.current = history;
  }
  const dispatch = useCallback((action: HistoryAction) => {
    latestHistory.current = historyReducer(latestHistory.current, action);
    dispatchHistory(action);
  }, []);
  /** The document every dispatched edit so far has produced, rendered or not. */
  const latest = useCallback(() => latestHistory.current.present, []);

  const [etag, setEtag] = useState(initial.etag);
  const [draftVersion, setDraftVersion] = useState(initial.version);
  const [savedDocument, setSavedDocument] = useState(initial.document);
  const save = useSaveDraft(target);
  const discard = useDiscardDraft(target);
  const complete = useCompleteDraft(target, imageIds);
  const dirty = canonical(history.present) !== canonical(savedDocument);

  /**
   * Ensure a persisted draft exists and return the token that owns it.
   *
   * Two things it must get right. A clean document still has to be materialised when nothing
   * is persisted yet — completing an unedited seed is a real action, and accepting an imported
   * source mask as truth verbatim is the common case. And concurrent callers must share one
   * flight: the idle autosave and an explicit save would otherwise both start from the same
   * token and the loser would collect a 412 it caused itself.
   */
  const inFlight = useRef<Promise<string> | null>(null);
  const persist = useCallback((): Promise<string> => {
    if (inFlight.current) return inFlight.current;
    if (!dirty && etag !== null) return Promise.resolve(etag);
    const flight = save
      .mutateAsync({ document: history.present, etag })
      .then((saved) => {
        setEtag(saved.etag);
        setDraftVersion(saved.version);
        setSavedDocument(saved.document);
        flash("Draft saved");
        return saved.etag as string;
      })
      .finally(() => {
        inFlight.current = null;
      });
    inFlight.current = flight;
    return flight;
  }, [dirty, etag, flash, history.present, save]);

  /**
   * Save, then freeze a revision. Resolves `true` when a revision was written; the mutations
   * expose their own errors, so a failure keeps the current image open and says so there.
   */
  const completeDraft = useCallback(async (): Promise<boolean> => {
    try {
      const currentEtag = await persist();
      const revisions = await complete.mutateAsync(currentEtag);
      const first = revisions[0];
      flash(
        revisions.length > 1
          ? `Completed revision ${first?.revision_no} on ${revisions.length} channels`
          : `Completed revision ${first?.revision_no}`,
      );
      return true;
    } catch {
      return false;
    }
  }, [complete, flash, persist]);

  /**
   * Throw the draft away. A 412 keeps the dialog open and turns its button into the explicit
   * force; the mutation's error is rendered inside it. Resolves `true` once discarded.
   */
  const discardDraft = useCallback(
    async (force: boolean): Promise<boolean> => {
      try {
        await discard.mutateAsync({ etag, force });
        flash("Draft discarded");
        return true;
      } catch {
        return false;
      }
    },
    [discard, etag, flash],
  );

  useBeforeUnload(
    useCallback(
      (event: BeforeUnloadEvent) => {
        if (!dirty) return;
        event.preventDefault();
        event.returnValue = "";
      },
      [dirty],
    ),
  );

  // A label PATCH answers 404 or 500, never 412, so the route folding it in beside these
  // cannot light up "Reload server draft" — that stays gated on an actual draft conflict.
  const error = save.error ?? complete.error;

  return {
    history,
    dispatch,
    latest,
    etag,
    draftVersion,
    dirty,
    persist,
    completeDraft,
    discardDraft,
    save,
    discard,
    complete,
    error,
  };
}

export type DraftSession = ReturnType<typeof useDraftSession>;

/**
 * Save a dirty document after it has been left alone for `AUTOSAVE_MS`.
 *
 * `held` is the caller's "not now": an open polygon is not a document anybody meant to save.
 * A failed save stops the loop rather than retrying into the same 412.
 */
export function useAutosave(session: DraftSession, held: boolean): void {
  const { dirty, persist, save, complete, history } = session;
  useEffect(() => {
    if (!dirty || held || save.isPending || save.error || complete.isPending) return;
    const timer = globalThis.setTimeout(() => {
      void persist();
    }, AUTOSAVE_MS);
    return () => globalThis.clearTimeout(timer);
  }, [complete.isPending, dirty, history.present, held, persist, save.error, save.isPending]);
}
