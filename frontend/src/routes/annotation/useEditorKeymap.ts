/**
 * The editor's window shortcuts: `EDITOR_BINDINGS` resolved to actions.
 *
 * The page-level guard — text entry, lists, an open dialog — is `useHotkeys`'s. This handler
 * used to carry its own, which did not know about dialogs: with Copy or Discard open, `C`
 * completed the document and Backspace deleted the selected region behind it. Modified keys
 * are let through because ⌘S and ⌘Z are bound; every other binding is a bare key, so a held
 * modifier resolves to nothing rather than turning ⌘V into Select.
 *
 * Which key means what is data (`editorKeys.ts`), shared with the shortcut sheet. What each
 * command *does*, including when it declines — J/K with unsaved work, Enter with fewer than
 * three vertices — is the caller's `actions`, so the conditions stay beside the state they
 * read.
 */

import { type RefObject, useEffect, useRef } from "react";

import type { Label } from "../../api/client";
import type { AnnotationCanvasHandle } from "../../components/annotation/AnnotationCanvas";
import {
  CANVAS_BINDINGS,
  classSlotFor,
  type EditorCommand,
  PEEK_MS,
  windowBindingFor,
} from "../../components/annotation/editorKeys";
import { useHotkeys } from "../../hooks/useHotkeys";
import type { ChannelPanes } from "./useChannelPanes";
import type { DocumentCommands } from "./useDocumentCommands";
import type { DraftSession } from "./useDraftSession";
import type { QueueNavigation } from "./useQueueNavigation";
import type { SegmentAssistSession } from "./useSegmentAssistSession";
import type { Workspace } from "./useWorkspace";

export type EditorKeyActions = Record<EditorCommand, (event: KeyboardEvent) => void>;

/**
 * Keys the focused canvas owns — its cursor arrows, Space and Enter. Checked with modifiers
 * ignored, so a held Shift (ten-pixel steps) is still the canvas's.
 */
function ownedByCanvas(event: KeyboardEvent): boolean {
  const target = event.target;
  if (!(target instanceof HTMLElement) || !target.closest("[data-annotation-canvas]")) {
    return false;
  }
  const bare = { key: event.key, metaKey: false, ctrlKey: false, altKey: false, shiftKey: false };
  return CANVAS_BINDINGS.some((binding) => binding.match(bare));
}

/**
 * What each command does in this editor, and when it declines.
 *
 * Every condition the original `if`/`else` chain carried is here, beside the command it
 * guards: J/K and the arrows wait for a save, Enter needs three vertices, Backspace takes a
 * vertex off an open ring before it deletes a region, `C` does not start a second completion.
 */
export function editorKeyActions({
  session,
  commands,
  queue,
  panes,
  assist,
  workspace,
  canvas,
  applyLabel,
  complete,
  openShortcuts,
}: {
  session: DraftSession;
  commands: DocumentCommands;
  queue: QueueNavigation;
  panes: ChannelPanes;
  assist: SegmentAssistSession;
  workspace: Workspace;
  canvas: RefObject<AnnotationCanvasHandle | null>;
  applyLabel: (label: Label) => void;
  complete: () => Promise<void>;
  openShortcuts: () => void;
}): EditorKeyActions {
  const { pendingPoints, selectedId } = commands;
  const { setTool, brushSize, setBrushSize } = workspace;
  return {
    save: (event) => {
      event.preventDefault();
      void session.persist();
    },
    undo: (event) => {
      event.preventDefault();
      commands.undo();
    },
    redo: (event) => {
      event.preventDefault();
      commands.redo();
    },
    "tool.select": () => setTool("select"),
    "tool.polygon": () => setTool("polygon"),
    "tool.box": () => setTool("box"),
    "tool.brush": () => setTool("brush"),
    "tool.eraser": () => setTool("eraser"),
    "tool.assist": () => setTool("assist"),
    "polygon.close": (event) => {
      if (pendingPoints.length < 3) return;
      event.preventDefault();
      commands.finishPolygon();
    },
    cancel: () => {
      // Cancel the current thing, and only that. Escape used to also drop back to Select,
      // which made "escape to start a fresh region" cost a second keystroke to get the brush
      // back — V is the way to Select, and it always was.
      if (pendingPoints.length > 0) {
        commands.setPendingPoints([]);
      } else {
        commands.setSelectedId(null);
        assist.clear();
      }
    },
    delete: (event) => {
      if (pendingPoints.length > 0) {
        // Undo does not reach a ring that has not been committed yet, so the only way back
        // from a misplaced vertex used to be Escape and starting over.
        event.preventDefault();
        commands.setPendingPoints((points) => points.slice(0, -1));
      } else if (selectedId) {
        event.preventDefault();
        commands.removeSelected();
      }
    },
    "queue.next": () => {
      if (!session.dirty) queue.openNext();
    },
    "queue.previous": () => {
      if (!session.dirty) queue.openPrevious();
    },
    "channel.previous": () => void panes.openChannel(panes.activeIndex - 1),
    "channel.next": () => void panes.openChannel(panes.activeIndex + 1),
    "brush.smaller": (event) => setBrushSize(brushSize - (event.shiftKey ? 10 : 1)),
    "brush.larger": (event) => setBrushSize(brushSize + (event.shiftKey ? 10 : 1)),
    complete: () => {
      if (!session.complete.isPending) void complete();
    },
    "regions.toggle": () => workspace.setRegionsHidden((hidden) => !hidden),
    "view.fit": () => canvas.current?.fit(),
    "view.actual": () => canvas.current?.actualPixels(),
    "label.normal": () => applyLabel("normal"),
    "label.defect": () => applyLabel("defect"),
    "label.unlabeled": () => applyLabel("unlabeled"),
    "class.pick": (event) => {
      const slot = classSlotFor(event.key);
      if (slot !== null) commands.pickClass(slot);
    },
    shortcuts: (event) => {
      event.preventDefault();
      openShortcuts();
    },
  };
}

export function useEditorKeymap(actions: EditorKeyActions): void {
  /** When the current hold-binding press began, or `null` when the key is not down. */
  const held = useRef<{ key: string; command: EditorCommand; startedAt: number } | null>(null);
  const current = useRef(actions);
  current.current = actions;

  useHotkeys(
    (event) => {
      if (ownedByCanvas(event)) return;
      const binding = windowBindingFor(event);
      if (!binding) return;
      if (binding.hold) {
        // Press flips, and a *long* press flips back on release: tapping toggles, holding
        // shows the other state for as long as it is held. `repeat` is what makes the hold
        // work at all; without it the browser's auto-repeat would deliver a stream of toggles.
        if (event.repeat) return;
        held.current = {
          key: event.key.toLowerCase(),
          command: binding.command,
          startedAt: event.timeStamp,
        };
      }
      current.current[binding.command](event);
    },
    { allowModifiers: true },
  );

  useEffect(() => {
    const onKeyUp = (event: KeyboardEvent) => {
      const press = held.current;
      if (!press || event.key.toLowerCase() !== press.key) return;
      held.current = null;
      // Only a press this handler actually saw is a press it may undo: one that began in a
      // text field, or in another window, never set `held`.
      if (event.timeStamp - press.startedAt > PEEK_MS) current.current[press.command](event);
    };
    globalThis.addEventListener("keyup", onKeyUp);
    return () => globalThis.removeEventListener("keyup", onKeyUp);
  }, []);
}
