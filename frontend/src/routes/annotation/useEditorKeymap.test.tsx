/**
 * The keymap as the editor wires it: a real window `keydown` reaches the action its binding
 * names, through `useHotkeys`'s guard, and `H` flips back on a long release.
 */

import { renderHook } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { WINDOW_BINDINGS, type EditorCommand } from "../../components/annotation/editorKeys";
import { type EditorKeyActions, useEditorKeymap } from "./useEditorKeymap";

function spies(): EditorKeyActions & Record<EditorCommand, ReturnType<typeof vi.fn>> {
  return Object.fromEntries(
    WINDOW_BINDINGS.map((binding) => [binding.command, vi.fn()]),
  ) as unknown as EditorKeyActions & Record<EditorCommand, ReturnType<typeof vi.fn>>;
}

function press(key: string, init: KeyboardEventInit = {}, target: EventTarget = window) {
  const event = new KeyboardEvent("keydown", { key, bubbles: true, cancelable: true, ...init });
  target.dispatchEvent(event);
  return event;
}

function release(key: string) {
  window.dispatchEvent(new KeyboardEvent("keyup", { key, bubbles: true }));
}

afterEach(() => {
  document.body.innerHTML = "";
});

describe("useEditorKeymap", () => {
  it("routes each key to its command's action", () => {
    const actions = spies();
    renderHook(() => useEditorKeymap(actions));

    press("b");
    press("z", { metaKey: true });
    press("z", { metaKey: true, shiftKey: true });
    press("j");
    press("?", { shiftKey: true });

    expect(actions["tool.brush"]).toHaveBeenCalledTimes(1);
    expect(actions.undo).toHaveBeenCalledTimes(1);
    expect(actions.redo).toHaveBeenCalledTimes(1);
    expect(actions["queue.next"]).toHaveBeenCalledTimes(1);
    expect(actions.shortcuts).toHaveBeenCalledTimes(1);
  });

  it("leaves a modified key the editor did not bind to the browser", () => {
    const actions = spies();
    renderHook(() => useEditorKeymap(actions));
    press("v", { metaKey: true });
    press("d", { metaKey: true });
    expect(actions["tool.select"]).not.toHaveBeenCalled();
    expect(actions["label.defect"]).not.toHaveBeenCalled();
  });

  it("acts on nothing behind an open dialog", () => {
    const actions = spies();
    renderHook(() => useEditorKeymap(actions));
    const dialog = document.createElement("div");
    dialog.setAttribute("role", "dialog");
    document.body.append(dialog);
    press("c");
    press("Backspace");
    expect(actions.complete).not.toHaveBeenCalled();
    expect(actions.delete).not.toHaveBeenCalled();
  });

  it("leaves the arrows, Enter and Space to a focused canvas", () => {
    const actions = spies();
    renderHook(() => useEditorKeymap(actions));
    const canvas = document.createElement("div");
    canvas.setAttribute("data-annotation-canvas", "");
    document.body.append(canvas);
    press("ArrowRight", {}, canvas);
    press("Enter", {}, canvas);
    press("j", {}, canvas);
    expect(actions["queue.next"]).toHaveBeenCalledTimes(1);
    expect(actions["polygon.close"]).not.toHaveBeenCalled();
  });

  it("uses the latest actions without re-subscribing", () => {
    const first = spies();
    const second = spies();
    const { rerender } = renderHook(({ actions }) => useEditorKeymap(actions), {
      initialProps: { actions: first },
    });
    rerender({ actions: second });
    press("p");
    expect(first["tool.polygon"]).not.toHaveBeenCalled();
    expect(second["tool.polygon"]).toHaveBeenCalledTimes(1);
  });

  describe("H", () => {
    it("toggles once on a tap", () => {
      const actions = spies();
      renderHook(() => useEditorKeymap(actions));
      press("h");
      release("h");
      expect(actions["regions.toggle"]).toHaveBeenCalledTimes(1);
    });

    it("flips back on the release of a hold, and ignores auto-repeat", () => {
      const actions = spies();
      renderHook(() => useEditorKeymap(actions));
      const down = press("h");
      press("h", { repeat: true });
      press("h", { repeat: true });
      expect(actions["regions.toggle"]).toHaveBeenCalledTimes(1);
      // A keyup whose timestamp is well past the peek threshold.
      const up = new KeyboardEvent("keyup", { key: "h", bubbles: true });
      Object.defineProperty(up, "timeStamp", { value: down.timeStamp + 1000 });
      window.dispatchEvent(up);
      expect(actions["regions.toggle"]).toHaveBeenCalledTimes(2);
    });

    it("does not undo a press it never saw", () => {
      const actions = spies();
      renderHook(() => useEditorKeymap(actions));
      const input = document.createElement("input");
      document.body.append(input);
      press("h", {}, input);
      release("h");
      expect(actions["regions.toggle"]).not.toHaveBeenCalled();
    });
  });
});
