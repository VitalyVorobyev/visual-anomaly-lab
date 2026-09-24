/**
 * The editor's key list is the keymap, so it is asserted as one: every command is reachable,
 * no keystroke is claimed twice, and a held modifier reaches only what asked for it.
 */

import { describe, expect, it } from "vitest";

import {
  EDITOR_BINDINGS,
  WINDOW_BINDINGS,
  type EditorCommand,
  type KeyLike,
  canvasBindingFor,
  windowBindingFor,
  withKeys,
} from "./editorKeys";

function key(value: string, modifiers: Partial<KeyLike> = {}): KeyLike {
  return { key: value, metaKey: false, ctrlKey: false, altKey: false, shiftKey: false, ...modifiers };
}

const command = (event: KeyLike) => windowBindingFor(event)?.command;

describe("window bindings", () => {
  it.each<[KeyLike, EditorCommand]>([
    [key("v"), "tool.select"],
    [key("V", { shiftKey: true }), "tool.select"],
    [key("p"), "tool.polygon"],
    [key("b"), "tool.brush"],
    [key("e"), "tool.eraser"],
    [key("a"), "tool.assist"],
    [key("s", { metaKey: true }), "save"],
    [key("s", { ctrlKey: true }), "save"],
    [key("z", { metaKey: true }), "undo"],
    [key("z", { metaKey: true, shiftKey: true }), "redo"],
    [key("Enter"), "polygon.close"],
    [key("Escape"), "cancel"],
    [key("Backspace"), "delete"],
    [key("Delete"), "delete"],
    [key("j"), "queue.next"],
    [key("ArrowRight"), "queue.next"],
    [key("k"), "queue.previous"],
    [key("ArrowLeft"), "queue.previous"],
    [key("["), "channel.previous"],
    [key("]"), "channel.next"],
    [key(","), "brush.smaller"],
    [key("<", { shiftKey: true }), "brush.smaller"],
    [key("."), "brush.larger"],
    [key(">", { shiftKey: true }), "brush.larger"],
    [key("c"), "complete"],
    [key("h"), "regions.toggle"],
    [key("0"), "view.fit"],
    [key("1"), "view.actual"],
    [key("n"), "label.normal"],
    [key("d"), "label.defect"],
    [key("u"), "label.unlabeled"],
    [key("?", { shiftKey: true }), "shortcuts"],
  ])("%o → %s", (event, expected) => {
    expect(command(event)).toBe(expected);
  });

  it("gives a held ⌘ or Ctrl only to the bindings that ask for it", () => {
    // ⌘V is paste, ⌘D is the bookmark, Ctrl+U is view-source: the browser's, never a tool or
    // a relabel.
    expect(command(key("v", { metaKey: true }))).toBeUndefined();
    expect(command(key("d", { metaKey: true }))).toBeUndefined();
    expect(command(key("u", { ctrlKey: true }))).toBeUndefined();
    expect(command(key("c", { metaKey: true }))).toBeUndefined();
  });

  it("never answers Alt", () => {
    expect(command(key("v", { altKey: true }))).toBeUndefined();
    expect(command(key("s", { metaKey: true, altKey: true }))).toBeUndefined();
  });

  it("binds every command exactly once", () => {
    const commands = WINDOW_BINDINGS.map((binding) => binding.command);
    expect(new Set(commands).size).toBe(commands.length);
  });

  it("claims no keystroke twice", () => {
    for (const binding of WINDOW_BINDINGS) {
      const others = WINDOW_BINDINGS.filter((other) => other !== binding);
      for (const shown of binding.keys) {
        // The printed key, pressed bare, must reach this binding and no other.
        const bare = shown.length === 1 ? key(shown.toLowerCase()) : undefined;
        if (!bare || !binding.match(bare)) continue;
        expect(others.filter((other) => other.match(bare)).map((other) => other.command)).toEqual([]);
      }
    }
  });
});

describe("canvas bindings", () => {
  it("own the arrows, Enter and Space", () => {
    expect(canvasBindingFor(key("ArrowUp"))?.command).toBe("cursor.move");
    expect(canvasBindingFor(key("ArrowUp", { shiftKey: true }))?.command).toBe("cursor.move");
    expect(canvasBindingFor(key("Enter"))?.command).toBe("polygon.close");
    expect(canvasBindingFor(key(" "))?.command).toBe("tool.apply");
    expect(canvasBindingFor(key("v"))).toBeUndefined();
  });
});

describe("the sheet's source", () => {
  it("documents every binding, pointer gestures included", () => {
    for (const binding of EDITOR_BINDINGS) {
      expect(binding.keys.length).toBeGreaterThan(0);
      expect(binding.description).not.toBe("");
    }
    expect(EDITOR_BINDINGS.some((binding) => binding.scope === "pointer")).toBe(true);
  });

  it("names a control's keys from the same list", () => {
    expect(withKeys("Select", "tool.select")).toBe("Select (V)");
    expect(withKeys("Redo", "redo")).toBe("Redo (⇧⌘Z)");
  });
});
