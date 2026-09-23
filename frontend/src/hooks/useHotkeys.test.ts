/**
 * The one guard every screen's shortcuts go through.
 *
 * Each case is a bug one of the screens actually had: a relabel on ⌘D, a page turn from a
 * focused slider, a document completed behind an open dialog.
 */

import { afterEach, describe, expect, it } from "vitest";

import { hotkeyBlocked } from "./useHotkeys";

function press(key: string, target: Element, init: KeyboardEventInit = {}): KeyboardEvent {
  const event = new KeyboardEvent("keydown", { key, bubbles: true, cancelable: true, ...init });
  Object.defineProperty(event, "target", { value: target });
  return event;
}

function mount(html: string): HTMLElement {
  const host = document.createElement("div");
  host.innerHTML = html;
  document.body.append(host);
  return host;
}

afterEach(() => {
  document.body.innerHTML = "";
});

describe("when a key is a shortcut", () => {
  it("lets a bare key on the page through", () => {
    expect(hotkeyBlocked(press("d", document.body))).toBe(false);
  });

  it("leaves modified keys to the browser unless a screen asks for them", () => {
    expect(hotkeyBlocked(press("d", document.body, { metaKey: true }))).toBe(true);
    expect(hotkeyBlocked(press("u", document.body, { ctrlKey: true }))).toBe(true);
    expect(hotkeyBlocked(press("s", document.body, { metaKey: true }), { allowModifiers: true })).toBe(
      false,
    );
  });

  it("never fires while text is being entered", () => {
    const host = mount(`<input id="a" /><textarea id="b"></textarea><div id="c" contenteditable="true"><span id="d"></span></div>`);
    for (const id of ["a", "b", "d"]) {
      expect(hotkeyBlocked(press("n", host.querySelector(`#${id}`)!))).toBe(true);
    }
  });

  it("leaves the arrows to a focused slider, but not its letters", () => {
    // A Radix thumb is a span with role=slider — the tag-name guard missed it entirely.
    const host = mount(`<span role="slider" tabindex="0" id="thumb"></span>`);
    const thumb = host.querySelector("#thumb")!;
    expect(hotkeyBlocked(press("ArrowRight", thumb))).toBe(true);
    expect(hotkeyBlocked(press("d", thumb))).toBe(false);
  });

  it("leaves every key to an open list, where letters are typeahead", () => {
    const host = mount(`<div role="listbox"><div role="option" id="o"></div></div>`);
    expect(hotkeyBlocked(press("d", host.querySelector("#o")!))).toBe(true);
  });

  it("does nothing behind an open dialog", () => {
    mount(`<div role="dialog" aria-modal="true"><button id="ok"></button></div>`);
    expect(hotkeyBlocked(press("c", document.body))).toBe(true);
    expect(hotkeyBlocked(press("Backspace", document.body))).toBe(true);
  });

  it("respects a key another handler already consumed", () => {
    const event = press("ArrowLeft", document.body);
    event.preventDefault();
    expect(hotkeyBlocked(event)).toBe(true);
  });
});
