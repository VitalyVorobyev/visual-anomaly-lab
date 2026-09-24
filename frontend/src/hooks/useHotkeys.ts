/**
 * Window-level shortcuts, behind one guard.
 *
 * Every screen with shortcuts used to write its own `keydown` listener and its own idea of
 * when a key is not a command, and each got a different part of it wrong. The sample page
 * relabelled on ⌘D and Ctrl+U. The experiment sample page paged away while its own slider
 * had focus, because an arrow on a Radix thumb is a `span[role=slider]`, not an `<input>`.
 * The annotation editor completed the document on `C` and deleted a region on Backspace
 * while a dialog was open over it. The rule is one rule, so it lives here once.
 *
 * A key is *not* a shortcut when:
 *
 * - something already handled it (`defaultPrevented`);
 * - text is being entered — an input, a textarea, a select, anything contentEditable;
 * - focus is in a list or menu, where every key is typeahead or navigation;
 * - it is a navigation key and focus is on a widget that owns those keys — a slider, a tab
 *   strip, a radio group, a spin button;
 * - a dialog is open: a shortcut must never act on the page *behind* a question;
 * - a modifier is held and the screen did not ask for modified keys — ⌘D is the browser's.
 */

import { useEffect, useRef } from "react";

const TEXT_ENTRY = "input, textarea, select, [contenteditable=''], [contenteditable='true']";
const OWNS_EVERY_KEY = "[role='listbox'], [role='menu'], [role='combobox']";
const OWNS_NAVIGATION =
  "[role='slider'], [role='tablist'], [role='radiogroup'], [role='spinbutton'], [role='scrollbar']";
const NAVIGATION_KEYS = new Set([
  "ArrowLeft",
  "ArrowRight",
  "ArrowUp",
  "ArrowDown",
  "Home",
  "End",
  "PageUp",
  "PageDown",
]);
const OPEN_DIALOG = "[role='dialog'], [role='alertdialog']";

export interface HotkeyOptions {
  /**
   * Let ⌘/Ctrl/Alt combinations through to the handler. Off by default, because most
   * screens bind bare letters and a held modifier means the key belongs to the browser.
   * A screen that binds ⌘S or ⌘Z opts in and checks the modifier itself.
   */
  allowModifiers?: boolean;
}

/** True when this key event must not be treated as a shortcut. Pure, so it is testable. */
export function hotkeyBlocked(
  event: KeyboardEvent,
  options: HotkeyOptions = {},
  doc: Document | undefined = globalThis.document,
): boolean {
  if (event.defaultPrevented) return true;
  if (!options.allowModifiers && (event.metaKey || event.ctrlKey || event.altKey)) return true;

  const target = event.target instanceof Element ? event.target : null;
  if (target) {
    if (target.closest(TEXT_ENTRY)) return true;
    if (target instanceof HTMLElement && target.isContentEditable) return true;
    if (target.closest(OWNS_EVERY_KEY)) return true;
    if (NAVIGATION_KEYS.has(event.key) && target.closest(OWNS_NAVIGATION)) return true;
  }

  return doc?.querySelector(OPEN_DIALOG) != null;
}

/**
 * Subscribe `handler` to window `keydown`, filtered by `hotkeyBlocked`.
 *
 * The handler is held in a ref so the listener is attached once: a screen that re-renders
 * on every pointer move would otherwise add and remove a window listener at the same rate.
 */
export function useHotkeys(
  handler: (event: KeyboardEvent) => void,
  options: HotkeyOptions = {},
): void {
  const current = useRef(handler);
  current.current = handler;
  const allowModifiers = options.allowModifiers ?? false;

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (hotkeyBlocked(event, { allowModifiers })) return;
      current.current(event);
    };
    globalThis.addEventListener("keydown", onKey);
    return () => globalThis.removeEventListener("keydown", onKey);
  }, [allowModifiers]);
}
