/**
 * The annotation editor's keys, as data.
 *
 * One list is both the keymap and its documentation: `useEditorKeymap` resolves a keystroke
 * by walking it, and the shortcut sheet renders it. A key that is bound but not listed, or
 * listed but not bound, cannot exist — which is the drift the footer's hand-written hint line
 * used to have (it never mentioned `,`/`.`, `0` or `1`).
 *
 * `window` bindings reach the editor through `useHotkeys` and its one guard. `canvas`
 * bindings are handled by the focused canvas itself, which owns the arrows, Space and Enter
 * while it has focus. `pointer` entries are gestures, listed so the sheet is complete; they
 * never match a key.
 */

export type EditorCommand =
  | "save"
  | "undo"
  | "redo"
  | "tool.select"
  | "tool.polygon"
  | "tool.box"
  | "tool.brush"
  | "tool.eraser"
  | "tool.assist"
  | "polygon.close"
  | "cancel"
  | "delete"
  | "queue.next"
  | "queue.previous"
  | "channel.previous"
  | "channel.next"
  | "brush.smaller"
  | "brush.larger"
  | "complete"
  | "regions.toggle"
  | "view.fit"
  | "view.actual"
  | "label.normal"
  | "label.defect"
  | "label.unlabeled"
  | "class.pick"
  | "shortcuts";

export type CanvasCommand = "cursor.move" | "tool.apply" | "polygon.close";

/** The part of a `KeyboardEvent` a binding reads, so matching is testable without a DOM. */
export type KeyLike = Pick<KeyboardEvent, "key" | "metaKey" | "ctrlKey" | "altKey" | "shiftKey">;

export type KeyGroup = "Tools" | "Document" | "Navigation" | "View" | "Label" | "Canvas" | "Pointer";

interface BindingBase {
  /** What the sheet prints, one chip per alternative. */
  keys: string[];
  description: string;
  group: KeyGroup;
}

export interface WindowBinding extends BindingBase {
  scope: "window";
  command: EditorCommand;
  match: (event: KeyLike) => boolean;
  /**
   * Press flips, and a press held past `PEEK_MS` flips back on release. Only `H` has it:
   * tapping toggles the mask, holding shows the other state for as long as it is held.
   */
  hold?: boolean;
}

export interface CanvasBinding extends BindingBase {
  scope: "canvas";
  command: CanvasCommand;
  match: (event: KeyLike) => boolean;
}

export interface PointerBinding extends BindingBase {
  scope: "pointer";
}

export type EditorBinding = WindowBinding | CanvasBinding | PointerBinding;

/** Longer than this and an `H` press was somebody looking, not somebody switching. */
export const PEEK_MS = 250;

/** Letters compare case-insensitively (Shift+V is still Select); named keys compare exactly. */
function normal(key: string): string {
  return key.length === 1 ? key.toLowerCase() : key;
}

function commandHeld(event: KeyLike): boolean {
  return event.metaKey || event.ctrlKey;
}

/** A key with no ⌘/Ctrl. Shift is allowed: `<` and `>` are how Shift reaches `,` and `.`. */
function bare(...keys: string[]) {
  return (event: KeyLike) => !commandHeld(event) && keys.includes(normal(event.key));
}

function withCommand(key: string, shift: boolean | undefined = undefined) {
  return (event: KeyLike) =>
    commandHeld(event) &&
    normal(event.key) === key &&
    (shift === undefined || event.shiftKey === shift);
}

const ARROWS = ["ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight"];

/**
 * The digits that pick a class, in class order: `2` is the first class, `9` the eighth.
 * `0` and `1` are the view's (Fit, 1:1) and stay so, which is why the run starts at two.
 */
const CLASS_KEYS = ["2", "3", "4", "5", "6", "7", "8", "9"];

/** The position in class order a digit picks, or `null` for any other key. */
export function classSlotFor(key: string): number | null {
  const slot = CLASS_KEYS.indexOf(key);
  return slot < 0 ? null : slot;
}

/** The key that picks the class at `position` in class order, if it has one. */
export function classKeyAt(position: number): string | null {
  return CLASS_KEYS[position] ?? null;
}

export const EDITOR_BINDINGS: readonly EditorBinding[] = [
  { scope: "window", command: "tool.select", keys: ["V"], description: "Select, move and reshape", group: "Tools", match: bare("v") },
  { scope: "window", command: "tool.polygon", keys: ["P"], description: "Polygon", group: "Tools", match: bare("p") },
  { scope: "window", command: "tool.box", keys: ["R"], description: "Box", group: "Tools", match: bare("r") },
  { scope: "window", command: "tool.brush", keys: ["B"], description: "Brush", group: "Tools", match: bare("b") },
  { scope: "window", command: "tool.eraser", keys: ["E"], description: "Eraser", group: "Tools", match: bare("e") },
  { scope: "window", command: "tool.assist", keys: ["A"], description: "Contour assist (MobileSAM)", group: "Tools", match: bare("a") },
  // Not `[` and `]`, the conventional pair — those are channel navigation here and have been
  // longer. Shift jumps by ten so the whole range is a few keystrokes.
  { scope: "window", command: "class.pick", keys: ["2–9"], description: "Class for new regions, in class order (the first eight)", group: "Tools", match: bare(...CLASS_KEYS) },
  { scope: "window", command: "brush.smaller", keys: [",", "<"], description: "Brush smaller (Shift: by ten)", group: "Tools", match: bare(",", "<") },
  { scope: "window", command: "brush.larger", keys: [".", ">"], description: "Brush larger (Shift: by ten)", group: "Tools", match: bare(".", ">") },

  { scope: "window", command: "save", keys: ["⌘S"], description: "Save the draft", group: "Document", match: withCommand("s") },
  { scope: "window", command: "undo", keys: ["⌘Z"], description: "Undo", group: "Document", match: withCommand("z", false) },
  { scope: "window", command: "redo", keys: ["⇧⌘Z"], description: "Redo", group: "Document", match: withCommand("z", true) },
  { scope: "window", command: "polygon.close", keys: ["Enter"], description: "Close the open polygon (three vertices or more)", group: "Document", match: bare("Enter") },
  { scope: "window", command: "cancel", keys: ["Esc"], description: "Drop the open polygon, else clear the selection and assist prompts", group: "Document", match: bare("Escape") },
  { scope: "window", command: "delete", keys: ["⌫", "Del"], description: "Remove the last open vertex, else delete the selected region", group: "Document", match: bare("Backspace", "Delete") },
  { scope: "window", command: "complete", keys: ["C"], description: "Complete: save, freeze a revision and open the next", group: "Document", match: bare("c") },

  { scope: "window", command: "queue.next", keys: ["J", "→"], description: "Next in the queue (once saved)", group: "Navigation", match: bare("j", "ArrowRight") },
  { scope: "window", command: "queue.previous", keys: ["K", "←"], description: "Previous in the queue (once saved)", group: "Navigation", match: bare("k", "ArrowLeft") },
  { scope: "window", command: "channel.previous", keys: ["["], description: "Previous channel", group: "Navigation", match: bare("[") },
  { scope: "window", command: "channel.next", keys: ["]"], description: "Next channel", group: "Navigation", match: bare("]") },

  { scope: "window", command: "regions.toggle", keys: ["H"], description: "Hide or show the regions; hold to peek", group: "View", match: bare("h"), hold: true },
  { scope: "window", command: "view.fit", keys: ["0"], description: "Fit the image", group: "View", match: bare("0") },
  { scope: "window", command: "view.actual", keys: ["1"], description: "Actual pixels (1:1)", group: "View", match: bare("1") },
  { scope: "window", command: "shortcuts", keys: ["?"], description: "This list", group: "View", match: bare("?") },

  { scope: "window", command: "label.normal", keys: ["N"], description: "Label the sample normal", group: "Label", match: bare("n") },
  { scope: "window", command: "label.defect", keys: ["D"], description: "Label the sample defect", group: "Label", match: bare("d") },
  { scope: "window", command: "label.unlabeled", keys: ["U"], description: "Label the sample unlabeled", group: "Label", match: bare("u") },

  { scope: "canvas", command: "cursor.move", keys: ["←", "↑", "→", "↓"], description: "Move the pixel cursor, or nudge the selected region under Select (Shift: 10 px)", group: "Canvas", match: (event) => !commandHeld(event) && ARROWS.includes(event.key) },
  { scope: "canvas", command: "polygon.close", keys: ["Enter"], description: "Close the polygon, or place a vertex at the cursor", group: "Canvas", match: (event) => !commandHeld(event) && event.key === "Enter" },
  { scope: "canvas", command: "tool.apply", keys: ["Space"], description: "Apply the drawing tool at the cursor (Shift: negative assist point)", group: "Canvas", match: (event) => !commandHeld(event) && event.key === " " },

  { scope: "pointer", keys: ["Right-drag"], description: "Pan, under every tool", group: "Pointer" },
  { scope: "pointer", keys: ["Drag"], description: "Pan under Select, when not on a region", group: "Pointer" },
  { scope: "pointer", keys: ["Wheel"], description: "Zoom about the pointer", group: "Pointer" },
  { scope: "pointer", keys: ["Double-click"], description: "Select: toggle Fit. Polygon: close the ring", group: "Pointer" },
  { scope: "pointer", keys: ["Shift-click"], description: "Assist: a background point", group: "Pointer" },
];

export const WINDOW_BINDINGS = EDITOR_BINDINGS.filter(
  (binding): binding is WindowBinding => binding.scope === "window",
);

export const CANVAS_BINDINGS = EDITOR_BINDINGS.filter(
  (binding): binding is CanvasBinding => binding.scope === "canvas",
);

/**
 * The window binding a keystroke resolves to, if any.
 *
 * Alt never reaches a command, and a held ⌘/Ctrl reaches only the bindings that ask for it —
 * so ⌘V stays the browser's paste rather than turning into Select.
 */
export function windowBindingFor(event: KeyLike): WindowBinding | undefined {
  if (event.altKey) return undefined;
  return WINDOW_BINDINGS.find((binding) => binding.match(event));
}

/** A control's name with the keys that reach it, for a tooltip: `Select (V)`, `Undo (⌘Z)`. */
export function withKeys(label: string, command: EditorCommand): string {
  const binding = WINDOW_BINDINGS.find((candidate) => candidate.command === command);
  return binding ? `${label} (${binding.keys.join(" / ")})` : label;
}

export function canvasBindingFor(event: KeyLike): CanvasBinding | undefined {
  if (event.altKey) return undefined;
  return CANVAS_BINDINGS.find((binding) => binding.match(event));
}
