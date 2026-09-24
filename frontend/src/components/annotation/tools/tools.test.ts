/**
 * The editing tools, as the pure functions they are.
 *
 * Every rule the scene used to carry inline — a polygon closing on its first vertex, a
 * double-click leaving no duplicate vertex, a box normalised whichever corner it began from, a
 * Shift-click being a negative prompt — is asserted here without Konva or a DOM.
 */

import { describe, expect, it } from "vitest";

import { TOOLS, type ToolContext } from ".";

const context = (overrides: Partial<ToolContext> = {}): ToolContext => ({
  pendingPoints: [],
  scale: 1,
  assistMode: "point",
  ...overrides,
});

const TRIANGLE = [
  { x: 10, y: 10 },
  { x: 50, y: 10 },
  { x: 50, y: 50 },
];

describe("brush and eraser", () => {
  it("are one gesture", () => {
    expect(TOOLS.brush).toBe(TOOLS.eraser);
  });

  it("collects a trail in place and hands the whole of it over on release", () => {
    const tool = TOOLS.brush;
    const down = tool.down(context(), { x: 1, y: 2 }, { shiftKey: false });
    expect(down.effects).toEqual([]);
    const gesture = down.gesture;
    if (gesture?.kind !== "stroke") throw new Error("expected a stroke");
    const points = gesture.points;

    const moved = tool.move(context(), gesture, { x: 3, y: 4 });
    tool.move(context(), gesture, { x: 5, y: 6 });

    // The same arrays, appended: a stroke is never copied per sample.
    expect(moved.gesture).toBe(gesture);
    expect(gesture.points).toBe(points);
    expect(gesture.flat).toEqual([1, 2, 3, 4, 5, 6]);
    expect(tool.up(context(), gesture)).toEqual([
      { type: "stroke", points: [{ x: 1, y: 2 }, { x: 3, y: 4 }, { x: 5, y: 6 }] },
    ]);
  });

  it("stamps once at the keyboard cursor", () => {
    expect(TOOLS.brush.key(context(), { x: 7, y: 8 }, { key: " ", shiftKey: false })).toEqual([
      { type: "stroke", points: [{ x: 7, y: 8 }] },
    ]);
  });

  it("draws with the primary button rather than panning", () => {
    expect(TOOLS.brush.pansWithPrimary).toBe(false);
  });
});

describe("polygon", () => {
  const tool = TOOLS.polygon;

  it("clears the selection and adds a vertex on an empty-scene click", () => {
    expect(tool.down(context(), { x: 5, y: 5 }, { shiftKey: false })).toEqual({
      gesture: null,
      effects: [{ type: "deselect" }, { type: "vertex", point: { x: 5, y: 5 } }],
    });
  });

  it("closes on a click back on the first vertex, once there are three", () => {
    const step = tool.down(context({ pendingPoints: TRIANGLE }), { x: 12, y: 11 }, { shiftKey: false });
    expect(step.effects).toEqual([{ type: "deselect" }, { type: "closePolygon" }]);
    expect(tool.snaps(context({ pendingPoints: TRIANGLE }), { x: 12, y: 11 })).toBe(true);
    expect(tool.snaps(context({ pendingPoints: TRIANGLE.slice(0, 2) }), { x: 10, y: 10 })).toBe(false);
  });

  it("drops the second click of a double-click, and the double-click closes the ring", () => {
    const step = tool.down(context({ pendingPoints: TRIANGLE }), { x: 50, y: 50 }, { shiftKey: false });
    expect(step.effects).toEqual([{ type: "deselect" }]);
    expect(tool.doubleClick(context({ pendingPoints: TRIANGLE }))).toEqual([{ type: "closePolygon" }]);
    expect(tool.doubleClick(context({ pendingPoints: TRIANGLE.slice(0, 2) }))).toEqual([]);
  });

  it("keeps the snap target a constant size on screen", () => {
    // 9 screen pixels: at 4x that is a little over two source pixels.
    const zoomed = context({ pendingPoints: TRIANGLE, scale: 4 });
    expect(tool.snaps(zoomed, { x: 12, y: 10 })).toBe(true);
    expect(tool.snaps(zoomed, { x: 14, y: 10 })).toBe(false);
  });

  it("closes on Enter and places a vertex on Space", () => {
    expect(tool.key(context({ pendingPoints: TRIANGLE }), { x: 0, y: 0 }, { key: "Enter", shiftKey: false })).toEqual([
      { type: "closePolygon" },
    ]);
    expect(tool.key(context({ pendingPoints: TRIANGLE }), { x: 1, y: 1 }, { key: " ", shiftKey: false })).toEqual([
      { type: "vertex", point: { x: 1, y: 1 } },
    ]);
  });
});

describe("assist", () => {
  const tool = TOOLS.assist;

  it("prompts positive on a click and negative on a Shift-click", () => {
    expect(tool.down(context(), { x: 3, y: 4 }, { shiftKey: false }).effects).toEqual([
      { type: "assistPoint", point: { x: 3, y: 4, kind: "positive" } },
    ]);
    expect(tool.down(context(), { x: 3, y: 4 }, { shiftKey: true }).effects).toEqual([
      { type: "assistPoint", point: { x: 3, y: 4, kind: "negative" } },
    ]);
  });

  it("spans a box that is the same box whichever corner the drag began from", () => {
    const box = context({ assistMode: "box" });
    const down = tool.down(box, { x: 40, y: 30 }, { shiftKey: false });
    expect(down.effects).toEqual([{ type: "assistBox", box: { x0: 40, y0: 30, x1: 40, y1: 30 } }]);
    if (!down.gesture) throw new Error("expected a box gesture");
    expect(tool.move(box, down.gesture, { x: 10, y: 50 }).effects).toEqual([
      { type: "assistBox", box: { x0: 10, y0: 30, x1: 40, y1: 50 } },
    ]);
  });

  it("leaves Enter and Space to the page in box mode, where there is nothing to place", () => {
    expect(tool.key(context({ assistMode: "box" }), { x: 0, y: 0 }, { key: " ", shiftKey: false })).toBeNull();
    expect(tool.key(context(), { x: 2, y: 2 }, { key: " ", shiftKey: true })).toEqual([
      { type: "assistPoint", point: { x: 2, y: 2, kind: "negative" } },
    ]);
  });
});

describe("box", () => {
  const tool = TOOLS.box;

  it("commits the box a drag spans, whichever corner it began from", () => {
    const down = tool.down(context(), { x: 40, y: 30 }, { shiftKey: false });
    expect(down.effects).toEqual([{ type: "deselect" }]);
    if (!down.gesture) throw new Error("expected a box gesture");
    const moved = tool.move(context(), down.gesture, { x: 10.5, y: 50 });
    expect(moved.effects).toEqual([]);
    if (!moved.gesture) throw new Error("expected the gesture to continue");
    expect(tool.up(context(), moved.gesture)).toEqual([
      { type: "box", rect: { x: 10.5, y: 30, width: 29.5, height: 20 } },
    ]);
  });

  it("commits nothing for a drag without area", () => {
    const down = tool.down(context(), { x: 5, y: 5 }, { shiftKey: false });
    if (!down.gesture) throw new Error("expected a box gesture");
    expect(tool.up(context(), down.gesture)).toEqual([]);
    // A drag along one axis is a line, not a box.
    const flat = tool.move(context(), down.gesture, { x: 20, y: 5 });
    if (!flat.gesture) throw new Error("expected the gesture to continue");
    expect(tool.up(context(), flat.gesture)).toEqual([]);
  });

  it("draws rather than pans, and leaves the keys to the page", () => {
    expect(tool.pansWithPrimary).toBe(false);
    expect(tool.key(context(), { x: 0, y: 0 }, { key: " ", shiftKey: false })).toBeNull();
  });
});

describe("select", () => {
  it("pans with the primary button, owns no key, and toggles Fit on a double-click", () => {
    const tool = TOOLS.select;
    expect(tool.pansWithPrimary).toBe(true);
    expect(tool.key(context(), { x: 0, y: 0 }, { key: " ", shiftKey: false })).toBeNull();
    expect(tool.doubleClick(context())).toEqual([{ type: "toggleFit" }]);
  });
});
