/**
 * The stage editor's coordinate boundary, proved through the DOM.
 *
 * The view is held at four screen pixels per source pixel with no offset, so a half-pixel error
 * is two CSS pixels and cannot hide in a tolerance. happy-dom lays nothing out, so the stage's
 * viewport sits at client (0, 0): client `(42, 22)` is stage point `(42 / 4 - 0.5, 22 / 4 - 0.5)
 * = (10, 5)` — the centre of pixel (10, 5) — which the document records as `(10.5, 5.5)`.
 *
 * Presses go to `[data-stage-surface]`, the stage's one press target. A drag's moves are read
 * from `window`, as the surface reads them; a release is dispatched on the viewport, where the
 * pointer is, and bubbles to `window` from there.
 */

import { fireEvent, render, screen } from "@testing-library/react";
import type { ComponentProps } from "react";
import { describe, expect, it, vi } from "vitest";

import type {
  AnnotationDocument,
  AnnotationLabel,
  BoxShape,
  PolygonShape,
} from "../../api/client";
import { AnnotationStage } from "./AnnotationStage";

type Props = ComponentProps<typeof AnnotationStage>;

const VIEW = { scale: 4, tx: 0, ty: 0 };

const labels: AnnotationLabel[] = [
  {
    id: 1,
    dataset_id: 1,
    key: "defect",
    name: "Defect",
    color: "#cc33cc",
    position: 0,
    created_at: "2030-01-01T00:00:00Z",
  },
];

const box: BoxShape = {
  id: "box",
  label_key: "defect",
  kind: "box",
  operation: "add",
  x: 10,
  y: 5,
  width: 10,
  height: 10,
};

const ring: PolygonShape = {
  id: "ring",
  label_key: "defect",
  kind: "polygon",
  operation: "add",
  points: [
    { x: 10, y: 10 },
    { x: 30, y: 10 },
    { x: 20, y: 30 },
  ],
};

function documentWith(...shapes: AnnotationDocument["shapes"]): AnnotationDocument {
  return { schema_version: 1, image_width: 64, image_height: 48, base: "empty", shapes };
}

function renderStage(overrides: Partial<Props> = {}) {
  const handlers = {
    onView: vi.fn(),
    onSelect: vi.fn(),
    onPoint: vi.fn(),
    onReshapePolygon: vi.fn(),
    onReshapeBox: vi.fn(),
    onMoveShape: vi.fn(),
    onBrush: vi.fn(),
    onFinishPolygon: vi.fn(),
    onBox: vi.fn(),
    onAssistPoint: vi.fn(),
    onAssistBox: vi.fn(),
  };
  const props: Props = {
    imageId: 1,
    document: documentWith(),
    labels,
    selectedId: null,
    tool: "polygon",
    pendingPoints: [],
    brushSize: 1,
    assistMode: "point",
    assistPoints: [],
    assistBox: null,
    assistShape: null,
    view: VIEW,
    ...handlers,
    ...overrides,
  };
  const result = render(<AnnotationStage {...props} />);
  return { ...result, ...handlers };
}

function surface(): Element {
  const element = document.querySelector("[data-stage-surface]");
  if (!element) throw new Error("no tool surface");
  return element;
}

const mouse = { pointerId: 1, pointerType: "mouse" };

function press(target: Element, x: number, y: number, button = 0) {
  fireEvent.pointerDown(target, { ...mouse, clientX: x, clientY: y, button });
}

function moveTo(x: number, y: number) {
  fireEvent.pointerMove(window, { ...mouse, clientX: x, clientY: y });
}

function release(x: number, y: number, button = 0) {
  fireEvent.pointerUp(screen.getByRole("application"), { ...mouse, clientX: x, clientY: y, button });
}

function readout(): string {
  return document.querySelector("[data-readout-slot]")?.textContent ?? "";
}

/** A handle that captures the pointer: happy-dom has no pointer to capture. */
function capturing<T extends Element>(element: T): T {
  element.setPointerCapture = vi.fn();
  return element;
}

describe("AnnotationStage: the coordinate boundary", () => {
  it("records a polygon click at the area coordinate under the pointer", () => {
    const { onPoint } = renderStage({ tool: "polygon" });
    press(surface(), 42, 22);
    release(42, 22);
    expect(onPoint).toHaveBeenCalledExactlyOnceWith({ x: 10.5, y: 5.5 });
  });

  it("draws a box between the area coordinates of its corners", () => {
    const { onBox } = renderStage({ tool: "box" });
    press(surface(), 82, 62);
    moveTo(60, 40);
    moveTo(42, 22);
    release(42, 22);
    expect(onBox).toHaveBeenCalledExactlyOnceWith({ x: 10.5, y: 5.5, width: 10, height: 10 });
  });

  it("hands the brush its trail in area coordinates", () => {
    const { onBrush } = renderStage({ tool: "brush" });
    press(surface(), 42, 22);
    moveTo(46, 22);
    moveTo(46, 30);
    release(46, 30);
    expect(onBrush).toHaveBeenCalledExactlyOnceWith([
      { x: 10.5, y: 5.5 },
      { x: 11.5, y: 5.5 },
      { x: 11.5, y: 7.5 },
    ]);
  });

  it("selects the region under a press and moves it by the drag over the scale", () => {
    const { onSelect, onMoveShape } = renderStage({ tool: "select", document: documentWith(box) });
    // Client (60, 40) is stage (14.5, 9.5), area (15, 10): inside the box [10, 20) x [5, 15).
    press(surface(), 60, 40);
    expect(onSelect).toHaveBeenCalledExactlyOnceWith("box");
    moveTo(70, 44);
    moveTo(80, 48);
    release(80, 48);
    expect(onMoveShape).toHaveBeenCalledExactlyOnceWith("box", 5, 2);
  });

  it("selects without moving when the press stays inside the click slop", () => {
    const { onSelect, onMoveShape } = renderStage({ tool: "select", document: documentWith(box) });
    press(surface(), 60, 40);
    moveTo(62, 41);
    release(62, 41);
    expect(onSelect).toHaveBeenCalledWith("box");
    expect(onMoveShape).not.toHaveBeenCalled();
  });

  it("leaves a press on bare image to the stage under Select", () => {
    const { onSelect, onMoveShape } = renderStage({ tool: "select", document: documentWith(box) });
    press(surface(), 200, 150);
    release(200, 150);
    expect(onMoveShape).not.toHaveBeenCalled();
    // Declined by the surface, the press is the stage's: a click without movement is a
    // background click, which deselects.
    expect(onSelect).toHaveBeenCalledExactlyOnceWith(null);
  });

  it("drags a polygon vertex to the area coordinate under the pointer, committed once", () => {
    const { onReshapePolygon } = renderStage({
      tool: "select",
      document: documentWith(ring),
      selectedId: "ring",
    });
    const vertex = capturing(screen.getByRole("button", { name: "Region point 1" }));
    fireEvent.pointerDown(vertex, { ...mouse, button: 0, clientX: 40, clientY: 40 });
    fireEvent.pointerMove(vertex, { ...mouse, clientX: 60, clientY: 70 });
    fireEvent.pointerMove(vertex, { ...mouse, clientX: 82, clientY: 82 });
    expect(onReshapePolygon).not.toHaveBeenCalled();
    fireEvent.pointerUp(vertex, { ...mouse, clientX: 82, clientY: 82 });
    expect(onReshapePolygon).toHaveBeenCalledExactlyOnceWith("ring", [
      { x: 20.5, y: 20.5 },
      { x: 30, y: 10 },
      { x: 20, y: 30 },
    ]);
  });

  it("puts a polygon's vertices where the document has them", () => {
    renderStage({ tool: "select", document: documentWith(ring), selectedId: "ring" });
    // Vertex (10, 10) in the area convention is the stage's (9.5, 9.5).
    const vertex = screen.getByRole("button", { name: "Region point 1" });
    expect(vertex.getAttribute("cx")).toBe("9.5");
    expect(vertex.getAttribute("cy")).toBe("9.5");
  });

  it("resizes a box by its corner to the area coordinate under the pointer", () => {
    const { onReshapeBox } = renderStage({
      tool: "select",
      document: documentWith(box),
      selectedId: "box",
    });
    const corner = capturing(document.querySelector('[data-handle="se"]')!);
    fireEvent.pointerDown(corner, { ...mouse, button: 0, clientX: 80, clientY: 60 });
    fireEvent.pointerMove(corner, { ...mouse, clientX: 100, clientY: 80 });
    fireEvent.pointerUp(corner, { ...mouse, clientX: 100, clientY: 80 });
    // The corner moved from (20, 15) to client (100, 80) = area (25, 20).
    expect(onReshapeBox).toHaveBeenCalledExactlyOnceWith("box", { x: 10, y: 5, width: 15, height: 15 });
  });

  it("reads the pixel under the pointer, and the region on top of it", () => {
    renderStage({ tool: "select", document: documentWith(box) });
    const viewport = screen.getByRole("application");
    // Client 41 is area x 10.25: inside pixel 10, a quarter pixel from its left edge, where a
    // missing half pixel would read pixel 9.
    fireEvent.pointerMove(viewport, { ...mouse, clientX: 41, clientY: 22 });
    expect(readout()).toBe("10, 5 · mask 1 · region 1 · 400%");
    // Client 39.5 is area x 9.875: pixel 9, just left of the box, where a doubled half pixel
    // would read pixel 10.
    fireEvent.pointerMove(viewport, { ...mouse, clientX: 39.5, clientY: 22 });
    expect(readout()).toBe("9, 5 · mask 0 · 400%");
  });
});

describe("AnnotationStage: the keyboard's target", () => {
  it("takes focus on a press, so the arrows that follow nudge rather than navigate", () => {
    const { onMoveShape } = renderStage({ tool: "select", document: documentWith(box), selectedId: "box" });
    const region = screen.getByRole("region");
    press(surface(), 200, 150);
    release(200, 150);
    expect(document.activeElement).toBe(region);
    fireEvent.keyDown(region, { key: "ArrowRight", shiftKey: true });
    expect(onMoveShape).toHaveBeenCalledExactlyOnceWith("box", 10, 0);
  });
});

describe("AnnotationStage: what does not draw", () => {
  it("gives a right press to the view, not the tool", () => {
    const { onPoint, onSelect } = renderStage({ tool: "polygon" });
    press(surface(), 42, 22, 2);
    release(42, 22, 2);
    expect(onPoint).not.toHaveBeenCalled();
    expect(onSelect).not.toHaveBeenCalled();
  });

  it("has no tool surface and no handles in a pane that does not edit", () => {
    const { onPoint, onSelect } = renderStage({
      tool: "select",
      editable: false,
      document: documentWith(ring, box),
      selectedId: "ring",
    });
    expect(document.querySelector("[data-stage-surface]")).toBeNull();
    expect(screen.queryByRole("button", { name: /^Region/ })).toBeNull();
    fireEvent.pointerDown(screen.getByRole("application"), { ...mouse, button: 0, clientX: 60, clientY: 40 });
    release(60, 40);
    expect(onPoint).not.toHaveBeenCalled();
    expect(onSelect).not.toHaveBeenCalled();
  });

  it("keeps the gesture's drafts while the regions are hidden", () => {
    const pendingPoints = [
      { x: 10.5, y: 5.5 },
      { x: 20.5, y: 5.5 },
    ];
    const shown = renderStage({ document: documentWith(box), pendingPoints });
    expect(screen.getByRole("img", { name: /^Regions/ })).toBeTruthy();
    expect(document.querySelector('[data-draft="polygon"]')).not.toBeNull();
    shown.unmount();

    renderStage({ document: documentWith(box), pendingPoints, showRegions: false });
    expect(screen.queryByRole("img", { name: /^Regions/ })).toBeNull();
    expect(document.querySelector('[data-draft="polygon"]')).not.toBeNull();
  });

  it("closes an open ring on a double-click under Polygon", () => {
    const { onFinishPolygon } = renderStage({
      tool: "polygon",
      pendingPoints: ring.points,
    });
    fireEvent.doubleClick(surface(), { clientX: 42, clientY: 22 });
    expect(onFinishPolygon).toHaveBeenCalledOnce();
  });
});
