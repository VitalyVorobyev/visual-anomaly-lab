/**
 * The annotation editor's behavioural contract, as an end-to-end spec.
 *
 *     bun run test:e2e
 *
 * It pins what the editor *does* with a pointer and a keyboard, and nothing about what draws it:
 * the same spec must pass on the Konva scene and on any renderer that replaces it. So it never
 * touches a `<canvas>`, a Konva node or a pixel of the screenshot. It reaches the scene through
 * what the app itself exposes:
 *
 * - the canvas is the region named "Annotation canvas" (`[data-annotation-canvas]`);
 * - the source pixel under the pointer, the mask value there, the region on top and the zoom are
 *   the on-screen **pixel readout** (`readout()` below is the one place that knows its wording);
 * - the document is read back from the backend after a ⌘S / autosave, i.e. what the editor
 *   actually persisted, in source pixels (`saved()`).
 *
 * Coordinates are never assumed. `calibrate()` finds where the source frame sits on screen by
 * bisecting the readout (the pointer is over the frame or it is not), which yields the screen
 * position of the frame's top-left corner and the screen pixels per source pixel; `screenFor`
 * is built on that. Every test starts from a discarded draft on image 1 of the seeded dataset
 * (`scripts/e2e-seed.py`), a 256x256 synthetic image, and `global-setup.ts` runs the backend.
 *
 * Coordinates in the document are the area convention: continuous source coordinates with
 * pixel (i, j) covering [i, i+1) x [j, j+1). A click lands at the continuous point under the
 * pointer, so assertions allow the screen quantisation of one pointer step (`tolerance`).
 */

import { expect, test, type APIRequestContext, type Locator, type Page } from "@playwright/test";

import { API_PORT } from "./global-setup";

const API = `http://127.0.0.1:${API_PORT}`;
const IMAGE_ID = 1;
const EDITOR_PATH = "/#/datasets/1/annotate/1/1";
const SAVE = "ControlOrMeta+s";
const UNDO = "ControlOrMeta+z";

interface Point {
  x: number;
  y: number;
}

interface Shape {
  id: string;
  kind: "polygon" | "box" | "bitmap";
  operation: "add" | "subtract";
  points?: Point[];
  x?: number;
  y?: number;
  width?: number;
  height?: number;
}

interface Doc {
  image_width: number;
  image_height: number;
  shapes: Shape[];
}

interface Reading {
  x: number;
  y: number;
  mask: 0 | 1;
  region: number | null;
}

/** Where the source frame is on screen: its top-left corner and screen pixels per source pixel. */
interface Mapping {
  left: number;
  top: number;
  scale: number;
}

type Tool = "Select" | "Polygon" | "Box" | "Brush" | "Eraser";

/** The editor open on the seeded image, with the helpers every case shares. */
class Editor {
  mapping: Mapping | null = null;

  constructor(
    readonly page: Page,
    readonly request: APIRequestContext,
    readonly canvas: Locator,
  ) {}

  // --- the readout: the app's own account of where the pointer is -------------------------

  /** Two frames: whatever the pointer did has been rendered. */
  private async settle(): Promise<void> {
    await this.page.evaluate(
      () => new Promise<void>((done) => requestAnimationFrame(() => requestAnimationFrame(() => done()))),
    );
  }

  /** The readout's pixel reading (null when the pointer is off the frame) and zoom in percent. */
  async readout(): Promise<{ reading: Reading | null; zoom: number | null; fit: boolean }> {
    const text = await this.canvas.innerText();
    const pixel = /(\d+), (\d+) · mask ([01])(?: · region (\d+))?/.exec(text);
    const zoom = /(Fit )?(\d+(?:\.\d+)?)%/.exec(text);
    return {
      reading: pixel
        ? {
            x: Number(pixel[1]),
            y: Number(pixel[2]),
            mask: Number(pixel[3]) as 0 | 1,
            region: pixel[4] === undefined ? null : Number(pixel[4]),
          }
        : null,
      zoom: zoom ? Number(zoom[2]) : null,
      fit: zoom?.[1] !== undefined,
    };
  }

  async hover(at: Point): Promise<void> {
    await this.page.mouse.move(at.x, at.y);
    await this.settle();
  }

  async readAt(at: Point) {
    await this.hover(at);
    return this.readout();
  }

  // --- calibration -----------------------------------------------------------------------

  /**
   * Find the frame on screen. The readout has a reading exactly when the pointer is over the
   * frame, so the frame's left, right and top edges are the places where that flips; each is
   * bisected to a fraction of a screen pixel. Needs the whole frame visible (Fit, or a pan).
   */
  async calibrate(): Promise<Mapping> {
    const box = (await this.canvas.boundingBox())!;
    const centre = { x: box.x + box.width / 2, y: box.y + box.height / 2 };
    const doc = await this.serverDocument();
    const over = async (at: Point) => (await this.readAt(at)).reading !== null;

    // `inside` is a coordinate over the frame, `outside` one that is not; returns the edge.
    const edge = async (inside: number, outside: number, at: (value: number) => Point) => {
      for (let step = 0; step < 24 && Math.abs(inside - outside) > 0.01; step += 1) {
        const middle = (inside + outside) / 2;
        if (await over(at(middle))) inside = middle;
        else outside = middle;
      }
      return (inside + outside) / 2;
    };

    expect(await over(centre), "the frame is under the middle of the canvas").toBe(true);
    const left = await edge(centre.x, box.x, (x) => ({ x, y: centre.y }));
    const right = await edge(centre.x, box.x + box.width, (x) => ({ x, y: centre.y }));
    const scale = (right - left) / doc.image_width;
    const top = await edge(centre.y, box.y, (y) => ({ x: centre.x, y }));
    this.mapping = { left, top, scale };

    // The mapping must agree with the readout away from the edges it was fitted on.
    const probe = { x: 200, y: 56 };
    const check = await this.readAt(this.screenFor({ x: probe.x + 0.5, y: probe.y + 0.5 }));
    expect(check.reading, "calibration is consistent with the readout").toMatchObject(probe);
    return this.mapping;
  }

  get scale(): number {
    if (!this.mapping) throw new Error("not calibrated");
    return this.mapping.scale;
  }

  /** How far a coordinate may differ from the one asked for: a pointer cannot say less than a pixel. */
  get tolerance(): number {
    return 1 / this.scale;
  }

  /** The client point of a source point, under the current calibration. */
  screenFor(source: Point): Point {
    if (!this.mapping) throw new Error("not calibrated");
    return {
      x: this.mapping.left + source.x * this.mapping.scale,
      y: this.mapping.top + source.y * this.mapping.scale,
    };
  }

  // --- gestures --------------------------------------------------------------------------

  async tool(name: Tool): Promise<void> {
    const button = this.page.getByRole("button", { name: new RegExp(`^${name}\\b`) });
    await button.click();
    await expect(button).toHaveAttribute("aria-pressed", "true");
  }

  async click(source: Point): Promise<void> {
    const at = this.screenFor(source);
    await this.page.mouse.click(at.x, at.y);
  }

  /** A left or right drag along source points, pressed at the first and released at the last. */
  async drag(path: Point[], button: "left" | "right" = "left"): Promise<void> {
    const [first, ...rest] = path.map((point) => this.screenFor(point));
    await this.dragScreen(first!, rest, button);
  }

  async dragScreen(from: Point, path: Point[], button: "left" | "right" = "left"): Promise<void> {
    await this.page.mouse.move(from.x, from.y);
    await this.page.mouse.down({ button });
    let previous = from;
    for (const next of path) {
      await this.page.mouse.move(next.x, next.y, { steps: 8 });
      previous = next;
    }
    await this.page.mouse.up({ button });
    await this.page.mouse.move(previous.x, previous.y);
    await this.settle();
  }

  /** Focus the canvas, as a reader's click on it does; the canvas owns the arrow keys. */
  async focusCanvas(): Promise<void> {
    await this.canvas.focus();
  }

  // --- what the editor persisted -----------------------------------------------------------

  async serverDocument(): Promise<Doc> {
    const response = await this.request.get(`${API}/api/images/${IMAGE_ID}/annotations/draft`);
    expect(response.ok()).toBe(true);
    return ((await response.json()) as { document: Doc }).document;
  }

  /**
   * Save with the keyboard, then read the persisted document once it satisfies `until`.
   * The idle autosave may get there first, which is the same document; the wait is on the
   * app's state, never on a delay.
   */
  async saved(until: (doc: Doc) => boolean = () => true): Promise<Doc> {
    await this.page.keyboard.press(SAVE);
    let latest: Doc | null = null;
    await expect
      .poll(
        async () => {
          latest = await this.serverDocument();
          return until(latest);
        },
        { timeout: 10_000, message: "the saved draft reaches the expected state" },
      )
      .toBe(true);
    return latest!;
  }

  async shapeCount(): Promise<number> {
    return (await this.serverDocument()).shapes.length;
  }
}

/** Throw the image's draft away; 404 means there was none. The seeded state has no draft. */
async function discardDraft(request: APIRequestContext): Promise<void> {
  const discarded = await request.delete(`${API}/api/images/${IMAGE_ID}/annotations/draft`, {
    headers: { "If-Match": "*" },
  });
  expect([204, 404]).toContain(discarded.status());
}

async function openEditor(page: Page, request: APIRequestContext): Promise<Editor> {
  await discardDraft(request);

  await page.goto(EDITOR_PATH);
  const canvas = page.getByRole("region", { name: /^Annotation canvas/ });
  await expect(canvas).toBeVisible();
  await expect(canvas.getByText(/^Loading source image/)).toHaveCount(0);
  const editor = new Editor(page, request, canvas);
  await editor.calibrate();
  return editor;
}

/** Every coordinate of `shape` is within `tolerance` of `expected`, in order. */
function expectPoints(actual: Point[] | undefined, expected: Point[], tolerance: number): void {
  expect(actual, "the polygon's points").toBeDefined();
  expect(actual).toHaveLength(expected.length);
  expected.forEach((point, index) => {
    expect(Math.abs(actual![index]!.x - point.x)).toBeLessThanOrEqual(tolerance);
    expect(Math.abs(actual![index]!.y - point.y)).toBeLessThanOrEqual(tolerance);
  });
}

function expectBox(
  shape: Shape | undefined,
  expected: { x: number; y: number; width: number; height: number },
  tolerance: number,
): void {
  expect(shape?.kind).toBe("box");
  for (const key of ["x", "y", "width", "height"] as const) {
    expect(Math.abs(shape![key]! - expected[key]), `box ${key}`).toBeLessThanOrEqual(tolerance);
  }
}

const A = { x: 60, y: 60 };
const B = { x: 180, y: 70 };
const C = { x: 120, y: 170 };

test.describe("annotation editor contract", () => {
  let editor: Editor;

  test.beforeEach(async ({ page, request }) => {
    editor = await openEditor(page, request);
  });

  // The backend is shared with the screenshot suite, whose baseline is the seeded state.
  test.afterEach(async ({ request }) => {
    await discardDraft(request);
  });

  test.describe("polygon tool", () => {
    test("three clicks and a click on the first vertex close one three-point polygon", async () => {
      await editor.tool("Polygon");
      await editor.click(A);
      await editor.click(B);
      await editor.click(C);
      await editor.click(A);

      const doc = await editor.saved((d) => d.shapes.length > 0);
      expect(doc.shapes).toHaveLength(1);
      expect(doc.shapes[0]!.kind).toBe("polygon");
      expectPoints(doc.shapes[0]!.points, [A, B, C], editor.tolerance);
    });

    test("a double-click closes the ring without leaving a duplicate vertex", async () => {
      await editor.tool("Polygon");
      await editor.click(A);
      await editor.click(B);
      const last = editor.screenFor(C);
      await editor.page.mouse.dblclick(last.x, last.y);

      const doc = await editor.saved((d) => d.shapes.length > 0);
      expect(doc.shapes).toHaveLength(1);
      expectPoints(doc.shapes[0]!.points, [A, B, C], editor.tolerance);
    });

    test("Backspace removes the last open vertex and Enter closes the ring", async () => {
      const D = { x: 40, y: 150 };
      await editor.tool("Polygon");
      await editor.click(A);
      await editor.click(B);
      await editor.click(C);
      await editor.page.keyboard.press("Backspace");
      await editor.click(D);
      await editor.page.keyboard.press("Enter");

      const doc = await editor.saved((d) => d.shapes.length > 0);
      expect(doc.shapes).toHaveLength(1);
      expectPoints(doc.shapes[0]!.points, [A, B, D], editor.tolerance);
    });

    test("a right-drag pans the view and adds no vertex", async () => {
      // Zoomed in first: whether a fitted image may be panned at all is the viewer's policy,
      // not part of this contract. The pan is read off the readout at a fixed screen point.
      await editor.calibrate();
      await editor.tool("Polygon");
      const at = editor.screenFor({ x: 128, y: 128 });
      await editor.hover(at);
      for (let notch = 0; notch < 4; notch += 1) await editor.page.mouse.wheel(0, -120);
      const before = await editor.readAt(at);
      const scale = before.zoom! / 100;
      await editor.dragScreen(at, [{ x: at.x + 60, y: at.y }], "right");

      const after = await editor.readAt(at);
      expect(after.zoom).toBe(before.zoom);
      expect(Math.abs(before.reading!.x - after.reading!.x - 60 / scale)).toBeLessThanOrEqual(1);
      expect(Math.abs(before.reading!.y - after.reading!.y)).toBeLessThanOrEqual(1);

      // No vertex was placed by the drag: three clicks make a three-point ring, not four.
      await editor.page.keyboard.press("0");
      await expect.poll(async () => (await editor.readout()).fit).toBe(true);
      await editor.calibrate();
      await editor.click(A);
      await editor.click(B);
      await editor.click(C);
      await editor.page.keyboard.press("Enter");
      const doc = await editor.saved((d) => d.shapes.length > 0);
      expect(doc.shapes).toHaveLength(1);
      expectPoints(doc.shapes[0]!.points, [A, B, C], editor.tolerance);
    });
  });

  test.describe("box tool", () => {
    test("a drag from bottom-right to top-left is the same normalised box", async () => {
      await editor.tool("Box");
      await editor.drag([
        { x: 200, y: 180 },
        { x: 120, y: 120 },
        { x: 80, y: 60 },
      ]);

      const doc = await editor.saved((d) => d.shapes.length > 0);
      expect(doc.shapes).toHaveLength(1);
      expectBox(doc.shapes[0], { x: 80, y: 60, width: 120, height: 120 }, editor.tolerance);
    });
  });

  test.describe("brush and eraser", () => {
    test("the brush sets mask pixels and the eraser clears them", async () => {
      const along = (x: number) => ({ x: x + 0.5, y: 100.5 });
      const mask = async (x: number) => (await editor.readAt(editor.screenFor(along(x)))).reading?.mask;

      expect((await editor.readAt(editor.screenFor(along(130)))).reading?.mask).toBe(0);

      await editor.tool("Brush");
      await editor.drag([along(60), along(130), along(200)]);
      await expect.poll(() => mask(130)).toBe(1);
      expect(await mask(70)).toBe(1);
      expect(await mask(190)).toBe(1);
      expect((await editor.readAt(editor.screenFor({ x: 130.5, y: 200.5 }))).reading?.mask).toBe(0);
      expect((await editor.readAt(editor.screenFor(along(130)))).reading?.region).toBe(1);

      // The cut is short enough to spare both ends of the stroke, whatever the brush width.
      await editor.tool("Eraser");
      await editor.drag([along(110), along(150)]);
      await expect.poll(() => mask(130)).toBe(0);
      expect(await mask(70)).toBe(1);
      expect(await mask(190)).toBe(1);
    });
  });

  test.describe("select tool", () => {
    const BOX = { x: 80, y: 60, width: 120, height: 120 };
    const inside = { x: 140, y: 120 };

    async function drawBox() {
      await editor.tool("Box");
      await editor.drag([
        { x: BOX.x, y: BOX.y },
        { x: BOX.x + BOX.width, y: BOX.y + BOX.height },
      ]);
      const drawn = await editor.saved((d) => d.shapes.length > 0);
      await editor.tool("Select");
      return drawn.shapes[0]!;
    }

    const deleteButton = () => editor.page.getByRole("button", { name: "Delete selected region" });

    test("a click selects a region and a click on the empty scene clears it", async () => {
      await drawBox();
      // Drawing selected it; Escape clears the selection.
      await expect(deleteButton()).toBeVisible();
      await editor.page.keyboard.press("Escape");
      await expect(deleteButton()).toHaveCount(0);

      await editor.click(inside);
      await expect(deleteButton()).toBeVisible();
      expect((await editor.readAt(editor.screenFor(inside))).reading?.region).toBe(1);

      await editor.click({ x: 20, y: 230 });
      await expect(deleteButton()).toHaveCount(0);
    });

    test("a drag moves the region by the screen offset over the scale", async () => {
      await drawBox();
      const dx = 60;
      const dy = 36;
      const from = editor.screenFor(inside);
      await editor.dragScreen(from, [{ x: from.x + dx, y: from.y + dy }]);

      const moved = { x: BOX.x + dx / editor.scale, y: BOX.y + dy / editor.scale };
      const doc = await editor.saved((d) => d.shapes[0]?.x !== BOX.x);
      expect(doc.shapes).toHaveLength(1);
      expectBox(doc.shapes[0], { ...moved, width: BOX.width, height: BOX.height }, editor.tolerance);
    });

    test("arrow keys nudge the selection by one pixel and Shift by ten", async () => {
      const start = await drawBox();
      await editor.click(inside);
      await editor.focusCanvas();

      await editor.page.keyboard.press("ArrowRight");
      await editor.page.keyboard.press("Shift+ArrowDown");
      await editor.page.keyboard.press("ArrowLeft");
      await editor.page.keyboard.press("ArrowLeft");
      await editor.page.keyboard.press("Shift+ArrowRight");
      await editor.page.keyboard.press("ArrowUp");

      // x: +1 -1 -1 +10 = +9; y: +10 -1 = +9. The steps are whole pixels, so they are exact
      // offsets from wherever the drawn box began.
      const doc = await editor.saved((d) => Math.abs((d.shapes[0]?.x ?? 0) - (start.x! + 9)) < 1e-6);
      expect(doc.shapes).toHaveLength(1);
      expectBox(
        doc.shapes[0],
        { x: start.x! + 9, y: start.y! + 9, width: start.width!, height: start.height! },
        1e-6,
      );
    });

    test("one vertex drag is one undo step", async () => {
      await editor.tool("Polygon");
      await editor.click(A);
      await editor.click(B);
      await editor.click(C);
      await editor.click(A);
      await editor.saved((d) => d.shapes.length > 0);

      // The ring is selected; Select makes its vertices handles.
      await editor.tool("Select");
      const dx = 40;
      const dy = 25;
      const vertex = editor.screenFor(A);
      await editor.dragScreen(vertex, [{ x: vertex.x + dx, y: vertex.y + dy }]);

      const dragged = { x: A.x + dx / editor.scale, y: A.y + dy / editor.scale };
      const afterDrag = await editor.saved((d) => Math.abs((d.shapes[0]?.points?.[0]?.x ?? 0) - A.x) > 1);
      expectPoints(afterDrag.shapes[0]!.points, [dragged, B, C], editor.tolerance);

      // One undo returns the vertex to where it was, not part-way: the drag is one commit.
      await editor.page.keyboard.press(UNDO);
      const afterUndo = await editor.saved(
        (d) => Math.abs((d.shapes[0]?.points?.[0]?.x ?? 0) - A.x) <= editor.tolerance,
      );
      expect(afterUndo.shapes).toHaveLength(1);
      expectPoints(afterUndo.shapes[0]!.points, [A, B, C], editor.tolerance);
    });
  });

  test.describe("view", () => {
    test("the wheel zooms about the pointer, 0 fits and 1 is actual pixels", async () => {
      const fitted = await editor.readout();
      expect(fitted.fit).toBe(true);
      const fitZoom = (await editor.readAt(editor.screenFor({ x: 128, y: 128 }))).zoom!;

      const anchor = editor.screenFor({ x: 90.5, y: 90.5 });
      const before = (await editor.readAt(anchor)).reading;
      await editor.page.mouse.wheel(0, -120);
      await editor.page.mouse.wheel(0, -120);
      await expect.poll(async () => (await editor.readout()).zoom!).toBeGreaterThan(fitZoom * 1.15);
      const zoomed = await editor.readout();
      expect(zoomed.fit).toBe(false);
      // The pixel under the pointer stays under it.
      expect((await editor.readAt(anchor)).reading).toMatchObject({ x: before!.x, y: before!.y });
      // Wheel steps are multiplicative: out again by as many returns to the same zoom.
      await editor.page.mouse.wheel(0, 120);
      await editor.page.mouse.wheel(0, 120);
      await expect.poll(async () => (await editor.readout()).zoom!).toBeLessThan(fitZoom * 1.05);

      await editor.page.mouse.wheel(0, -120);
      await expect.poll(async () => (await editor.readout()).fit).toBe(false);
      await editor.page.keyboard.press("0");
      await expect.poll(async () => (await editor.readout()).fit).toBe(true);
      expect((await editor.readout()).zoom).toBe(fitted.zoom);

      await editor.page.keyboard.press("1");
      await expect.poll(async () => (await editor.readout()).zoom).toBe(100);
      expect((await editor.readout()).fit).toBe(false);
    });

    test("a double-click under Select toggles Fit and the previous view", async () => {
      await editor.hover(editor.screenFor({ x: 128, y: 128 }));
      await editor.page.mouse.wheel(0, -120);
      await editor.page.mouse.wheel(0, -120);
      await expect.poll(async () => (await editor.readout()).fit).toBe(false);
      const zoomed = (await editor.readout()).zoom;

      const at = editor.screenFor({ x: 20, y: 230 });
      await editor.page.mouse.dblclick(at.x, at.y);
      await expect.poll(async () => (await editor.readout()).fit).toBe(true);
      await editor.page.mouse.dblclick(at.x, at.y);
      await expect.poll(async () => (await editor.readout()).fit).toBe(false);
      expect((await editor.readout()).zoom).toBe(zoomed);
    });
  });

  test.describe("reference pane", () => {
    // The seeded sample has one channel, and the side-by-side pane needs a second one of the
    // same part. A multi-channel dataset in the seed would add a dataset to every list in the
    // screenshot baseline, so this stays skipped until a spec-local fixture can carry it.
    test("clicks in the reference pane do not edit the document", async ({ page }) => {
      const sideBySide = page.getByRole("radio", { name: "Side by side" });
      test.skip((await sideBySide.count()) === 0, "the seeded sample has a single channel");

      await sideBySide.click();
      const reference = page.getByRole("region", { name: /^Reference channel/ });
      const box = (await reference.boundingBox())!;
      await editor.tool("Select");
      await page.mouse.click(box.x + box.width / 2, box.y + box.height / 2);
      await page.mouse.dblclick(box.x + box.width / 2, box.y + box.height / 2);
      await editor.tool("Polygon");
      await page.mouse.click(box.x + 100, box.y + 100);
      await page.mouse.click(box.x + 200, box.y + 100);
      await page.mouse.click(box.x + 150, box.y + 200);
      await page.keyboard.press("Enter");
      await page.keyboard.press(SAVE);
      expect((await editor.serverDocument()).shapes).toHaveLength(0);
    });
  });
});
