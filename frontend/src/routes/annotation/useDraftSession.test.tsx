/**
 * The draft session and the document commands on it, driven the way the editor drives them.
 *
 * Requests are answered by the API client's own middleware (see `useAnnotations.test.ts` for
 * why not `globalThis.fetch`), so the real mutations run and nothing reaches a network.
 * Painting is mocked: `paintStroke` decodes a PNG through a canvas this DOM does not have,
 * and the thing under test is *when* its result is committed, not what it paints.
 */

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, renderHook, waitFor } from "@testing-library/react";
import { createElement, type ReactNode } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { api } from "../../api/client";
import type { AnnotationDocument, AnnotationLabel, BitmapShape } from "../../api/client";
import type { EditorTool } from "../../components/annotation/AnnotationCanvas";
import type { DraftEnvelope } from "../../hooks/useAnnotations";
import { useDocumentCommands } from "./useDocumentCommands";
import { isConflict, useDraftSession } from "./useDraftSession";

const pending: { target: BitmapShape; resolve: (value: BitmapShape) => void }[] = [];

vi.mock("../../api/annotationBitmap", async (importOriginal) => ({
  ...(await importOriginal<typeof import("../../api/annotationBitmap")>()),
  // Each paint waits for the test to release it, and appends one mark to the region's PNG so
  // the committed document says which strokes landed.
  // Rasterising needs a canvas this DOM does not have; a new region is a fixed stub.
  bitmapStroke: vi.fn(() => ({ ...REGION, id: "minted", png_base64: "minted" })),
  paintStroke: vi.fn(
    (target: BitmapShape) =>
      new Promise<BitmapShape>((resolve) => {
        pending.push({ target, resolve });
      }),
  ),
}));

const REGION: BitmapShape = {
  id: "region",
  label_key: "defect",
  kind: "bitmap",
  operation: "add",
  x: 0,
  y: 0,
  width: 4,
  height: 4,
  png_base64: "base",
};

const DOCUMENT = {
  schema_version: 1,
  image_width: 16,
  image_height: 16,
  base: "empty",
  shapes: [REGION],
} as AnnotationDocument;

const LABELS: AnnotationLabel[] = [
  { key: "defect", name: "Defect", color: "#c026d3", position: 0 } as AnnotationLabel,
];

const flash = vi.fn();
const noTool = () => undefined;
const IMAGE_IDS = [7];

interface Reply {
  status: number;
  body: unknown;
  etag?: string;
}

let replies: Reply[] = [];
let calls: { method: string; url: string }[] = [];
const middleware = {
  onRequest({ request }: { request: Request }) {
    calls.push({ method: request.method, url: request.url });
    const reply = replies.shift() ?? { status: 500, body: { detail: "unexpected request" } };
    return new Response(JSON.stringify(reply.body), {
      status: reply.status,
      headers: {
        "content-type": "application/json",
        ...(reply.etag ? { etag: reply.etag } : {}),
      },
    });
  },
};

beforeEach(() => {
  replies = [];
  calls = [];
  pending.length = 0;
  flash.mockClear();
  api.use(middleware);
});

afterEach(() => {
  api.eject(middleware);
});

function saved(document: AnnotationDocument, version: number): Reply {
  return {
    status: version === 1 ? 201 : 200,
    body: { image_id: 7, document, version, updated_at: "now" },
    etag: `"annotation-draft-7-v${version}"`,
  };
}

function wrapper({ children }: { children: ReactNode }) {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false, gcTime: 0 }, mutations: { retry: false } },
  });
  return createElement(QueryClientProvider, { client }, children);
}

function renderEditor(initial: DraftEnvelope = { document: DOCUMENT, version: null, etag: null }) {
  return renderHook(
    ({ tool }: { tool: EditorTool }) => {
      const session = useDraftSession({
        target: { scope: "image", imageId: 7 },
        initial,
        imageIds: IMAGE_IDS,
        flash,
      });
      const commands = useDocumentCommands({
        history: session.history,
        dispatch: session.dispatch,
        latest: session.latest,
        labels: LABELS,
        tool,
        setTool: noTool,
        brushSize: 3,
        flash,
      });
      return { session, commands };
    },
    { wrapper, initialProps: { tool: "brush" as EditorTool } },
  );
}

/**
 * Release the oldest paint in flight. The result is the region it was *handed*, with `mark`
 * appended — so the committed PNG spells out which document each stroke was painted from.
 */
async function finishPaint(mark: string) {
  await waitFor(() => expect(pending.length).toBeGreaterThan(0));
  const paint = pending.shift();
  await act(async () => {
    paint?.resolve({ ...paint.target, png_base64: `${paint.target.png_base64}${mark}` });
  });
}

describe("useDraftSession", () => {
  it("opens clean, and a commit makes it dirty", () => {
    const { result } = renderEditor();
    expect(result.current.session.dirty).toBe(false);
    expect(result.current.session.etag).toBeNull();
    act(() => result.current.commands.moveShape("region", 2, 0));
    expect(result.current.session.dirty).toBe(true);
    expect(result.current.session.history.present.shapes[0]).toMatchObject({ x: 2 });
  });

  it("knows a dispatched edit before it has rendered", () => {
    const { result } = renderEditor();
    act(() => {
      result.current.commands.moveShape("region", 1, 0);
      // Same tick, no render yet: a closure's `history.present` is still the old document,
      // and `latest()` already is not.
      expect(result.current.session.history.present.shapes[0]).toMatchObject({ x: 0 });
      expect(result.current.session.latest().shapes[0]).toMatchObject({ x: 1 });
      result.current.commands.moveShape("region", 1, 0);
    });
    // Two moves, each from the one before: neither was built from a stale document.
    expect(result.current.session.history.present.shapes[0]).toMatchObject({ x: 2 });
  });

  it("creates the draft on the first save, holds its token, and is clean after", async () => {
    const { result } = renderEditor();
    act(() => result.current.commands.moveShape("region", 3, 0));
    const moved = result.current.session.history.present;
    replies = [saved(moved, 1)];

    await act(async () => {
      await expect(result.current.session.persist()).resolves.toBe('"annotation-draft-7-v1"');
    });

    expect(calls).toEqual([{ method: "POST", url: expect.stringContaining("/images/7/annotations/draft") }]);
    expect(result.current.session.etag).toBe('"annotation-draft-7-v1"');
    expect(result.current.session.draftVersion).toBe(1);
    expect(result.current.session.dirty).toBe(false);
    expect(flash).toHaveBeenCalledWith("Draft saved");
  });

  it("shares one flight between concurrent saves", async () => {
    const { result } = renderEditor();
    act(() => result.current.commands.moveShape("region", 3, 0));
    replies = [saved(result.current.session.history.present, 1)];
    await act(async () => {
      const first = result.current.session.persist();
      const second = result.current.session.persist();
      expect(second).toBe(first);
      await first;
    });
    expect(calls).toHaveLength(1);
  });

  it("answers a clean, persisted document with the token it has, and writes nothing", async () => {
    const { result } = renderEditor({ document: DOCUMENT, version: 4, etag: '"v4"' });
    await act(async () => {
      await expect(result.current.session.persist()).resolves.toBe('"v4"');
    });
    expect(calls).toEqual([]);
  });

  it("surfaces a 412 as a conflict and keeps the local edit", async () => {
    const { result } = renderEditor({ document: DOCUMENT, version: 4, etag: '"v4"' });
    act(() => result.current.commands.moveShape("region", 3, 0));
    replies = [{ status: 412, body: { detail: "The draft changed in another window." } }];
    await act(async () => {
      await expect(result.current.session.persist()).rejects.toThrow();
    });
    await waitFor(() => expect(isConflict(result.current.session.error)).toBe(true));
    expect(result.current.session.dirty).toBe(true);
    expect(result.current.session.history.present.shapes[0]).toMatchObject({ x: 3 });
  });

  it("completes by saving first, then freezing a revision", async () => {
    const { result } = renderEditor();
    act(() => result.current.commands.moveShape("region", 3, 0));
    replies = [
      saved(result.current.session.history.present, 1),
      { status: 200, body: { revision_no: 2, image_id: 7, document: DOCUMENT } },
    ];
    let completed = false;
    await act(async () => {
      completed = await result.current.session.completeDraft();
    });
    expect(completed).toBe(true);
    expect(calls.map((call) => call.method)).toEqual(["POST", "POST"]);
    expect(calls[1]?.url).toContain("/images/7/annotations/complete");
    expect(flash).toHaveBeenLastCalledWith("Completed revision 2");
  });

  it("reports a failed completion rather than throwing", async () => {
    const { result } = renderEditor({ document: DOCUMENT, version: 4, etag: '"v4"' });
    replies = [{ status: 409, body: { detail: "nope" } }];
    let completed = true;
    await act(async () => {
      completed = await result.current.session.completeDraft();
    });
    expect(completed).toBe(false);
    await waitFor(() => expect(result.current.session.error?.message).toContain("nope"));
  });
});

describe("useDocumentCommands: undo and redo", () => {
  it("steps one edit at a time, both ways, and a new edit drops the redo", () => {
    const { result } = renderEditor();
    act(() => result.current.commands.moveShape("region", 1, 0));
    act(() => result.current.commands.moveShape("region", 1, 0));
    expect(result.current.commands.canUndo).toBe(true);
    expect(result.current.commands.canRedo).toBe(false);

    act(() => result.current.commands.undo());
    expect(result.current.session.history.present.shapes[0]).toMatchObject({ x: 1 });
    act(() => result.current.commands.undo());
    expect(result.current.session.history.present.shapes[0]).toMatchObject({ x: 0 });
    expect(result.current.commands.canUndo).toBe(false);
    expect(result.current.session.dirty).toBe(false);

    act(() => result.current.commands.redo());
    expect(result.current.session.history.present.shapes[0]).toMatchObject({ x: 1 });
    expect(result.current.commands.canRedo).toBe(true);

    act(() => result.current.commands.moveShape("region", 0, 5));
    expect(result.current.commands.canRedo).toBe(false);
  });

  it("deletes the selected region and clears the selection", () => {
    const { result } = renderEditor();
    act(() => result.current.commands.setSelectedId("region"));
    act(() => result.current.commands.removeSelected());
    expect(result.current.session.history.present.shapes).toEqual([]);
    expect(result.current.commands.selectedId).toBeNull();
    act(() => result.current.commands.undo());
    expect(result.current.session.history.present.shapes).toHaveLength(1);
  });

  it("closes an open polygon into one committed, selected region", () => {
    const { result } = renderEditor();
    act(() => {
      result.current.commands.addPendingPoint({ x: 1, y: 1 });
      result.current.commands.addPendingPoint({ x: 9, y: 1 });
    });
    act(() => result.current.commands.finishPolygon());
    // Two vertices is not a ring.
    expect(result.current.session.history.present.shapes).toHaveLength(1);
    act(() => result.current.commands.addPendingPoint({ x: 9, y: 9 }));
    act(() => result.current.commands.finishPolygon());
    const shapes = result.current.session.history.present.shapes;
    expect(shapes).toHaveLength(2);
    expect(shapes[1]).toMatchObject({ kind: "polygon", operation: "add" });
    expect(result.current.commands.selectedId).toBe(shapes[1]?.id);
    expect(result.current.commands.pendingPoints).toEqual([]);
  });
});

describe("useDocumentCommands: strokes", () => {
  it("applies strokes in flight together one after another, dropping none", async () => {
    const { result } = renderEditor();
    act(() => result.current.commands.setSelectedId("region"));

    // Two strokes issued in the same render, as Space auto-repeat on the canvas does. Both
    // closures see the same `history.present`; before the fix the second to finish won.
    let first: Promise<void> = Promise.resolve();
    let second: Promise<void> = Promise.resolve();
    act(() => {
      first = result.current.commands.applyStroke([{ x: 1, y: 1 }]);
      second = result.current.commands.applyStroke([{ x: 2, y: 2 }]);
    });

    await finishPaint("+a");
    await act(async () => first);
    // Painted from the region the first stroke left, not from the one both closures saw.
    await finishPaint("+b");
    await act(async () => second);

    expect(result.current.session.history.present.shapes).toEqual([
      { ...REGION, png_base64: "base+a+b" },
    ]);
    // One undo step per stroke.
    act(() => result.current.commands.undo());
    expect(result.current.session.history.present.shapes[0]).toMatchObject({ png_base64: "base+a" });
  });

  it("repaints against the new document when an undo lands while a stroke decodes", async () => {
    const { result } = renderEditor();
    act(() => result.current.commands.setSelectedId("region"));
    act(() => result.current.commands.moveShape("region", 1, 0));

    let stroke: Promise<void> = Promise.resolve();
    act(() => {
      stroke = result.current.commands.applyStroke([{ x: 1, y: 1 }]);
    });
    await waitFor(() => expect(pending).toHaveLength(1));

    // The move is undone while the paint is still decoding.
    act(() => result.current.commands.undo());
    await finishPaint("+painted-before-undo");

    // The stale result is not committed; the stroke is painted again against the undone
    // document, and only that one lands.
    await waitFor(() => expect(pending).toHaveLength(1));
    expect(pending[0]?.target).toMatchObject({ x: 0 });
    await finishPaint("+painted-after-undo");
    await act(async () => stroke);

    const present = result.current.session.history.present;
    expect(present.shapes[0]).toMatchObject({ png_base64: "base+painted-after-undo" });
    // The undone move stayed undone.
    expect(present.shapes[0]).toMatchObject({ x: 0 });
  });

  it("extends the region the previous stroke minted, even before it has rendered", async () => {
    const { result } = renderEditor({
      document: { ...DOCUMENT, shapes: [] },
      version: null,
      etag: null,
    });
    let first: Promise<void> = Promise.resolve();
    let second: Promise<void> = Promise.resolve();
    act(() => {
      // Nothing selected: the first stroke starts a region (synchronously rasterised), and the
      // second, queued behind it, must extend *that* region rather than start another.
      first = result.current.commands.applyStroke([{ x: 5, y: 5 }]);
      second = result.current.commands.applyStroke([{ x: 6, y: 6 }]);
    });
    await act(async () => first);
    const minted = result.current.session.latest().shapes[0];
    expect(minted?.id).toBe("minted");
    await finishPaint("+extended");
    await act(async () => second);
    const shapes = result.current.session.history.present.shapes;
    expect(shapes).toEqual([{ ...REGION, id: "minted", png_base64: "minted+extended" }]);
    expect(result.current.commands.selectedId).toBe(minted?.id);
  });
});
