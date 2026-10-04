/** The controlled, source-coordinate annotation scene, on one `@vitavision/stage2d` stage.
 *
 * The stage owns the view, the pan, the wheel and the hit-tests, never annotation state. All
 * geometry comes in through props and every edit leaves through a callback as source-image
 * pixels in the document's area convention, so the same document can be saved, rendered on the
 * backend and evaluated without a view transform leaking into truth. `stageFrame` is the one
 * place the stage's pixel-centre convention is converted.
 *
 * The pieces, each doing one job:
 *
 * - **tools** (`tools/`) turn pointer and key input into effects, as pure functions;
 * - the **scene** (`StageScene`) draws the committed document, and re-renders only when it,
 *   the selection or the view changes;
 * - the **tool surface and editors** (`StageTools`) take presses for the tool in hand and edit
 *   the selected region;
 * - the **drafts** (`LiveDrafts`) draw the gesture in progress and the cursors, fed by a small
 *   store (`liveStore`) so a pointer move re-renders them and nothing else.
 *
 * This component wires them together: the wrapper the editor's keys and its readouts live on,
 * and the translation of tool effects into the callbacks below.
 */

import {
  useCallback,
  useEffect,
  useImperativeHandle,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
  type KeyboardEvent,
  type Ref,
} from "react";
import {
  ImageLayer,
  ImageStage,
  formatScale,
  insideImage,
  toImage,
  toScreen,
  useStage,
  type ClampOptions,
  type StageHandle,
  type StageMouseButton,
  type StageView,
  type StageViewChange,
} from "@vitavision/stage2d";

import type {
  AnnotationDocument,
  AnnotationLabel,
  AnnotationPoint,
  AssistBox,
  AssistPoint,
  BitmapShape,
} from "../../api/client";
import { decodeShapeMask, imageMask } from "../../api/annotationBitmap";
import { defined } from "../../api/defined";
import { imageUrl, sourceMaskUrl } from "../../api/imageUrl";
import { canvasBindingFor } from "./editorKeys";
import { applyEdit, type LiveEdit } from "./liveEdit";
import { createLiveStore, useLive, type LiveStore } from "./liveStore";
import { readPixel } from "./pixelReadout";
import { useScenePalette } from "./scenePalette";
import { fromStage, toStage } from "./stageFrame";
import { StageScene } from "./StageScene";
import { LiveDrafts, ShapeEditors, ToolSurface } from "./StageTools";
import { TOOLS, type BoxRect, type EditorTool, type ToolEffect } from "./tools";
import { useHtmlImage } from "./useHtmlImage";

export interface AnnotationStageHandle {
  fit: () => void;
  actualPixels: () => void;
  /** Zoom about the centre of the pane, within the pane's zoom range. */
  zoomBy: (factor: number) => void;
}

interface Props {
  imageId: number;
  /**
   * A second channel of the same sample, composited over the first at `overlayOpacity`.
   *
   * This is a registration check, not decoration: a multi-shot rig triggers its exposures
   * milliseconds apart, and whether the part moved between them decides whether one
   * annotation can be shared across the channels at all. Blending them is the only way to
   * see a few pixels of drift; side by side, it is invisible.
   */
  overlayImageId?: number | undefined;
  overlayOpacity?: number;
  /** How strongly annotated regions are painted over the photograph. */
  maskOpacity?: number;
  /**
   * Whether the annotated regions are drawn at all.
   *
   * Not the same question as `maskOpacity`, and not reachable through it. The opacity dims
   * *fills*; a region's outline and the selected region's handles carry no opacity of their
   * own, so even at zero the photograph would be under a wireframe. Deciding whether a marked
   * defect is really there means seeing the pixels with nothing on them.
   *
   * What is still drawn while this is off is what the reader is doing *now*: the live brush
   * trail, the pending polygon, the assist points and their un-accepted suggestion, and the
   * keyboard cursor. Those are not the overlay.
   */
  showRegions?: boolean;
  /** What a screen reader calls this surface. Panes beside the editor are not the editor. */
  label?: string;
  /**
   * Whether this pane edits at all.
   *
   * A reference pane is a photograph with the document drawn over it, not a second editor:
   * no tool surface, no handles, and its regions take no press. Without this a region would
   * still take a click and start a drag that snapped back on release — an affordance that lies.
   */
  editable?: boolean;
  document: AnnotationDocument;
  labels: AnnotationLabel[];
  selectedId: string | null;
  tool: EditorTool;
  pendingPoints: AnnotationPoint[];
  brushSize: number;
  assistMode: "point" | "box";
  assistPoints: AssistPoint[];
  assistBox: AssistBox | null;
  assistShape: BitmapShape | null;
  /** The view, shared with any pane beside this one; `null` opens at Fit. */
  view: StageView | null;
  onView: (view: StageView, change: StageViewChange) => void;
  onSelect: (shapeId: string | null) => void;
  onPoint: (point: AnnotationPoint) => void;
  /** A polygon's ring after a vertex was dragged, inserted or removed. */
  onReshapePolygon: (shapeId: string, points: AnnotationPoint[]) => void;
  /** A box after a corner or edge was dragged, or its interior moved. */
  onReshapeBox: (shapeId: string, rect: BoxRect) => void;
  /** A whole-region offset in source pixels: a drag, or an arrow-key nudge. */
  onMoveShape: (shapeId: string, dx: number, dy: number) => void;
  onBrush: (points: AnnotationPoint[]) => void;
  onFinishPolygon: () => void;
  /** A box drawn with the box tool, normalised and with area. */
  onBox: (rect: BoxRect) => void;
  onAssistPoint: (point: AssistPoint) => void;
  onAssistBox: (box: AssistBox | null) => void;
  ref?: Ref<AnnotationStageHandle> | undefined;
}

const ARROW_STEPS: Partial<Record<string, [number, number]>> = {
  ArrowUp: [0, -1],
  ArrowDown: [0, 1],
  ArrowLeft: [-1, 0],
  ArrowRight: [1, 0],
};

/** Fit leaves a margin, in CSS pixels, so the frame's edge is visible against the canvas. */
const FIT = { padding: 24 };
/**
 * The view keeps some image point at or past the pane's centre, nothing more: a wheel step holds
 * the pixel under the pointer while the image is still narrower than the pane, and an edge or a
 * corner can be brought to the middle of the screen to be traced there.
 */
const CLAMP: ClampOptions = { panBounds: "center" };
/** Under Select the primary button pans bare image too (`pansWithPrimary`); otherwise it draws. */
const SELECT_PAN: readonly StageMouseButton[] = ["left", "middle", "right"];
const TOOL_PAN: readonly StageMouseButton[] = ["middle", "right"];

export function AnnotationStage({
  imageId,
  overlayImageId,
  overlayOpacity = 0.5,
  maskOpacity = 0.45,
  showRegions = true,
  label = "Annotation canvas",
  editable = true,
  document,
  labels,
  selectedId,
  tool,
  pendingPoints,
  brushSize,
  assistMode,
  assistPoints,
  assistBox,
  assistShape,
  view,
  onView,
  onSelect,
  onPoint,
  onReshapePolygon,
  onReshapeBox,
  onMoveShape,
  onBrush,
  onFinishPolygon,
  onBox,
  onAssistPoint,
  onAssistBox,
  ref,
}: Props) {
  const palette = useScenePalette();
  const toolModule = TOOLS[tool];
  // Selecting, dragging and reshaping are all the same permission.
  const interactive = editable && tool === "select";
  const stageRef = useRef<StageHandle>(null);
  const regionRef = useRef<HTMLDivElement>(null);
  const [store] = useState<LiveStore>(() =>
    createLiveStore({
      gesture: null,
      revision: 0,
      keyboardPoint: { x: document.image_width / 2, y: document.image_height / 2 },
      keyboardFocused: false,
      pointer: null,
    }),
  );
  const image = useMemo(
    () => ({ width: document.image_width, height: document.image_height }),
    [document.image_width, document.image_height],
  );
  const masks = useDecodedMasks(document);
  const src = imageUrl(imageId, "full");
  const [photo, setPhoto] = useState<{ src: string; failed: boolean } | null>(null);
  // Requested with CORS: the pixels are read back to tint them, and an image from the
  // sidecar's origin drawn without it taints the canvas and makes `getImageData` throw.
  const baseMaskSource = useHtmlImage(
    document.base === "source_mask" ? sourceMaskUrl(imageId) : undefined,
    "anonymous",
  );
  // The import as one byte per pixel, for the readout. Decoded once per base, not per move.
  const baseValues = useMemo(() => {
    if (baseMaskSource === null) return null;
    try {
      return imageMask(baseMaskSource, document.image_width, document.image_height);
    } catch {
      // A base layer that cannot be read is context lost, not an editor lost.
      return null;
    }
  }, [baseMaskSource, document.image_width, document.image_height]);

  // The edit in progress, and the document as it looks meanwhile. The ref is what an editor's
  // commit reads: it can arrive in the same event as the change it commits, before a render.
  const [edit, setEdit] = useState<LiveEdit | null>(null);
  const latestEditRef = useRef<LiveEdit | null>(null);
  const reportEdit = useCallback((next: LiveEdit | null) => {
    latestEditRef.current = next;
    setEdit(next);
  }, []);
  const readEdit = useCallback(() => latestEditRef.current, []);
  const shown = useMemo(() => applyEdit(document, edit), [document, edit]);
  const order = useMemo(
    () => new Map(document.shapes.map((shape, index) => [shape.id, index])),
    [document.shapes],
  );
  const selectedShape = shown.shapes.find((shape) => shape.id === selectedId);

  // The view on screen, for the handle and for re-reading the pointer when the view moves.
  const viewRef = useRef(view);
  useLayoutEffect(() => {
    viewRef.current = view;
  });

  useImperativeHandle(
    ref,
    () => ({
      fit: () => stageRef.current?.fit(),
      actualPixels: () => stageRef.current?.zoomTo(1),
      zoomBy: (factor: number) => {
        const current = viewRef.current;
        if (current) stageRef.current?.zoomTo(current.scale * factor);
      },
    }),
    [],
  );

  const handleView = (next: StageView, change: StageViewChange) => {
    // The view moved under a still pointer — a wheel step, a key — so what it is over has
    // changed, and the readout must say so before the pointer next moves.
    const previous = viewRef.current;
    const pointer = store.get().pointer;
    if (previous && pointer) {
      const under = toImage(next, toScreen(previous, toStage(pointer)));
      store.set({ pointer: insideImage(under, image) ? fromStage(under) : null });
    }
    onView(next, change);
  };

  const emit = (effects: ToolEffect[]) => {
    for (const effect of effects) {
      if (effect.type === "deselect") onSelect(null);
      else if (effect.type === "stroke") onBrush(effect.points);
      else if (effect.type === "vertex") onPoint(effect.point);
      else if (effect.type === "closePolygon") onFinishPolygon();
      else if (effect.type === "box") onBox(effect.rect);
      else if (effect.type === "assistPoint") onAssistPoint(effect.point);
      else if (effect.type === "assistBox") onAssistBox(effect.box);
      // The stage's own double-click toggles Fit under Select; a tool asking for it fits.
      else stageRef.current?.fit();
    }
  };

  const onCanvasKeyDown = (event: KeyboardEvent<HTMLDivElement>) => {
    const binding = canvasBindingFor(event);
    if (!binding) return;
    if (binding.command === "cursor.move") {
      const direction = ARROW_STEPS[event.key];
      if (!direction) return;
      event.preventDefault();
      event.stopPropagation();
      const step = event.shiftKey ? 10 : 1;
      // With a region selected the arrows are the precise tool, not the cursor: a copy taken
      // from another channel lands a handful of pixels out, and that is a nudge, not a drag.
      if (tool === "select" && selectedId) {
        onMoveShape(selectedId, direction[0] * step, direction[1] * step);
        return;
      }
      const point = store.get().keyboardPoint;
      store.set({
        keyboardPoint: {
          x: clamp(point.x + direction[0] * step, 0, document.image_width),
          y: clamp(point.y + direction[1] * step, 0, document.image_height),
        },
      });
      return;
    }
    const effects = toolModule.key(
      { pendingPoints, scale: viewRef.current?.scale ?? 1, assistMode },
      store.get().keyboardPoint,
      { key: event.key, shiftKey: event.shiftKey },
    );
    if (!effects) return;
    event.preventDefault();
    event.stopPropagation();
    emit(effects);
  };

  return (
    <div
      ref={regionRef}
      role="region"
      tabIndex={0}
      data-annotation-canvas
      aria-label={label}
      aria-description="Use arrow keys to move the source-pixel cursor. Shift moves ten pixels. Space applies the current drawing tool; Enter closes a polygon after three points."
      // A press on the canvas makes it the keyboard's target, so the arrows, Space and Enter that
      // follow are the canvas's and not the page's (→ is the next sample there). The stage cancels
      // the presses it handles, which also cancels the focus a click would otherwise move; a
      // vertex handle, which does not, still takes focus after this.
      onPointerDownCapture={() => regionRef.current?.focus({ preventScroll: true })}
      onFocus={() => store.set({ keyboardFocused: true })}
      onBlur={() => store.set({ keyboardFocused: false })}
      onKeyDown={onCanvasKeyDown}
      className="relative min-h-0 min-w-0 flex-1 overflow-hidden bg-canvas focus-visible:-outline-offset-2 focus-visible:outline-2 focus-visible:outline-signal"
    >
      <ImageStage
        ref={stageRef}
        image={image}
        view={view}
        onView={handleView}
        // Space, the arrows and the digits are the editor's (`editorKeys`), not the stage's.
        shortcuts={false}
        initialView="fit"
        fit={FIT}
        clamp={CLAMP}
        panButton={toolModule.pansWithPrimary ? SELECT_PAN : TOOL_PAN}
        // Under Polygon a double-click closes the ring; only Select gives it to the view.
        doubleClickFit={tool === "select"}
        onHover={(point) => store.set({ pointer: point ? fromStage(point) : null })}
        className="rounded-none border-0"
        readout={
          <PixelReadout store={store} document={document} masks={masks} base={baseValues} />
        }
        banner={
          <StageBanner
            store={store}
            loading={photo?.src !== src}
            failed={photo?.src === src && photo.failed}
          />
        }
        {...defined({ onBackgroundClick: interactive ? () => onSelect(null) : undefined })}
      >
        <ImageLayer
          src={src}
          alt=""
          // Past 1:1 a source pixel covers several screen pixels, and smoothing would blur
          // exactly the edge being traced, so it is drawn as the square it is.
          pixelatedAbove={1}
          onLoad={() => setPhoto({ src, failed: false })}
          onError={() => setPhoto({ src, failed: true })}
        />
        {overlayImageId !== undefined && (
          <div
            className="pointer-events-none absolute inset-0"
            style={{ opacity: overlayOpacity }}
          >
            <ImageLayer src={imageUrl(overlayImageId, "full")} alt="" pixelatedAbove={1} />
          </div>
        )}
        <StageScene
          document={shown}
          labels={labels}
          palette={palette}
          baseImage={baseMaskSource}
          maskOpacity={maskOpacity}
          showRegions={showRegions}
          selectedId={selectedId}
          masks={masks}
          assistShape={assistShape}
          interactive={interactive}
          onSelect={onSelect}
          onMoveShape={onMoveShape}
          onEdit={reportEdit}
        />
        {editable && (
          <ToolSurface
            tool={tool}
            store={store}
            pendingPoints={pendingPoints}
            assistMode={assistMode}
            order={order}
            emit={emit}
            onSelect={onSelect}
            onMoveShape={onMoveShape}
            onEdit={reportEdit}
          />
        )}
        {interactive && showRegions && selectedShape && selectedShape.kind !== "bitmap" && (
          <ShapeEditors
            shape={selectedShape}
            width={document.image_width}
            height={document.image_height}
            onEdit={reportEdit}
            latestEdit={readEdit}
            onReshapePolygon={onReshapePolygon}
            onReshapeBox={onReshapeBox}
          />
        )}
        <LiveDrafts
          store={store}
          tool={tool}
          editable={editable}
          brushSize={brushSize}
          pendingPoints={pendingPoints}
          assistMode={assistMode}
          assistPoints={assistPoints}
          assistBox={assistBox}
        />
      </ImageStage>
    </div>
  );
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

/**
 * Every bitmap region's crop as one byte per pixel, keyed by its PNG, for the pixel readout
 * and for picking a bitmap by its painted pixels.
 *
 * Decoded when a region appears or changes, never per pointer move; a region whose PNG is
 * unchanged keeps its decode, and one that left the document is dropped.
 */
function useDecodedMasks(document: AnnotationDocument): ReadonlyMap<string, Uint8Array> {
  const [masks, setMasks] = useState<ReadonlyMap<string, Uint8Array>>(() => new Map());
  useEffect(() => {
    let cancelled = false;
    // Only a bitmap needs decoding; a polygon and a box are read from their outline.
    const bitmaps = document.shapes.filter((shape) => shape.kind === "bitmap");
    const wanted = new Set(bitmaps.map((shape) => shape.png_base64));
    const missing = bitmaps.filter((shape) => !masks.has(shape.png_base64));
    const stale = [...masks.keys()].some((key) => !wanted.has(key));
    if (missing.length === 0 && !stale) return;
    void Promise.all(
      missing.map(async (shape) => {
        try {
          return [shape.png_base64, await decodeShapeMask(shape)] as const;
        } catch {
          // An undecodable region reads as outside; the scene says the same by not drawing it.
          return null;
        }
      }),
    ).then((decoded) => {
      if (cancelled) return;
      const next = new Map([...masks].filter(([key]) => wanted.has(key)));
      for (const entry of decoded) if (entry) next.set(entry[0], entry[1]);
      setMasks(next);
    });
    return () => {
      cancelled = true;
    };
    // A decode landing changes `masks` and runs this again, which then finds nothing to do.
  }, [document.shapes, masks]);
  return masks;
}

/**
 * The pixel under the pointer, the mask value the document resolves to there, and the zoom as
 * screen pixels per source pixel — `100%` is 1:1, not fit.
 */
function PixelReadout({
  store,
  document,
  masks,
  base,
}: {
  store: LiveStore;
  document: AnnotationDocument;
  masks: ReadonlyMap<string, Uint8Array>;
  base: Uint8Array | null;
}) {
  const stage = useStage();
  const pointer = useLive(store, (state) => state.pointer);
  const reading = pointer ? readPixel(document, pointer, masks, base) : null;
  const zoom = formatScale(stage.view.scale);
  return (
    <div
      className="rounded-control border border-line bg-surface/90 px-2 py-1 font-mono text-[10px] text-fg-muted shadow-panel backdrop-blur-sm"
      aria-hidden
    >
      {reading && (
        <>
          <span className="text-fg">
            {reading.x}, {reading.y}
          </span>
          {" · mask "}
          <span className={reading.value ? "text-signal" : undefined}>{reading.value}</span>
          {reading.region !== null && ` · region ${reading.region}`}
          {" · "}
        </>
      )}
      {stage.isFit ? `Fit ${zoom}` : zoom}
    </div>
  );
}

/** Over the top-left: the photograph still loading, and the keyboard cursor while focused. */
function StageBanner({
  store,
  loading,
  failed,
}: {
  store: LiveStore;
  loading: boolean;
  failed: boolean;
}) {
  const focused = useLive(store, (state) => state.keyboardFocused);
  const point = useLive(store, (state) => state.keyboardPoint);
  const chip =
    "w-fit rounded-control border border-line bg-surface/90 px-2 py-1 text-[11px] text-fg-muted shadow-panel backdrop-blur-sm";
  return (
    <div className="flex flex-col gap-1">
      {failed ? (
        <div className={chip}>The source image did not load.</div>
      ) : (
        loading && <div className={chip}>Loading source image…</div>
      )}
      {focused && (
        <div className={`${chip} font-mono text-[10px]`}>
          {Math.round(point.x)}, {Math.round(point.y)} px · arrows move · Shift 10 px · Space draws
        </div>
      )}
    </div>
  );
}
