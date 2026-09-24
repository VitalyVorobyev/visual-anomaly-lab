/** The controlled, source-coordinate annotation scene.
 *
 * Konva owns rendering and hit testing, never state. All geometry comes in through props
 * and every edit leaves through a callback as source-image pixels. The same document can
 * therefore be saved, rendered on the backend and evaluated without a canvas transform
 * leaking into truth.
 *
 * Three pieces, each doing one job:
 *
 * - **tools** (`tools/`) turn pointer and key input into effects, as pure functions;
 * - the **static layer** (`SceneLayer`) draws the photograph and the committed document, and
 *   re-renders only when they or the view change;
 * - the **live layer** (`LiveLayer`) draws the gesture in progress and the cursors, fed by a
 *   small store (`liveStore`) so a pointer move re-renders it and nothing else.
 *
 * This component wires them together: it owns the view arithmetic (`canvasView`), the pan
 * gesture, and the translation of tool effects into the callbacks below.
 */

import type Konva from "konva";
import type { KonvaEventObject } from "konva/lib/Node";
import {
  forwardRef,
  useEffect,
  useImperativeHandle,
  useMemo,
  useRef,
  useState,
  type KeyboardEvent,
} from "react";
import { Stage } from "react-konva";

import type {
  AnnotationDocument,
  AnnotationLabel,
  AnnotationPoint,
  AssistBox,
  AssistPoint,
  BitmapShape,
} from "../../api/client";
import { formatScale } from "@vitavision/lab-ui";

import { decodeShapeMask, imageMask, tintedMask } from "../../api/annotationBitmap";
import { imageUrl, sourceMaskUrl } from "../../api/imageUrl";
import {
  type CanvasView,
  INITIAL_CANVAS_VIEW,
  actualPixelsView,
  clamp,
  fitScale,
  isFitView,
  smoothAt,
  toSource,
  toSourceUnclamped,
  viewOrigin,
  zoomAbout,
  zoomBy,
} from "./canvasView";
import { readPixel } from "./pixelReadout";
import { canvasBindingFor } from "./editorKeys";
import { LiveLayer } from "./LiveLayer";
import { createLiveStore, useLive, type LiveStore } from "./liveStore";
import { SceneLayer } from "./SceneLayer";
import { useScenePalette } from "./scenePalette";
import { TOOLS, type EditorTool, type ToolContext, type ToolEffect } from "./tools";
import { useHtmlImage } from "./useHtmlImage";

export type { EditorTool } from "./tools";
export { INITIAL_CANVAS_VIEW, type CanvasView } from "./canvasView";

export interface AnnotationCanvasHandle {
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
   * *fills*; a polygon's outline, its vertex handles and a bitmap's selection rectangle are
   * siblings that carry no opacity of their own, so even at zero the photograph would be
   * under a wireframe. Deciding whether a marked defect is really there means seeing the
   * pixels with nothing on them.
   *
   * What is still drawn while this is off is what the reader is doing *now*: the live brush
   * trail, the pending polygon, the assist points and their un-accepted suggestion, and the
   * keyboard cursor. Those are not the overlay.
   */
  showRegions?: boolean;
  /** What a screen reader calls this surface. Panes beside the editor are not the editor. */
  label?: string;
  /**
   * Whether this pane's shapes answer the pointer at all.
   *
   * A reference pane is a photograph with the document drawn over it, not a second editor.
   * Without this its regions would still take a click and start a drag that snapped back on
   * release — an affordance that lies.
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
  view: CanvasView;
  onView: (view: CanvasView) => void;
  onSelect: (shapeId: string | null) => void;
  onPoint: (point: AnnotationPoint) => void;
  onMovePoint: (shapeId: string, pointIndex: number, point: AnnotationPoint) => void;
  /** A whole-region offset in source pixels: a drag, or an arrow-key nudge. */
  onMoveShape: (shapeId: string, dx: number, dy: number) => void;
  onBrush: (points: AnnotationPoint[]) => void;
  onFinishPolygon: () => void;
  onAssistPoint: (point: AssistPoint) => void;
  onAssistBox: (box: AssistBox | null) => void;
}

const ARROW_STEPS: Partial<Record<string, [number, number]>> = {
  ArrowUp: [0, -1],
  ArrowDown: [0, 1],
  ArrowLeft: [-1, 0],
  ArrowRight: [1, 0],
};

interface Pan {
  x: number;
  y: number;
  panX: number;
  panY: number;
  moved: boolean;
  /** A primary-button click on the empty scene, without movement, clears the selection. */
  clearOnClick: boolean;
}

export const AnnotationCanvas = forwardRef<AnnotationCanvasHandle, Props>(function AnnotationCanvas({
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
  onMovePoint,
  onMoveShape,
  onBrush,
  onFinishPolygon,
  onAssistPoint,
  onAssistBox,
}, forwardedRef) {
  const palette = useScenePalette();
  const toolModule = TOOLS[tool];
  // Selecting, dragging and reshaping are all the same permission.
  const interactive = editable && tool === "select";
  const hostRef = useRef<HTMLDivElement>(null);
  const stageRef = useRef<Konva.Stage>(null);
  const [size, setSize] = useState({ width: 1, height: 1 });
  const panning = useRef<Pan | null>(null);
  const [grabbing, setGrabbing] = useState(false);
  const previousView = useRef<CanvasView | null>(null);
  const [store] = useState<LiveStore>(() =>
    createLiveStore({
      gesture: null,
      revision: 0,
      snapReady: false,
      keyboardPoint: { x: document.image_width / 2, y: document.image_height / 2 },
      keyboardFocused: false,
      pointer: null,
    }),
  );
  const masks = useDecodedMasks(document);
  const source = useHtmlImage(imageUrl(imageId, "full"));
  const overlay = useHtmlImage(
    overlayImageId === undefined ? undefined : imageUrl(overlayImageId, "full"),
  );
  // Requested with CORS: the pixels are read back to tint them, and an image from the
  // sidecar's origin drawn without it taints the canvas and makes `getImageData` throw.
  const baseMaskSource = useHtmlImage(
    document.base === "source_mask" ? sourceMaskUrl(imageId) : undefined,
    "anonymous",
  );

  useEffect(() => {
    const host = hostRef.current;
    if (!host) return;
    const observer = new ResizeObserver(([entry]) => {
      if (!entry) return;
      setSize({
        width: Math.max(1, Math.floor(entry.contentRect.width)),
        height: Math.max(1, Math.floor(entry.contentRect.height)),
      });
    });
    observer.observe(host);
    return () => observer.disconnect();
  }, []);

  const image = { width: document.image_width, height: document.image_height };
  const fit = fitScale(size, image);
  const scale = fit * view.zoom;
  const origin = viewOrigin(size, image, fit, view);
  // The import carries no class, so it takes the dataset's first one — which is the class
  // every region drawn on a single-class dataset gets too, so base and edits read as one.
  const baseColor = labels[0]?.color ?? palette.unknownLabel;
  const baseMask = useMemo(() => {
    if (baseMaskSource === null) return null;
    try {
      return tintedMask(baseMaskSource, document.image_width, document.image_height, baseColor);
    } catch {
      // A base layer that cannot be read is context lost, not an editor lost: this threw
      // inside render once, and the crash boundary took the whole workbench with it.
      return null;
    }
  }, [baseMaskSource, document.image_width, document.image_height, baseColor]);
  // The same import as one byte per pixel, for the readout. Decoded once per base, not per move.
  const baseValues = useMemo(() => {
    if (baseMaskSource === null) return null;
    try {
      return imageMask(baseMaskSource, document.image_width, document.image_height);
    } catch {
      return null;
    }
  }, [baseMaskSource, document.image_width, document.image_height]);
  const brushing = editable && (tool === "brush" || tool === "eraser");

  const context: ToolContext = { pendingPoints, scale, assistMode };

  const fitView = () => {
    previousView.current = view;
    onView(INITIAL_CANVAS_VIEW);
  };

  const actualPixels = () => {
    previousView.current = view;
    onView(actualPixelsView(size, image));
  };

  useImperativeHandle(forwardedRef, () => ({
    fit: fitView,
    actualPixels,
    zoomBy: (factor: number) => onView(zoomBy(view, factor, size, image)),
  }));

  const toggleFit = () => {
    if (isFitView(view) && previousView.current) {
      const previous = previousView.current;
      previousView.current = null;
      onView(previous);
      return;
    }
    previousView.current = view;
    onView(INITIAL_CANVAS_VIEW);
  };

  const emit = (effects: ToolEffect[]) => {
    for (const effect of effects) {
      if (effect.type === "deselect") onSelect(null);
      else if (effect.type === "stroke") onBrush(effect.points);
      else if (effect.type === "vertex") onPoint(effect.point);
      else if (effect.type === "closePolygon") onFinishPolygon();
      else if (effect.type === "assistPoint") onAssistPoint(effect.point);
      else if (effect.type === "assistBox") onAssistBox(effect.box);
      else toggleFit();
    }
  };

  const pointerPosition = () => stageRef.current?.getPointerPosition() ?? null;

  const onStageDown = (event: KonvaEventObject<MouseEvent | TouchEvent>) => {
    const pointer = pointerPosition();
    if (!pointer) return;
    const native = event.evt;
    const button = native instanceof MouseEvent ? native.button : 0;
    if (button === 2 || (button === 0 && toolModule.pansWithPrimary)) {
      panning.current = {
        x: pointer.x,
        y: pointer.y,
        panX: view.panX,
        panY: view.panY,
        moved: false,
        clearOnClick: button === 0,
      };
      setGrabbing(true);
      return;
    }
    const point = toSource(pointer, origin, scale, image);
    store.set({ keyboardPoint: point });
    const step = toolModule.down(context, point, {
      shiftKey: native instanceof MouseEvent && native.shiftKey,
    });
    if (step.gesture) store.set({ gesture: step.gesture, revision: store.get().revision + 1 });
    emit(step.effects);
  };

  const onStageMove = () => {
    const pointer = pointerPosition();
    if (!pointer) return;
    const pan = panning.current;
    if (pan) {
      onView({
        ...view,
        panX: pan.panX + pointer.x - pan.x,
        panY: pan.panY + pointer.y - pan.y,
      });
      if (!pan.moved && Math.hypot(pointer.x - pan.x, pointer.y - pan.y) > 2) pan.moved = true;
    }
    const point = toSource(pointer, origin, scale, image);
    store.set({ pointer: toSourceUnclamped(pointer, origin, scale) });
    const live = store.get();
    if (live.gesture) {
      const step = toolModule.move(context, live.gesture, point);
      store.set({ gesture: step.gesture, revision: live.revision + 1 });
      emit(step.effects);
    }
    // Light up the first vertex while the pointer is over it, so "click here to close" is
    // visible before the click.
    const snapReady = toolModule.snaps(context, point);
    if (snapReady !== live.snapReady) store.set({ snapReady });
  };

  const finishGesture = () => {
    const pan = panning.current;
    if (pan) {
      if (pan.clearOnClick && !pan.moved) onSelect(null);
      panning.current = null;
      setGrabbing(false);
    }
    const live = store.get();
    if (live.gesture) {
      store.set({ gesture: null, revision: live.revision + 1 });
      emit(toolModule.up(context, live.gesture));
    }
  };

  const onWheel = (event: KonvaEventObject<WheelEvent>) => {
    event.evt.preventDefault();
    const pointer = pointerPosition();
    if (!pointer) return;
    onView(zoomAbout(view, event.evt.deltaY > 0 ? 0.9 : 1.1, pointer, size, image));
    // The view moved under a still pointer, so what it is over has changed.
    store.set({ pointer: toSourceUnclamped(pointer, origin, scale) });
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
    const effects = toolModule.key(context, store.get().keyboardPoint, {
      key: event.key,
      shiftKey: event.shiftKey,
    });
    if (!effects) return;
    event.preventDefault();
    event.stopPropagation();
    emit(effects);
  };

  return (
    <div
      ref={hostRef}
      role="region"
      tabIndex={0}
      data-annotation-canvas
      aria-label={label}
      aria-description="Use arrow keys to move the source-pixel cursor. Shift moves ten pixels. Space applies the current drawing tool; Enter closes a polygon after three points."
      onFocus={() => store.set({ keyboardFocused: true })}
      onBlur={() => store.set({ keyboardFocused: false })}
      onKeyDown={onCanvasKeyDown}
      className="relative min-h-0 min-w-0 flex-1 overflow-hidden bg-canvas focus-visible:-outline-offset-2 focus-visible:outline-2 focus-visible:outline-signal"
    >
      <Stage
        ref={stageRef}
        width={size.width}
        height={size.height}
        onMouseDown={onStageDown}
        onTouchStart={onStageDown}
        onMouseMove={onStageMove}
        onTouchMove={onStageMove}
        onMouseUp={finishGesture}
        onMouseLeave={() => {
          finishGesture();
          store.set({ pointer: null });
        }}
        onTouchEnd={finishGesture}
        onWheel={onWheel}
        onContextMenu={(event) => event.evt.preventDefault()}
        onDblClick={() => emit(toolModule.doubleClick(context))}
        className={
          grabbing
            ? "cursor-grabbing"
            : brushing
              ? // The brush's own footprint is the cursor (`LiveLayer`); a crosshair on top of
                // it would be a second pointer claiming a different size.
                "cursor-none"
              : toolModule.cursor === "crosshair"
                ? "cursor-crosshair"
                : "cursor-grab"
        }
      >
        <SceneLayer
          document={document}
          labels={labels}
          palette={palette}
          source={source}
          overlay={overlay}
          overlayOpacity={overlayOpacity}
          baseMask={baseMask}
          maskOpacity={maskOpacity}
          showRegions={showRegions}
          selectedId={selectedId}
          interactive={interactive}
          assistShape={assistShape}
          originX={origin.x}
          originY={origin.y}
          scale={scale}
          smoothing={smoothAt(scale)}
          onSelect={onSelect}
          onMoveShape={onMoveShape}
          onMovePoint={onMovePoint}
        />
        <LiveLayer
          store={store}
          palette={palette}
          tool={tool}
          brushSize={brushSize}
          pendingPoints={pendingPoints}
          assistPoints={assistPoints}
          assistBox={assistBox}
          brushCursor={brushing && !grabbing}
          originX={origin.x}
          originY={origin.y}
          scale={scale}
        />
      </Stage>
      {!source && (
        <div className="pointer-events-none absolute inset-0 grid place-items-center text-sm text-white/55">
          Loading source image…
        </div>
      )}
      <KeyboardReadout store={store} />
      <PixelReadout
        store={store}
        document={document}
        masks={masks}
        base={baseValues}
        scale={scale}
        fitted={isFitView(view)}
      />
    </div>
  );
});

/**
 * Every bitmap region's crop as one byte per pixel, keyed by its PNG, for the pixel readout.
 *
 * Decoded when a region appears or changes, never per pointer move; a region whose PNG is
 * unchanged keeps its decode, and one that left the document is dropped.
 */
function useDecodedMasks(document: AnnotationDocument): ReadonlyMap<string, Uint8Array> {
  const [masks, setMasks] = useState<ReadonlyMap<string, Uint8Array>>(() => new Map());
  useEffect(() => {
    let cancelled = false;
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
    // `masks` is read, not watched: this runs when the document changes, and the decode it
    // starts is what changes `masks`.
  }, [document.shapes]);
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
  scale,
  fitted,
}: {
  store: LiveStore;
  document: AnnotationDocument;
  masks: ReadonlyMap<string, Uint8Array>;
  base: Uint8Array | null;
  scale: number;
  fitted: boolean;
}) {
  const pointer = useLive(store, (state) => state.pointer);
  const reading = pointer ? readPixel(document, pointer, masks, base) : null;
  return (
    <div
      className="pointer-events-none absolute bottom-3 left-3 rounded-control border border-line bg-surface/90 px-2 py-1 font-mono text-[10px] text-fg-muted shadow-panel backdrop-blur-sm"
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
      {fitted ? `Fit ${formatScale(scale)}` : formatScale(scale)}
    </div>
  );
}

/** Where the keyboard cursor is, while the canvas has focus. */
function KeyboardReadout({ store }: { store: LiveStore }) {
  const focused = useLive(store, (state) => state.keyboardFocused);
  const point = useLive(store, (state) => state.keyboardPoint);
  if (!focused) return null;
  return (
    <div className="pointer-events-none absolute bottom-3 left-1/2 -translate-x-1/2 rounded-control border border-line bg-surface/90 px-2 py-1 font-mono text-[10px] text-fg-muted shadow-panel backdrop-blur-sm">
      {Math.round(point.x)}, {Math.round(point.y)} px · arrows move · Shift 10 px · Space draws
    </div>
  );
}
