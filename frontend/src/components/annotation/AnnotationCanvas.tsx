/** The controlled, source-coordinate annotation scene.
 *
 * Konva owns rendering and hit testing, never state. All geometry comes in through props
 * and every edit leaves through a callback as source-image pixels. The same document can
 * therefore be saved, rendered on the backend and evaluated without a canvas transform
 * leaking into truth.
 */

import Konva from "konva";
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
import { Circle, Group, Image as KonvaImage, Layer, Line, Rect, Stage } from "react-konva";

import type {
  AnnotationDocument,
  AnnotationLabel,
  AnnotationPoint,
  AssistBox,
  AssistPoint,
  BitmapShape,
} from "../../api/client";
import { tintedMask } from "../../api/annotationBitmap";
import { polygonClick, snapTolerance } from "../../api/annotationPolygon";
import { imageUrl, maskUrl } from "../../api/imageUrl";
import { useScenePalette, withAlpha, type ScenePalette } from "./scenePalette";

export type EditorTool = "select" | "polygon" | "brush" | "eraser" | "assist";

export interface CanvasView {
  zoom: number;
  panX: number;
  panY: number;
}

export const INITIAL_CANVAS_VIEW: CanvasView = { zoom: 1, panX: 0, panY: 0 };

export interface AnnotationCanvasHandle {
  fit: () => void;
  actualPixels: () => void;
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
  // Selecting, dragging and reshaping are all the same permission.
  const interactive = editable && tool === "select";
  const hostRef = useRef<HTMLDivElement>(null);
  const stageRef = useRef<Konva.Stage>(null);
  const [size, setSize] = useState({ width: 1, height: 1 });
  const [panning, setPanning] = useState<{
    x: number;
    y: number;
    panX: number;
    panY: number;
    moved: boolean;
    clearOnClick: boolean;
  } | null>(null);
  const previousView = useRef<CanvasView | null>(null);
  const [brushPoints, setBrushPoints] = useState<AnnotationPoint[]>([]);
  const [boxStart, setBoxStart] = useState<AnnotationPoint | null>(null);
  /**
   * A vertex mid-drag, and whether the pointer is over the ring's first vertex.
   *
   * Transient gesture state, exactly like `brushPoints` and `boxStart` above it — the scene
   * still never becomes a second store of annotation truth. The vertex override is what makes
   * a drag *look* like a drag: the outline used to be read straight from the committed
   * document, which only changes on `dragEnd`, so the handle moved and the polygon it belonged
   * to stayed where it was until the mouse came up.
   */
  const [vertexDrag, setVertexDrag] = useState<{
    shapeId: string;
    index: number;
    point: AnnotationPoint;
  } | null>(null);
  const [snapReady, setSnapReady] = useState(false);
  const [keyboardFocused, setKeyboardFocused] = useState(false);
  const [keyboardPoint, setKeyboardPoint] = useState<AnnotationPoint>({
    x: document.image_width / 2,
    y: document.image_height / 2,
  });
  const source = useHtmlImage(imageUrl(imageId, "full"));
  const overlay = useHtmlImage(
    overlayImageId === undefined ? undefined : imageUrl(overlayImageId, "full"),
  );
  const baseMask = useHtmlImage(
    document.base === "source_mask" ? maskUrl(imageId) : undefined,
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

  const fit = Math.min(
    size.width / document.image_width,
    size.height / document.image_height,
  ) * 0.94;
  const scale = fit * view.zoom;
  const origin = {
    x: (size.width - document.image_width * fit) / 2 + view.panX,
    y: (size.height - document.image_height * fit) / 2 + view.panY,
  };
  const colors = useMemo(
    () => new Map(labels.map((label) => [label.key, label.color])),
    [labels],
  );

  const fitView = () => {
    previousView.current = view;
    onView(INITIAL_CANVAS_VIEW);
  };

  const actualPixels = () => {
    previousView.current = view;
    const actualOrigin = {
      x: (size.width - document.image_width) / 2,
      y: (size.height - document.image_height) / 2,
    };
    const fitOrigin = {
      x: (size.width - document.image_width * fit) / 2,
      y: (size.height - document.image_height * fit) / 2,
    };
    onView({
      zoom: 1 / fit,
      panX: actualOrigin.x - fitOrigin.x,
      panY: actualOrigin.y - fitOrigin.y,
    });
  };

  useImperativeHandle(forwardedRef, () => ({ fit: fitView, actualPixels }));

  const toggleFit = () => {
    const isFit =
      Math.abs(view.zoom - 1) < 0.0001 &&
      Math.abs(view.panX) < 0.5 &&
      Math.abs(view.panY) < 0.5;
    if (isFit && previousView.current) {
      const previous = previousView.current;
      previousView.current = null;
      onView(previous);
      return;
    }
    previousView.current = view;
    onView(INITIAL_CANVAS_VIEW);
  };

  const sourcePoint = (): AnnotationPoint | null => {
    const pointer = stageRef.current?.getPointerPosition();
    if (!pointer) return null;
    return {
      x: clamp((pointer.x - origin.x) / scale, 0, document.image_width),
      y: clamp((pointer.y - origin.y) / scale, 0, document.image_height),
    };
  };

  const onStageDown = (event: KonvaEventObject<MouseEvent | TouchEvent>) => {
    const pointer = stageRef.current?.getPointerPosition();
    if (!pointer) return;
    const native = event.evt;
    const button = native instanceof MouseEvent ? native.button : 0;
    const panGesture = button === 2 || (button === 0 && tool === "select");
    if (panGesture) {
      setPanning({
        x: pointer.x,
        y: pointer.y,
        panX: view.panX,
        panY: view.panY,
        moved: false,
        clearOnClick: button === 0,
      });
      return;
    }
    if (tool === "assist") {
      const point = sourcePoint();
      if (!point) return;
      setKeyboardPoint(point);
      if (assistMode === "point") {
        const negative = native instanceof MouseEvent && native.shiftKey;
        onAssistPoint({ ...point, kind: negative ? "negative" : "positive" });
      } else {
        setBoxStart(point);
        onAssistBox({ x0: point.x, y0: point.y, x1: point.x, y1: point.y });
      }
      return;
    }
    if (tool === "brush" || tool === "eraser") {
      const point = sourcePoint();
      if (point) {
        setKeyboardPoint(point);
        setBrushPoints([point]);
      }
      return;
    }
    // Source pixels are themselves Konva Image nodes, so a useful canvas click almost
    // never targets the Stage object. Shape handlers stop propagation; anything reaching
    // here is the image/background and is therefore an empty-scene gesture.
    onSelect(null);
    if (tool === "polygon") {
      const point = sourcePoint();
      if (!point) return;
      setKeyboardPoint(point);
      const decision = polygonClick(pendingPoints, point, snapTolerance(scale));
      // `ignore` is not a no-op worth skipping: it is what lets a double-click close a ring
      // without leaving the duplicate vertex its second click would otherwise add.
      if (decision === "close") onFinishPolygon();
      else if (decision === "add") onPoint(point);
    }
  };

  const onCanvasKeyDown = (event: KeyboardEvent<HTMLDivElement>) => {
    const directions: Partial<Record<string, [number, number]>> = {
      ArrowUp: [0, -1],
      ArrowDown: [0, 1],
      ArrowLeft: [-1, 0],
      ArrowRight: [1, 0],
    };
    const direction = directions[event.key];
    if (direction) {
      event.preventDefault();
      event.stopPropagation();
      const step = event.shiftKey ? 10 : 1;
      // With a region selected the arrows are the precise tool, not the cursor: a copy taken
      // from another channel lands a handful of pixels out, and that is a nudge, not a drag.
      if (tool === "select" && selectedId) {
        onMoveShape(selectedId, direction[0] * step, direction[1] * step);
        return;
      }
      setKeyboardPoint((point) => ({
        x: clamp(point.x + direction[0] * step, 0, document.image_width),
        y: clamp(point.y + direction[1] * step, 0, document.image_height),
      }));
      return;
    }
    if (event.key !== "Enter" && event.key !== " ") return;
    if (tool === "select" || (tool === "assist" && assistMode === "box")) return;
    event.preventDefault();
    event.stopPropagation();
    if (tool === "polygon") {
      if (event.key === "Enter" && pendingPoints.length >= 3) onFinishPolygon();
      else onPoint(keyboardPoint);
    } else if (tool === "brush" || tool === "eraser") {
      onBrush([keyboardPoint]);
    } else if (tool === "assist") {
      onAssistPoint({
        ...keyboardPoint,
        kind: event.shiftKey ? "negative" : "positive",
      });
    }
  };

  const onStageMove = () => {
    if (!panning) return;
    const pointer = stageRef.current?.getPointerPosition();
    if (!pointer) return;
    onView({
      ...view,
      panX: panning.panX + pointer.x - panning.x,
      panY: panning.panY + pointer.y - panning.y,
    });
    if (!panning.moved && Math.hypot(pointer.x - panning.x, pointer.y - panning.y) > 2) {
      setPanning({ ...panning, moved: true });
    }
  };

  const trackBrush = () => {
    if (tool !== "brush" && tool !== "eraser") return;
    if (brushPoints.length === 0) return;
    const point = sourcePoint();
    if (!point) return;
    setBrushPoints((points) => [...points, point]);
  };

  /** Light up the first vertex while the pointer is over it, so "click here to close" is visible. */
  const trackPolygonSnap = () => {
    if (tool !== "polygon" || pendingPoints.length < 3) {
      if (snapReady) setSnapReady(false);
      return;
    }
    const point = sourcePoint();
    const first = pendingPoints[0];
    const ready =
      point !== null &&
      first !== undefined &&
      polygonClick(pendingPoints, point, snapTolerance(scale)) === "close";
    if (ready !== snapReady) setSnapReady(ready);
  };

  const trackAssistBox = () => {
    if (tool !== "assist" || assistMode !== "box" || !boxStart) return;
    const point = sourcePoint();
    if (!point) return;
    onAssistBox({
      x0: Math.min(boxStart.x, point.x),
      y0: Math.min(boxStart.y, point.y),
      x1: Math.max(boxStart.x, point.x),
      y1: Math.max(boxStart.y, point.y),
    });
  };

  const finishGesture = () => {
    if (panning?.clearOnClick && !panning.moved) onSelect(null);
    setPanning(null);
    if (brushPoints.length > 0) onBrush(brushPoints);
    setBrushPoints([]);
    setBoxStart(null);
  };

  const onWheel = (event: KonvaEventObject<WheelEvent>) => {
    event.evt.preventDefault();
    const pointer = stageRef.current?.getPointerPosition();
    if (!pointer) return;
    const before = {
      x: (pointer.x - origin.x) / scale,
      y: (pointer.y - origin.y) / scale,
    };
    const zoom = clamp(view.zoom * (event.evt.deltaY > 0 ? 0.9 : 1.1), 0.25, 12);
    const nextScale = fit * zoom;
    const fitOrigin = {
      x: (size.width - document.image_width * fit) / 2,
      y: (size.height - document.image_height * fit) / 2,
    };
    onView({
      zoom,
      panX: pointer.x - before.x * nextScale - fitOrigin.x,
      panY: pointer.y - before.y * nextScale - fitOrigin.y,
    });
  };

  return (
    <div
      ref={hostRef}
      role="region"
      tabIndex={0}
      data-annotation-canvas
      aria-label={label}
      aria-description="Use arrow keys to move the source-pixel cursor. Shift moves ten pixels. Space applies the current drawing tool; Enter closes a polygon after three points."
      onFocus={() => setKeyboardFocused(true)}
      onBlur={() => setKeyboardFocused(false)}
      onKeyDown={onCanvasKeyDown}
      className="relative min-h-0 min-w-0 flex-1 overflow-hidden bg-canvas focus-visible:-outline-offset-2 focus-visible:outline-2 focus-visible:outline-signal"
    >
      <Stage
        ref={stageRef}
        width={size.width}
        height={size.height}
        onMouseDown={onStageDown}
        onTouchStart={onStageDown}
        onMouseMove={() => {
          onStageMove();
          trackBrush();
          trackPolygonSnap();
          trackAssistBox();
        }}
        onTouchMove={() => {
          onStageMove();
          trackBrush();
          trackPolygonSnap();
          trackAssistBox();
        }}
        onMouseUp={finishGesture}
        onMouseLeave={finishGesture}
        onTouchEnd={finishGesture}
        onWheel={onWheel}
        onContextMenu={(event) => event.evt.preventDefault()}
        onDblClick={() => {
          if (tool === "select") toggleFit();
          // The second click of the pair was dropped as a duplicate, so this closes the ring
          // the first click completed — a polygon tool's ordinary gesture, and the reason the
          // "Close" button in the inspector is gone.
          else if (tool === "polygon" && pendingPoints.length >= 3) onFinishPolygon();
        }}
        className={
          panning
            ? "cursor-grabbing"
            : tool === "polygon" || tool === "brush" || tool === "eraser" || tool === "assist"
              ? "cursor-crosshair"
              : "cursor-grab"
        }
      >
        <Layer imageSmoothingEnabled>
          <Group x={origin.x} y={origin.y} scaleX={scale} scaleY={scale}>
            <Rect
              width={document.image_width}
              height={document.image_height}
              fill={palette.canvas}
              shadowColor="#000000"
              shadowBlur={16 / scale}
              shadowOpacity={0.4}
            />
            {source && (
              <KonvaImage
                image={source}
                width={document.image_width}
                height={document.image_height}
                listening={false}
              />
            )}
            {overlay && (
              <KonvaImage
                image={overlay}
                width={document.image_width}
                height={document.image_height}
                opacity={overlayOpacity}
                listening={false}
              />
            )}
            {showRegions && baseMask && (
              // The imported base, at half the weight of the editable regions above it: it is
              // context for what is being edited, not one of the things being edited.
              <KonvaImage
                image={baseMask}
                width={document.image_width}
                height={document.image_height}
                opacity={maskOpacity * 0.5}
                listening={false}
              />
            )}

            {showRegions &&
              document.shapes.map((shape) => {
                if (shape.kind === "bitmap") {
                  return (
                    <BitmapLayer
                      key={shape.id}
                      shape={shape}
                      color={colors.get(shape.label_key) ?? palette.unknownLabel}
                      palette={palette}
                      opacity={maskOpacity}
                      scale={scale}
                      selected={shape.id === selectedId}
                      onSelect={() => onSelect(shape.id)}
                      onMove={(dx, dy) => onMoveShape(shape.id, dx, dy)}
                      selectable={interactive}
                    />
                  );
                }
                const color = colors.get(shape.label_key) ?? palette.unknownLabel;
                const selected = shape.id === selectedId;
                // The dragged vertex is applied here rather than committed on every mouse move:
                // the outline follows the handle, and undo still steps one drag at a time.
                const points =
                  vertexDrag?.shapeId === shape.id
                    ? shape.points.map((point, index) =>
                        index === vertexDrag.index ? vertexDrag.point : point,
                      )
                    : shape.points;
                return (
                  <Group
                    key={shape.id}
                    draggable={interactive}
                    // A click has to survive a shaking hand, or selecting a region would drag it
                    // a pixel and land in undo history.
                    dragDistance={4}
                    // Selection is claimed *here*, on the draggable node, and not on the line
                    // inside it. Konva starts a drag from a `mousedown` listener on the
                    // draggable node itself, reached by bubbling — so cancelling the bubble on
                    // the line would have selected the region and then silently refused to move
                    // it. Cancelling here still stops the stage's pan gesture, which is the only
                    // thing that had to stop.
                    onMouseDown={(event) => {
                      if (event.evt.button === 2) return;
                      event.cancelBubble = true;
                      onSelect(shape.id);
                    }}
                    onTap={(event) => {
                      event.cancelBubble = true;
                      onSelect(shape.id);
                    }}
                    onDragEnd={(event) => {
                      // `dragend` bubbles, and a selected polygon's vertices are draggable
                      // children — without this, dragging a vertex would also read the
                      // *vertex's* position as a whole-region offset and fling the shape across
                      // the frame.
                      if (event.target !== event.currentTarget) return;
                      const node = event.target;
                      const dx = node.x();
                      const dy = node.y();
                      // Konva moving the group *is* the live feedback; the committed document
                      // then carries the offset, so the node has to go back to the origin or it
                      // would be applied twice.
                      node.position({ x: 0, y: 0 });
                      onMoveShape(shape.id, dx, dy);
                    }}
                  >
                    <Line
                      points={points.flatMap((point) => [point.x, point.y])}
                      closed
                      fill={withAlpha(shape.operation === "add" ? color : palette.cut, maskOpacity)}
                      stroke={shape.operation === "add" ? color : palette.cut}
                      strokeWidth={(selected ? 2.5 : 1.5) / scale}
                      hitStrokeWidth={10 / scale}
                      listening={interactive}
                    />
                    {selected &&
                      points.map((point, index) => (
                        <Circle
                          key={`${shape.id}-${index}`}
                          x={point.x}
                          y={point.y}
                          radius={4.5 / scale}
                          fill={palette.frame}
                          stroke={color}
                          strokeWidth={1.5 / scale}
                          hitStrokeWidth={12 / scale}
                          // Silent under every other tool. A selected polygon stays on screen
                          // while brushing, and a vertex that still took the click swallowed the
                          // start of the stroke.
                          listening={interactive}
                          draggable={interactive}
                          onDragMove={(event) =>
                            setVertexDrag({
                              shapeId: shape.id,
                              index,
                              point: {
                                x: clamp(event.target.x(), 0, document.image_width),
                                y: clamp(event.target.y(), 0, document.image_height),
                              },
                            })
                          }
                          onDragEnd={(event) => {
                            event.cancelBubble = true;
                            setVertexDrag(null);
                            onMovePoint(shape.id, index, {
                              x: clamp(event.target.x(), 0, document.image_width),
                              y: clamp(event.target.y(), 0, document.image_height),
                            });
                          }}
                          onMouseDown={(event) => {
                            // The vertex owns this gesture: without cancelling, the group under
                            // it would drag the whole region at the same time.
                            event.cancelBubble = true;
                          }}
                        />
                      ))}
                  </Group>
                );
              })}

            {pendingPoints.length > 0 && (
              <Group listening={false}>
                <Line
                  points={pendingPoints.flatMap((point) => [point.x, point.y])}
                  stroke={palette.signal}
                  strokeWidth={2 / scale}
                  dash={[6 / scale, 4 / scale]}
                />
                {pendingPoints.map((point, index) => (
                  <Circle
                    key={`${point.x}-${point.y}-${index}`}
                    x={point.x}
                    y={point.y}
                    // The first vertex swells and fills while the pointer is over it, so
                    // "click here to close" is something the ring says rather than something
                    // a button elsewhere claims.
                    radius={(index === 0 ? (snapReady ? 8 : 5) : 3.5) / scale}
                    fill={index === 0 && snapReady ? palette.signal : palette.frame}
                    stroke={palette.signal}
                    strokeWidth={1.5 / scale}
                  />
                ))}
              </Group>
            )}
            {brushPoints.length > 0 && (
              <Line
                points={brushPoints.flatMap((point) => [point.x, point.y])}
                stroke={tool === "eraser" ? palette.cut : palette.signal}
                /* True size in source pixels, floored at one *screen* pixel. A one-pixel
                   brush at fit zoom is otherwise a fraction of a pixel wide, so the
                   gesture would leave no visible trail while it was being made. The
                   `n / scale` idiom is how every outline in this scene is drawn. */
                strokeWidth={Math.max(brushSize, 1 / scale)}
                lineCap="round"
                lineJoin="round"
                opacity={0.72}
                listening={false}
              />
            )}
            {assistShape && (
              <BitmapLayer
                shape={assistShape}
                color={palette.suggestion}
                palette={palette}
                opacity={maskOpacity}
                scale={scale}
                selected
                selectable={false}
                suggestion
                onSelect={() => undefined}
              />
            )}
            {assistBox && (
              <Rect
                x={assistBox.x0}
                y={assistBox.y0}
                width={assistBox.x1 - assistBox.x0}
                height={assistBox.y1 - assistBox.y0}
                stroke={palette.signal}
                strokeWidth={2 / scale}
                dash={[7 / scale, 4 / scale]}
                listening={false}
              />
            )}
            {assistPoints.map((point, index) => (
              <Group key={`assist-point-${index}`} listening={false}>
                <Circle
                  x={point.x}
                  y={point.y}
                  radius={6 / scale}
                  fill={point.kind === "positive" ? palette.positive : palette.negative}
                  stroke={palette.frame}
                  strokeWidth={1.5 / scale}
                />
                <Line
                  points={[-3 / scale, 0, 3 / scale, 0]}
                  x={point.x}
                  y={point.y}
                  stroke={palette.frame}
                  strokeWidth={1.5 / scale}
                />
                {point.kind === "positive" && (
                  <Line
                    points={[0, -3 / scale, 0, 3 / scale]}
                    x={point.x}
                    y={point.y}
                    stroke={palette.frame}
                    strokeWidth={1.5 / scale}
                  />
                )}
              </Group>
            ))}
            {keyboardFocused && tool !== "select" && (
              <Group listening={false}>
                <Circle
                  x={keyboardPoint.x}
                  y={keyboardPoint.y}
                  radius={7 / scale}
                  fill={withAlpha(palette.canvas, 0.67)}
                  stroke={palette.signal}
                  strokeWidth={2 / scale}
                />
                <Line
                  x={keyboardPoint.x}
                  y={keyboardPoint.y}
                  points={[-11 / scale, 0, 11 / scale, 0]}
                  stroke={palette.frame}
                  strokeWidth={1 / scale}
                />
                <Line
                  x={keyboardPoint.x}
                  y={keyboardPoint.y}
                  points={[0, -11 / scale, 0, 11 / scale]}
                  stroke={palette.frame}
                  strokeWidth={1 / scale}
                />
              </Group>
            )}
            <Rect
              width={document.image_width}
              height={document.image_height}
              stroke={withAlpha(palette.frame, 0.55)}
              strokeWidth={1 / scale}
              listening={false}
            />
          </Group>
        </Layer>
      </Stage>
      {!source && (
        <div className="pointer-events-none absolute inset-0 grid place-items-center text-sm text-white/55">
          Loading source image…
        </div>
      )}
      {keyboardFocused && (
        <div className="pointer-events-none absolute bottom-3 left-1/2 -translate-x-1/2 rounded-control border border-line bg-surface/90 px-2 py-1 font-mono text-[10px] text-fg-muted shadow-panel backdrop-blur-sm">
          {Math.round(keyboardPoint.x)}, {Math.round(keyboardPoint.y)} px · arrows move · Shift 10 px · Space draws
        </div>
      )}
    </div>
  );
});

function BitmapLayer({
  shape,
  color,
  palette,
  opacity,
  scale,
  selected,
  selectable,
  suggestion = false,
  onSelect,
  onMove,
}: {
  shape: BitmapShape;
  color: string;
  palette: ScenePalette;
  opacity: number;
  scale: number;
  selected: boolean;
  selectable: boolean;
  suggestion?: boolean;
  onSelect: () => void;
  onMove?: (dx: number, dy: number) => void;
}) {
  const cut = shape.operation === "subtract";
  const tint = suggestion ? palette.suggestion : cut ? palette.cut : color;
  const painted = useTintedMask(shape, tint);
  if (!painted) return null;
  return (
    <Group
      clipX={shape.x}
      clipY={shape.y}
      clipWidth={shape.width}
      clipHeight={shape.height}
      draggable={selectable && onMove !== undefined}
      dragDistance={4}
      onDragEnd={(event) => {
        const node = event.target;
        const dx = node.x();
        const dy = node.y();
        node.position({ x: 0, y: 0 });
        onMove?.(dx, dy);
      }}
      onMouseDown={(event) => {
        if (!selectable || event.evt.button === 2) return;
        event.cancelBubble = true;
        onSelect();
      }}
      listening={selectable}
    >
      <KonvaImage
        image={painted}
        x={shape.x}
        y={shape.y}
        width={shape.width}
        height={shape.height}
        opacity={suggestion ? Math.min(1, opacity + 0.2) : opacity}
      />
      {/* A cut is outlined even when it is not selected. Rendered like an added region it was
          indistinguishable from one, which is what made the eraser look like a second brush. */}
      {(cut || selected) && (
        <Rect
          x={shape.x}
          y={shape.y}
          width={shape.width}
          height={shape.height}
          stroke={selected ? (suggestion ? palette.suggestion : palette.signal) : palette.cut}
          strokeWidth={(selected ? 1.5 : 1) / scale}
          dash={cut && !selected ? [6 / scale, 4 / scale] : undefined}
        />
      )}
    </Group>
  );
}

/**
 * The shape's mask, painted in one colour with alpha taken from its luminance.
 *
 * Two bugs in one: the mask used to be drawn untinted, so a brush region was white-on-grey at a
 * third opacity — invisible — and an opaque backend-produced mask covered its whole crop
 * rectangle in flat grey. `tintedMask` fixes both by deriving alpha rather than trusting it.
 */
function useTintedMask(shape: BitmapShape, color: string): HTMLCanvasElement | null {
  const source = useHtmlImage(`data:image/png;base64,${shape.png_base64}`);
  const [painted, setPainted] = useState<HTMLCanvasElement | null>(null);

  useEffect(() => {
    if (!source) {
      setPainted(null);
      return;
    }
    setPainted(tintedMask(source, shape.width, shape.height, color));
  }, [source, shape.width, shape.height, color]);

  return painted;
}

function useHtmlImage(src: string | undefined): HTMLImageElement | null {
  const [image, setImage] = useState<HTMLImageElement | null>(null);
  useEffect(() => {
    setImage(null);
    if (!src) return;
    const next = new globalThis.Image();
    next.onload = () => setImage(next);
    next.onerror = () => setImage(null);
    next.src = src;
    return () => {
      next.onload = null;
      next.onerror = null;
    };
  }, [src]);
  return image;
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}
