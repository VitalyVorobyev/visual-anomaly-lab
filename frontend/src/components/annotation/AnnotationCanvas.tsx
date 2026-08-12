/** The controlled, source-coordinate annotation scene.
 *
 * Konva owns rendering and hit testing, never state. All geometry comes in through props
 * and every edit leaves through a callback as source-image pixels. The same document can
 * therefore be saved, rendered on the backend and evaluated without a canvas transform
 * leaking into truth.
 */

import Konva from "konva";
import type { KonvaEventObject } from "konva/lib/Node";
import { forwardRef, useEffect, useImperativeHandle, useMemo, useRef, useState } from "react";
import { Circle, Group, Image as KonvaImage, Layer, Line, Rect, Stage } from "react-konva";

import type {
  AnnotationDocument,
  AnnotationLabel,
  AnnotationPoint,
  BitmapShape,
} from "../../api/client";
import { imageUrl, maskUrl } from "../../api/imageUrl";

export type EditorTool = "select" | "polygon" | "brush" | "eraser";

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
  document: AnnotationDocument;
  labels: AnnotationLabel[];
  selectedId: string | null;
  tool: EditorTool;
  pendingPoints: AnnotationPoint[];
  brushRadius: number;
  view: CanvasView;
  onView: (view: CanvasView) => void;
  onSelect: (shapeId: string | null) => void;
  onPoint: (point: AnnotationPoint) => void;
  onMovePoint: (shapeId: string, pointIndex: number, point: AnnotationPoint) => void;
  onBrush: (points: AnnotationPoint[]) => void;
}

export const AnnotationCanvas = forwardRef<AnnotationCanvasHandle, Props>(function AnnotationCanvas({
  imageId,
  document,
  labels,
  selectedId,
  tool,
  pendingPoints,
  brushRadius,
  view,
  onView,
  onSelect,
  onPoint,
  onMovePoint,
  onBrush,
}, forwardedRef) {
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
  const source = useHtmlImage(imageUrl(imageId, "full"));
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
    if (tool === "brush" || tool === "eraser") {
      const point = sourcePoint();
      if (point) setBrushPoints([point]);
      return;
    }
    // Source pixels are themselves Konva Image nodes, so a useful canvas click almost
    // never targets the Stage object. Shape handlers stop propagation; anything reaching
    // here is the image/background and is therefore an empty-scene gesture.
    onSelect(null);
    if (tool === "polygon") {
      const point = sourcePoint();
      if (point) onPoint(point);
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

  const finishGesture = () => {
    if (panning?.clearOnClick && !panning.moved) onSelect(null);
    setPanning(null);
    if (brushPoints.length > 0) onBrush(brushPoints);
    setBrushPoints([]);
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
    <div ref={hostRef} className="relative min-h-0 min-w-0 flex-1 overflow-hidden bg-canvas">
      <Stage
        ref={stageRef}
        width={size.width}
        height={size.height}
        onMouseDown={onStageDown}
        onTouchStart={onStageDown}
        onMouseMove={() => {
          onStageMove();
          trackBrush();
        }}
        onTouchMove={() => {
          onStageMove();
          trackBrush();
        }}
        onMouseUp={finishGesture}
        onMouseLeave={finishGesture}
        onTouchEnd={finishGesture}
        onWheel={onWheel}
        onContextMenu={(event) => event.evt.preventDefault()}
        onDblClick={() => {
          if (tool === "select") toggleFit();
        }}
        className={
          panning
            ? "cursor-grabbing"
            : tool === "polygon" || tool === "brush" || tool === "eraser"
              ? "cursor-crosshair"
              : "cursor-grab"
        }
      >
        <Layer imageSmoothingEnabled>
          <Group x={origin.x} y={origin.y} scaleX={scale} scaleY={scale}>
            <Rect
              width={document.image_width}
              height={document.image_height}
              fill="#08090a"
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
            {baseMask && (
              <KonvaImage
                image={baseMask}
                width={document.image_width}
                height={document.image_height}
                opacity={0.22}
                listening={false}
              />
            )}

            {document.shapes.map((shape) => {
              if (shape.kind === "bitmap") {
                return (
                  <BitmapLayer
                    key={shape.id}
                    shape={shape}
                    selected={shape.id === selectedId}
                    onSelect={() => onSelect(shape.id)}
                  />
                );
              }
              const color = colors.get(shape.label_key) ?? "#0e8fa3";
              const selected = shape.id === selectedId;
              return (
                <Group key={shape.id}>
                  <Line
                    points={shape.points.flatMap((point) => [point.x, point.y])}
                    closed
                    fill={shape.operation === "add" ? `${color}44` : "#00000088"}
                    stroke={shape.operation === "add" ? color : "#f87171"}
                    strokeWidth={(selected ? 2.5 : 1.5) / scale}
                    hitStrokeWidth={10 / scale}
                    onMouseDown={(event) => {
                      if (event.evt.button === 2) return;
                      event.cancelBubble = true;
                      onSelect(shape.id);
                    }}
                    onTap={(event) => {
                      event.cancelBubble = true;
                      onSelect(shape.id);
                    }}
                  />
                  {selected &&
                    shape.points.map((point, index) => (
                      <Circle
                        key={`${shape.id}-${index}`}
                        x={point.x}
                        y={point.y}
                        radius={4.5 / scale}
                        fill="#ffffff"
                        stroke={color}
                        strokeWidth={1.5 / scale}
                        draggable={tool === "select"}
                        onDragEnd={(event) =>
                          onMovePoint(shape.id, index, {
                            x: clamp(event.target.x(), 0, document.image_width),
                            y: clamp(event.target.y(), 0, document.image_height),
                          })
                        }
                        onMouseDown={(event) => {
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
                  stroke="#3bc9db"
                  strokeWidth={2 / scale}
                  dash={[6 / scale, 4 / scale]}
                />
                {pendingPoints.map((point, index) => (
                  <Circle
                    key={`${point.x}-${point.y}-${index}`}
                    x={point.x}
                    y={point.y}
                    radius={index === 0 ? 5 / scale : 3.5 / scale}
                    fill="#ffffff"
                    stroke="#3bc9db"
                    strokeWidth={1.5 / scale}
                  />
                ))}
              </Group>
            )}
            {brushPoints.length > 0 && (
              <Line
                points={brushPoints.flatMap((point) => [point.x, point.y])}
                stroke={tool === "eraser" ? "#f87171" : "#3bc9db"}
                strokeWidth={brushRadius * 2}
                lineCap="round"
                lineJoin="round"
                opacity={0.72}
                listening={false}
              />
            )}
            <Rect
              width={document.image_width}
              height={document.image_height}
              stroke="#ffffff55"
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
    </div>
  );
});

function BitmapLayer({
  shape,
  selected,
  onSelect,
}: {
  shape: BitmapShape;
  selected: boolean;
  onSelect: () => void;
}) {
  const image = useHtmlImage(`data:image/png;base64,${shape.png_base64}`);
  if (!image) return null;
  return (
    <Group
      clipX={shape.x}
      clipY={shape.y}
      clipWidth={shape.width}
      clipHeight={shape.height}
      onMouseDown={(event) => {
        if (event.evt.button === 2) return;
        event.cancelBubble = true;
        onSelect();
      }}
    >
      <KonvaImage
        image={image}
        x={shape.x}
        y={shape.y}
        width={shape.width}
        height={shape.height}
        opacity={shape.operation === "add" ? 0.34 : 0.2}
      />
      {selected && (
        <Rect
          x={shape.x}
          y={shape.y}
          width={shape.width}
          height={shape.height}
          stroke="#3bc9db"
          strokeWidth={1.5}
        />
      )}
    </Group>
  );
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
