/** The controlled, source-coordinate annotation scene.
 *
 * Konva owns rendering and hit testing, never state. All geometry comes in through props
 * and every edit leaves through a callback as source-image pixels. The same document can
 * therefore be saved, rendered on the backend and evaluated without a canvas transform
 * leaking into truth.
 */

import Konva from "konva";
import type { KonvaEventObject } from "konva/lib/Node";
import { useEffect, useMemo, useRef, useState } from "react";
import { Circle, Group, Image as KonvaImage, Layer, Line, Rect, Stage } from "react-konva";

import type {
  AnnotationDocument,
  AnnotationLabel,
  AnnotationPoint,
  BitmapShape,
} from "../../api/client";
import { imageUrl, maskUrl } from "../../api/imageUrl";

export type EditorTool = "select" | "polygon" | "hand";

export interface CanvasView {
  zoom: number;
  panX: number;
  panY: number;
}

export const INITIAL_CANVAS_VIEW: CanvasView = { zoom: 1, panX: 0, panY: 0 };

interface Props {
  imageId: number;
  document: AnnotationDocument;
  labels: AnnotationLabel[];
  selectedId: string | null;
  tool: EditorTool;
  pendingPoints: AnnotationPoint[];
  view: CanvasView;
  onView: (view: CanvasView) => void;
  onSelect: (shapeId: string | null) => void;
  onPoint: (point: AnnotationPoint) => void;
  onMovePoint: (shapeId: string, pointIndex: number, point: AnnotationPoint) => void;
}

export function AnnotationCanvas({
  imageId,
  document,
  labels,
  selectedId,
  tool,
  pendingPoints,
  view,
  onView,
  onSelect,
  onPoint,
  onMovePoint,
}: Props) {
  const hostRef = useRef<HTMLDivElement>(null);
  const stageRef = useRef<Konva.Stage>(null);
  const [size, setSize] = useState({ width: 1, height: 1 });
  const [panning, setPanning] = useState<{
    x: number;
    y: number;
    panX: number;
    panY: number;
  } | null>(null);
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
    const panGesture =
      tool === "hand" || (native instanceof MouseEvent && native.button === 1);
    if (panGesture) {
      setPanning({ x: pointer.x, y: pointer.y, panX: view.panX, panY: view.panY });
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
        onMouseMove={onStageMove}
        onTouchMove={onStageMove}
        onMouseUp={() => setPanning(null)}
        onTouchEnd={() => setPanning(null)}
        onWheel={onWheel}
        className={tool === "hand" ? "cursor-grab" : tool === "polygon" ? "cursor-crosshair" : ""}
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
}

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
