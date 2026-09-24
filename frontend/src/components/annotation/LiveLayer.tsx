/**
 * The live layer: what the reader is doing *now* — the brush trail, the open polygon, the
 * assist prompts and the keyboard cursor.
 *
 * It subscribes to the live store, so a pointer move re-renders this layer and nothing else.
 * It never listens for hits: every gesture is read off the stage, and a live shape that took
 * the pointer would swallow the next click.
 *
 * All of it stays on screen while the regions are hidden: the overlay is the committed
 * document, and hiding it must not hide the stroke being made.
 */

import type Konva from "konva";
import { useLayoutEffect, useRef } from "react";
import { Circle, Group, Layer, Line, Rect } from "react-konva";

import { boxBetween } from "../../api/annotationState";
import type { AnnotationPoint, AssistBox, AssistPoint } from "../../api/client";
import { type LiveStore, useLive } from "./liveStore";
import { withAlpha, type ScenePalette } from "./scenePalette";
import type { EditorTool } from "./tools";

export function LiveLayer({
  store,
  palette,
  tool,
  brushSize,
  pendingPoints,
  assistPoints,
  assistBox,
  brushCursor,
  originX,
  originY,
  scale,
}: {
  store: LiveStore;
  palette: ScenePalette;
  tool: EditorTool;
  brushSize: number;
  pendingPoints: AnnotationPoint[];
  assistPoints: AssistPoint[];
  assistBox: AssistBox | null;
  /** Draw the brush's footprint at the pointer: a brush or eraser in an editable pane. */
  brushCursor: boolean;
  originX: number;
  originY: number;
  scale: number;
}) {
  const gesture = useLive(store, (state) => state.gesture);
  const revision = useLive(store, (state) => state.revision);
  const snapReady = useLive(store, (state) => state.snapReady);
  const keyboardPoint = useLive(store, (state) => state.keyboardPoint);
  const keyboardFocused = useLive(store, (state) => state.keyboardFocused);
  const pointer = useLive(store, (state) => (brushCursor ? state.pointer : null));
  const trail = gesture?.kind === "stroke" ? gesture.flat : null;
  // The box tool's drag; the assist box is drawn from its own prop, which outlives the drag.
  const drawnBox =
    tool === "box" && gesture?.kind === "box" && gesture.end
      ? boxBetween(gesture.start, gesture.end)
      : null;

  // The trail's array is appended in place, so its identity does not change as it grows and
  // react-konva would never hand the node the new points. Handing them over here, keyed on
  // the store's revision, costs nothing per sample — no copy of the trail is ever made.
  const trailRef = useRef<Konva.Line>(null);
  useLayoutEffect(() => {
    const line = trailRef.current;
    if (!line || !trail) return;
    line.points(trail);
    line.getLayer()?.batchDraw();
  }, [trail, revision]);

  return (
    <Layer listening={false}>
      <Group x={originX} y={originY} scaleX={scale} scaleY={scale}>
        {pendingPoints.length > 0 && (
          <Group>
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
                // The first vertex swells and fills while the pointer is over it, so "click
                // here to close" is something the ring says rather than something a button
                // elsewhere claims.
                radius={(index === 0 ? (snapReady ? 8 : 5) : 3.5) / scale}
                fill={index === 0 && snapReady ? palette.signal : palette.frame}
                stroke={palette.signal}
                strokeWidth={1.5 / scale}
              />
            ))}
          </Group>
        )}
        {trail && (
          <Line
            ref={trailRef}
            points={trail}
            stroke={tool === "eraser" ? palette.cut : palette.signal}
            /* True size in source pixels, floored at one *screen* pixel. A one-pixel brush at
               fit zoom is otherwise a fraction of a pixel wide, so the gesture would leave no
               visible trail while it was being made. The `n / scale` idiom is how every
               outline in this scene is drawn. */
            strokeWidth={Math.max(brushSize, 1 / scale)}
            lineCap="round"
            lineJoin="round"
            opacity={0.72}
          />
        )}
        {drawnBox && (
          <Rect
            x={drawnBox.x}
            y={drawnBox.y}
            width={drawnBox.width}
            height={drawnBox.height}
            stroke={palette.signal}
            strokeWidth={2 / scale}
            dash={[6 / scale, 4 / scale]}
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
          />
        )}
        {assistPoints.map((point, index) => (
          <Group key={`assist-point-${index}`}>
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
          <Group>
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
        {pointer && (
          // The footprint the stroke will paint: a disc of the brush's *diameter* in source
          // pixels, the same disc `rasterizeStroke` stamps. Two rings, dark under light, so it
          // reads over a bright specular surface and a dark field alike; the OS cursor is
          // hidden while it is shown, so there is one pointer, not two.
          <Group x={pointer.x} y={pointer.y}>
            <Circle
              radius={Math.max(brushSize / 2, 2 / scale)}
              stroke={palette.canvas}
              strokeWidth={3 / scale}
              opacity={0.7}
            />
            <Circle
              radius={Math.max(brushSize / 2, 2 / scale)}
              stroke={tool === "eraser" ? palette.cut : palette.signal}
              strokeWidth={1.25 / scale}
            />
            <Circle radius={1 / scale} fill={tool === "eraser" ? palette.cut : palette.signal} />
          </Group>
        )}
      </Group>
    </Layer>
  );
}
