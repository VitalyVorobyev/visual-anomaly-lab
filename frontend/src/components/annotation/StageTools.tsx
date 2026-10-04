/**
 * What the reader does on the stage: the tool surface that turns a press into the active
 * tool's gesture, the vertex and corner editors of the selected region, and the drafts that
 * draw a gesture while it is made.
 *
 * Points cross into the document's coordinates here and nowhere else (`stageFrame`): a press,
 * a drag and an editor's vertices come back through `fromStage`, and every draft goes out
 * through `toStage`.
 */

import { useMemo } from "react";
import {
  ContourEditor,
  DraftShape,
  PointSet,
  RectRoiEditor,
  StageSurface,
  overlayRole,
  useStage,
  useStageHitTest,
  type PointSetItem,
  type StageDrag,
  type StagePress,
} from "@vitavision/stage2d";

import type { AnnotationPoint, AssistBox, AssistPoint } from "../../api/client";
import { moveDrag, type LiveEdit } from "./liveEdit";
import { type LiveStore, useLive } from "./liveStore";
import { REGION_LAYER, topmostShape } from "./shapePick";
import {
  flatArrayToStage,
  flatToStage,
  frameBounds,
  fromStage,
  rectFromStage,
  rectToStage,
  toStage,
} from "./stageFrame";
import type { VectorShape } from "./sceneRuns";
import { TOOLS, type BoxRect, type EditorTool, type ToolContext, type ToolEffect } from "./tools";

const ERASER = "var(--defect)";
const SELECTION = overlayRole("selection");

/**
 * The stage's one press target, for the tool in hand.
 *
 * Under a drawing tool every primary press is the tool's: `ToolModule.down` on the press, and
 * the gesture it starts runs `move` and `up` as the drag goes. The surface spans the whole
 * viewport, so a vertex can be put exactly on the frame's border by pressing just outside it;
 * every point is clamped to the frame. Under Select a press on a region selects it and a drag
 * moves it; a press on bare image is declined, and the stage pans with it (and deselects on a
 * click, through its background click).
 */
export function ToolSurface({
  tool,
  store,
  pendingPoints,
  assistMode,
  order,
  emit,
  onSelect,
  onMoveShape,
  onEdit,
}: {
  tool: EditorTool;
  store: LiveStore;
  pendingPoints: AnnotationPoint[];
  assistMode: "point" | "box";
  /** Each region's position in the document, for "the one on top". */
  order: ReadonlyMap<string, number>;
  emit: (effects: ToolEffect[]) => void;
  onSelect: (shapeId: string | null) => void;
  onMoveShape: (shapeId: string, dx: number, dy: number) => void;
  onEdit: (edit: LiveEdit | null) => void;
}) {
  const stage = useStage();
  const { hitTestAll } = useStageHitTest();
  const toolModule = TOOLS[tool];
  const context: ToolContext = { pendingPoints, scale: stage.view.scale, assistMode };

  // Select: regions answer the press themselves, and bare image is the view's.
  if (toolModule.pansWithPrimary) {
    const select = (press: StagePress): StageDrag | undefined => {
      const hits = hitTestAll(press.point, press.radius).filter((hit) =>
        hit.layerId.startsWith(REGION_LAYER),
      );
      const id = topmostShape(hits, order);
      if (id === null) return undefined;
      onSelect(id);
      return moveDrag(id, press.point, press.client, onEdit, onMoveShape);
    };
    return <StageSurface cursor={stage.panning ? "grabbing" : toolModule.cursor} onPress={select} />;
  }

  const draw = (press: StagePress): StageDrag => {
    const point = fromStage(press.point);
    store.set({ keyboardPoint: point });
    const step = toolModule.down(context, point, { shiftKey: press.shiftKey });
    if (!step.gesture) {
      // A click — a vertex, a prompt. A finger's waits for the lift, once it is plain that the
      // touch was a tap and not the start of a pan.
      if (press.touch) {
        return {
          onEnd: (_point, _event, moved) => {
            if (!moved) emit(step.effects);
          },
        };
      }
      emit(step.effects);
      return {};
    }
    emit(step.effects);
    store.set({ gesture: step.gesture, revision: store.get().revision + 1 });
    return {
      claimsTouch: true,
      onMove: (next) => {
        const live = store.get();
        if (!live.gesture) return;
        const moved = toolModule.move(context, live.gesture, fromStage(next));
        store.set({ gesture: moved.gesture, revision: live.revision + 1 });
        emit(moved.effects);
      },
      onEnd: () => {
        const live = store.get();
        if (!live.gesture) return;
        store.set({ gesture: null, revision: live.revision + 1 });
        emit(toolModule.up(context, live.gesture));
      },
      // An interrupted gesture is dropped, not committed: nobody finished drawing it.
      onCancel: () => store.set({ gesture: null, revision: store.get().revision + 1 }),
    };
  };
  const brushing = tool === "brush" || tool === "eraser";
  return (
    <StageSurface
      extent="viewport"
      // The brush's own footprint is the cursor; a crosshair on top of it would be a second
      // pointer claiming a different size.
      cursor={stage.panning ? "grabbing" : brushing ? "none" : toolModule.cursor}
      onPress={draw}
      onDoubleClick={() => emit(toolModule.doubleClick(context))}
    />
  );
}

/**
 * The selected region's handles, under Select: a polygon's vertices (drag one, or Insert and
 * Delete on a focused one, or double-click the outline for a new one) and a box's corners and
 * edges. Each edit is previewed through `onEdit` and committed once, when it ends.
 */
export function ShapeEditors({
  shape,
  width,
  height,
  onEdit,
  latestEdit,
  onReshapePolygon,
  onReshapeBox,
}: {
  shape: VectorShape;
  width: number;
  height: number;
  onEdit: (edit: LiveEdit | null) => void;
  /** The edit as last reported, which an editor's commit may read before it has rendered. */
  latestEdit: () => LiveEdit | null;
  onReshapePolygon: (shapeId: string, points: AnnotationPoint[]) => void;
  onReshapeBox: (shapeId: string, rect: BoxRect) => void;
}) {
  // Exactly on the border is legal in the area convention: the frame is [0, W] x [0, H].
  const bounds = frameBounds(width, height);
  const id = shape.id;
  if (shape.kind === "polygon") {
    return (
      <ContourEditor
        key={id}
        points={shape.points.map(toStage)}
        editable
        label="Region"
        stroke={SELECTION}
        bounds={bounds}
        onChange={(points) => onEdit({ kind: "polygon", id, points: points.map(fromStage) })}
        onCommit={() => {
          const edit = latestEdit();
          onEdit(null);
          if (edit?.kind === "polygon" && edit.id === id) onReshapePolygon(id, edit.points);
        }}
      />
    );
  }
  return (
    <RectRoiEditor
      key={id}
      value={rectToStage(shape)}
      bounds={bounds}
      minSize={1}
      label="Region"
      onValueChange={(rect) => onEdit({ kind: "box", id, rect: rectFromStage(rect) })}
      onCommit={(rect) => {
        onEdit(null);
        onReshapeBox(id, rectFromStage(rect));
      }}
    />
  );
}

/**
 * What the reader is doing *now*: the open polygon, the box being dragged, the assist box and
 * prompts, the brush trail and footprint, and the keyboard cursor.
 *
 * It subscribes to the live store, so a pointer move re-renders this and nothing else, and it
 * takes no pointer events. All of it stays on screen while the regions are hidden: the overlay
 * is the committed document, and hiding it must not hide the stroke being made.
 */
export function LiveDrafts({
  store,
  tool,
  editable,
  brushSize,
  pendingPoints,
  assistMode,
  assistPoints,
  assistBox,
}: {
  store: LiveStore;
  tool: EditorTool;
  editable: boolean;
  brushSize: number;
  pendingPoints: AnnotationPoint[];
  assistMode: "point" | "box";
  assistPoints: AssistPoint[];
  assistBox: AssistBox | null;
}) {
  const stage = useStage();
  const gesture = useLive(store, (state) => state.gesture);
  // The trail is appended in place, so its identity does not change as it grows; the
  // revision is what says it did.
  useLive(store, (state) => state.revision);
  const keyboardPoint = useLive(store, (state) => state.keyboardPoint);
  const keyboardFocused = useLive(store, (state) => state.keyboardFocused);
  const brushing = editable && (tool === "brush" || tool === "eraser");
  const drawingRing = editable && pendingPoints.length > 0;
  // Read only while something follows the pointer, so a hover re-renders nothing otherwise.
  const pointer = useLive(store, (state) => (brushing || drawingRing ? state.pointer : null));
  const tint = tool === "eraser" ? ERASER : undefined;
  // "Click here to close" is something the ring says, before the click.
  const closing =
    pointer !== null &&
    TOOLS[tool].snaps({ pendingPoints, scale: stage.view.scale, assistMode }, pointer);
  const assistItems = useMemo<PointSetItem[]>(
    () =>
      assistPoints.map((point, index) => ({
        id: `assist-${index}`,
        ...toStage(point),
        // Include and exclude differ in shape as well as colour.
        kind: point.kind === "positive" ? "plus" : "cross",
        color: point.kind === "positive" ? "var(--normal)" : "var(--defect)",
      })),
    [assistPoints],
  );

  return (
    <>
      {pendingPoints.length > 0 && (
        <DraftShape
          shape={{
            kind: "polygon",
            points: flatToStage(pendingPoints),
            cursor: pointer === null ? undefined : toStage(pointer),
            closing,
          }}
        />
      )}
      {tool === "box" && gesture?.kind === "box" && gesture.end && (
        <DraftShape shape={{ kind: "rect", from: toStage(gesture.start), to: toStage(gesture.end) }} />
      )}
      {assistBox && (
        <DraftShape
          shape={{
            kind: "rect",
            from: toStage({ x: assistBox.x0, y: assistBox.y0 }),
            to: toStage({ x: assistBox.x1, y: assistBox.y1 }),
          }}
        />
      )}
      {assistItems.length > 0 && <PointSet items={assistItems} label="Assist prompts" />}
      {gesture?.kind === "stroke" && (
        // The trail at the brush's true width in source pixels: the pixels it will paint.
        <DraftShape
          shape={{ kind: "stroke", points: flatArrayToStage(gesture.flat), width: brushSize }}
          stroke={tint}
        />
      )}
      {brushing && pointer && (
        // The disc the stroke stamps, of the brush's *diameter* in source pixels; the OS cursor
        // is hidden while it shows, so there is one pointer, not two.
        <DraftShape shape={{ kind: "brush", ...toStage(pointer), diameter: brushSize }} stroke={tint} />
      )}
      {keyboardFocused && tool !== "select" && (
        <DraftShape shape={{ kind: "point", ...toStage(keyboardPoint) }} />
      )}
    </>
  );
}
