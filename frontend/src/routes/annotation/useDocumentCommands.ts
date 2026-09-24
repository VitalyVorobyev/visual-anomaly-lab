/**
 * Everything the reader does *to* the document: draw, reshape, move, erase, delete, relabel,
 * trace, undo and redo — and the selection and open polygon those gestures act on.
 *
 * Every edit becomes one `commit` on the draft session's history, so one gesture is one undo
 * step. The Konva scene never holds a second copy of the truth; it reports gestures in source
 * pixels and this hook turns them into documents.
 */

import type { Dispatch } from "react";
import { useCallback, useState } from "react";

import {
  type AnnotationHistory,
  type HistoryAction,
  nextShapeId,
  replaceShape,
  translateShape,
  withPolygonPoint,
  withShape,
  withoutShape,
} from "../../api/annotationState";
import {
  bitmapStroke,
  paintStroke,
  strokeBounds,
  strokeTargets,
  traceBitmapShape,
} from "../../api/annotationBitmap";
import type {
  AnnotationLabel,
  AnnotationPoint,
  AnnotationShape,
  BitmapShape,
  PolygonShape,
} from "../../api/client";
import type { EditorTool } from "../../components/annotation/AnnotationCanvas";
import type { Flash } from "./useFlashMessage";

export function useDocumentCommands({
  history,
  dispatch,
  labels,
  tool,
  setTool,
  brushSize,
  flash,
}: {
  history: AnnotationHistory;
  dispatch: Dispatch<HistoryAction>;
  labels: AnnotationLabel[];
  tool: EditorTool;
  setTool: (tool: EditorTool) => void;
  brushSize: number;
  flash: Flash;
}) {
  // Every shape is minted as an addition. A cut is drawn and then turned into one in the
  // Selection panel, on the shape that drawing it already selected — which is the same
  // document, one control, and no state that has to be remembered between two strokes.
  const operation = "add" as const;
  const [labelKey, setLabelKey] = useState(labels[0]?.key ?? "defect");
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [pendingPoints, setPendingPoints] = useState<AnnotationPoint[]>([]);
  const [traceError, setTraceError] = useState<string | null>(null);
  const [tracing, setTracing] = useState(false);
  const selected = history.present.shapes.find((shape) => shape.id === selectedId) ?? null;

  const finishPolygon = useCallback(() => {
    if (pendingPoints.length < 3) return;
    const polygon: PolygonShape = {
      id: nextShapeId(),
      label_key: labelKey,
      kind: "polygon",
      operation,
      points: pendingPoints,
    };
    dispatch({ type: "commit", document: withShape(history.present, polygon) });
    setPendingPoints([]);
    setSelectedId(polygon.id);
    // The tool stays where it is. Most parts carry more than one defect, and dropping back to
    // Select after every ring meant pressing P again for each of them.
  }, [dispatch, history.present, labelKey, operation, pendingPoints]);

  const addPendingPoint = useCallback((point: AnnotationPoint) => {
    setPendingPoints((points) => [...points, point]);
  }, []);

  const moveShape = useCallback(
    (shapeId: string, dx: number, dy: number) => {
      dispatch({ type: "commit", document: translateShape(history.present, shapeId, dx, dy) });
    },
    [dispatch, history.present],
  );

  const movePoint = useCallback(
    (shapeId: string, pointIndex: number, point: AnnotationPoint) => {
      dispatch({
        type: "commit",
        document: withPolygonPoint(history.present, shapeId, pointIndex, point),
      });
    },
    [dispatch, history.present],
  );

  /**
   * One brush or eraser gesture.
   *
   * The rule, stated in the inspector so it is never a guess: **a stroke extends the selected
   * region**, and with nothing selected the brush starts one. Every gesture used to mint a new
   * shape, so a defect painted in three strokes was three regions.
   *
   * **The eraser never creates.** It takes paint off the selected region, or — with nothing
   * selected — off every painted region it passes over. It used to append a `subtract` layer
   * instead, which is a region: the tool for removing things added one, and said so in the
   * region list. Cutting a hole through a *polygon* is still possible, but by drawing the
   * region and turning it into a Subtract in the Selection panel, rather than as the
   * eraser's side effect.
   */
  const applyStroke = useCallback(
    async (points: AnnotationPoint[]) => {
      const geometry = {
        points,
        size: brushSize,
        imageWidth: history.present.image_width,
        imageHeight: history.present.image_height,
      };
      const target = selected?.kind === "bitmap" ? selected : null;

      if (tool === "eraser") {
        // The selection scopes the eraser exactly as it scopes the brush; without one, the
        // pointer itself is the scope.
        const targets = target
          ? [target]
          : strokeTargets(
              history.present.shapes,
              strokeBounds(points, brushSize, geometry.imageWidth, geometry.imageHeight),
            );
        if (targets.length === 0) {
          flash(
            selected?.kind === "polygon"
              ? "The eraser takes paint off painted regions. Reshape this polygon by its vertices, or delete it."
              : "Nothing painted here to erase.",
          );
          return;
        }
        const painted = await Promise.all(
          targets.map((shape) => paintStroke(shape, { ...geometry, erase: true })),
        );
        let next = history.present;
        let removed = 0;
        targets.forEach((shape, index) => {
          const result = painted[index];
          if (result === undefined) return;
          if (result === null) {
            next = withoutShape(next, shape.id);
            removed += 1;
            if (shape.id === selectedId) setSelectedId(null);
            return;
          }
          next = replaceShape(next, shape.id, [result]);
        });
        dispatch({ type: "commit", document: next });
        if (removed > 0) flash(removed === 1 ? "Region erased" : `${removed} regions erased`);
        return;
      }

      if (target) {
        const painted = await paintStroke(target, { ...geometry, erase: false });
        // A brush cannot empty a region, but `paintStroke` promises `null` for an empty result
        // and honouring it here keeps the one contract rather than two.
        if (painted === null) {
          dispatch({ type: "commit", document: withoutShape(history.present, target.id) });
          setSelectedId(null);
          return;
        }
        dispatch({
          type: "commit",
          document: replaceShape(history.present, target.id, [painted]),
        });
        return;
      }
      const shape = bitmapStroke({ ...geometry, labelKey, operation });
      if (!shape) return;
      dispatch({ type: "commit", document: withShape(history.present, shape) });
      setSelectedId(shape.id);
    },
    [brushSize, dispatch, flash, history.present, labelKey, operation, selected, selectedId, tool],
  );

  const removeSelected = useCallback(() => {
    if (!selectedId) return;
    dispatch({
      type: "commit",
      document: withoutShape(history.present, selectedId),
    });
    setSelectedId(null);
  }, [dispatch, history.present, selectedId]);

  const updateSelected = (patch: Partial<Pick<AnnotationShape, "label_key" | "operation">>) => {
    if (!selectedId) return;
    dispatch({
      type: "commit",
      document: {
        ...history.present,
        shapes: history.present.shapes.map((shape) =>
          shape.id === selectedId ? { ...shape, ...patch } : shape,
        ),
      },
    });
  };

  const traceSelected = async () => {
    if (!selected || selected.kind !== "bitmap") return;
    setTracing(true);
    setTraceError(null);
    try {
      const polygons = await traceBitmapShape(selected);
      if (polygons.length === 0) throw new Error("No contour could be derived from this region.");
      dispatch({
        type: "commit",
        document: replaceShape(history.present, selected.id, polygons),
      });
      setSelectedId(polygons[0]?.id ?? null);
      setTool("select");
      flash(polygons.length === 1 ? "Editable contour created" : `${polygons.length} editable contours created`);
    } catch (error) {
      setTraceError(error instanceof Error ? error.message : "Contour tracing failed.");
    } finally {
      setTracing(false);
    }
  };

  /**
   * Put a MobileSAM suggestion into the document, as the compact mask or traced into editable
   * contours. Resolves `false` when the conversion failed; the reason is in `traceError`.
   */
  const acceptSuggestion = async (shape: BitmapShape, asContour: boolean): Promise<boolean> => {
    try {
      if (asContour) {
        const polygons = await traceBitmapShape(shape);
        if (polygons.length === 0) throw new Error("No editable contour could be derived.");
        dispatch({
          type: "commit",
          document: {
            ...history.present,
            shapes: [...history.present.shapes, ...polygons],
          },
        });
        setSelectedId(polygons[0]?.id ?? null);
      } else {
        dispatch({ type: "commit", document: withShape(history.present, shape) });
        setSelectedId(shape.id);
      }
      setTool("select");
      return true;
    } catch (error) {
      setTraceError(error instanceof Error ? error.message : "Suggestion conversion failed.");
      return false;
    }
  };

  return {
    labelKey,
    setLabelKey,
    operation,
    selectedId,
    setSelectedId,
    selected,
    pendingPoints,
    setPendingPoints,
    addPendingPoint,
    finishPolygon,
    moveShape,
    movePoint,
    applyStroke,
    removeSelected,
    updateSelected,
    traceSelected,
    tracing,
    traceError,
    acceptSuggestion,
    canUndo: history.past.length > 0,
    canRedo: history.future.length > 0,
    undo: useCallback(() => dispatch({ type: "undo" }), [dispatch]),
    redo: useCallback(() => dispatch({ type: "redo" }), [dispatch]),
  };
}

export type DocumentCommands = ReturnType<typeof useDocumentCommands>;
