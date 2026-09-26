/**
 * Everything the reader does *to* the document: draw, reshape, move, erase, delete, relabel,
 * trace, undo and redo — and the selection and open polygon those gestures act on.
 *
 * Every edit becomes one `commit` on the draft session's history, so one gesture is one undo
 * step. The Konva scene never holds a second copy of the truth; it reports gestures in source
 * pixels and this hook turns them into documents.
 *
 * **Every edit is built from `latest()`, never from a render's `history.present`.** Most edits
 * are synchronous and the two are the same thing, but painting and tracing await a decode,
 * and a document captured before the await is a document from the past by the time it is
 * committed. See `applyStroke`.
 */

import { useCallback, useRef, useState } from "react";

import {
  type AnnotationHistory,
  type HistoryAction,
  nextShapeId,
  replaceShape,
  translateShape,
  withShapePoint,
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
  AnnotationDocument,
  AnnotationLabel,
  AnnotationPoint,
  AnnotationShape,
  BitmapShape,
  BoxShape,
  PolygonShape,
} from "../../api/client";
import type { EditorTool } from "../../components/annotation/AnnotationCanvas";
import type { BoxRect } from "../../components/annotation/tools";
import type { Flash } from "./useFlashMessage";

/** How often a stroke is repainted because the document moved under it before it gives up. */
const STROKE_ATTEMPTS = 3;

interface StrokeRequest {
  points: AnnotationPoint[];
  erase: boolean;
  size: number;
  labelKey: string;
}

export function useDocumentCommands({
  history,
  dispatch,
  latest,
  labels,
  tool,
  setTool,
  brushSize,
  flash,
}: {
  history: AnnotationHistory;
  dispatch: (action: HistoryAction) => void;
  /** The document every dispatched edit has produced so far, rendered or not. */
  latest: () => AnnotationDocument;
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
  const [selectedId, setSelectedState] = useState<string | null>(null);
  const [pendingPoints, setPendingPoints] = useState<AnnotationPoint[]>([]);
  const [traceError, setTraceError] = useState<string | null>(null);
  const [tracing, setTracing] = useState(false);
  const selected = history.present.shapes.find((shape) => shape.id === selectedId) ?? null;

  // The selection, readable by a stroke that runs after the render that set it. Written
  // eagerly by `setSelectedId` for the same reason `latest()` exists: a brush stroke queued
  // behind the one that minted a region has to extend that region, not start another.
  const selection = useRef(selectedId);
  const renderedSelection = useRef(selectedId);
  if (renderedSelection.current !== selectedId) {
    renderedSelection.current = selectedId;
    selection.current = selectedId;
  }
  const setSelectedId = useCallback((id: string | null) => {
    selection.current = id;
    setSelectedState(id);
  }, []);

  const commit = useCallback(
    (document: AnnotationDocument) => dispatch({ type: "commit", document }),
    [dispatch],
  );

  const finishPolygon = useCallback(() => {
    if (pendingPoints.length < 3) return;
    const polygon: PolygonShape = {
      id: nextShapeId(),
      label_key: labelKey,
      kind: "polygon",
      operation,
      points: pendingPoints,
    };
    commit(withShape(latest(), polygon));
    setPendingPoints([]);
    setSelectedId(polygon.id);
    // The tool stays where it is. Most parts carry more than one defect, and dropping back to
    // Select after every ring meant pressing P again for each of them.
  }, [commit, labelKey, latest, operation, pendingPoints, setSelectedId]);

  /** A box drawn with the box tool: minted, selected, and the tool stays in hand. */
  const addBox = useCallback(
    (rect: BoxRect) => {
      const box: BoxShape = { id: nextShapeId(), label_key: labelKey, kind: "box", operation, ...rect };
      commit(withShape(latest(), box));
      setSelectedId(box.id);
    },
    [commit, labelKey, latest, operation, setSelectedId],
  );

  /**
   * The class for new regions, by its position in class order — what the digit keys pick.
   * A position past the last class picks nothing, and says so.
   */
  const pickClass = useCallback(
    (position: number) => {
      const label = labels[position];
      if (!label) {
        flash(`There is no class ${position + 1}.`);
        return;
      }
      setLabelKey(label.key);
      flash(`New regions: ${label.name}`);
    },
    [flash, labels],
  );

  const addPendingPoint = useCallback((point: AnnotationPoint) => {
    setPendingPoints((points) => [...points, point]);
  }, []);

  const moveShape = useCallback(
    (shapeId: string, dx: number, dy: number) => {
      commit(translateShape(latest(), shapeId, dx, dy));
    },
    [commit, latest],
  );

  const movePoint = useCallback(
    (shapeId: string, pointIndex: number, point: AnnotationPoint) => {
      commit(withShapePoint(latest(), shapeId, pointIndex, point));
    },
    [commit, latest],
  );

  /**
   * Paint one stroke into whatever document is current *when it runs*.
   *
   * Returns `false` when the document changed while the paint was being decoded, so the
   * caller paints again against the new one rather than committing over it.
   */
  const paint = useCallback(
    async ({ points, erase, size, labelKey: key }: StrokeRequest): Promise<boolean> => {
      const base = latest();
      const selectedShape = base.shapes.find((shape) => shape.id === selection.current) ?? null;
      const geometry = {
        points,
        size,
        imageWidth: base.image_width,
        imageHeight: base.image_height,
      };
      const target = selectedShape?.kind === "bitmap" ? selectedShape : null;

      if (erase) {
        // The selection scopes the eraser exactly as it scopes the brush; without one, the
        // pointer itself is the scope.
        const targets = target
          ? [target]
          : strokeTargets(
              base.shapes,
              strokeBounds(points, size, geometry.imageWidth, geometry.imageHeight),
            );
        if (targets.length === 0) {
          flash(
            selectedShape?.kind === "polygon" || selectedShape?.kind === "box"
              ? `The eraser takes paint off painted regions. Reshape this ${selectedShape.kind} by its vertices, or delete it.`
              : "Nothing painted here to erase.",
          );
          return true;
        }
        const painted = await Promise.all(
          targets.map((shape) => paintStroke(shape, { ...geometry, erase: true })),
        );
        if (latest() !== base) return false;
        let next = base;
        let removed = 0;
        targets.forEach((shape, index) => {
          const result = painted[index];
          if (result === undefined) return;
          if (result === null) {
            next = withoutShape(next, shape.id);
            removed += 1;
            if (shape.id === selection.current) setSelectedId(null);
            return;
          }
          next = replaceShape(next, shape.id, [result]);
        });
        commit(next);
        if (removed > 0) flash(removed === 1 ? "Region erased" : `${removed} regions erased`);
        return true;
      }

      if (target) {
        const painted = await paintStroke(target, { ...geometry, erase: false });
        if (latest() !== base) return false;
        // A brush cannot empty a region, but `paintStroke` promises `null` for an empty result
        // and honouring it here keeps the one contract rather than two.
        if (painted === null) {
          commit(withoutShape(base, target.id));
          setSelectedId(null);
          return true;
        }
        commit(replaceShape(base, target.id, [painted]));
        return true;
      }
      const shape = bitmapStroke({ ...geometry, labelKey: key, operation });
      if (!shape) return true;
      commit(withShape(base, shape));
      setSelectedId(shape.id);
      return true;
    },
    [commit, flash, latest, operation, setSelectedId],
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
   *
   * **Strokes run one at a time, each against the document the previous one left.** Painting
   * into an existing region awaits a PNG decode, and this used to commit a document built from
   * the `history.present` its closure had captured before the await. Anything dispatched in
   * between was overwritten: two strokes in flight at once — Space held on the focused canvas
   * auto-repeats, and every repeat is a stroke — both painted from the same region and the
   * last to finish won, dropping the others; an undo pressed while a stroke decoded came
   * straight back when the stroke committed on top of the pre-undo document. Now the strokes
   * are queued, each reads `latest()` when it starts, and one that finds the document moved
   * when its decode returns is painted again rather than committed over the move.
   *
   * The tool, size and class are the gesture's own, captured when it was made; the document
   * and the selection are read when it is applied.
   */
  const strokes = useRef<Promise<void>>(Promise.resolve());
  const applyStroke = useCallback(
    (points: AnnotationPoint[]): Promise<void> => {
      const request: StrokeRequest = {
        points,
        erase: tool === "eraser",
        size: brushSize,
        labelKey,
      };
      const run = strokes.current.then(async () => {
        for (let attempt = 0; attempt < STROKE_ATTEMPTS; attempt += 1) {
          if (await paint(request)) return;
        }
        flash("The document kept changing under that stroke; paint it again.");
      });
      // The queue must outlive a stroke that fails to decode, or every later one would wait on
      // a rejection forever.
      strokes.current = run.catch(() => undefined);
      return run;
    },
    [brushSize, flash, labelKey, paint, tool],
  );

  const removeSelected = useCallback(() => {
    if (!selectedId) return;
    commit(withoutShape(latest(), selectedId));
    setSelectedId(null);
  }, [commit, latest, selectedId, setSelectedId]);

  const updateSelected = (patch: Partial<Pick<AnnotationShape, "label_key" | "operation">>) => {
    if (!selectedId) return;
    const document = latest();
    commit({
      ...document,
      shapes: document.shapes.map((shape) =>
        shape.id === selectedId ? { ...shape, ...patch } : shape,
      ),
    });
  };

  const traceSelected = async () => {
    if (!selected || selected.kind !== "bitmap") return;
    setTracing(true);
    setTraceError(null);
    try {
      const polygons = await traceBitmapShape(selected);
      if (polygons.length === 0) throw new Error("No contour could be derived from this region.");
      // Replaced in the document as it is now. If the region was edited while it was being
      // traced, `replaceShape` finds it by id and the newer edit is what gets replaced — the
      // reader asked for *this region* as contours, and it is still selected.
      commit(replaceShape(latest(), selected.id, polygons));
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
        const document = latest();
        commit({ ...document, shapes: [...document.shapes, ...polygons] });
        setSelectedId(polygons[0]?.id ?? null);
      } else {
        commit(withShape(latest(), shape));
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
    setPendingPoints: setPendingPoints,
    addPendingPoint,
    finishPolygon,
    addBox,
    pickClass,
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
