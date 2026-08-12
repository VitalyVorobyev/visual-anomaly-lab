/** Pure editing operations for annotation documents.
 *
 * Keeping history independent of Konva makes undo/redo deterministic and testable. The
 * scene is controlled: it renders this document and emits another document; it never
 * becomes a second, hidden source of annotation truth.
 */

import type {
  AnnotationDocument,
  AnnotationPoint,
  AnnotationShape,
  PolygonShape,
} from "./client";

export interface AnnotationHistory {
  past: AnnotationDocument[];
  present: AnnotationDocument;
  future: AnnotationDocument[];
}

export type HistoryAction =
  | { type: "replace"; document: AnnotationDocument }
  | { type: "commit"; document: AnnotationDocument }
  | { type: "undo" }
  | { type: "redo" };

export function createHistory(document: AnnotationDocument): AnnotationHistory {
  return { past: [], present: document, future: [] };
}

export function historyReducer(
  state: AnnotationHistory,
  action: HistoryAction,
): AnnotationHistory {
  if (action.type === "replace") return createHistory(action.document);
  if (action.type === "commit") {
    if (canonical(state.present) === canonical(action.document)) return state;
    return {
      past: [...state.past, state.present].slice(-100),
      present: action.document,
      future: [],
    };
  }
  if (action.type === "undo") {
    const previous = state.past.at(-1);
    if (!previous) return state;
    return {
      past: state.past.slice(0, -1),
      present: previous,
      future: [state.present, ...state.future],
    };
  }
  const next = state.future[0];
  if (!next) return state;
  return {
    past: [...state.past, state.present],
    present: next,
    future: state.future.slice(1),
  };
}

export function withShape(
  document: AnnotationDocument,
  shape: AnnotationShape,
): AnnotationDocument {
  return { ...document, shapes: [...document.shapes, shape] };
}

export function withoutShape(document: AnnotationDocument, shapeId: string): AnnotationDocument {
  return { ...document, shapes: document.shapes.filter((shape) => shape.id !== shapeId) };
}

export function replaceShape(
  document: AnnotationDocument,
  shapeId: string,
  replacements: AnnotationShape[],
): AnnotationDocument {
  const index = document.shapes.findIndex((shape) => shape.id === shapeId);
  if (index < 0) return document;
  return {
    ...document,
    shapes: [
      ...document.shapes.slice(0, index),
      ...replacements,
      ...document.shapes.slice(index + 1),
    ],
  };
}

export function withPolygonPoint(
  document: AnnotationDocument,
  shapeId: string,
  pointIndex: number,
  point: AnnotationPoint,
): AnnotationDocument {
  return {
    ...document,
    shapes: document.shapes.map((shape) => {
      if (shape.id !== shapeId || shape.kind !== "polygon") return shape;
      const points = [...shape.points];
      points[pointIndex] = point;
      return { ...shape, points } satisfies PolygonShape;
    }),
  };
}

export function isPolygon(shape: AnnotationShape): shape is PolygonShape {
  return shape.kind === "polygon";
}

export function canonical(document: AnnotationDocument): string {
  return JSON.stringify(document);
}

export function nextShapeId(): string {
  return globalThis.crypto.randomUUID();
}
