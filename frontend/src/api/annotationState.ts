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

/**
 * Move one region without deforming it.
 *
 * The offset is clamped once, against the shape's own extent, rather than per coordinate.
 * Clamping each point independently would let a polygon pushed against an edge collapse
 * against it, one vertex at a time, and a document that had been merely dragged too far would
 * come back a different shape.
 *
 * Moving a region is not a nicety here: a copy taken from another channel of the same part
 * lands a few pixels out, because the exposures are milliseconds apart on a moving line.
 */
export function translateShape(
  document: AnnotationDocument,
  shapeId: string,
  dx: number,
  dy: number,
): AnnotationDocument {
  const shape = document.shapes.find((candidate) => candidate.id === shapeId);
  if (!shape) return document;

  const extent =
    shape.kind === "polygon"
      ? {
          minX: Math.min(...shape.points.map((point) => point.x)),
          minY: Math.min(...shape.points.map((point) => point.y)),
          maxX: Math.max(...shape.points.map((point) => point.x)),
          maxY: Math.max(...shape.points.map((point) => point.y)),
        }
      : {
          minX: shape.x,
          minY: shape.y,
          maxX: shape.x + shape.width,
          maxY: shape.y + shape.height,
        };
  const offsetX = clamp(dx, -extent.minX, document.image_width - extent.maxX);
  const offsetY = clamp(dy, -extent.minY, document.image_height - extent.maxY);
  if (offsetX === 0 && offsetY === 0) return document;

  return {
    ...document,
    shapes: document.shapes.map((candidate) => {
      if (candidate.id !== shapeId) return candidate;
      if (candidate.kind === "polygon") {
        return {
          ...candidate,
          points: candidate.points.map((point) => ({
            x: point.x + offsetX,
            y: point.y + offsetY,
          })),
        } satisfies PolygonShape;
      }
      // A bitmap's crop is integer source pixels, so the offset it can take is too.
      return {
        ...candidate,
        x: candidate.x + Math.round(offsetX),
        y: candidate.y + Math.round(offsetY),
      };
    }),
  };
}

function clamp(value: number, min: number, max: number): number {
  // A shape wider than the frame cannot satisfy both bounds; refusing to move it at all is
  // better than picking one edge to snap it to.
  if (min > max) return 0;
  return Math.min(max, Math.max(min, value));
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
