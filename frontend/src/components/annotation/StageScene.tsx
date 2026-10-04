/**
 * The committed scene: the imported base, every region and the MobileSAM suggestion — what
 * changes when the *document*, the selection or the view changes, and not when the pointer
 * moves.
 *
 * Regions are painted in document order (`sceneRuns`): each stretch of polygons and boxes is
 * one `AreaSet`, filled even-odd as completion rasterises them, and each bitmap a tinted raster
 * between them, so a cut paints over what it cuts. A bitmap's crop rectangle is outlined when
 * it is a cut (dashed — rendered like an added region it read as a second brush) or selected.
 *
 * Every region layer answers the stage's hit-test under the `REGION_LAYER` prefix, which is how
 * Select tells a region from an assist prompt or the suggestion's outline.
 */

import { memo, useMemo } from "react";

import {
  AreaSet,
  PolylineSet,
  STAGE_HIT_PRIORITY,
  useStage,
  useStageDrag,
  useStageHitLayer,
  type AreaSetItem,
  type PolylineSetItem,
  type StagePointerEvent,
} from "@vitavision/stage2d";

import { shapeOutline } from "../../api/annotationState";
import type { AnnotationDocument, AnnotationLabel, AnnotationShape, BitmapShape } from "../../api/client";
import { defined } from "../../api/defined";
import { moveDrag, type LiveEdit } from "./liveEdit";
import { MaskRaster, ShapeRaster } from "./MaskRaster";
import type { ScenePalette } from "./scenePalette";
import { sceneRuns, type VectorShape } from "./sceneRuns";
import { pickBitmap, REGION_LAYER } from "./shapePick";
import { flatToStage, fromStage, rectToStage } from "./stageFrame";

/** SVG paints a token directly; a raster needs it resolved (`ScenePalette`). */
const CUT = "var(--defect)";
const UNKNOWN_LABEL = "var(--signal)";
const SUGGESTION = "var(--warn)";
const SUGGESTION_ID = "suggestion";

interface Props {
  document: AnnotationDocument;
  labels: AnnotationLabel[];
  palette: ScenePalette;
  /** The imported base mask, decoded; `null` when there is none or it has not loaded. */
  baseImage: HTMLImageElement | null;
  maskOpacity: number;
  showRegions: boolean;
  selectedId: string | null;
  /** Every bitmap's decoded crop, for picking painted pixels rather than the rectangle. */
  masks: ReadonlyMap<string, Uint8Array>;
  /** An un-accepted MobileSAM suggestion, previewed over the regions. */
  assistShape: BitmapShape | null;
  /**
   * Selecting and moving are this pane's (Select, in an editable pane): regions show hover,
   * and a press that reaches a region without passing the tool surface — on the selected
   * polygon's outline, where the vertex editor's insert target lies — selects and moves it.
   */
  interactive: boolean;
  onSelect: (shapeId: string | null) => void;
  onMoveShape: (shapeId: string, dx: number, dy: number) => void;
  onEdit: (edit: LiveEdit | null) => void;
}

export const StageScene = memo(function StageScene({
  document,
  labels,
  palette,
  baseImage,
  maskOpacity,
  showRegions,
  selectedId,
  masks,
  assistShape,
  interactive,
  onSelect,
  onMoveShape,
  onEdit,
}: Props) {
  const stage = useStage();
  const startDrag = useStageDrag();
  const colors = useMemo(() => new Map(labels.map((label) => [label.key, label.color])), [labels]);
  // Built when the document changes, not when the view does: a new item list re-indexes
  // and re-batches its layer.
  const runs = useMemo(
    () =>
      sceneRuns(document.shapes).map((run) =>
        run.kind === "vector" ? { ...run, items: regionItems(run.shapes, colors) } : run,
      ),
    [document.shapes, colors],
  );
  const selected = useMemo(() => (selectedId === null ? [] : [selectedId]), [selectedId]);
  const outlines = useMemo(
    () => bitmapOutlines(showRegions ? document.shapes : [], selectedId, assistShape),
    [showRegions, document.shapes, selectedId, assistShape],
  );
  // The import carries no class, so it takes the dataset's first one — which is the class
  // every region drawn on a single-class dataset gets too, so base and edits read as one.
  const baseColor = labels[0]?.color ?? palette.unknownLabel;

  const onRegionPress = interactive
    ? (id: string | number, event: StagePointerEvent) => {
        const shapeId = String(id);
        const client = { x: event.clientX, y: event.clientY };
        onSelect(shapeId);
        startDrag(event, moveDrag(shapeId, stage.toImage(client), client, onEdit, onMoveShape));
      }
    : undefined;

  return (
    <>
      {showRegions && baseImage && (
        // The imported base, at half the weight of the editable regions above it: it is
        // context for what is being edited, not one of the things being edited.
        <MaskRaster
          image={baseImage}
          rect={{ x: 0, y: 0, width: document.image_width, height: document.image_height }}
          color={baseColor}
          opacity={maskOpacity * 0.5}
        />
      )}
      {showRegions &&
        runs.map((run) =>
          run.kind === "vector" ? (
            <AreaSet
              key={run.key}
              items={run.items}
              label="Regions"
              layerId={`${REGION_LAYER}:${run.key}`}
              selectedIds={selected}
              selectionFill="item"
              fillOpacity={maskOpacity}
              fillRule="evenodd"
              paintOrder="items"
              halo={false}
              // Hover is an affordance of Select alone: under a drawing tool a lit outline
              // would promise a click that draws instead.
              {...defined({ hoveredId: interactive ? undefined : null, onItemPress: onRegionPress })}
            />
          ) : (
            <ShapeRaster
              key={run.key}
              shape={run.shape}
              color={
                run.shape.operation === "subtract"
                  ? palette.cut
                  : (colors.get(run.shape.label_key) ?? palette.unknownLabel)
              }
              opacity={maskOpacity}
            />
          ),
        )}
      {showRegions && <BitmapHits shapes={document.shapes} masks={masks} />}
      {assistShape && (
        <ShapeRaster
          shape={assistShape}
          color={palette.suggestion}
          opacity={Math.min(1, maskOpacity + 0.2)}
        />
      )}
      {outlines.length > 0 && (
        <PolylineSet
          items={outlines}
          selected={selected}
          interactive={false}
          layerId={`${REGION_LAYER}:outlines`}
          label="Painted region outlines"
        />
      )}
    </>
  );
});

/** A vector run as `AreaSet` items: each region's ring in stage coordinates, in its class colour. */
function regionItems(
  shapes: readonly VectorShape[],
  colors: ReadonlyMap<string, string>,
): AreaSetItem[] {
  return shapes.map((shape) => {
    const color =
      shape.operation === "subtract" ? CUT : (colors.get(shape.label_key) ?? UNKNOWN_LABEL);
    return { id: shape.id, points: flatToStage(shapeOutline(shape)), stroke: color, fill: color };
  });
}

/**
 * The crop rectangles worth outlining: every cut (dashed), the selected bitmap (which
 * `PolylineSet` draws in the selection colour, cut or not), and the suggestion.
 */
function bitmapOutlines(
  shapes: readonly AnnotationShape[],
  selectedId: string | null,
  suggestion: BitmapShape | null,
): PolylineSetItem[] {
  const items: PolylineSetItem[] = shapes
    .filter(
      (shape): shape is BitmapShape =>
        shape.kind === "bitmap" && (shape.operation === "subtract" || shape.id === selectedId),
    )
    .map((shape) => ({ id: shape.id, points: rectRing(shape), closed: true, stroke: CUT, dashed: true }));
  if (suggestion) {
    items.push({ id: SUGGESTION_ID, points: rectRing(suggestion), closed: true, stroke: SUGGESTION });
  }
  return items;
}

function rectRing(shape: BitmapShape): number[] {
  const { x, y, width, height } = rectToStage(shape);
  return [x, y, x + width, y, x + width, y + height, x, y + height];
}

/** The bitmaps' answer to the stage's hit-test: painted pixels, from the decoded masks. */
function BitmapHits({
  shapes,
  masks,
}: {
  shapes: readonly AnnotationShape[];
  masks: ReadonlyMap<string, Uint8Array>;
}) {
  useStageHitLayer({
    layerId: `${REGION_LAYER}:bitmaps`,
    priority: STAGE_HIT_PRIORITY.area,
    pick: (point, radius) => pickBitmap(shapes, masks, fromStage(point), radius),
  });
  return null;
}
