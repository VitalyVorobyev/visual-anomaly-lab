/**
 * The static layer: the photograph, a blended second channel, the imported base mask and every
 * committed region — the things that change when the *document* or the *view* changes, and
 * not when the pointer moves.
 *
 * Memoised, and fed only primitives and stable callbacks, so a gesture in progress (drawn by
 * `LiveLayer`) never re-renders it. A vertex being dragged is the one transient state kept
 * here, because it deforms a committed outline and the outline lives on this layer.
 */

import { memo, useEffect, useMemo, useState } from "react";
import { Circle, Group, Image as KonvaImage, Layer, Line, Rect } from "react-konva";

import { tintedMask } from "../../api/annotationBitmap";
import type {
  AnnotationDocument,
  AnnotationLabel,
  AnnotationPoint,
  BitmapShape,
} from "../../api/client";
import { clamp } from "./canvasView";
import { withAlpha, type ScenePalette } from "./scenePalette";
import { useHtmlImage } from "./useHtmlImage";

interface Props {
  document: AnnotationDocument;
  labels: AnnotationLabel[];
  palette: ScenePalette;
  source: HTMLImageElement | null;
  overlay: HTMLImageElement | null;
  overlayOpacity: number;
  /** The imported base, already tinted. */
  baseMask: HTMLCanvasElement | null;
  maskOpacity: number;
  showRegions: boolean;
  selectedId: string | null;
  /** Selecting, dragging and reshaping are all the same permission. */
  interactive: boolean;
  /** An un-accepted MobileSAM suggestion, previewed over the regions. */
  assistShape: BitmapShape | null;
  originX: number;
  originY: number;
  scale: number;
  /** Resample the photograph smoothly; off when a source pixel is larger than a screen pixel. */
  smoothing: boolean;
  onSelect: (shapeId: string | null) => void;
  onMoveShape: (shapeId: string, dx: number, dy: number) => void;
  onMovePoint: (shapeId: string, pointIndex: number, point: AnnotationPoint) => void;
}

export const SceneLayer = memo(function SceneLayer({
  document,
  labels,
  palette,
  source,
  overlay,
  overlayOpacity,
  baseMask,
  maskOpacity,
  showRegions,
  selectedId,
  interactive,
  assistShape,
  originX,
  originY,
  scale,
  smoothing,
  onSelect,
  onMoveShape,
  onMovePoint,
}: Props) {
  /**
   * A vertex mid-drag. The override is what makes a drag *look* like a drag: the outline used
   * to be read straight from the committed document, which only changes on `dragEnd`, so the
   * handle moved and the polygon it belonged to stayed where it was until the mouse came up.
   */
  const [vertexDrag, setVertexDrag] = useState<{
    shapeId: string;
    index: number;
    point: AnnotationPoint;
  } | null>(null);
  const colors = useMemo(
    () => new Map(labels.map((label) => [label.key, label.color])),
    [labels],
  );
  const width = document.image_width;
  const height = document.image_height;
  const inFrame = (x: number, y: number) => ({
    x: clamp(x, 0, width),
    y: clamp(y, 0, height),
  });

  return (
    <Layer imageSmoothingEnabled={smoothing}>
      <Group x={originX} y={originY} scaleX={scale} scaleY={scale}>
        <Rect
          width={width}
          height={height}
          fill={palette.canvas}
          shadowColor="#000000"
          shadowBlur={16 / scale}
          shadowOpacity={0.4}
        />
        {source && <KonvaImage image={source} width={width} height={height} listening={false} />}
        {overlay && (
          <KonvaImage
            image={overlay}
            width={width}
            height={height}
            opacity={overlayOpacity}
            listening={false}
          />
        )}
        {showRegions && baseMask && (
          // The imported base, at half the weight of the editable regions above it: it is
          // context for what is being edited, not one of the things being edited.
          <KonvaImage
            image={baseMask}
            width={width}
            height={height}
            opacity={maskOpacity * 0.5}
            listening={false}
          />
        )}

        {showRegions &&
          document.shapes.map((shape) => {
            const color = colors.get(shape.label_key) ?? palette.unknownLabel;
            if (shape.kind === "bitmap") {
              return (
                <BitmapLayer
                  key={shape.id}
                  shape={shape}
                  color={color}
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
                // A click has to survive a shaking hand, or selecting a region would drag it a
                // pixel and land in undo history.
                dragDistance={4}
                // Selection is claimed *here*, on the draggable node, and not on the line
                // inside it. Konva starts a drag from a `mousedown` listener on the draggable
                // node itself, reached by bubbling — so cancelling the bubble on the line would
                // have selected the region and then silently refused to move it. Cancelling
                // here still stops the stage's pan gesture, which is the only thing that had
                // to stop.
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
                  // `dragend` bubbles, and a selected polygon's vertices are draggable children
                  // — without this, dragging a vertex would also read the *vertex's* position
                  // as a whole-region offset and fling the shape across the frame.
                  if (event.target !== event.currentTarget) return;
                  const node = event.target;
                  const dx = node.x();
                  const dy = node.y();
                  // Konva moving the group *is* the live feedback; the committed document then
                  // carries the offset, so the node has to go back to the origin or it would be
                  // applied twice.
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
                      // Silent under every other tool. A selected polygon stays on screen while
                      // brushing, and a vertex that still took the click swallowed the start of
                      // the stroke.
                      listening={interactive}
                      draggable={interactive}
                      onDragMove={(event) =>
                        setVertexDrag({
                          shapeId: shape.id,
                          index,
                          point: inFrame(event.target.x(), event.target.y()),
                        })
                      }
                      onDragEnd={(event) => {
                        event.cancelBubble = true;
                        setVertexDrag(null);
                        onMovePoint(shape.id, index, inFrame(event.target.x(), event.target.y()));
                      }}
                      onMouseDown={(event) => {
                        // The vertex owns this gesture: without cancelling, the group under it
                        // would drag the whole region at the same time.
                        event.cancelBubble = true;
                      }}
                    />
                  ))}
              </Group>
            );
          })}

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
        <Rect
          width={width}
          height={height}
          stroke={withAlpha(palette.frame, 0.55)}
          strokeWidth={1 / scale}
          listening={false}
        />
      </Group>
    </Layer>
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
