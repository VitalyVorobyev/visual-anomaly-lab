/**
 * The one image viewer: a photograph, raster layers over it, vector shapes over those, and
 * anything else a screen needs drawn in image pixels — all inside one `ImageStage`.
 *
 * There used to be three stacks. The dataset sample viewer was on the deprecated
 * `ZoomPanCanvas`, whose zoom was relative to the frame and so meant a different thing in
 * every window; the two result viewers were on `ImageStage`, each assembling the same
 * photograph-plus-layers stack by hand. This is that stack, once. The stage is laid out at
 * the image's own pixel size, so every layer at `inset-0` is registered with the
 * photograph by construction, and `tierFor` picks the preview or the lossless full image
 * from a scale in real units.
 */

import { useLayoutEffect, useRef, type MutableRefObject, type ReactNode } from "react";

import { ImageStage, StageToolbar, cn, useStage, type StageView } from "@vitavision/lab-ui";

import { imageUrl, tierFor } from "../../api/imageUrl";
import { VectorLayer, type VectorShape } from "./VectorLayer";

export interface RasterLayer {
  key: string;
  src: string;
  className?: string;
  /** 0–1, for a layer whose weight the reader sets (Explore's opacity slider). */
  opacity?: number;
}

type Projection = (client: { x: number; y: number }) => { x: number; y: number };

/** A click on the image that was not a pan, in image pixels, with the modifier it carried. */
export type StagePick = (point: { x: number; y: number }, modifiers: { shiftKey: boolean }) => void;

export function SampleStage({
  image,
  alt,
  view,
  onView,
  layers = [],
  shapes = [],
  children,
  label,
  onHover,
  readout,
  panKeys = true,
  onPick,
}: {
  image: { id: number; width: number; height: number };
  alt: string;
  view: StageView | null;
  onView: (view: StageView) => void;
  /** Source-frame PNGs, stacked in order over the photograph. */
  layers?: readonly RasterLayer[];
  /** Boxes and polygons, over every raster layer. */
  shapes?: readonly VectorShape[];
  /** Anything else in image pixels — a peak marker, a measurement. */
  children?: ReactNode;
  label: string;
  onHover?: (point: { x: number; y: number } | null) => void;
  readout?: ReactNode;
  /** False where the arrow keys belong to the screen (a sample list), not to panning. */
  panKeys?: boolean;
  /**
   * A click that landed on the picture and did not become a pan. The stage reports a
   * background click in client coordinates; its own transform turns that into image pixels,
   * so a screen asking "where did they click" never re-derives the view arithmetic.
   */
  onPick?: StagePick;
}) {
  const toImage = useRef<Projection | null>(null);
  return (
    <ImageStage
      image={{ width: image.width, height: image.height }}
      view={view}
      onView={onView}
      onHover={onHover}
      panKeys={panKeys}
      label={label}
      toolbar={<StageToolbar />}
      readout={readout}
      onBackgroundClick={
        onPick
          ? (event) => {
              const project = toImage.current;
              if (!project) return;
              const point = project({ x: event.clientX, y: event.clientY });
              const inside =
                point.x >= 0 && point.y >= 0 && point.x < image.width && point.y < image.height;
              if (inside) onPick(point, { shiftKey: event.shiftKey });
            }
          : undefined
      }
    >
      <img
        src={imageUrl(image.id, tierFor(view))}
        alt={alt}
        draggable={false}
        className="absolute inset-0 h-full w-full"
      />
      {layers.map((layer) => (
        <img
          key={layer.key}
          src={layer.src}
          alt=""
          aria-hidden
          draggable={false}
          className={cn("pointer-events-none absolute inset-0 h-full w-full", layer.className)}
          style={layer.opacity === undefined ? undefined : { opacity: layer.opacity }}
        />
      ))}
      <VectorLayer width={image.width} height={image.height} shapes={shapes} />
      {children}
      {onPick && <StageProjection target={toImage} />}
    </ImageStage>
  );
}

/** Hands the stage's own client-to-image projection to `SampleStage`, which sits outside it. */
function StageProjection({ target }: { target: MutableRefObject<Projection | null> }) {
  const stage = useStage();
  useLayoutEffect(() => {
    target.current = stage.toImage;
    return () => {
      target.current = null;
    };
  }, [stage.toImage, target]);
  return null;
}
