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

import type { ReactNode } from "react";

import { ImageStage, StageToolbar, cn, type StageView } from "@vitavision/lab-ui";

import { imageUrl, tierFor } from "../../api/imageUrl";
import { VectorLayer, type VectorShape } from "./VectorLayer";

export interface RasterLayer {
  key: string;
  src: string;
  className?: string;
}

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
  banner,
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
  /** Over the top-left, outside the transform: a pending or stale state. */
  banner?: ReactNode;
}) {
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
      banner={banner}
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
        />
      ))}
      <VectorLayer width={image.width} height={image.height} shapes={shapes} />
      {children}
    </ImageStage>
  );
}
