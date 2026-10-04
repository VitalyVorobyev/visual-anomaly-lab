/**
 * A binary mask painted in one colour, as a stage layer: a painted region, the imported base,
 * a MobileSAM suggestion.
 *
 * A canvas positioned in CSS pixels inside the stage's box, which is laid out at the image's
 * own size, so `left: x` puts the crop's first column exactly over source column `x` — the
 * area convention, with nothing to convert. Its pixels are `tintedMask`'s: alpha from the
 * mask's luminance, never from the PNG's own alpha, so an opaque mask the backend wrote is an
 * overlay and not a filled rectangle. Drawn as squares at every zoom, since the point of a mask
 * is which pixels it holds.
 */

import { useEffect, useRef } from "react";

import { tintedMask } from "../../api/annotationBitmap";
import type { BitmapShape } from "../../api/client";
import { useHtmlImage } from "./useHtmlImage";

export interface MaskRect {
  x: number;
  y: number;
  width: number;
  height: number;
}

export function MaskRaster({
  image,
  rect,
  color,
  opacity,
}: {
  /** The decoded mask, or `null` while it loads (nothing is drawn). */
  image: CanvasImageSource | null;
  /** Where the mask lies, in source pixels. */
  rect: MaskRect;
  /** A literal colour: a canvas cannot read a custom property (`useScenePalette`). */
  color: string;
  opacity: number;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const { x, y, width, height } = rect;

  useEffect(() => {
    const target = canvasRef.current;
    const context = target?.getContext("2d");
    if (!target || !context) return;
    context.clearRect(0, 0, width, height);
    if (!image) return;
    try {
      context.drawImage(tintedMask(image, width, height, color), 0, 0);
    } catch {
      // A mask that cannot be read (a tainted canvas) is context lost, not an editor lost:
      // it is left undrawn, as the readout leaves it unread.
    }
  }, [image, width, height, color]);

  return (
    <canvas
      ref={canvasRef}
      width={width}
      height={height}
      aria-hidden
      data-mask-raster=""
      className="pointer-events-none absolute"
      style={{ left: x, top: y, width, height, opacity, imageRendering: "pixelated" }}
    />
  );
}

/** A region's (or a suggestion's) mask, decoded from its PNG. */
export function ShapeRaster({
  shape,
  color,
  opacity,
}: {
  shape: BitmapShape;
  color: string;
  opacity: number;
}) {
  const image = useHtmlImage(`data:image/png;base64,${shape.png_base64}`);
  return <MaskRaster image={image} rect={shape} color={color} opacity={opacity} />;
}
