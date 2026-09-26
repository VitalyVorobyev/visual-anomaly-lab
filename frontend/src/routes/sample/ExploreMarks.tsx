/**
 * What Explore draws in image pixels, inside the stage's transform: the clicked points, and a
 * MobileSAM candidate tinted as a suggestion.
 *
 * Points are dots of constant on-screen size — a positive in the `normal` tone, a negative in
 * `defect`, the editor's own convention for include and exclude. The candidate is a cropped
 * binary PNG in the source frame, painted in the suggestion colour through the same
 * `tintedMask` the editor uses, so a candidate reads the same here and after it is carried
 * there.
 */

import { useEffect, useState } from "react";

import { MeasureOverlay, useStage, type MeasurePrimitive } from "@vitavision/lab-ui";

import { tintedMask } from "../../api/annotationBitmap";
import type { BitmapShape } from "../../api/client";
import type { ImagePoint } from "../../api/explore";
import { useScenePalette } from "../../components/annotation/scenePalette";

const DOT_PX = 5;

export function ExploreMarks({
  width,
  height,
  positives,
  negatives,
  candidate,
  opacity,
}: {
  width: number;
  height: number;
  positives: readonly ImagePoint[];
  negatives: readonly ImagePoint[];
  candidate: BitmapShape | null;
  opacity: number;
}) {
  const stage = useStage();
  const radius = stage.imageLength(DOT_PX);
  // Coordinates only: a SAM point also carries a `kind`, which must not become the primitive's.
  const dot =
    (tone: "normal" | "defect") =>
    (point: ImagePoint): MeasurePrimitive => ({
      kind: "point",
      x: point.x,
      y: point.y,
      radius,
      tone,
    });
  const primitives: MeasurePrimitive[] = [
    ...positives.map(dot("normal")),
    ...negatives.map(dot("defect")),
  ];
  return (
    <>
      {candidate && <CandidateMask shape={candidate} opacity={opacity} />}
      {primitives.length > 0 && (
        <MeasureOverlay
          nativeWidth={width}
          nativeHeight={height}
          primitives={primitives}
          strokeScale={stage.view.scale}
        />
      )}
    </>
  );
}

function CandidateMask({ shape, opacity }: { shape: BitmapShape; opacity: number }) {
  const palette = useScenePalette();
  const [painted, setPainted] = useState<string | null>(null);

  useEffect(() => {
    let live = true;
    const source = new globalThis.Image();
    source.onload = () => {
      if (!live) return;
      setPainted(tintedMask(source, shape.width, shape.height, palette.suggestion).toDataURL());
    };
    source.src = `data:image/png;base64,${shape.png_base64}`;
    return () => {
      live = false;
    };
  }, [shape, palette.suggestion]);

  if (!painted) return null;
  return (
    <img
      src={painted}
      alt=""
      aria-hidden
      draggable={false}
      className="pointer-events-none absolute"
      style={{
        left: shape.x,
        top: shape.y,
        width: shape.width,
        height: shape.height,
        opacity: opacity * 0.7,
        imageRendering: "pixelated",
      }}
    />
  );
}
