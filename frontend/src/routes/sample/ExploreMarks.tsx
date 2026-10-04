/**
 * What Explore draws in image pixels, inside the stage's transform: the clicked points, and a
 * MobileSAM candidate tinted as a suggestion.
 *
 * Points are dots of constant on-screen size — a positive in the `normal` tone, a negative in
 * `defect`, the editor's own convention for include and exclude. The candidate is a cropped
 * binary PNG in the source frame, painted in the suggestion colour by the same raster the
 * editor uses, so a candidate reads the same here and after it is carried there.
 */

import { MeasureOverlay, useStage, type MeasurePrimitive } from "@vitavision/stage2d";

import type { BitmapShape } from "../../api/client";
import type { ImagePoint } from "../../api/explore";
import { ShapeRaster } from "../../components/annotation/MaskRaster";
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
  return <ShapeRaster shape={shape} color={palette.suggestion} opacity={opacity * 0.7} />;
}
