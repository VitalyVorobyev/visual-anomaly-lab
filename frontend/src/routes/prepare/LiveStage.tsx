/**
 * The live stage: one source image with its crop, and the frame a method will be fed.
 *
 * Left, the source with the extractor's own box dashed and the padded crop solid — the gap
 * between the two *is* the padding. Right, the prepared frame exactly as a build writes it,
 * with the resized crop outlined so the contain-pad bands are visible. Both answer to the
 * same unsaved recipe; the previous answer stays up while the next is computed, marked
 * pending on the stage rather than blanked.
 */

import { Badge, cn, ErrorBox, ImageStage, ReadoutStrip, Skeleton, StageToolbar, type StageView } from "@vitavision/lab-ui";
import { useEffect, useState } from "react";

import type { RegionLivePreview } from "../../api/client";
import { SampleStage } from "../../components/viewer/SampleStage";
import { VectorLayer, type VectorShape } from "../../components/viewer/VectorLayer";

/** The image on the stage. Its size is known from the filmstrip, or from the first answer. */
export interface StageTarget {
  imageId: number;
  width?: number | undefined;
  height?: number | undefined;
  label: string;
}

const PANE = "relative transition-opacity h-[min(52vh,32rem)] min-h-64 overflow-hidden rounded-panel border border-line bg-raised";

export function LiveStage({
  target,
  preview,
  pending,
  error,
  blocked,
}: {
  target: StageTarget | undefined;
  /** The latest answer; it may describe the previous image or recipe while `pending`. */
  preview: RegionLivePreview | undefined;
  pending: boolean;
  error: Error | null;
  /** Why no preview can be asked for right now — an invalid option, say. */
  blocked: string | null;
}) {
  const [sourceView, setSourceView] = useState<StageView | null>(null);
  const [preparedView, setPreparedView] = useState<StageView | null>(null);
  const imageId = target?.imageId;
  useEffect(() => {
    setSourceView(null);
    setPreparedView(null);
  }, [imageId]);

  const current = preview !== undefined && preview.image_id === imageId ? preview : undefined;
  const width = target?.width ?? current?.source_width;
  const height = target?.height ?? current?.source_height;
  const pendingBanner = pending ? <Badge tone="info">Preparing…</Badge> : undefined;
  const failed = current?.status === "failed";

  return (
    <div className="flex flex-col gap-3">
      <div className="grid gap-3 md:grid-cols-2">
        <figure className="flex min-w-0 flex-col gap-1.5">
          <figcaption className="flex items-center justify-between gap-2 text-xs text-fg-muted">
            <span className="truncate">
              Source · <span className="font-mono text-fg">{target?.label ?? "—"}</span>
            </span>
            {width !== undefined && height !== undefined && (
              <span className="font-mono">
                {width}×{height}
              </span>
            )}
          </figcaption>
          <div className={cn(PANE, pending && "opacity-80")} aria-busy={pending}>
            {target !== undefined && width !== undefined && height !== undefined ? (
              <SampleStage
                image={{ id: target.imageId, width, height }}
                alt={`Source image ${target.label}`}
                label={`Source image ${target.label}`}
                view={sourceView}
                onView={setSourceView}
                shapes={current ? sourceShapes(current) : []}
                panKeys={false}
                banner={pendingBanner}
              />
            ) : (
              <Skeleton className="absolute inset-0" />
            )}
          </div>
        </figure>

        <figure className="flex min-w-0 flex-col gap-1.5">
          <figcaption className="flex items-center justify-between gap-2 text-xs text-fg-muted">
            <span className="truncate">Prepared · as the method reads it</span>
            {current && (
              <span className="font-mono">
                {current.width}×{current.height}
              </span>
            )}
          </figcaption>
          <div className={cn(PANE, pending && "opacity-80")} aria-busy={pending}>
            {current?.prepared_png ? (
              <ImageStage
                image={{ width: current.width, height: current.height }}
                view={preparedView}
                onView={setPreparedView}
                panKeys={false}
                label="Prepared frame"
                toolbar={<StageToolbar />}
                banner={pendingBanner}
              >
                <img
                  src={current.prepared_png}
                  alt={`Prepared frame of ${target?.label ?? "the image"}`}
                  draggable={false}
                  className="absolute inset-0 h-full w-full [image-rendering:pixelated]"
                />
                <VectorLayer width={current.width} height={current.height} shapes={preparedShapes(current)} />
              </ImageStage>
            ) : failed ? (
              <div className="grid h-full place-items-center px-6 text-center text-sm text-defect">
                No prepared frame: this image fails under this configuration.
              </div>
            ) : blocked !== null || (error !== null && !pending) ? (
              <div className="grid h-full place-items-center px-6 text-center text-sm text-fg-muted">
                {blocked ?? "No prepared frame: the preview could not be computed."}
              </div>
            ) : (
              <Skeleton className="absolute inset-0" />
            )}
          </div>
        </figure>
      </div>

      {failed && current?.error && <ErrorBox>{current.error}</ErrorBox>}
      {error && <ErrorBox>{error.message}</ErrorBox>}

      <ReadoutStrip items={readout(current)} />
    </div>
  );
}

function sourceShapes(preview: RegionLivePreview): VectorShape[] {
  const shapes: VectorShape[] = [];
  if (preview.region) {
    const { left, top, right, bottom } = preview.region;
    shapes.push({
      id: "region",
      kind: "box",
      x: left,
      y: top,
      width: right - left,
      height: bottom - top,
      tone: "muted",
      dashed: true,
    });
  }
  const crop = preview.transform;
  if (crop) {
    shapes.push({
      id: "crop",
      kind: "box",
      x: crop.crop_left,
      y: crop.crop_top,
      width: crop.crop_right - crop.crop_left,
      height: crop.crop_bottom - crop.crop_top,
      tone: "signal",
      label: preview.united > 1 ? `crop · shared by ${preview.united}` : "crop",
    });
  }
  return shapes;
}

function preparedShapes(preview: RegionLivePreview): VectorShape[] {
  const t = preview.transform;
  if (!t || (t.pad_left === 0 && t.pad_top === 0 && t.pad_right === 0 && t.pad_bottom === 0)) {
    return [];
  }
  return [
    {
      id: "resized",
      kind: "box",
      x: t.pad_left,
      y: t.pad_top,
      width: t.resized_width,
      height: t.resized_height,
      tone: "muted",
      dashed: true,
    },
  ];
}

function readout(preview: RegionLivePreview | undefined) {
  if (!preview) return [{ value: null }];
  const t = preview.transform;
  const coverage = preview.extractor_metadata["coverage_fraction"];
  return [
    {
      label: "crop",
      value: t ? `${t.crop_right - t.crop_left}×${t.crop_bottom - t.crop_top} at ${t.crop_left},${t.crop_top}` : "—",
    },
    { label: "resized", value: t ? `${t.resized_width}×${t.resized_height}` : "—" },
    {
      label: "confidence",
      value:
        preview.extractor_confidence === null || preview.extractor_confidence === undefined
          ? "—"
          : preview.extractor_confidence.toFixed(3),
    },
    {
      label: "coverage",
      value: typeof coverage === "number" ? `${(coverage * 100).toFixed(1)}%` : "—",
    },
    { label: "time", value: `${Math.round(preview.elapsed_ms)} ms` },
  ];
}
