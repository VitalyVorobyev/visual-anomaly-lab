/**
 * One sample, as one experiment saw it: the source image, the anomaly map over it, and
 * the ground-truth outline where a mask exists.
 *
 * The overlay is three stacked `<img>` elements at the same size, with opacity applied in
 * CSS. That is deliberate and is what ADR-0007 asks for: the map is stored raw, the
 * server colormaps it against the *run's* range so two images are comparable, and the
 * blend happens at view time — so moving the slider is instant and costs no round trip.
 *
 * The map's PNG carries alpha that already follows the score, so the source image shows
 * through untouched wherever the model found nothing. An opaque colormap over a
 * photograph tints the whole frame and makes the quiet regions look as processed as the
 * loud one — which is the opposite of what an overlay is for.
 *
 * Alignment comes for free from the preprocessing bridge. Every method resizes straight
 * to the configured size without preserving aspect ratio, so the map is a plain stretch
 * back onto the source and there are no letterbox offsets to reconstruct here.
 */

import { useState } from "react";
import { Link, useParams } from "react-router";

import { anomalyMapUrl, imageUrl, maskUrl } from "../api/imageUrl";
import type { ImageScore } from "../api/client";
import { Badge, Empty, ErrorBox, Panel } from "../components/ui";
import { useExperiment, useSampleImages } from "../hooks/useExperiments";

export function ExperimentSampleRoute() {
  const { experimentId: rawExperiment, sampleId: rawSample } = useParams();
  const experimentId = rawExperiment === undefined ? undefined : Number(rawExperiment);
  const sampleId = rawSample === undefined ? undefined : Number(rawSample);

  const experiment = useExperiment(experimentId);
  const images = useSampleImages(experimentId, sampleId);

  const [opacity, setOpacity] = useState(0.6);
  const [showMask, setShowMask] = useState(true);

  if (images.error) return <ErrorBox>{images.error.message}</ErrorBox>;
  if (images.isPending) return <p className="text-sm text-slate-500">Loading…</p>;
  if (!images.data || images.data.length === 0) {
    return <Empty>This experiment scored no images for this sample.</Empty>;
  }

  const anyMask = images.data.some((image) => image.has_mask);

  return (
    <div className="flex flex-col gap-6">
      <header className="flex flex-wrap items-center gap-3">
        <h2 className="text-lg font-semibold tracking-tight">Sample {sampleId}</h2>
        <span className="font-mono text-xs text-slate-500">{experiment.data?.model_type}</span>
        <Link
          to={`/experiments/${experimentId}`}
          className="ml-auto text-sm text-slate-500 hover:underline"
        >
          Back to results
        </Link>
      </header>

      <Panel title="Overlay">
        <div className="flex flex-wrap items-center gap-6">
          <label className="flex items-center gap-3 text-sm">
            <span className="text-xs font-medium tracking-wide text-slate-500 uppercase">
              Anomaly map
            </span>
            <input
              type="range"
              aria-label="Anomaly map opacity"
              min={0}
              max={1}
              step={0.02}
              value={opacity}
              onChange={(event) => setOpacity(Number(event.target.value))}
            />
            <span className="w-10 font-mono text-xs">{Math.round(opacity * 100)}%</span>
          </label>

          <label className="flex items-center gap-2 text-sm">
            <input
              type="checkbox"
              aria-label="Ground truth"
              checked={showMask}
              disabled={!anyMask}
              onChange={(event) => setShowMask(event.target.checked)}
            />
            <span className={anyMask ? "" : "text-slate-400"}>
              Ground-truth outline{anyMask ? "" : " (none for this sample)"}
            </span>
          </label>
        </div>
      </Panel>

      <div className="grid gap-6 md:grid-cols-2">
        {images.data.map((image) => (
          <ChannelView
            key={image.image_id}
            image={image}
            experimentId={experimentId as number}
            opacity={opacity}
            showMask={showMask}
          />
        ))}
      </div>
    </div>
  );
}

function ChannelView({
  image,
  experimentId,
  opacity,
  showMask,
}: {
  image: ImageScore;
  experimentId: number;
  opacity: number;
  showMask: boolean;
}) {
  return (
    <figure className="flex flex-col gap-2">
      <figcaption className="flex items-center gap-2 text-sm">
        <span className="font-medium">{image.channel ?? "single view"}</span>
        {image.has_mask && <Badge tone="info">annotated</Badge>}
        <span className="ml-auto font-mono text-xs">score {image.score.toFixed(4)}</span>
      </figcaption>

      <div className="relative overflow-hidden rounded border border-slate-200 bg-slate-950 dark:border-slate-700">
        <img
          src={imageUrl(image.image_id, "preview")}
          alt={`Channel ${image.channel ?? "single view"}`}
          className="block w-full"
        />
        {image.has_map && (
          <img
            src={anomalyMapUrl(image.image_id, experimentId)}
            alt=""
            aria-hidden
            // Plain alpha compositing. The map arrives as RGBA with its *own* alpha
            // already following the score, so it is transparent wherever the model found
            // nothing; this slider scales that, rather than fighting it with a blend mode.
            className="pointer-events-none absolute inset-0 h-full w-full"
            style={{ opacity }}
          />
        )}
        {showMask && image.has_mask && (
          <img
            src={maskUrl(image.image_id)}
            alt=""
            aria-hidden
            className="pointer-events-none absolute inset-0 h-full w-full"
          />
        )}
      </div>

      <p className="font-mono text-xs text-slate-500">
        {image.inference_ms.toFixed(1)} ms
        {image.has_map ? "" : " · no anomaly map"}
      </p>
    </figure>
  );
}
