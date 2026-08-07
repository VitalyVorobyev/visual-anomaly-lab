/**
 * One sample, as one experiment saw it: the source image, the anomaly map over it, the
 * ground-truth outline where a mask exists, and whatever per-image diagnostics the method
 * recorded beside them.
 *
 * The overlay is stacked `<img>` elements at the same size, with opacity applied in CSS.
 * That is deliberate and is what ADR-0007 asks for: the map is stored raw, the server
 * colormaps it against the *run's* range so two images are comparable, and the blend
 * happens at view time — so moving the slider is instant and costs no round trip.
 *
 * The map's PNG carries alpha that already follows the score, so the source image shows
 * through untouched wherever the model found nothing. An opaque colormap over a
 * photograph tints the whole frame and makes the quiet regions look as processed as the
 * loud one — which is the opposite of what an overlay is for.
 *
 * Alignment comes for free from the preprocessing bridge. Every method resizes straight
 * to the configured size without preserving aspect ratio, so the map is a plain stretch
 * back onto the source and there are no letterbox offsets to reconstruct here.
 *
 * **The diagnostic panes are generic over the index (ADR-0018).** For
 * `efficientad_anomalib` they happen to be the student-teacher and autoencoder errors,
 * which is the comparison that makes the method legible — a hit in one branch means
 * something different from a hit in the other. Nothing here knows that: it draws every
 * image-scoped `map` the run recorded, so `pixel_reference` gets its unsmoothed z-map and
 * M6's method gets whatever it emits, with no code written here.
 */

import { useState } from "react";
import { useParams } from "react-router";

import { budgetNote, imageScoped, ofKinds } from "../api/diagnostics";
import { diagnosticPayloadUrl } from "../api/diagnostics";
import { anomalyMapUrl, imageUrl, maskUrl } from "../api/imageUrl";
import type { DiagnosticEntry, ImageScore } from "../api/client";
import { Badge, Empty, ErrorBox, PageHeader, Panel, ReadoutStrip, SkeletonRows, Slider, Switch } from "../components/ui";
import { useDiagnostics, useExperiment, useSampleImages } from "../hooks/useExperiments";

export function ExperimentSampleRoute() {
  const { experimentId: rawExperiment, sampleId: rawSample } = useParams();
  const experimentId = rawExperiment === undefined ? undefined : Number(rawExperiment);
  const sampleId = rawSample === undefined ? undefined : Number(rawSample);

  const experiment = useExperiment(experimentId);
  const images = useSampleImages(experimentId, sampleId);
  const diagnostics = useDiagnostics(experimentId);

  const [opacity, setOpacity] = useState(0.6);
  const [showMask, setShowMask] = useState(true);

  if (images.error) return <ErrorBox>{images.error.message}</ErrorBox>;
  if (images.isPending) return <SkeletonRows rows={5} />;
  if (!images.data || images.data.length === 0) {
    return <Empty>This experiment scored no images for this sample.</Empty>;
  }

  const anyMask = images.data.some((image) => image.has_mask);
  const anyDiagnostic = images.data.some(
    (image) => imageScoped(diagnostics.data, image.image_id).length > 0,
  );
  const note = budgetNote(diagnostics.data);

  return (
    <div className="flex flex-col gap-6">
      <PageHeader
        back={{ to: `/experiments/${experimentId}`, label: "Back to results" }}
        title={`Sample ${sampleId}`}
        meta={
          <ReadoutStrip
            items={[
              {
                label: "experiment",
                value: experiment.data?.name,
                to: `/experiments/${experimentId}`,
              },
              { label: "method", value: experiment.data?.model_type },
              { label: "channels", value: images.data.length },
            ]}
          />
        }
      />

      <Panel title="Overlay">
        <div className="flex flex-wrap items-center gap-x-8 gap-y-4">
          <div className="flex min-w-64 flex-1 items-center gap-3">
            <span className="shrink-0 text-xs font-medium text-fg">Anomaly map</span>
            <Slider
              aria-label="Anomaly map opacity"
              min={0}
              max={1}
              step={0.02}
              value={opacity}
              onValueChange={setOpacity}
              readout={
                <span className="inline-block w-9 text-right">
                  {Math.round(opacity * 100)}%
                </span>
              }
            />
          </div>

          <Switch
            checked={showMask}
            disabled={!anyMask}
            onCheckedChange={setShowMask}
            label={
              <span className={anyMask ? undefined : "text-fg-subtle"}>
                Ground-truth outline{anyMask ? "" : " — none for this sample"}
              </span>
            }
          />
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

      <Panel title="Diagnostics">
        {anyDiagnostic ? (
          <div className="flex flex-col gap-6">
            <p className="text-xs text-fg-muted">
              What the method recorded about this particular image, beside the combined map
              it produced. Every pane is drawn on the run's own scale, so the same region
              looks the same across images.
            </p>
            {images.data.map((image) => (
              <DiagnosticRow
                key={image.image_id}
                image={image}
                experimentId={experimentId as number}
                entries={ofKinds(imageScoped(diagnostics.data, image.image_id), ["map", "image"])}
                showMask={showMask}
              />
            ))}
          </div>
        ) : (
          <Empty>
            {/* "That image was not one of the first twelve" is a different fact from "this
                model records nothing", and a blank panel says neither. */}
            {note ??
              "This run recorded no per-image diagnostics. A method reports them by calling " +
                "ctx.emit_diagnostic during predict."}
          </Empty>
        )}
      </Panel>
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

      <div className="relative overflow-hidden rounded border border-line bg-[#08090a] ">
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

      <p className="font-mono text-xs text-fg-muted">
        {image.inference_ms.toFixed(1)} ms
        {image.has_map ? "" : " · no anomaly map"}
      </p>
    </figure>
  );
}

/**
 * One image's diagnostics, laid out beside the combined map they decompose.
 *
 * The combined map comes first deliberately: the question these panes answer is "which
 * branch produced that", and the comparison only works if the thing being decomposed is
 * on screen next to its parts.
 */
function DiagnosticRow({
  image,
  experimentId,
  entries,
  showMask,
}: {
  image: ImageScore;
  experimentId: number;
  entries: DiagnosticEntry[];
  showMask: boolean;
}) {
  if (entries.length === 0) return null;

  return (
    <section className="flex flex-col gap-2">
      <h3 className="text-sm font-medium">
        {image.channel ?? "single view"}
        <span className="ml-2 font-mono text-xs text-fg-muted">
          score {image.score.toFixed(4)}
        </span>
      </h3>

      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
        {image.has_map && (
          <DiagnosticPane
            title="Combined anomaly map"
            description="What the score was read off — the map the evaluation layer measured."
            // Opaque here, unlike the overlay above: beside an opaque per-branch map, an
            // alpha-scaled one puts the two panes on different scales and a clean image
            // renders as an empty box.
            src={anomalyMapUrl(image.image_id, experimentId, false)}
            maskSrc={showMask && image.has_mask ? maskUrl(image.image_id) : undefined}
          />
        )}
        {entries.map((entry) => (
          <DiagnosticPane
            key={entry.key}
            title={entry.title}
            description={entry.description ?? undefined}
            src={diagnosticPayloadUrl(experimentId, entry)}
            maskSrc={showMask && image.has_mask ? maskUrl(image.image_id) : undefined}
          />
        ))}
      </div>
    </section>
  );
}

function DiagnosticPane({
  title,
  description,
  src,
  maskSrc,
}: {
  title: string;
  description?: string;
  src: string;
  maskSrc?: string;
}) {
  return (
    <figure className="flex flex-col gap-1">
      <div className="relative overflow-hidden rounded border border-line bg-[#08090a] ">
        <img src={src} alt={title} className="block w-full" loading="lazy" />
        {maskSrc && (
          <img
            src={maskSrc}
            alt=""
            aria-hidden
            className="pointer-events-none absolute inset-0 h-full w-full"
          />
        )}
      </div>
      <figcaption className="flex flex-col gap-0.5">
        <span className="text-xs font-medium">{title}</span>
        {description && (
          <span className="text-[11px] text-fg-muted">{description}</span>
        )}
      </figcaption>
    </figure>
  );
}
