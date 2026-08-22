/**
 * One sample, as one experiment saw it — built around the image rather than around its
 * chrome.
 *
 * The previous version spent three stacked rows and a titled panel on a single opacity
 * slider before the photograph started, then drew it at half width in a two-column grid
 * that a one-image dataset never fills. The image is the content here; anything that
 * pushes it below the fold is a defect. So: one header line, one toolbar line, and the
 * canvas takes what is left of the viewport.
 *
 * **Three things are drawn over the source, and they answer different questions.** The
 * heatmap says *how anomalous, everywhere* and has no edge. The segmentation says *where*,
 * with an edge, and is the only form that can be laid against the ground-truth outline and
 * read as agreement or disagreement — which is the comparison this page exists for. The
 * ground truth says what was annotated. All three are rendered server-side against the
 * run's own range, so nothing about colormap or threshold is baked into the stored map
 * (ADR-0007).
 *
 * Layers stack as absolutely-positioned `<img>` elements inside **one** zoom/pan transform,
 * so they cannot drift apart at the zoom level where a reader is judging whether the
 * prediction lands on the defect.
 *
 * Alignment is explicit. Every method sees the experiment's pinned prepared artifact, and
 * the shared inference boundary projects its map through that image's stored transform
 * before persistence. The browser therefore receives source-frame layers and never has to
 * reconstruct crop, resize or letterbox offsets.
 *
 * **The diagnostic panes are generic over the index (ADR-0018).** For
 * `efficientad_anomalib` they happen to be the student-teacher and autoencoder errors,
 * which is the comparison that makes the method legible. Nothing here knows that: it draws
 * every image-scoped `map` the run recorded, so `pixel_reference` gets its unsmoothed
 * z-map and M6's method gets whatever it emits, with no code written here.
 */

import { useEffect, useRef, useState } from "react";
import { Link, useNavigate, useParams, useSearchParams } from "react-router";
import { ChevronLeft, ChevronRight, Microscope } from "lucide-react";

import { imageScoped, isOnDemand, missingNote, ofKinds } from "../api/diagnostics";
import { diagnosticPayloadUrl } from "../api/diagnostics";
import { anomalyMapUrl, imageUrl, maskUrl, predictionUrl } from "../api/imageUrl";
import type { DiagnosticEntry, ImageScore, MapScale, SampleVerdict } from "../api/client";
import type { ResultsState } from "../api/resultsState";
import { cutValue, readResultsState, writeResultsState } from "../api/resultsState";
import { Badge, Button, Disclosure, Empty, ErrorBox, FULL_TIER_ZOOM, RESET_VIEW, SkeletonRows, Tooltip, ZoomPanCanvas, cn, type View } from "@vitavision/lab-ui";
import { useAnomalyValues, useSourceValues } from "../hooks/useMapValues";
import {
  useDiagnoseImage,
  useDiagnostics,
  useExperiment,
  useSampleImages,
} from "../hooks/useExperiments";
import { MapScaleReadout, OverlayControls } from "./experiment/OverlayControls";
import { ValueReadout } from "./experiment/ValueReadout";
import type { HoverPosition } from "./experiment/ValueReadout";
import { OUTCOME_LABEL, OUTCOME_TONE } from "./experiment/ResultsPanel";
import { useVerdicts } from "./experiment/useVerdicts";

export function ExperimentSampleRoute() {
  const { experimentId: rawExperiment, sampleId: rawSample } = useParams();
  const experimentId = rawExperiment === undefined ? undefined : Number(rawExperiment);
  const sampleId = rawSample === undefined ? undefined : Number(rawSample);

  const navigate = useNavigate();
  const [params, setParams] = useSearchParams();
  const state = readResultsState(params);

  const experiment = useExperiment(experimentId);
  const images = useSampleImages(experimentId, sampleId);
  const diagnostics = useDiagnostics(experimentId);

  // The same query the gallery ran, rebuilt from the URL — a cache hit rather than a
  // second fetch, and the reason this page knows which sample comes next *under the
  // filter that is on screen* rather than in raw id order.
  const verdicts = useVerdicts(experimentId, state);

  const [view, setView] = useState<View>(RESET_VIEW);

  const update = (next: Partial<ResultsState>) =>
    setParams(writeResultsState({ ...state, ...next }), { replace: true });

  const neighbours = stepThrough(verdicts.shown, sampleId);
  const goTo = (target: SampleVerdict | undefined) => {
    if (!target || experimentId === undefined) return;
    navigate({
      pathname: `/experiments/${experimentId}/samples/${target.sample_id}`,
      search: writeResultsState(state).toString(),
    });
  };

  // Reset the zoom on arrival: carrying a 6x pan onto a different part shows a corner of
  // it with no way to tell that is what happened.
  useEffect(() => {
    setView(RESET_VIEW);
  }, [sampleId]);

  /*
   * Arrow keys step through the filtered set, matching the labelling keys the dataset
   * browser already binds; `0` fits, matching its viewer.
   *
   * The handler is held in a ref and the effect has an explicit empty dependency array.
   * Without one it re-subscribed a global listener on *every* render — which worked, but
   * meant a component that re-renders on each pointermove was adding and removing a window
   * listener at the same rate.
   */
  const shortcuts = useRef<(event: KeyboardEvent) => void>(() => {});
  shortcuts.current = (event: KeyboardEvent) => {
    if (event.metaKey || event.ctrlKey || event.altKey) return;
    const target = event.target as HTMLElement | null;
    if (target && /^(INPUT|TEXTAREA|SELECT)$/.test(target.tagName)) return;
    if (event.key === "ArrowLeft") goTo(neighbours.previous);
    if (event.key === "ArrowRight") goTo(neighbours.next);
    if (event.key === "0") setView(RESET_VIEW);
  };

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => shortcuts.current(event);
    globalThis.addEventListener("keydown", onKey);
    return () => globalThis.removeEventListener("keydown", onKey);
  }, []);

  if (images.error) return <ErrorBox>{images.error.message}</ErrorBox>;
  if (images.isPending) return <SkeletonRows rows={5} />;
  if (!images.data || images.data.length === 0) {
    return <Empty>This experiment scored no images for this sample.</Empty>;
  }

  const anyMask = images.data.some((image) => image.has_mask);
  const anyMap = images.data.some((image) => image.has_map);
  const anyDiagnostic = images.data.some(
    (image) => imageScoped(diagnostics.data, image.image_id).length > 0,
  );
  const note = missingNote(diagnostics.data);
  const range = experiment.data?.map_range;
  const cut = cutValue(state, range);
  const verdict = verdicts.shown.find((entry) => entry.sample_id === sampleId);

  return (
    // `min-h-0` all the way down, or the canvas's `flex-1` is computed against content
    // height and the image grows the page instead of filling it.
    <div className="flex min-h-0 flex-1 flex-col gap-3">
      {/* One line. Everything that was three rows of chrome, in the order it is read:
          where you are, what this sample is, and how to get to the next one. */}
      <header className="flex flex-wrap items-center gap-x-3 gap-y-2">
        <Link
          to={{
            pathname: `/experiments/${experimentId}`,
            search: writeResultsState(state).toString(),
          }}
          className="text-sm text-fg-muted hover:text-fg hover:underline"
        >
          ← Results
        </Link>
        <h1 className="text-base font-semibold">Sample {sampleId}</h1>
        {verdict && (
          <>
            <Badge tone={verdict.label === "defect" ? "defect" : "normal"}>{verdict.label}</Badge>
            <Badge tone={OUTCOME_TONE[verdict.outcome] ?? "neutral"}>
              {OUTCOME_LABEL[verdict.outcome] ?? verdict.outcome}
            </Badge>
          </>
        )}
        <span className="font-mono text-xs text-fg-muted">
          score {images.data[0]?.score.toFixed(4)}
          {" · "}
          {images.data[0]?.inference_ms.toFixed(1)} ms
        </span>

        <span className="ml-auto flex items-center gap-2">
          <span className="font-mono text-[11px] text-fg-subtle">
            {neighbours.position === null
              ? verdicts.label
              : `${neighbours.position} of ${verdicts.shown.length} · ${verdicts.label}`}
          </span>
          <Tooltip content="Previous (←)">
            <Button
              variant="ghost"
              disabled={!neighbours.previous}
              onClick={() => goTo(neighbours.previous)}
              aria-label="Previous sample"
            >
              <ChevronLeft className="size-4" />
            </Button>
          </Tooltip>
          <Tooltip content="Next (→)">
            <Button
              variant="ghost"
              disabled={!neighbours.next}
              onClick={() => goTo(neighbours.next)}
              aria-label="Next sample"
            >
              <ChevronRight className="size-4" />
            </Button>
          </Tooltip>
        </span>
      </header>

      <OverlayControls
        state={state}
        onChange={update}
        range={range}
        hasMask={anyMask}
        hasMap={anyMap}
      />

      {/* Columns from the image count, never from a constant: a one-image sample is one
          full-width canvas, and the old `md:grid-cols-2` is why it was half of one. */}
      <div
        className="grid min-h-0 flex-1 gap-3"
        style={{
          gridTemplateColumns: `repeat(${Math.min(images.data.length, 3)}, minmax(0, 1fr))`,
        }}
      >
        {images.data.map((image) => (
          <ChannelView
            key={image.image_id}
            image={image}
            experimentId={experimentId as number}
            state={state}
            cut={cut}
            range={range}
            view={view}
            onView={setView}
            single={images.data.length === 1}
          />
        ))}
      </div>

      {/* Below the fold by design. These decompose the map above; they must not compete
          with it for the first screen. */}
      <Disclosure summary="Per-branch diagnostics" count={anyDiagnostic ? undefined : 0}>
        <div className="flex flex-col gap-4">
          <div className="flex flex-wrap items-start justify-between gap-3">
            <p className="max-w-prose text-xs text-fg-muted">
              {anyDiagnostic ? (
                <>
                  What the method recorded about this particular image, beside the combined
                  map it produced. Every pane is drawn on the run's own scale, so the same
                  region looks the same across images.
                </>
              ) : (
                /* "That image was not one of the ones kept" is a different fact from "this
                   model records nothing", and a blank panel said neither. */
                note
              )}
            </p>
            {experiment.data?.produces_diagnostics === true && (
              <DiagnoseButton
                experimentId={experimentId as number}
                imageIds={images.data.map((image) => image.image_id)}
                again={anyDiagnostic}
              />
            )}
          </div>

          {anyDiagnostic &&
            images.data.map((image) => (
              <DiagnosticRow
                key={image.image_id}
                image={image}
                experimentId={experimentId as number}
                entries={ofKinds(imageScoped(diagnostics.data, image.image_id), ["map", "image"])}
                showMask={state.truth}
              />
            ))}
        </div>
      </Disclosure>
    </div>
  );
}

/**
 * Where this sample sits in the filtered set, and what is either side of it.
 *
 * `null` position for a sample that is not in the set at all — reached by a direct link,
 * or filtered out by a change made while it was open. It still renders; it just has no
 * neighbours, which is honest rather than a silent jump to somewhere else in the list.
 */
export function stepThrough(
  ordered: SampleVerdict[],
  sampleId: number | undefined,
): { previous: SampleVerdict | undefined; next: SampleVerdict | undefined; position: number | null } {
  const index = ordered.findIndex((entry) => entry.sample_id === sampleId);
  if (index < 0) return { previous: undefined, next: undefined, position: null };
  return {
    previous: ordered[index - 1],
    next: ordered[index + 1],
    position: index + 1,
  };
}

function ChannelView({
  image,
  experimentId,
  state,
  cut,
  range,
  view,
  onView,
  single,
}: {
  image: ImageScore;
  experimentId: number;
  state: ResultsState;
  cut: number | null;
  range: MapScale | null | undefined;
  view: View;
  onView: (view: View) => void;
  single: boolean;
}) {
  /*
   * Nothing is fetched until the pointer is actually over this canvas — `hovered` gates
   * both queries, so a reader who never hovers pays nothing, and one who does pays one
   * fetch per plane for the whole time the sample is open.
   */
  const [hover, setHover] = useState<HoverPosition | null>(null);
  const mapValues = useAnomalyValues(experimentId, image.image_id, hover !== null && image.has_map);
  // Every colour plane in one payload, with the count in its header — a mono experiment
  // and an RGB one are the same code path, and neither is encoded here.
  const sourceValues = useSourceValues(experimentId, image.image_id, hover !== null);

  const layers: { key: string; src: string; className?: string }[] = [];
  if (state.heatmap && image.has_map) {
    layers.push({
      key: "heatmap",
      src: anomalyMapUrl(image.image_id, experimentId),
      // The PNG's own alpha already follows the score, so the photograph shows through
      // untouched wherever the model found nothing. A fixed opacity scales that rather
      // than fighting it — which is all the old 0–100 slider ever did.
      className: "opacity-80",
    });
  }
  if (state.region && image.has_map && cut !== null) {
    layers.push({ key: "region", src: predictionUrl(image.image_id, experimentId, cut) });
  }
  if (state.truth && image.has_mask) {
    layers.push({ key: "truth", src: maskUrl(image.image_id) });
  }

  return (
    <figure className="flex min-h-0 flex-col gap-1.5">
      {!single && (
        <figcaption className="flex items-center gap-2 text-xs">
          <span className="font-medium">{image.channel ?? "single view"}</span>
          {image.has_mask && <Badge tone="info">annotated</Badge>}
          <span className="ml-auto font-mono">{image.score.toFixed(4)}</span>
        </figcaption>
      )}

      {/* The frame is shaped like the picture, not like the column. `object-contain` in a
          full-width box centres the image between two black bars and spends the window on
          them; an aspect-ratio box with `min-h-0` fits the image to whichever of height or
          width runs out first and takes no more room than that. */}
      <div className="flex min-h-0 flex-1 items-center justify-center">
        <ZoomPanCanvas
          view={view}
          onView={onView}
          className="max-h-full max-w-full"
          style={{ aspectRatio: aspectOf(image), height: "100%" }}
          label={`${image.channel ?? "single view"} · ${view.zoom.toFixed(1)}×`}
          nativeWidth={image.width}
          onHover={setHover}
        >
          <img
            src={imageUrl(image.image_id, view.zoom > FULL_TIER_ZOOM ? "full" : "preview")}
            alt={`Channel ${image.channel ?? "single view"}`}
            draggable={false}
            className="absolute inset-0 h-full w-full object-fill"
          />
          {/* Every layer is already in source coordinates, projected by the backend through
              this image's pinned region transform. Filling one box is therefore exact. */}
          {layers.map((layer) => (
            <img
              key={layer.key}
              src={layer.src}
              alt=""
              aria-hidden
              draggable={false}
              className={cn(
                "pointer-events-none absolute inset-0 h-full w-full object-fill",
                layer.className,
              )}
            />
          ))}
        </ZoomPanCanvas>
      </div>

      <div className="flex flex-col gap-0.5">
        {/* The measurement, then the scale it sits on. Both are needed to read the map:
            one says what this pixel is, the other says what "hot" means for this run. */}
        <ValueReadout
          position={hover}
          map={mapValues.data}
          source={sourceValues.data}
          range={range}
        />
        <div className="flex items-baseline justify-between gap-3">
          <MapScaleReadout scale={image.map_scale} range={range} />
          {!image.has_map && <span className="text-xs text-fg-subtle">no anomaly map</span>}
        </div>
      </div>
    </figure>
  );
}

/**
 * Ask the method about this sample's images, now (ADR-0026).
 *
 * Every image of the sample, in one press: a two-channel part is one thing to look at, and
 * making the reader diagnose each view separately would be an implementation detail
 * surfacing as a chore.
 *
 * The pending text names the model load rather than showing a spinner alone. The first
 * request of a session genuinely takes seconds, and an unexplained wait is the difference
 * between "this is loading" and "this is broken".
 */
function DiagnoseButton({
  experimentId,
  imageIds,
  again,
}: {
  experimentId: number;
  imageIds: number[];
  again: boolean;
}) {
  const diagnose = useDiagnoseImage(experimentId);
  const [error, setError] = useState<string | null>(null);

  const run = () => {
    setError(null);
    // Sequential, not concurrent: the resident serves one request at a time by design, so
    // firing them together would only queue them behind each other with less to read.
    void imageIds
      .reduce(
        (chain, imageId) => chain.then(() => diagnose.mutateAsync(imageId).then(() => undefined)),
        Promise.resolve(),
      )
      .catch((cause: unknown) => {
        setError(cause instanceof Error ? cause.message : String(cause));
      });
  };

  return (
    <span className="flex flex-col items-end gap-1">
      <Button onClick={run} disabled={diagnose.isPending}>
        <Microscope className="size-4" />
        {again ? "Diagnose again" : "Diagnose this image"}
      </Button>
      {diagnose.isPending && (
        <span className="text-[11px] text-fg-muted">
          Running the method on this image — the first request loads the model.
        </span>
      )}
      {error !== null && <span className="max-w-xs text-[11px] text-warn">{error}</span>}
    </span>
  );
}

/** The frame's shape, from the source's own pixels. Square until the API says otherwise. */
function aspectOf(image: ImageScore): string {
  if (image.width > 0 && image.height > 0) return `${image.width} / ${image.height}`;
  return "1 / 1";
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
      <h3 className="flex items-center gap-2 text-sm font-medium">
        {image.channel ?? "single view"}
        <span className="font-mono text-xs text-fg-muted">score {image.score.toFixed(4)}</span>
        {entries.some(isOnDemand) && (
          <Tooltip content="Recorded on request rather than sampled by the run. It survives a reload and is cleared with the other diagnostics.">
            <Badge tone="neutral">on demand</Badge>
          </Tooltip>
        )}
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
