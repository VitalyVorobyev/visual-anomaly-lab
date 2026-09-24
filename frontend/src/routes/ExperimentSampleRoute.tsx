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
 * Layers stack as absolutely-positioned `<img>` elements inside **one** `ImageStage`
 * transform, so they cannot drift apart at the zoom level where a reader is judging whether
 * the prediction lands on the defect. The stage is laid out at the image's own pixel size,
 * which is what makes that registration structural rather than a matter of every layer
 * being given the same aspect-ratio box: the predecessor's frame was clamped by
 * `max-w-full` while its height stayed full, so a column narrower than the picture drew
 * every layer squashed — identically squashed, and therefore invisible as a fault.
 *
 * Alignment is explicit. Every method sees the experiment's pinned prepared artifact, and
 * the shared inference boundary projects its map through that image's stored transform
 * before persistence. The browser therefore receives source-frame layers and never has to
 * reconstruct crop, resize or letterbox offsets.
 *
 * **The diagnostic panes are generic over the index (ADR-0018).** For
 * `efficientad_custom` they happen to be the student-teacher and autoencoder errors,
 * which is the comparison that makes the method legible. Nothing here knows that: it draws
 * every image-scoped `map` the run recorded, so `pixel_reference` gets its unsmoothed
 * z-map and M6's method gets whatever it emits, with no code written here.
 */

import { useEffect, useState } from "react";
import { Link, useNavigate, useParams, useSearchParams } from "react-router";
import { ChevronLeft, ChevronRight, Microscope } from "lucide-react";

import { imageScoped, isOnDemand, missingNote, ofKinds } from "../api/diagnostics";
import { diagnosticPayloadUrl } from "../api/diagnostics";
import { anomalyMapUrl, maskUrl, predictionUrl } from "../api/imageUrl";
import type { DiagnosticEntry, ImageScore, MapScale, SampleVerdict } from "../api/client";
import type { ResultsState } from "../api/resultsState";
import { cutValue, readResultsState, resolveSubset, writeResultsState } from "../api/resultsState";
import { Badge, Button, Disclosure, Empty, ErrorBox, SkeletonRows, StageReadout, Tooltip, type StageView } from "@vitavision/lab-ui";
import { SampleStage, type RasterLayer } from "../components/viewer/SampleStage";
import { useHotkeys } from "../hooks/useHotkeys";
import { LabelLayer } from "../components/viewer/LabelLayer";
import { useAnomalyValues, useLabelPlane, useSourceValues } from "../hooks/useMapValues";
import {
  useDiagnoseImage,
  useDiagnostics,
  useExperiment,
  useSampleImages,
} from "../hooks/useExperiments";
import { MapScaleReadout, OverlayControls } from "./experiment/OverlayControls";
import { PeakMarker } from "./experiment/PeakMarker";
import { ValueReadout } from "./experiment/ValueReadout";
import { OUTCOME_LABEL, OUTCOME_TONE, localizationBadge } from "./experiment/ResultsPanel";
import { useVerdicts } from "./experiment/useVerdicts";

export function ExperimentSampleRoute() {
  const { experimentId: rawExperiment, sampleId: rawSample } = useParams();
  const experimentId = rawExperiment === undefined ? undefined : Number(rawExperiment);
  const sampleId = rawSample === undefined ? undefined : Number(rawSample);

  const navigate = useNavigate();
  const [params, setParams] = useSearchParams();
  const experiment = useExperiment(experimentId);
  // Resolved exactly as the experiment page resolves it, so a link that carried no subset
  // still pages through the subset the results tabs show — not every scored subset at once.
  // Read under the run's own task, so a segmentation run opens on its label map.
  const task = experiment.data?.task;
  const state = resolveSubset(
    readResultsState(params, task),
    experiment.data?.scored_subsets ?? [],
  );
  const images = useSampleImages(experimentId, sampleId);
  const diagnostics = useDiagnostics(experimentId);

  // The same query the gallery ran, rebuilt from the URL — a cache hit rather than a
  // second fetch, and the reason this page knows which sample comes next *under the
  // filter that is on screen* rather than in raw id order.
  const verdicts = useVerdicts(experimentId, state, task);

  // `null` is not "no view": it is `ImageStage`'s own opening view — 1:1 where the picture
  // fits, fit otherwise — resolved once the viewport has been measured.
  const [view, setView] = useState<StageView | null>(null);

  const update = (next: Partial<ResultsState>) =>
    setParams(writeResultsState({ ...state, ...next }, task), { replace: true });

  const neighbours = stepThrough(verdicts.shown, sampleId);
  const goTo = (target: SampleVerdict | undefined) => {
    if (!target || experimentId === undefined) return;
    navigate({
      pathname: `/experiments/${experimentId}/samples/${target.sample_id}`,
      search: writeResultsState(state, task).toString(),
    });
  };

  // Reset the zoom on arrival: carrying a 6x pan onto a different part shows a corner of
  // it with no way to tell that is what happened.
  useEffect(() => {
    setView(null);
  }, [sampleId]);

  /*
   * Arrow keys step through the filtered set, matching the labelling keys the dataset
   * browser already binds. The zoom keys are the stage's own — `0`, `1`, `+`, `-` are bound
   * by `ImageStage`, and `panKeys={false}` is what keeps the arrows for this list.
   *
   * The guard is `useHotkeys`'s. This screen's own matched tag names only, so an arrow on
   * the focused cut slider — a `span[role=slider]` — also paged to the next sample.
   */
  useHotkeys((event) => {
    if (event.key === "ArrowLeft") goTo(neighbours.previous);
    if (event.key === "ArrowRight") goTo(neighbours.next);
  });

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
  // A supervised segmentation run draws its label maps instead of a cut and a mask (ADR-0039).
  const classes =
    experiment.data?.task === "semantic_segmentation" ? experiment.data.classes : undefined;
  const verdict = verdicts.shown.find((entry) => entry.sample_id === sampleId);
  // Beside the outcome rather than folded into it: "caught it, from the wrong pixels" is a
  // different finding from "caught it", and only this badge can say so.
  const localization = localizationBadge(verdict?.localized);

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
            search: writeResultsState(state, task).toString(),
          }}
          className="text-sm text-fg-muted hover:text-fg hover:underline"
        >
          ← Results
        </Link>
        {/* The sample's own name, as every other screen prints it — not its row id. */}
        <h1 className="font-mono text-sm font-semibold">
          {verdict ? `${verdict.group_key}/${verdict.external_id}` : `Sample ${sampleId}`}
        </h1>
        {verdict && (
          <>
            <Badge tone={verdict.label === "defect" ? "defect" : "normal"}>{verdict.label}</Badge>
            <Badge tone={OUTCOME_TONE[verdict.outcome] ?? "neutral"}>
              {OUTCOME_LABEL[verdict.outcome] ?? verdict.outcome}
            </Badge>
            {localization && <Badge tone={localization.tone}>{localization.label}</Badge>}
          </>
        )}
        <span className="font-mono text-xs text-fg-muted">
          score {images.data[0]?.score.toFixed(4)}
          {" · "}
          {images.data[0]?.inference_ms.toFixed(1)} ms
        </span>

        {/* A mistake is only useful if it can be fixed from where it was found: a wrong
            label, or a mask that missed the defect. Both used to be a manual trip back
            through the dataset. */}
        {experiment.data && (
          <span className="flex items-center gap-3 text-xs">
            <Link
              to={`/datasets/${experiment.data.dataset_id}/samples/${sampleId}`}
              className="text-signal underline-offset-2 hover:underline focus-visible:outline-2 focus-visible:outline-signal"
            >
              Open sample
            </Link>
            {images.data[0] && (
              <Link
                to={`/datasets/${experiment.data.dataset_id}/annotate/${sampleId}/${images.data[0].image_id}`}
                className="text-signal underline-offset-2 hover:underline focus-visible:outline-2 focus-visible:outline-signal"
              >
                Annotate
              </Link>
            )}
          </span>
        )}

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
        classes={classes}
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
            labelled={classes !== undefined}
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
  labelled,
}: {
  image: ImageScore;
  experimentId: number;
  state: ResultsState;
  cut: number | null;
  range: MapScale | null | undefined;
  view: StageView | null;
  onView: (view: StageView) => void;
  single: boolean;
  /** A supervised segmentation run: prediction and truth are label maps, not a cut and a mask. */
  labelled: boolean;
}) {
  /*
   * Nothing is fetched until the pointer is actually over this canvas — `hovered` gates
   * both queries, so a reader who never hovers pays nothing, and one who does pays one
   * fetch per plane for the whole time the sample is open.
   *
   * The stage reports **image pixels**, which is what the readout beside it prints and what
   * every number the backend returns is in. `valueAt` speaks fractions of the source frame,
   * so the conversion happens at that one call rather than being carried around as a second
   * coordinate system.
   */
  const [hover, setHover] = useState<{ x: number; y: number } | null>(null);
  const mapValues = useAnomalyValues(experimentId, image.image_id, hover !== null && image.has_map);
  // Every colour plane in one payload, with the count in its header — a mono experiment
  // and an RGB one are the same code path, and neither is encoded here.
  const sourceValues = useSourceValues(experimentId, image.image_id, hover !== null);
  // A supervised run's label maps, fetched only while their layer is on.
  const predictedLabels = useLabelPlane(
    experimentId,
    image.image_id,
    false,
    labelled && state.region,
  );
  const trueLabels = useLabelPlane(experimentId, image.image_id, true, labelled && state.truth);

  const layers: RasterLayer[] = [];
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
  if (!labelled && state.region && image.has_map && cut !== null) {
    layers.push({ key: "region", src: predictionUrl(image.image_id, experimentId, cut) });
  }
  if (!labelled && state.truth && image.has_mask) {
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

      {/* The stage fills the column and lays its own content out at the image's pixel size,
          so nothing here has to describe the picture's shape — which is what the previous
          `max-w-full` plus `aspectRatio` box got wrong the moment the column was narrower
          than the image. */}
      <div className="relative min-h-0 flex-1 overflow-hidden">
        <SampleStage
          image={{ id: image.image_id, width: image.width, height: image.height }}
          alt={`Channel ${image.channel ?? "single view"}`}
          view={view}
          onView={onView}
          onHover={setHover}
          // The arrows step through the filtered sample list; the stage keeps 0/1/+/-.
          panKeys={false}
          label={`${image.channel ?? "single view"} canvas`}
          readout={<StageReadout cursor={hover} />}
          // Every layer is already in source coordinates, projected by the backend through
          // this image's pinned region transform, and the stage is laid out at exactly
          // those coordinates. Filling it is therefore exact.
          layers={layers}
        >
          {/* A vector layer among the raster ones, and inside the same transform for the
              same reason: a marker that drifts from the heatmap it marks is worse than no
              marker. It draws itself in image coordinates. */}
          {state.peak && !labelled && <PeakMarker image={image} />}
          {labelled && state.region && predictedLabels.data && (
            <LabelLayer plane={predictedLabels.data} style="prediction" />
          )}
          {labelled && state.truth && trueLabels.data && (
            <LabelLayer plane={trueLabels.data} style="truth" />
          )}
        </SampleStage>
      </div>

      <div className="flex flex-col gap-0.5">
        {/* The measurement, then the scale it sits on. Both are needed to read the map:
            one says what this pixel is, the other says what "hot" means for this run. */}
        <ValueReadout
          position={
            hover === null ? null : { u: hover.x / image.width, v: hover.y / image.height }
          }
          map={mapValues.data}
          source={sourceValues.data}
          range={range}
        />
        <div className="flex items-baseline justify-between gap-3">
          <MapScaleReadout scale={image.map_scale} range={range} />
          {labelled ? (
            <LabelNote
              predicted={state.region ? predictedLabels.error : null}
              truth={state.truth ? trueLabels.error : null}
            />
          ) : (
            !image.has_map && <span className="text-xs text-fg-subtle">no anomaly map</span>
          )}
        </div>
      </div>
    </figure>
  );
}

/**
 * Why a label layer that is on draws nothing. A 404 is a fact — no label map, or truth that
 * does not answer for every pinned class — and anything else is a failure worth printing.
 */
function LabelNote({ predicted, truth }: { predicted: Error | null; truth: Error | null }) {
  const says = (error: Error | null, absent: string) =>
    error === null ? null : error.message.startsWith("404") ? absent : error.message;
  const notes = [
    says(predicted, "no label map"),
    says(truth, "no truth for every class of this run"),
  ].filter((note): note is string => note !== null);
  if (notes.length === 0) return null;
  return <span className="text-xs text-fg-subtle">{notes.join(" · ")}</span>;
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

/**
 * One image's diagnostics, laid out beside the combined map they decompose.
 *
 * The combined map comes first deliberately: the question these panes answer is "which
 * branch produced that", and the comparison only works if the thing being decomposed is
 * on screen next to its parts.
 *
 * **Two frames live in this row, and each pane's ground truth is fetched in its own.** A
 * diagnostic is a prepared-frame quantity by nature — `models/diagnostics.py` projects
 * nothing, deliberately, because a per-branch error map means what it means on the grid the
 * branch computed it on — and each pane is drawn at the array's own prepared or grid size.
 * The source-frame outline the overlay above uses is therefore the wrong picture beside
 * one, off by exactly the pinned crop and letterbox, so those panes ask the mask endpoint
 * for `frame=prepared`. The combined map is the exception and keeps the source-frame
 * outline, because the stored map was projected before it was written.
 *
 * It went unnoticed because of a coincidence: an identity extractor into a prepared size
 * that keeps the source's aspect ratio differs from the source frame by a *uniform scale*
 * alone, and both pictures are stretched into the same pane box, so they land on top of
 * each other. Any real crop, or any letterbox, and they do not.
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
            // Source-frame, unlike everything beside it: the stored map was projected
            // through the pinned transform before it was written, and this route renders it
            // at the source's own size. So this one pane wants the source-frame outline.
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
            maskSrc={
              showMask && image.has_mask
                ? maskUrl(image.image_id, { experimentId })
                : undefined
            }
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
