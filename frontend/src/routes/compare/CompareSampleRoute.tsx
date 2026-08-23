/**
 * One sample, one channel, every run's map beside each other.
 *
 * The screen ADR-0028 was written for, and every rule in that record is visible here:
 *
 * - **Each map is drawn on its own run's range**, which the server recorded when that run
 *   scored. Nothing is renormalized into a shared scale, because there is no shared scale —
 *   one method's 8 and another's 1.3 are the same claim on unrelated axes, and a shared
 *   range would render the lower-scaled method as flat background.
 * - **Both ranges are printed** under every pane, so the reader can see that the panes are
 *   not on one axis rather than having to know it.
 * - **One cut slider drives every segmentation, as a fraction.** A fraction resolves to a
 *   different number in each run, and that is exactly what makes it the same operating
 *   point in all of them.
 * - **One zoom and pan drives every pane**, because the comparison is spatial: where the
 *   methods fire and where they disagree. Panning one and not the others would make that
 *   unreadable at exactly the magnification where it matters.
 *
 * N panes, not two. The layout is a grid and the milestone's "A/B" is its N = 2 case.
 */

import { useState } from "react";
import { useParams, useSearchParams } from "react-router";

import type { ComparedRun, ComparedSample, ImageScore } from "../../api/client";
import type { CompareState } from "../../api/compareState";
import { cutFor, readCompareState, writeCompareState } from "../../api/compareState";
import { preferredImageIndex } from "../../api/defaultChannel";
import { anomalyMapUrl, imageUrl, maskUrl, predictionUrl } from "../../api/imageUrl";
import { Badge, Empty, ErrorBox, FULL_TIER_ZOOM, PageHeader, RESET_VIEW, SkeletonRows, Slider, Tabs, ToggleChip, ZoomPanCanvas, cn, type View } from "@vitavision/lab-ui";
import { useDataset } from "../../hooks/useCatalog";
import { useComparison, useSampleImageSets } from "../../hooks/useComparison";
import {
  HEATMAP_SWATCH,
  MapScaleReadout,
  PREDICTION_SWATCH,
  TRUTH_SWATCH,
} from "../experiment/OverlayControls";
import { OUTCOME_TONE } from "../experiment/ResultsPanel";

export function CompareSampleRoute() {
  const { sampleId: raw } = useParams();
  const sampleId = raw === undefined ? undefined : Number(raw);
  const [params, setParams] = useSearchParams();
  const state = readCompareState(params);
  const update = (next: Partial<CompareState>) => {
    setParams(writeCompareState({ ...state, ...next }), { replace: true });
  };

  const comparison = useComparison({
    ids: state.ids,
    subset: state.subset,
    at: state.at,
    recallTarget: state.recallTarget,
  });
  const images = useSampleImageSets(state.ids, sampleId);
  // Read before the early returns below, because a hook cannot be called after one. It is
  // idle until the report names its dataset, and then answers from the cache the rest of
  // the app has already filled.
  const dataset = useDataset(comparison.data?.dataset_id);
  const [channel, setChannel] = useState<number | null>(null);
  // One view for every pane. The whole point of the screen is that the panes are aligned.
  const [view, setView] = useState<View>(RESET_VIEW);

  const report = comparison.data;
  const back = `/compare?${writeCompareState({ ...state, view: "samples" }).toString()}`;

  if (comparison.isPending || images.some((query) => query.isPending)) {
    return <SkeletonRows rows={5} />;
  }
  if (comparison.error) return <ErrorBox>{comparison.error.message}</ErrorBox>;
  if (!report || sampleId === undefined) return <Empty>No such comparison.</Empty>;

  const row = report.samples.find((entry) => entry.sample_id === sampleId);
  /* The channel list comes from whichever run scored this sample first. Every run sees the
     same images — one dataset, one split — so this is the sample's channel list, not a
     particular run's opinion of it. */
  const channels = images.find((query) => (query.data?.length ?? 0) > 0)?.data ?? [];
  // Until the reader picks one, the dataset's own default answers — the same channel the
  // grid, the viewer and the annotation queue open on, so a part looks like itself here too.
  const active = channel ?? preferredImageIndex(channels, dataset.data?.default_channel);
  const image = channels[active];

  return (
    <div className="flex min-h-0 flex-1 flex-col gap-3 px-6 py-4">
      <PageHeader
        back={{ to: back, label: "Comparison" }}
        title={row ? row.external_id : `Sample ${sampleId}`}
        actions={
          row && (
            <Badge tone={row.label === "defect" ? "defect" : row.label === "normal" ? "normal" : "unlabeled"}>
              {row.label}
            </Badge>
          )
        }
      />

      <div className="flex flex-wrap items-center gap-x-6 gap-y-2">
        {channels.length > 1 && (
          <ChannelStrip images={channels} active={active} onSelect={setChannel} />
        )}

        <div className="flex flex-wrap items-center gap-1.5">
          <ToggleChip
            checked={state.heatmap}
            onCheckedChange={(heatmap) => update({ heatmap })}
            swatch={HEATMAP_SWATCH}
            title="How anomalous, everywhere — each pane on its own run's range"
          >
            heatmap
          </ToggleChip>
          <ToggleChip
            checked={state.region}
            onCheckedChange={(region) => update({ region })}
            swatch={PREDICTION_SWATCH}
            title="Where each map crosses the shared fraction of its own range"
          >
            prediction
          </ToggleChip>
          <ToggleChip
            checked={state.truth}
            onCheckedChange={(truth) => update({ truth })}
            swatch={TRUTH_SWATCH}
            title="The annotated region, identical in every pane"
          >
            ground truth
          </ToggleChip>
        </div>

        {state.region && (
          <div className="flex min-w-56 flex-1 items-center gap-2">
            <span className="shrink-0 text-xs text-fg-muted">cut</span>
            <Slider
              aria-label="Segmentation cut, as a fraction of each run's own map range"
              min={0}
              max={1}
              step={0.01}
              value={state.cut}
              onValueChange={(cut) => update({ cut })}
              readout={
                <span className="inline-block w-28 whitespace-nowrap text-right font-mono text-xs">
                  {/* A fraction, not a value: the number this resolves to is different in
                      every pane, and each pane prints its own below. */}
                  {(state.cut * 100).toFixed(0)}% of range
                </span>
              }
            />
          </div>
        )}
      </div>

      {image === undefined ? (
        <Empty>None of these runs scored this sample.</Empty>
      ) : (
        <div
          className="grid min-h-0 flex-1 gap-4"
          style={{ gridTemplateColumns: `repeat(${Math.min(report.runs.length, 3)}, minmax(0, 1fr))` }}
        >
          {report.runs.map((run, index) => (
            <RunPane
              key={run.id}
              run={run}
              row={row}
              index={index}
              image={images[index]?.data?.[active]}
              fallback={image}
              state={state}
              view={view}
              onView={setView}
            />
          ))}
        </div>
      )}
    </div>
  );
}

function ChannelStrip({
  images,
  active,
  onSelect,
}: {
  images: ImageScore[];
  active: number;
  onSelect: (index: number) => void;
}) {
  // Position is the identity, not the name: two images of one sample may share a channel
  // name and one may have none at all (ADR-0005).
  return (
    <Tabs
      label="Channels"
      active={String(active)}
      onSelect={(id) => onSelect(Number(id))}
      items={images.map((image, index) => ({
        id: String(index),
        label: image.channel ?? "unassigned",
      }))}
    />
  );
}

function RunPane({
  run,
  row,
  index,
  image,
  fallback,
  state,
  view,
  onView,
}: {
  run: ComparedRun;
  row: ComparedSample | undefined;
  index: number;
  image: ImageScore | undefined;
  /** The same image from a run that did score it, so the photograph is drawn regardless. */
  fallback: ImageScore;
  state: CompareState;
  view: View;
  onView: (view: View) => void;
}) {
  const shown = image ?? fallback;
  const cut = cutFor(state, run.map_range);
  const outcome = row?.outcomes[index] ?? null;
  const score = row?.scores[index] ?? null;

  const layers: { key: string; src: string; className?: string }[] = [];
  if (image && state.heatmap && image.has_map) {
    layers.push({
      key: "heatmap",
      src: anomalyMapUrl(image.image_id, run.id),
      className: "opacity-80",
    });
  }
  if (image && state.region && image.has_map && cut !== null) {
    layers.push({ key: "region", src: predictionUrl(image.image_id, run.id, cut) });
  }
  if (state.truth && shown.has_mask) {
    layers.push({ key: "truth", src: maskUrl(shown.image_id) });
  }

  return (
    <figure className="flex min-h-0 flex-col gap-1.5">
      <figcaption className="flex items-baseline gap-2">
        <span className="min-w-0 truncate text-sm font-semibold text-fg">{run.name}</span>
        <span className="font-mono text-[11px] text-fg-subtle">{run.model_type}</span>
        {outcome && <Badge tone={OUTCOME_TONE[outcome] ?? "neutral"}>{outcome}</Badge>}
        <span className="ml-auto font-mono text-xs tabular-nums">
          {score === null ? <span className="text-fg-subtle">—</span> : score.toFixed(4)}
        </span>
      </figcaption>

      <div className="flex min-h-0 flex-1 items-center justify-center">
        <ZoomPanCanvas
          view={view}
          onView={onView}
          className="max-h-full max-w-full"
          style={{ aspectRatio: `${shown.width} / ${shown.height}`, height: "100%" }}
          label={`${run.name} · ${view.zoom.toFixed(1)}×`}
          nativeWidth={shown.width}
        >
          <img
            src={imageUrl(shown.image_id, view.zoom > FULL_TIER_ZOOM ? "full" : "preview")}
            alt={run.name}
            draggable={false}
            className="absolute inset-0 h-full w-full object-fill"
          />
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

      {/* Both scales, under every pane. Without them two panes look like one axis, which is
          the misreading this whole design exists to prevent. */}
      <div className="flex flex-col gap-0.5">
        <MapScaleReadout scale={image?.map_scale} range={run.map_range} />
        {cut !== null && state.region && (
          <p className="font-mono text-[11px] text-fg-subtle">cut {cut.toFixed(3)}</p>
        )}
        {image === undefined && (
          <p className="text-xs text-fg-subtle">This run did not score this sample.</p>
        )}
        {image !== undefined && !image.has_map && (
          <p className="text-xs text-fg-subtle">This run recorded no anomaly map.</p>
        )}
      </div>
    </figure>
  );
}
