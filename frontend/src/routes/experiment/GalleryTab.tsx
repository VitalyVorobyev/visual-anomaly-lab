/**
 * Every scored sample, as a picture, ranked and filterable.
 *
 * This is the screen the workbench was missing. Until now the only route from a finished
 * run to an anomaly map was: open the results panel, wait for the threshold report, find
 * the verdict table, click a row. Four steps and a scroll to answer "what does this model
 * actually do", on a tool whose entire subject is images. Here the maps *are* the page.
 *
 * The outcome strip is the filter that matters — "mistakes" is false positives and false
 * negatives together, which is the first thing anyone wants and the one thing a ranked
 * list makes hardest to assemble.
 *
 * Two things are deliberately not done here. **Nothing re-applies the threshold**: the
 * server tags every row with its outcome and this sorts and filters what it was given
 * (§12 rule 3). And **the tiles are one request each for pixels and none for metadata** —
 * the preview endpoint hands over one image per sample in a single call, because a
 * grid of 270 tiles that each fetch their own identity is 270 requests to draw one screen.
 */

import { useMemo } from "react";
import { Link } from "react-router";

import type { MapScale, SamplePreview, SampleVerdict } from "../../api/client";
import { anomalyMapUrl, imageUrl, maskUrl, predictionUrl } from "../../api/imageUrl";
import type { Outcome, ResultsState } from "../../api/resultsState";
import { MISTAKE_OUTCOMES, cutValue, writeResultsState } from "../../api/resultsState";
import { Tabs } from "../../components/Tabs";
import { Badge, Empty, ErrorBox, SegmentedControl, Skeleton, cn } from "../../components/ui";
import { useSamplePreviews } from "../../hooks/useExperiments";
import { OverlayControls } from "./OverlayControls";
import { OUTCOME_LABEL, OUTCOME_TONE } from "./ResultsPanel";
import type { Verdicts } from "./useVerdicts";

/** `undefined` is every sample; `mistakes` is the pair anyone actually looks for. */
const FILTERS: { id: string; label: string; outcomes: readonly Outcome[] | undefined }[] = [
  { id: "all", label: "all", outcomes: undefined },
  { id: "mistakes", label: "mistakes", outcomes: MISTAKE_OUTCOMES },
  { id: "tp", label: "true positive", outcomes: ["tp"] },
  { id: "fn", label: "false negative", outcomes: ["fn"] },
  { id: "fp", label: "false positive", outcomes: ["fp"] },
  { id: "tn", label: "true negative", outcomes: ["tn"] },
  { id: "unlabeled", label: "unlabeled", outcomes: ["unlabeled"] },
];

export function GalleryTab({
  experimentId,
  state,
  onChange,
  verdicts,
  range,
}: {
  experimentId: number;
  state: ResultsState;
  onChange: (next: Partial<ResultsState>) => void;
  verdicts: Verdicts;
  range: MapScale | null | undefined;
}) {
  const previews = useSamplePreviews(experimentId, state.subset);

  const bySample = useMemo(() => {
    const index = new Map<number, SamplePreview>();
    for (const preview of previews.data ?? []) index.set(preview.sample_id, preview);
    return index;
  }, [previews.data]);

  const active = state.mistakesOnly ? "mistakes" : (state.outcome ?? "all");
  // Already filtered and ordered by `useVerdicts`, which is also what the sample page's
  // prev/next walks — so "the next one" means the same thing on both screens.
  const ordered = verdicts.shown;

  const counts = useMemo(() => {
    const tally = new Map<string, number>();
    for (const entry of verdicts.all) {
      tally.set(entry.outcome, (tally.get(entry.outcome) ?? 0) + 1);
    }
    return tally;
  }, [verdicts.all]);

  const countFor = (filter: (typeof FILTERS)[number]) =>
    filter.outcomes === undefined
      ? verdicts.all.length
      : filter.outcomes.reduce((total, outcome) => total + (counts.get(outcome) ?? 0), 0);

  if (verdicts.error) return <ErrorBox>{verdicts.error.message}</ErrorBox>;

  const anyMask = (previews.data ?? []).some((preview) => preview.has_mask);
  const anyMap = (previews.data ?? []).some((preview) => preview.has_map);
  const cut = cutValue(state, range);

  return (
    <div className="flex min-h-0 flex-1 flex-col gap-3">
      <div className="flex flex-wrap items-center gap-x-4 gap-y-2">
        <Tabs
          label="Outcome filter"
          active={active}
          onSelect={(id) => {
            const filter = FILTERS.find((entry) => entry.id === id);
            onChange({
              mistakesOnly: id === "mistakes",
              outcome: filter?.outcomes?.length === 1 ? filter.outcomes[0] : undefined,
            });
          }}
          items={FILTERS.map((filter) => ({
            id: filter.id,
            label: filter.label,
            count: countFor(filter),
            disabled: countFor(filter) === 0 && filter.id !== "all",
          }))}
        />
        <SegmentedControl
          aria-label="Rank order"
          value={state.sort}
          options={[
            { value: "score-desc", label: "most anomalous" },
            { value: "score-asc", label: "least" },
          ]}
          onValueChange={(sort) =>
            onChange({ sort: sort === "score-asc" ? "score-asc" : "score-desc" })
          }
        />
      </div>

      <OverlayControls
        state={state}
        onChange={onChange}
        range={range}
        hasMask={anyMask}
        hasMap={anyMap}
      />

      {verdicts.isPending || previews.isPending ? (
        <div className="grid grid-cols-[repeat(auto-fill,minmax(11rem,1fr))] gap-3">
          {Array.from({ length: 12 }, (_, index) => (
            <Skeleton key={index} className="aspect-square" />
          ))}
        </div>
      ) : ordered.length === 0 ? (
        <Empty>Nothing in this subset matches that filter.</Empty>
      ) : (
        <ul className="grid grid-cols-[repeat(auto-fill,minmax(11rem,1fr))] gap-3">
          {ordered.map((verdict) => (
            <Tile
              key={verdict.sample_id}
              experimentId={experimentId}
              verdict={verdict}
              preview={bySample.get(verdict.sample_id)}
              state={state}
              cut={cut}
            />
          ))}
        </ul>
      )}
    </div>
  );
}

function Tile({
  experimentId,
  verdict,
  preview,
  state,
  cut,
}: {
  experimentId: number;
  verdict: SampleVerdict;
  preview: SamplePreview | undefined;
  state: ResultsState;
  cut: number | null;
}) {
  const search = writeResultsState(state).toString();

  return (
    <li>
      <Link
        to={{
          pathname: `/experiments/${experimentId}/samples/${verdict.sample_id}`,
          search,
        }}
        className={cn(
          "group flex flex-col gap-1 rounded-panel focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-signal",
        )}
      >
        <div className="relative aspect-square overflow-hidden rounded border border-line bg-[#08090a] transition-colors group-hover:border-line-strong">
          {preview === undefined ? (
            <Skeleton className="h-full w-full" />
          ) : (
            <>
              <img
                src={imageUrl(preview.image_id, "thumb")}
                alt=""
                loading="lazy"
                className="h-full w-full object-cover"
              />
              {state.heatmap && preview.has_map && (
                <img
                  src={anomalyMapUrl(preview.image_id, experimentId)}
                  alt=""
                  aria-hidden
                  loading="lazy"
                  className="pointer-events-none absolute inset-0 h-full w-full object-cover opacity-80"
                />
              )}
              {state.region && preview.has_map && cut !== null && (
                <img
                  src={predictionUrl(preview.image_id, experimentId, cut)}
                  alt=""
                  aria-hidden
                  loading="lazy"
                  className="pointer-events-none absolute inset-0 h-full w-full object-cover"
                />
              )}
              {state.truth && preview.has_mask && (
                <img
                  src={maskUrl(preview.image_id)}
                  alt=""
                  aria-hidden
                  loading="lazy"
                  className="pointer-events-none absolute inset-0 h-full w-full object-cover"
                />
              )}
            </>
          )}
          <span className="absolute right-1 bottom-1 rounded bg-black/70 px-1 font-mono text-[10px] text-white tabular-nums">
            {verdict.score.toFixed(3)}
          </span>
        </div>

        <span className="flex items-center gap-1.5 overflow-hidden">
          <Badge tone={OUTCOME_TONE[verdict.outcome] ?? "neutral"}>
            {SHORT_OUTCOME[verdict.outcome] ?? OUTCOME_LABEL[verdict.outcome] ?? verdict.outcome}
          </Badge>
          {/* The external id alone. The group key is a directory path shared by every
              sample in the set, so at tile width it truncates to the part they have in
              common and identifies nothing. The full pair is on the sample page. */}
          <span
            className="truncate font-mono text-[11px] text-fg-muted"
            title={`${verdict.group_key}/${verdict.external_id}`}
          >
            {verdict.external_id}
          </span>
        </span>
      </Link>
    </li>
  );
}

/** The tile is 11rem wide; "false negative" does not fit beside an identifier. */
const SHORT_OUTCOME: Record<string, string> = {
  tp: "TP",
  tn: "TN",
  fp: "FP",
  fn: "FN",
  unlabeled: "—",
};
