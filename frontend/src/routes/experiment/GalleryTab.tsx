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

import type { MapScale, SamplePreview, SampleVerdict, Subset, Task } from "../../api/client";
import {
  anomalyMapUrl,
  boxMapUrl,
  imageUrl,
  labelMapUrl,
  maskUrl,
  predictionUrl,
} from "../../api/imageUrl";
import type { ResultsState } from "../../api/resultsState";
import { cutValue, writeResultsState } from "../../api/resultsState";
import { Badge, Empty, ErrorBox, SegmentedControl, Select, Skeleton, Tabs, cn } from "@vitavision/lab-ui";
import { boxToneColours } from "../../components/viewer/boxTones";
import { useSamplePreviews } from "../../hooks/useExperiments";
import { OverlayControls } from "./OverlayControls";
import { OUTCOME_LABEL, OUTCOME_TONE } from "./ResultsPanel";
import { taskView, type OutcomeFilter } from "./taskViews";
import type { Verdicts } from "./useVerdicts";

export function GalleryTab({
  experimentId,
  state,
  onChange,
  verdicts,
  subsets,
  range,
  task,
  targetLabel,
  classes,
}: {
  experimentId: number;
  /**
   * A supervised run's pinned classes. Given, the overlay row is the label-map set with its
   * legend and each tile draws the run's label maps instead of a cut — or, for an object
   * detection run, its boxes at the run's resolved cut, and the legend adds their tones.
   */
  classes?: readonly string[] | undefined;
  /** Decides the outcome strip and its words (`taskViews.tsx`). */
  task: Task | undefined;
  targetLabel: string | null;
  state: ResultsState;
  onChange: (next: Partial<ResultsState>) => void;
  verdicts: Verdicts;
  /** The subsets this run scored; the picker appears only when there is a choice. */
  subsets: readonly Subset[];
  range: MapScale | null | undefined;
}) {
  const previews = useSamplePreviews(experimentId, state.subset);
  const view = taskView(task);
  const FILTERS = view.filters;

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

  const countFor = (filter: OutcomeFilter) =>
    filter.outcomes === undefined
      ? verdicts.all.length
      : filter.outcomes.reduce((total, outcome) => total + (counts.get(outcome) ?? 0), 0);

  if (verdicts.error) return <ErrorBox>{verdicts.error.message}</ErrorBox>;

  const anyMask = (previews.data ?? []).some((preview) => preview.has_mask);
  const anyMap = (previews.data ?? []).some((preview) => preview.has_map);
  const cut = cutValue(state, range);
  const boxed = task === "object_detection" && classes !== undefined;
  // Read once for the grid, not per tile: the tones the server draws every tile's boxes in.
  const tones = boxed ? boxToneColours() : undefined;

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
            { value: "score-desc", label: view.rank.desc },
            { value: "score-asc", label: view.rank.asc },
          ]}
          onValueChange={(sort) =>
            onChange({ sort: sort === "score-asc" ? "score-asc" : "score-desc" })
          }
        />
        {subsets.length > 1 && (
          <Select
            className="w-32"
            aria-label="Subset"
            value={state.subset ?? subsets.at(-1) ?? ""}
            options={subsets.map((name) => ({ value: name, label: name }))}
            onValueChange={(value) => onChange({ subset: value as Subset, threshold: undefined })}
          />
        )}
      </div>

      {/* The badges are a verdict against something — a threshold, a class — and the reader
          needs to know which, or an outcome reads as a fact about the sample. */}
      {!verdicts.isPending && (
        <p className="text-xs text-fg-muted">{view.outcomeNote(verdicts, targetLabel)}</p>
      )}

      <OverlayControls
        state={state}
        onChange={onChange}
        range={range}
        hasMask={anyMask}
        hasMap={anyMap}
        classes={boxed ? undefined : classes}
        boxes={boxed ? { classes, rule: verdicts.rationale } : undefined}
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
              task={task}
              cut={cut}
              classes={boxed ? undefined : classes}
              tones={tones}
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
  task,
  cut,
  classes,
  tones,
}: {
  experimentId: number;
  verdict: SampleVerdict;
  preview: SamplePreview | undefined;
  state: ResultsState;
  task: Task | undefined;
  cut: number | null;
  classes: readonly string[] | undefined;
  /** A detection run's box tones; given, the tile draws its boxes. */
  tones: readonly [string, string, string] | undefined;
}) {
  const search = writeResultsState(state, task).toString();

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
              {tones !== undefined ? (
                /* The boxes, drawn by the server as one SVG at the source's size and
                   stretched as the thumbnail is — the sample page's `VectorLayer`, one
                   picture per tile instead of a box list. */
                (state.region || state.truth) && (
                  <img
                    src={boxMapUrl(preview.image_id, experimentId, tones, {
                      predictions: state.region,
                      truth: state.truth,
                    })}
                    alt=""
                    aria-hidden
                    loading="lazy"
                    data-boxes
                    className="pointer-events-none absolute inset-0 h-full w-full object-cover"
                  />
                )
              ) : classes !== undefined ? (
                <>
                  {/* The label maps, drawn by the server at the thumbnail's size in the
                      colours `labelMapUrl` names — the sample page's `LabelLayer`, one image
                      per layer instead of a value plane. */}
                  {state.region && (
                    <img
                      src={labelMapUrl(preview.image_id, experimentId, classes, false)}
                      alt=""
                      aria-hidden
                      loading="lazy"
                      data-labels="prediction"
                      className="pointer-events-none absolute inset-0 h-full w-full object-cover"
                    />
                  )}
                  {state.truth && preview.has_mask && (
                    <img
                      src={labelMapUrl(preview.image_id, experimentId, classes, true)}
                      alt=""
                      aria-hidden
                      loading="lazy"
                      data-labels="truth"
                      className="pointer-events-none absolute inset-0 h-full w-full object-cover"
                    />
                  )}
                </>
              ) : (
                <>
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
          {/* Only the failure, and only here: a tile is 11rem wide, so a badge on every
              localized sample would spend the identifier's width restating the common
              case. The rare row that fired off target is the one worth finding. */}
          {verdict.localized === false && (
            <Badge tone="warning" className="shrink-0">
              off target
            </Badge>
          )}
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
