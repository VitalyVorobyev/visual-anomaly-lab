/**
 * The dataset browser: a virtualized thumbnail grid with filters and bulk labelling.
 *
 * Three things keep this fast enough to be pleasant, which is the M2 exit criterion:
 *
 *  * **Only thumbnails are fetched on this path.** A 256 px WebP is about 4 KB against
 *    several megabytes for the source image, and the full tier is never requested here.
 *  * **Rows are virtualized.** Only the visible ones are in the DOM, so the grid costs
 *    the same at 3 000 samples as at 30.
 *  * **Bulk labelling is a filter, not a list of ids.** "Label all 203 matching" sends the
 *    filters and lets the server resolve them, so it labels the whole matching set rather
 *    than the one page the client happens to be holding.
 *
 * The channel filter is built from the dataset's own channel dictionary. Nothing in this
 * file knows how many channels there are, and a two-channel sample renders through the
 * same code as a three-channel one.
 */

import { useVirtualizer } from "@tanstack/react-virtual";
import { useEffect, useRef, useState } from "react";
import { Link, useParams, useSearchParams } from "react-router";

import type { BrowseState } from "../api/browseState";
import {
  PAGE_SIZE,
  readBrowseState,
  toBulkFilters,
  toSampleQuery,
  writeBrowseState,
} from "../api/browseState";
import { imageUrl } from "../api/imageUrl";
import type { Label, SampleSummary, Subset } from "../api/client";
import { Badge, Button, CountRun, Empty, ErrorBox, Panel, inputClasses } from "../components/ui";
import { useDataset, useSamples, useSetLabels, useSplits } from "../hooks/useCatalog";

const COLUMNS = 6;
const ROW_HEIGHT = 132;

const LABELS: Label[] = ["normal", "defect", "unlabeled"];

const LABEL_TONE: Record<Label, "normal" | "defect" | "unlabeled"> = {
  normal: "normal",
  defect: "defect",
  unlabeled: "unlabeled",
};

export function DatasetRoute() {
  const params = useParams();
  const datasetId = Number(params["datasetId"]);

  const [searchParams, setSearchParams] = useSearchParams();
  const browse = readBrowseState(searchParams);
  const search = writeBrowseState(browse).toString();

  const dataset = useDataset(datasetId);
  const splits = useSplits(datasetId);
  const page = useSamples(datasetId, toSampleQuery(browse));
  const setLabels = useSetLabels(datasetId);

  const [selected, setSelected] = useState<ReadonlySet<number>>(new Set());
  const [labellingAll, setLabellingAll] = useState(false);

  // Any change of filter or page makes the previous selection meaningless — the samples
  // it refers to may not even be on screen any more.
  useEffect(() => {
    setSelected(new Set());
    setLabellingAll(false);
  }, [search]);

  if (dataset.error) return <ErrorBox>{dataset.error.message}</ErrorBox>;
  if (dataset.isPending || !dataset.data) return <Empty>Loading…</Empty>;

  const detail = dataset.data;
  const items = page.data?.items ?? [];
  const total = page.data?.total ?? 0;

  /** Changing a filter returns to the first page; the old offset would be meaningless. */
  const setFilter = (patch: Partial<BrowseState>) =>
    setSearchParams(writeBrowseState({ ...browse, ...patch, offset: 0 }));

  const setOffset = (offset: number) => setSearchParams(writeBrowseState({ ...browse, offset }));

  const labelSelected = (label: Label) =>
    setLabels.mutate(
      { label, sample_ids: [...selected] },
      { onSuccess: () => setSelected(new Set()) },
    );

  const labelEverythingMatching = (label: Label) =>
    setLabels.mutate(
      { label, filters: toBulkFilters(browse) },
      { onSuccess: () => setLabellingAll(false) },
    );

  return (
    <div className="flex flex-col gap-4">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div>
          <h2 className="text-lg font-semibold tracking-tight">{detail.name}</h2>
          <p className="font-mono text-xs break-all text-slate-400">{detail.root_path}</p>
        </div>
        <div className="flex items-center gap-2">
          <CountRun
            counts={[
              ["normal", detail.label_counts["normal"] ?? 0, "normal"],
              ["defect", detail.label_counts["defect"] ?? 0, "defect"],
              ["unlabeled", detail.label_counts["unlabeled"] ?? 0, "unlabeled"],
            ]}
          />
          <Link to={`/datasets/${datasetId}/splits`}>
            <Button>Splits ({detail.splits})</Button>
          </Link>
        </div>
      </div>

      <Panel title="Filters">
        <div className="flex flex-wrap gap-3">
          <select
            aria-label="Label"
            className={inputClasses}
            value={browse.label ?? ""}
            onChange={(event) =>
              setFilter({ label: (event.target.value || undefined) as Label | undefined })
            }
          >
            <option value="">Any label</option>
            {LABELS.map((label) => (
              <option key={label} value={label}>
                {label}
              </option>
            ))}
          </select>

          {/* Options come from the dataset's channel dictionary, whatever its length. */}
          <select
            aria-label="Channel"
            className={inputClasses}
            value={browse.channelId ?? ""}
            onChange={(event) =>
              setFilter({
                channelId: event.target.value === "" ? undefined : Number(event.target.value),
              })
            }
          >
            <option value="">Any channel</option>
            {detail.channels.map((channel) => (
              <option key={channel.id} value={channel.id}>
                {channel.name}
              </option>
            ))}
          </select>

          <select
            aria-label="Split"
            className={inputClasses}
            value={browse.splitId ?? ""}
            onChange={(event) =>
              setFilter({
                splitId: event.target.value === "" ? undefined : Number(event.target.value),
                // A subset without a split is meaningless.
                ...(event.target.value === "" ? { subset: undefined } : {}),
              })
            }
          >
            <option value="">No split</option>
            {(splits.data ?? []).map((split) => (
              <option key={split.id} value={split.id}>
                {split.name}
              </option>
            ))}
          </select>

          <select
            aria-label="Subset"
            className={inputClasses}
            value={browse.subset ?? ""}
            disabled={browse.splitId === undefined}
            onChange={(event) =>
              setFilter({ subset: (event.target.value || undefined) as Subset | undefined })
            }
          >
            <option value="">Any subset</option>
            <option value="train">train</option>
            <option value="val">val</option>
            <option value="test">test</option>
          </select>
        </div>
      </Panel>

      {page.error && <ErrorBox>{page.error.message}</ErrorBox>}
      {setLabels.error && <ErrorBox>{setLabels.error.message}</ErrorBox>}

      {page.data && (
        <>
          <div className="flex flex-wrap items-center justify-between gap-2 text-sm text-slate-500 dark:text-slate-400">
            <span className="flex flex-wrap items-center gap-2">
              <span>
                {total} sample{total === 1 ? "" : "s"}
                {total > PAGE_SIZE &&
                  ` · showing ${browse.offset + 1}–${Math.min(browse.offset + PAGE_SIZE, total)}`}
              </span>

              {/* The count in the sentence is the confirmation: you cannot press the
                  label without having read how many samples it will touch. */}
              {total > 0 &&
                (labellingAll ? (
                  <span className="flex flex-wrap items-center gap-1.5">
                    <span className="text-slate-700 dark:text-slate-200">
                      Label all {total} as
                    </span>
                    {LABELS.map((label) => (
                      <Button
                        key={label}
                        variant="primary"
                        disabled={setLabels.isPending}
                        onClick={() => labelEverythingMatching(label)}
                      >
                        {label}
                      </Button>
                    ))}
                    <Button onClick={() => setLabellingAll(false)}>Cancel</Button>
                  </span>
                ) : (
                  <Button onClick={() => setLabellingAll(true)}>Label all {total}…</Button>
                ))}
            </span>

            {total > PAGE_SIZE && (
              <span className="flex gap-2">
                <Button
                  disabled={browse.offset === 0}
                  onClick={() => setOffset(Math.max(0, browse.offset - PAGE_SIZE))}
                >
                  Previous
                </Button>
                <Button
                  disabled={browse.offset + PAGE_SIZE >= total}
                  onClick={() => setOffset(browse.offset + PAGE_SIZE)}
                >
                  Next
                </Button>
              </span>
            )}
          </div>

          {items.length === 0 ? (
            <Empty>No samples match these filters.</Empty>
          ) : (
            <SampleGrid
              datasetId={datasetId}
              samples={items}
              search={search}
              selected={selected}
              onSelected={setSelected}
            />
          )}
        </>
      )}

      {selected.size > 0 && (
        <div className="sticky bottom-4 z-10 flex flex-wrap items-center justify-between gap-3 rounded-lg border border-slate-300 bg-white/95 px-4 py-3 shadow-lg backdrop-blur dark:border-slate-600 dark:bg-slate-900/95">
          <span className="text-sm font-medium">
            {selected.size} selected
            <span className="ml-2 font-normal text-slate-500 dark:text-slate-400">
              shift-click for a range · ⌘/ctrl-click to toggle
            </span>
          </span>
          <span className="flex flex-wrap gap-2">
            {LABELS.map((label) => (
              <Button
                key={label}
                variant="primary"
                disabled={setLabels.isPending}
                onClick={() => labelSelected(label)}
              >
                {label}
              </Button>
            ))}
            <Button onClick={() => setSelected(new Set())}>Clear</Button>
          </span>
        </div>
      )}
    </div>
  );
}

function SampleGrid({
  datasetId,
  samples,
  search,
  selected,
  onSelected,
}: {
  datasetId: number;
  samples: SampleSummary[];
  search: string;
  selected: ReadonlySet<number>;
  onSelected: (selected: ReadonlySet<number>) => void;
}) {
  const scrollRef = useRef<HTMLDivElement>(null);
  const anchor = useRef<number | null>(null);
  const rowCount = Math.ceil(samples.length / COLUMNS);

  const virtualizer = useVirtualizer({
    count: rowCount,
    getScrollElement: () => scrollRef.current,
    estimateSize: () => ROW_HEIGHT,
    overscan: 3,
  });

  /**
   * File-manager selection semantics, so nobody has to learn a new idiom: plain click
   * still opens the sample, ⌘/ctrl toggles one, and shift extends from the last one
   * touched. Range extension works on the *displayed* order, which is the order the user
   * can see.
   */
  const select = (index: number, event: { shiftKey: boolean; metaKey: boolean; ctrlKey: boolean }) => {
    const sample = samples[index];
    if (!sample) return;
    const next = new Set(selected);

    if (event.shiftKey && anchor.current !== null) {
      const [from, to] = [anchor.current, index].sort((a, b) => a - b) as [number, number];
      for (const item of samples.slice(from, to + 1)) next.add(item.id);
    } else if (next.has(sample.id)) {
      next.delete(sample.id);
      anchor.current = index;
    } else {
      next.add(sample.id);
      anchor.current = index;
    }

    onSelected(next);
  };

  const selectAll = () => {
    onSelected(new Set(samples.map((sample) => sample.id)));
    anchor.current = null;
  };

  return (
    <div
      ref={scrollRef}
      data-testid="sample-grid"
      tabIndex={-1}
      onKeyDown={(event) => {
        if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "a") {
          event.preventDefault();
          selectAll();
        }
      }}
      className="h-[60vh] overflow-y-auto rounded-lg border border-slate-200 dark:border-slate-700"
    >
      <div style={{ height: virtualizer.getTotalSize(), position: "relative" }}>
        {virtualizer.getVirtualItems().map((row) => (
          <div
            key={row.key}
            className="absolute inset-x-0 grid gap-2 p-2"
            style={{
              gridTemplateColumns: `repeat(${COLUMNS}, minmax(0, 1fr))`,
              transform: `translateY(${row.start}px)`,
              height: ROW_HEIGHT,
            }}
          >
            {samples
              .slice(row.index * COLUMNS, row.index * COLUMNS + COLUMNS)
              .map((sample, column) => {
                const index = row.index * COLUMNS + column;
                return (
                  <SampleTile
                    key={sample.id}
                    datasetId={datasetId}
                    sample={sample}
                    search={search}
                    selected={selected.has(sample.id)}
                    onSelect={(event) => select(index, event)}
                  />
                );
              })}
          </div>
        ))}
      </div>
    </div>
  );
}

function SampleTile({
  datasetId,
  sample,
  search,
  selected,
  onSelect,
}: {
  datasetId: number;
  sample: SampleSummary;
  search: string;
  selected: boolean;
  onSelect: (event: { shiftKey: boolean; metaKey: boolean; ctrlKey: boolean }) => void;
}) {
  const cover = sample.images[0];

  return (
    <Link
      // The browse filters travel with the link so the viewer can page through the same
      // set the grid is showing.
      to={`/datasets/${datasetId}/samples/${sample.id}${search ? `?${search}` : ""}`}
      onClick={(event) => {
        // A modified click selects instead of navigating; a plain click still opens.
        if (event.shiftKey || event.metaKey || event.ctrlKey) {
          event.preventDefault();
          onSelect(event);
        }
      }}
      className={`relative flex flex-col overflow-hidden rounded border ${
        selected
          ? "border-slate-900 ring-2 ring-slate-900 dark:border-slate-100 dark:ring-slate-100"
          : "border-slate-200 hover:border-slate-400 dark:border-slate-700"
      }`}
      title={`${sample.group_key}/${sample.external_id}`}
    >
      <input
        type="checkbox"
        aria-label={`Select ${sample.group_key}/${sample.external_id}`}
        checked={selected}
        // The checkbox is the discoverable path to selection; the modifier keys are the
        // fast one. Both end up in the same place.
        onClick={(event) => {
          event.preventDefault();
          event.stopPropagation();
          onSelect(event);
        }}
        onChange={() => undefined}
        className="absolute top-1 left-1 z-10 h-4 w-4 cursor-pointer accent-slate-900 dark:accent-slate-100"
      />
      <div className="flex h-20 items-center justify-center bg-slate-100 dark:bg-slate-800">
        {cover ? (
          <img
            // The grid never asks for a full-resolution decode.
            src={imageUrl(cover.id, "thumb")}
            alt=""
            loading="lazy"
            className="h-full w-full object-contain"
          />
        ) : (
          <span className="text-xs text-slate-400">no image</span>
        )}
      </div>
      <div className="flex items-center justify-between gap-1 px-1.5 py-1">
        <span className="truncate font-mono text-xs">{sample.external_id}</span>
        <span className="flex items-center gap-1">
          {/* The channel count is whatever this sample has. */}
          <span className="font-mono text-[10px] text-slate-400">{sample.images.length}ch</span>
          <Badge tone={LABEL_TONE[sample.label]}>{sample.label.slice(0, 3)}</Badge>
        </span>
      </div>
    </Link>
  );
}
