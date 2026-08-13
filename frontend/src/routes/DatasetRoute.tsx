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
import { useParams, useSearchParams } from "react-router";

import type { BrowseState } from "../api/browseState";
import {
  PAGE_SIZE,
  readBrowseState,
  toBulkFilters,
  toSampleQuery,
  writeBrowseState,
} from "../api/browseState";
import type { DatasetDetail, Label, SampleSummary, SplitDetail, Subset } from "../api/client";
import { SlidersHorizontal } from "lucide-react";

import {
  Button,
  Disclosure,
  Empty,
  ErrorBox,
  Field,
  Select,
  SkeletonRows,
} from "../components/ui";
import { useDataset, useSamples, useSetLabels, useSplits } from "../hooks/useCatalog";
import { SampleTile, type SelectModifiers } from "./dataset/SampleTile";

const COLUMNS = 6;
const ROW_HEIGHT = 132;

/** What stands in for the virtual grid while there is nothing to virtualize. */
const PLACEHOLDER_SCROLLER =
  "min-h-0 flex-1 overflow-y-auto overscroll-contain p-4 [scrollbar-gutter:stable]";

const LABELS: Label[] = ["normal", "defect", "unlabeled"];

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

  // No early return: the band above belongs to the layout and must stay put while this
  // loads. The filter rail carries the wait instead, because it is the only part of this
  // screen that needs the dataset — it builds the channel picker from the channel dictionary.
  const detail = dataset.data;
  const items = page.data?.items ?? [];
  const total = page.data?.total ?? 0;
  const activeFilters = [browse.label, browse.channelId, browse.splitId, browse.subset].filter(
    (value) => value !== undefined,
  ).length;

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

  const filters = detail ? (
    <DatasetFilters
      browse={browse}
      detail={detail}
      splits={splits.data ?? []}
      onChange={setFilter}
    />
  ) : (
    <SkeletonRows rows={4} />
  );

  return (
    <div className="flex min-h-0 flex-1 flex-col bg-ground">
      <div className="border-b border-line bg-surface px-4 py-2 lg:hidden">
        <Disclosure
          summary={
            <span className="flex items-center gap-1.5">
              <SlidersHorizontal className="size-3.5" aria-hidden /> Filters
            </span>
          }
          count={activeFilters > 0 ? activeFilters : undefined}
        >
          <div className="pb-2">
            {filters}
            {activeFilters > 0 && (
              <Button className="mt-3" size="sm" onClick={() => setSearchParams({})}>
                Clear filters
              </Button>
            )}
          </div>
        </Disclosure>
      </div>

      <div className="mx-auto flex min-h-0 w-full max-w-[100rem] flex-1">
        <aside
          aria-label="Dataset filters"
          className="hidden w-64 shrink-0 border-r border-line bg-surface lg:flex lg:flex-col"
        >
          <div className="flex items-center justify-between border-b border-line px-4 py-3">
            <div className="flex items-center gap-2">
              <SlidersHorizontal className="size-4 text-fg-muted" aria-hidden />
              <h2 className="text-sm font-semibold tracking-tight">Filters</h2>
              {activeFilters > 0 && (
                <span className="font-mono text-xs text-fg-subtle">{activeFilters}</span>
              )}
            </div>
            {activeFilters > 0 && (
              <Button variant="ghost" size="sm" onClick={() => setSearchParams({})}>
                Clear
              </Button>
            )}
          </div>
          {/* A peer column beside the grid, not a scroller stacked inside it: without this
              the rail is clipped by the workspace's `overflow-hidden` and a long channel
              dictionary becomes unreachable. */}
          <div className="min-h-0 flex-1 overflow-y-auto overscroll-contain p-4">{filters}</div>
        </aside>

        <section className="relative flex min-h-0 min-w-0 flex-1 flex-col bg-surface">
          {dataset.error && <div className="p-4"><ErrorBox>{dataset.error.message}</ErrorBox></div>}
          {page.error && <div className="p-4"><ErrorBox>{page.error.message}</ErrorBox></div>}
          {setLabels.error && (
            <div className="p-4"><ErrorBox>{setLabels.error.message}</ErrorBox></div>
          )}

          {/* The grid is this tab's one scroller, so a state that renders no grid has to
              stand in for it — otherwise the gutter it reserves blinks in and out. */}
          {!page.data && (
            <div data-scroll="tab" className={PLACEHOLDER_SCROLLER}>
              {page.isPending && <SkeletonRows rows={6} />}
            </div>
          )}

          {page.data && (
            <>
              <div className="flex min-h-14 flex-wrap items-center justify-between gap-2 border-b border-line px-4 py-2.5 text-sm text-fg-muted">
                <span className="flex flex-wrap items-center gap-2">
                  <span>
                    {total} sample{total === 1 ? "" : "s"}
                    {total > PAGE_SIZE &&
                      ` · ${browse.offset + 1}–${Math.min(browse.offset + PAGE_SIZE, total)}`}
                  </span>

                  {total > 0 &&
                    (labellingAll ? (
                      <span className="flex flex-wrap items-center gap-1.5">
                        <span className="text-fg">Label all {total} as</span>
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
                <div data-scroll="tab" className={PLACEHOLDER_SCROLLER}>
                  <Empty>No samples match these filters.</Empty>
                </div>
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
            <div className="absolute inset-x-4 bottom-4 z-10 flex flex-wrap items-center justify-between gap-3 rounded-panel border border-line-strong bg-overlay/95 px-4 py-3 shadow-lg shadow-black/10 backdrop-blur">
              <span className="text-sm font-medium">
                {selected.size} selected
                <span className="ml-2 font-normal text-fg-muted">
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
        </section>
      </div>
    </div>
  );
}

function DatasetFilters({
  browse,
  detail,
  splits,
  onChange,
}: {
  browse: BrowseState;
  detail: DatasetDetail;
  splits: SplitDetail[];
  onChange: (patch: Partial<BrowseState>) => void;
}) {
  return (
    <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-1">
      <Field as="group" label="Label">
        <Select
          aria-label="Label"
          value={browse.label ?? ""}
          placeholder="Any label"
          unsetLabel="Any label"
          options={LABELS.map((label) => ({ value: label, label }))}
          onValueChange={(value) =>
            onChange({ label: (value || undefined) as Label | undefined })
          }
        />
      </Field>

      <Field as="group" label="Channel">
        <Select
          aria-label="Channel"
          value={browse.channelId === undefined ? "" : String(browse.channelId)}
          placeholder="Any channel"
          unsetLabel="Any channel"
          options={detail.channels.map((channel) => ({
            value: String(channel.id),
            label: channel.name,
          }))}
          onValueChange={(value) =>
            onChange({ channelId: value === "" ? undefined : Number(value) })
          }
        />
      </Field>

      <Field as="group" label="Split">
        <Select
          aria-label="Split"
          value={browse.splitId === undefined ? "" : String(browse.splitId)}
          placeholder="No split"
          unsetLabel="No split"
          options={splits.map((split) => ({ value: String(split.id), label: split.name }))}
          onValueChange={(value) =>
            onChange({
              splitId: value === "" ? undefined : Number(value),
              ...(value === "" ? { subset: undefined } : {}),
            })
          }
        />
      </Field>

      <Field as="group" label="Subset">
        <Select
          aria-label="Subset"
          value={browse.subset ?? ""}
          placeholder={browse.splitId === undefined ? "Pick a split first" : "Any subset"}
          unsetLabel="Any subset"
          disabled={browse.splitId === undefined}
          options={[
            { value: "train", label: "train" },
            { value: "val", label: "val" },
            { value: "test", label: "test" },
          ]}
          onValueChange={(value) =>
            onChange({ subset: (value || undefined) as Subset | undefined })
          }
        />
      </Field>
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
  const select = (index: number, event: SelectModifiers) => {
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
      /* This tab's one scroll region. It stays inside `SampleGrid` because the virtualizer
         measures the element it holds a ref to; hoisting it would measure the wrong box. */
      data-scroll="tab"
      tabIndex={-1}
      onKeyDown={(event) => {
        if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "a") {
          event.preventDefault();
          selectAll();
        }
      }}
      className="min-h-0 flex-1 overflow-y-auto overscroll-contain [scrollbar-gutter:stable]"
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
