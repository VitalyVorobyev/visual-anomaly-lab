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
import type { DatasetDetail, Label, SampleSummary, SplitDetail, Subset } from "../api/client";
import { Scissors, SlidersHorizontal } from "lucide-react";

import {
  Badge,
  Button,
  CountRun,
  Disclosure,
  Empty,
  ErrorBox,
  Field,
  PageHeader,
  ReadoutStrip,
  Select,
  SkeletonRows,
} from "../components/ui";
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
  if (dataset.isPending || !dataset.data) return <SkeletonRows rows={6} />;

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

  const filters = (
    <DatasetFilters
      browse={browse}
      detail={detail}
      splits={splits.data ?? []}
      onChange={setFilter}
    />
  );

  return (
    <div className="flex min-h-0 flex-1 flex-col bg-ground">
      <header className="border-b border-line bg-ground px-5 py-4 lg:px-6">
        <div className="mx-auto flex w-full max-w-[100rem] flex-col gap-3">
          <PageHeader
            title={detail.name}
            actions={
              <Link to={`/datasets/${datasetId}/splits`}>
                <Button icon={<Scissors />}>Splits ({detail.splits})</Button>
              </Link>
            }
            meta={
              <ReadoutStrip
                items={[
                  { label: "samples", value: detail.samples },
                  { label: "images", value: detail.images },
                  ...(detail.channels.length > 0
                    ? [{ label: "channels", value: detail.channels.length }]
                    : []),
                  { value: detail.root_path },
                ]}
              />
            }
          />
          <CountRun
            counts={[
              ["normal", detail.label_counts["normal"] ?? 0, "normal"],
              ["defect", detail.label_counts["defect"] ?? 0, "defect"],
              ["unlabeled", detail.label_counts["unlabeled"] ?? 0, "unlabeled"],
            ]}
          />
        </div>
      </header>

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
          <div className="p-4">{filters}</div>
        </aside>

        <section className="relative flex min-h-0 min-w-0 flex-1 flex-col bg-surface">
          {page.error && <div className="p-4"><ErrorBox>{page.error.message}</ErrorBox></div>}
          {setLabels.error && (
            <div className="p-4"><ErrorBox>{setLabels.error.message}</ErrorBox></div>
          )}

          {page.isPending && !page.data && <div className="p-4"><SkeletonRows rows={6} /></div>}

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
                <div className="p-4"><Empty>No samples match these filters.</Empty></div>
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
      className="min-h-0 flex-1 overflow-y-auto overscroll-contain"
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
      className={`relative flex flex-col overflow-hidden rounded-control border transition-colors ${
        selected ? "border-signal ring-2 ring-signal" : "border-line hover:border-line-strong"
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
        className="absolute top-1 left-1 z-10 h-4 w-4 cursor-pointer accent-signal"
      />
      <div className="flex h-20 items-center justify-center bg-raised">
        {cover ? (
          <img
            // The grid never asks for a full-resolution decode.
            src={imageUrl(cover.id, "thumb")}
            alt=""
            loading="lazy"
            className="h-full w-full object-contain"
          />
        ) : (
          <span className="text-xs text-fg-subtle">no image</span>
        )}
      </div>
      <div className="flex items-center justify-between gap-1 px-1.5 py-1">
        <span className="truncate font-mono text-xs">{sample.external_id}</span>
        <span className="flex items-center gap-1">
          {/* The channel count is whatever this sample has. */}
          <span className="font-mono text-[10px] text-fg-subtle">{sample.images.length}ch</span>
          <Badge tone={LABEL_TONE[sample.label]}>{sample.label.slice(0, 3)}</Badge>
        </span>
      </div>
    </Link>
  );
}
