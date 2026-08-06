/**
 * The dataset browser: a virtualized thumbnail grid with filters.
 *
 * Two things keep this fast enough to be pleasant, which is the M2 exit criterion:
 *
 *  * **Only thumbnails are fetched on this path.** A 256 px WebP is about 4 KB against
 *    3.9 MB for the source image, and the full tier is never requested here.
 *  * **Rows are virtualized.** Only the visible ones are in the DOM, so the grid costs
 *    the same at 3 000 samples as at 30.
 *
 * The channel filter is built from the dataset's own channel dictionary. Nothing in this
 * file knows how many channels there are, and a two-channel sample renders through the
 * same code as a three-channel one.
 */

import { useVirtualizer } from "@tanstack/react-virtual";
import { useRef, useState } from "react";
import { Link, useParams } from "react-router";

import { imageUrl } from "../api/imageUrl";
import type { Label, SampleSummary, Subset } from "../api/client";
import { Badge, Button, CountRun, Empty, ErrorBox, Panel, inputClasses } from "../components/ui";
import { useDataset, useSamples, useSplits } from "../hooks/useCatalog";

const PAGE_SIZE = 200;
const COLUMNS = 6;
const ROW_HEIGHT = 132;

const LABEL_TONE: Record<Label, "normal" | "defect" | "unlabeled"> = {
  normal: "normal",
  defect: "defect",
  unlabeled: "unlabeled",
};

export function DatasetRoute() {
  const params = useParams();
  const datasetId = Number(params["datasetId"]);

  const [label, setLabel] = useState<Label | "">("");
  const [channelId, setChannelId] = useState<number | "">("");
  const [splitId, setSplitId] = useState<number | "">("");
  const [subset, setSubset] = useState<Subset | "">("");
  const [offset, setOffset] = useState(0);

  const dataset = useDataset(datasetId);
  const splits = useSplits(datasetId);
  const page = useSamples(datasetId, {
    label: label || undefined,
    channelId: channelId === "" ? undefined : channelId,
    splitId: splitId === "" ? undefined : splitId,
    subset: subset || undefined,
    limit: PAGE_SIZE,
    offset,
  });

  if (dataset.error) return <ErrorBox>{dataset.error.message}</ErrorBox>;
  if (dataset.isPending || !dataset.data) return <Empty>Loading…</Empty>;

  const detail = dataset.data;
  const reset = <T,>(setter: (value: T) => void) => (value: T) => {
    setter(value);
    setOffset(0);
  };

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
            value={label}
            onChange={(event) => reset(setLabel)(event.target.value as Label | "")}
          >
            <option value="">Any label</option>
            <option value="normal">normal</option>
            <option value="defect">defect</option>
            <option value="unlabeled">unlabeled</option>
          </select>

          {/* Options come from the dataset's channel dictionary, whatever its length. */}
          <select
            aria-label="Channel"
            className={inputClasses}
            value={channelId}
            onChange={(event) =>
              reset(setChannelId)(event.target.value === "" ? "" : Number(event.target.value))
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
            value={splitId}
            onChange={(event) => {
              reset(setSplitId)(event.target.value === "" ? "" : Number(event.target.value));
              if (event.target.value === "") setSubset("");
            }}
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
            value={subset}
            disabled={splitId === ""}
            onChange={(event) => reset(setSubset)(event.target.value as Subset | "")}
          >
            <option value="">Any subset</option>
            <option value="train">train</option>
            <option value="val">val</option>
            <option value="test">test</option>
          </select>
        </div>
      </Panel>

      {page.error && <ErrorBox>{page.error.message}</ErrorBox>}
      {page.data && (
        <>
          <div className="flex items-center justify-between text-sm text-slate-500 dark:text-slate-400">
            <span>
              {page.data.total} sample{page.data.total === 1 ? "" : "s"}
              {page.data.total > PAGE_SIZE &&
                ` · showing ${offset + 1}–${Math.min(offset + PAGE_SIZE, page.data.total)}`}
            </span>
            {page.data.total > PAGE_SIZE && (
              <span className="flex gap-2">
                <Button disabled={offset === 0} onClick={() => setOffset(Math.max(0, offset - PAGE_SIZE))}>
                  Previous
                </Button>
                <Button
                  disabled={offset + PAGE_SIZE >= page.data.total}
                  onClick={() => setOffset(offset + PAGE_SIZE)}
                >
                  Next
                </Button>
              </span>
            )}
          </div>

          {page.data.items.length === 0 ? (
            <Empty>No samples match these filters.</Empty>
          ) : (
            <SampleGrid datasetId={datasetId} samples={page.data.items} />
          )}
        </>
      )}
    </div>
  );
}

function SampleGrid({ datasetId, samples }: { datasetId: number; samples: SampleSummary[] }) {
  const scrollRef = useRef<HTMLDivElement>(null);
  const rowCount = Math.ceil(samples.length / COLUMNS);

  const virtualizer = useVirtualizer({
    count: rowCount,
    getScrollElement: () => scrollRef.current,
    estimateSize: () => ROW_HEIGHT,
    overscan: 3,
  });

  return (
    <div
      ref={scrollRef}
      data-testid="sample-grid"
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
              .map((sample) => (
                <SampleTile key={sample.id} datasetId={datasetId} sample={sample} />
              ))}
          </div>
        ))}
      </div>
    </div>
  );
}

function SampleTile({ datasetId, sample }: { datasetId: number; sample: SampleSummary }) {
  const cover = sample.images[0];

  return (
    <Link
      to={`/datasets/${datasetId}/samples/${sample.id}`}
      className="flex flex-col overflow-hidden rounded border border-slate-200 hover:border-slate-400 dark:border-slate-700"
      title={`${sample.group_key}/${sample.external_id}`}
    >
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
