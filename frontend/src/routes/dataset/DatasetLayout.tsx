/**
 * The dataset workspace: one band, rendered once, above five tabs.
 *
 * Each of the five tabs used to build its own header and render its own copy of the
 * section strip, and they disagreed on every axis -- two different layouts, two measures,
 * two paddings, a back link on three of them, an action on two, a `meta` line whose shape
 * depended on which page you were on. `PageHeader` reserves space for none of its optional
 * slots, so the strip below sat at a different offset per tab and visibly jumped when you
 * moved between them. Three tabs also early-returned a skeleton *before* their header, so
 * the strip vanished during load and popped back in.
 *
 * So the band belongs to the layout, not to the tabs. It carries the dataset's identity --
 * the only `<h1>` on any dataset screen, the facts, the label counts -- one action that is
 * a property of the dataset rather than of the tab, and the strip. Its height is fixed by
 * construction: the primary button sets the first row, `min-h-8` sets the second whether or
 * not the counts have anything to say, and the readout is held to one line. Nothing above
 * the strip moves, on any tab, in any state.
 *
 * **The facts are one run, counted in samples.** It used to read `samples 1100 · images
 * 1100 · /Users/…/VisA…` with the label counts stranded on the row below -- two counting
 * units for one dataset, and an absolute path as permanent furniture. `label_counts` and
 * split membership are stored per *sample* (ADR-0005), so the sample is the denominator
 * the counts beside them already use; the image count only says something a reader cannot
 * infer when a sample holds more than one, which is exactly when the channel count is
 * shown instead. The path and the adapter did not disappear -- they moved behind the
 * information mark, which is where a fact you consult belongs when it is not a fact you
 * read.
 *
 * Built by hand rather than through `PageHeader`, deliberately: that component stacks
 * `meta` *under* the title, and the whole point here is two rows of known height.
 *
 * Below the band it behaves like the app's other layouts in `App.tsx` -- it does not scroll,
 * and the tab names its own single scroll region (`TabScroll`, or, where the surface is
 * full-bleed, the virtual grid's own).
 */

import { Info, Plus } from "lucide-react";
import { Link, Outlet, useParams } from "react-router";

import { DatasetSectionNav } from "../../components/DatasetSectionNav";
import {
  Button,
  CountRun,
  InfoHint,
  ReadoutStrip,
  Skeleton,
  Tooltip,
} from "../../components/ui";
import { useDataset } from "../../hooks/useCatalog";

export function DatasetLayout() {
  const datasetId = Number(useParams()["datasetId"]);
  // Deliberately not gated on `isPending` or `error`: the band is the one thing on this
  // screen that must never be absent. A failed dataset read is reported by the tab, which
  // is where the reader is looking.
  const detail = useDataset(datasetId).data;

  return (
    <div
      data-layout="dataset"
      className="flex min-h-0 flex-1 flex-col overflow-hidden bg-ground"
    >
      <header
        data-band="dataset"
        className="shrink-0 border-b border-line bg-ground px-5 lg:px-6"
      >
        <div className="mx-auto flex w-full max-w-[100rem] flex-col gap-2 pt-3">
          <div className="flex items-center justify-between gap-x-4">
            <div className="flex min-w-0 items-center gap-x-3">
              <h1 className="shrink-0 text-xl font-semibold tracking-tight text-fg">
                {detail ? detail.name : <Skeleton className="h-6 w-44" />}
              </h1>
              {detail && (
                <>
                  {/* Dropped rather than wrapped below `lg`. `ReadoutStrip` wraps, and a
                      wrapped readout makes the band a line taller — which is the thing
                      this layout exists to prevent. The name carries a narrow window; the
                      facts come back when there is room for them on one line. */}
                  <ReadoutStrip
                    className="hidden min-w-0 lg:flex"
                    items={[
                      { value: `${detail.samples} samples` },
                      // Only when a sample is more than one image. At one channel per
                      // sample the number restates the sample count, and the reader has
                      // to decide whether it does before they can ignore it.
                      ...(detail.channels.length > 1
                        ? [{ value: `${detail.channels.length} channels` }]
                        : []),
                    ]}
                  />
                  <CountRun
                    counts={[
                      ["normal", detail.label_counts["normal"] ?? 0, "normal"],
                      ["defect", detail.label_counts["defect"] ?? 0, "defect"],
                      ["unlabeled", detail.label_counts["unlabeled"] ?? 0, "unlabeled"],
                    ]}
                  />
                  <span className="hidden shrink-0 lg:block">
                    <InfoHint icon={Info} label="Where this dataset came from">
                      <DatasetProvenance detail={detail} />
                    </InfoHint>
                  </span>
                </>
              )}
            </div>

            <Link className="shrink-0" to={`/datasets/${datasetId}/experiments/new`}>
              <Button variant="primary" icon={<Plus />}>
                New experiment
              </Button>
            </Link>
          </div>

          {/* No bottom padding: the strip's underline sits on this header's own border. The
              description shares this row with the strip rather than taking one of its own,
              so a dataset that has one is not a taller screen than a dataset that has not. */}
          <div className="flex min-h-8 items-end justify-between gap-x-6">
            <DatasetSectionNav datasetId={datasetId} />
            {detail?.description && (
              <Tooltip content={detail.description}>
                <p className="hidden min-w-0 shrink truncate pb-2 text-xs text-fg-muted lg:block">
                  {detail.description}
                </p>
              </Tooltip>
            )}
          </div>
        </div>
      </header>

      <Outlet />
    </div>
  );
}

/**
 * How this dataset came to look like this (ADR-0006), on demand rather than on screen.
 *
 * The root path is the reason this moved: it is long, it is absolute, it is the same on
 * every tab, and it answers a question nobody asks twice. `break-all` because a path has
 * no spaces to wrap at and the tooltip is capped at `max-w-xs`.
 */
function DatasetProvenance({
  detail,
}: {
  detail: { root_path: string; adapter: string | null; created_at: string };
}) {
  return (
    <dl className="flex flex-col gap-1.5">
      <div>
        <dt className="text-fg-subtle">Source</dt>
        <dd className="font-mono break-all">{detail.root_path}</dd>
      </div>
      {detail.adapter && (
        <div>
          <dt className="text-fg-subtle">Read by</dt>
          <dd className="font-mono">{detail.adapter}</dd>
        </div>
      )}
      <div>
        <dt className="text-fg-subtle">Imported</dt>
        <dd className="font-mono">{detail.created_at.slice(0, 10)}</dd>
      </div>
    </dl>
  );
}
