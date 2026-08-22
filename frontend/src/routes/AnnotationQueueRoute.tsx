/** Dataset-first entry into manual annotation.
 *
 * This is deliberately a queue, not another asset browser. It answers “what can I work
 * on next?” and hands the chosen unit to the full-height editor without interposing a
 * configuration form.
 *
 * A *unit* is one image or one whole sample, depending on the dataset's annotation scope
 * (ADR-0036). Under sample scope a three-channel part is one card and one job, because one
 * completion writes truth to all three of its images.
 */

import { ArrowRight, Check, PenTool } from "lucide-react";
import { Link, useParams, useSearchParams } from "react-router";

import { queueUnits } from "../api/annotationQueue";
import type { Label, OpenDraftUnit } from "../api/client";
import { imageUrl } from "../api/imageUrl";
import {
  Badge,
  Button,
  Callout,
  Empty,
  ErrorBox,
  SegmentedControl,
  SkeletonRows,
} from "@vitavision/lab-ui";
import { useAnnotationScope, useSetAnnotationScope } from "../hooks/useAnnotations";
import { useDataset, useSamples } from "../hooks/useCatalog";
import { TabScroll } from "./dataset/TabScroll";

const QUEUE_PAGE = 120;

export function AnnotationQueueRoute() {
  const datasetId = Number(useParams()["datasetId"]);
  const [searchParams, setSearchParams] = useSearchParams();
  const offset = Math.max(0, Number(searchParams.get("offset") ?? 0) || 0);
  const label = (searchParams.get("label") ?? "") as Label | "";
  const todo = searchParams.get("todo") === "1";

  const dataset = useDataset(datasetId);
  const scope = useAnnotationScope(datasetId);
  const setScope = useSetAnnotationScope(datasetId);
  const samples = useSamples(datasetId, {
    limit: QUEUE_PAGE,
    offset,
    label: label || undefined,
    annotated: todo ? false : undefined,
  });

  const perSample = dataset.data?.annotation_scope === "sample";
  const multiChannel = (scope.data?.multi_image_samples ?? 0) > 0;
  const units = queueUnits(samples.data?.items ?? [], perSample);
  const first = units[0];

  /** Filters live in the URL so a queue position survives a reload and can be shared. */
  const withParams = (changes: Record<string, string | null>) => {
    const next = new URLSearchParams(searchParams);
    // Any filter change reshapes the queue, so a page offset into the old one is a lie.
    // Dropped first so that paging itself can set a new one.
    next.delete("offset");
    for (const [key, value] of Object.entries(changes)) {
      if (value === null) next.delete(key);
      else next.set(key, value);
    }
    return next;
  };
  const query = new URLSearchParams({ offset: String(offset) }).toString();

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <TabScroll measure="wide" className="flex flex-col gap-4">
        {/* The queue's own controls, beside the queue rather than in the band above: the
            band carries what is true of the dataset, these are true of this page. */}
        <div className="flex min-h-8 flex-wrap items-center gap-3">
          <SegmentedControl
            aria-label="Which samples"
            value={label || "all"}
            options={[
              { value: "all", label: "All" },
              { value: "defect", label: "Defect" },
              { value: "normal", label: "Normal" },
            ]}
            onValueChange={(value) =>
              setSearchParams(withParams({ label: value === "all" ? null : value }))
            }
          />
          <SegmentedControl
            aria-label="Annotation state"
            value={todo ? "todo" : "any"}
            options={[
              { value: "any", label: "Any state" },
              { value: "todo", label: "Not yet annotated" },
            ]}
            onValueChange={(value) =>
              setSearchParams(withParams({ todo: value === "todo" ? "1" : null }))
            }
          />

          <span className="ml-auto flex items-center gap-3">
            {multiChannel && scope.data && (
              <SegmentedControl
                aria-label="Annotation scope"
                value={perSample ? "sample" : "image"}
                options={[
                  { value: "image", label: "Per image" },
                  { value: "sample", label: "Per sample" },
                ]}
                disabled={setScope.isPending || (!perSample && !scope.data.can_use_sample_scope)}
                onValueChange={(value) => setScope.mutate(value === "sample" ? "sample" : "image")}
              />
            )}
            {first && (
              <Link
                to={`/datasets/${datasetId}/annotate/${first.sample.id}/${first.image.id}?${query}`}
              >
                <Button variant="primary" icon={<PenTool />}>
                  Start queue
                </Button>
              </Link>
            )}
          </span>
        </div>

        {setScope.error && <ErrorBox>{setScope.error.message}</ErrorBox>}
        {perSample && multiChannel && (
          <Callout tone="info" title="One annotation per part">
            Editing any channel edits all of them. Completing renders once and writes the same
            mask to every image of the sample, so each channel still carries its own truth.
          </Callout>
        )}
        {/* Why the control above is dead, in text rather than only in a tooltip: a disabled
            control with no reachable explanation is the same as no control. */}
        {!perSample && multiChannel && scope.data && !scope.data.can_use_sample_scope && (
          <Callout tone="warning" title="This dataset cannot share one annotation per part">
            <ul className="list-disc pl-4">
              {scope.data.blockers.map((blocker) => (
                <li key={blocker}>{blocker}</li>
              ))}
            </ul>
            <OpenDraftLinks datasetId={datasetId} units={scope.data.open_draft_units} />
          </Callout>
        )}
        {/* The other direction is blocked by the same thing and used to say so only in a 409
            after the click. */}
        {perSample && scope.data && scope.data.open_draft_units.length > 0 && (
          <Callout tone="info" title="Annotation work is in progress">
            Going back to per-image editing needs an empty desk, the same way getting here did.
            <OpenDraftLinks datasetId={datasetId} units={scope.data.open_draft_units} />
          </Callout>
        )}

        {dataset.error && <ErrorBox>{dataset.error.message}</ErrorBox>}
        {samples.error && <ErrorBox>{samples.error.message}</ErrorBox>}
        {samples.isPending && !samples.data && <SkeletonRows rows={8} />}
        {samples.data && units.length === 0 && (
          <Empty>
            {todo || label
              ? "Nothing matches these filters."
              : "This dataset has no images to annotate."}
          </Empty>
        )}
        {units.length > 0 && (
          <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-5 xl:grid-cols-6">
            {units.map(({ sample, image }, index) => (
              <Link
                key={image.id}
                to={`/datasets/${datasetId}/annotate/${sample.id}/${image.id}?${query}`}
                className="group overflow-hidden rounded-panel border border-line bg-surface transition-colors hover:border-line-strong focus-visible:outline-2 focus-visible:outline-signal"
              >
                <div className="relative aspect-[4/3] bg-raised">
                  <img
                    src={imageUrl(image.id, "thumb")}
                    alt=""
                    loading="lazy"
                    className="h-full w-full object-contain"
                  />
                  <span className="absolute top-2 left-2 rounded-control bg-overlay/90 px-1.5 py-0.5 font-mono text-[10px] text-fg-muted">
                    {offset + index + 1}
                  </span>
                  {sample.annotation !== "none" && (
                    <span
                      title={
                        sample.annotation === "complete"
                          ? "Every image of this sample has ground truth"
                          : "Some images of this sample still have no ground truth"
                      }
                      className={`absolute top-2 right-2 grid size-5 place-items-center rounded-full bg-overlay/90 ${
                        sample.annotation === "complete" ? "text-normal" : "text-warn"
                      }`}
                    >
                      <Check className="size-3.5" />
                    </span>
                  )}
                  <ArrowRight className="absolute right-2 bottom-2 size-4 text-fg opacity-0 transition-opacity group-hover:opacity-100" />
                </div>
                <div className="flex items-center justify-between gap-2 px-2.5 py-2">
                  <span className="min-w-0 truncate font-mono text-xs">{sample.external_id}</span>
                  <span className="flex shrink-0 items-center gap-1.5">
                    {perSample && sample.images.length > 1 ? (
                      <span className="text-[10px] text-fg-subtle">
                        {sample.images.length} ch
                      </span>
                    ) : (
                      image.channel && (
                        <span className="text-[10px] text-fg-subtle">{image.channel}</span>
                      )
                    )}
                    <Badge tone={sample.label}>{sample.label.slice(0, 3)}</Badge>
                  </span>
                </div>
              </Link>
            ))}
          </div>
        )}
      </TabScroll>

      {samples.data && samples.data.total > QUEUE_PAGE && (
        <footer className="shrink-0 border-t border-line bg-surface px-5 py-2.5 lg:px-6">
          <div className="mx-auto flex max-w-[100rem] items-center justify-between">
            <span className="text-xs text-fg-muted">
              {offset + 1}–{Math.min(offset + QUEUE_PAGE, samples.data.total)} of{" "}
              {samples.data.total}
              {units.length > 0 && ` · ${units.length} on this page`}
            </span>
            <span className="flex gap-2">
              <Button
                disabled={offset === 0}
                onClick={() =>
                  setSearchParams(
                    withParams({ offset: String(Math.max(0, offset - QUEUE_PAGE)) }),
                  )
                }
              >
                Previous
              </Button>
              <Button
                disabled={offset + QUEUE_PAGE >= samples.data.total}
                onClick={() =>
                  setSearchParams(withParams({ offset: String(offset + QUEUE_PAGE) }))
                }
              >
                Next
              </Button>
            </span>
          </div>
        </footer>
      )}
    </div>
  );
}

/**
 * The units standing in the way, each one a click from being finished or discarded.
 *
 * There is deliberately no "discard all". A draft row now exists only because somebody saved
 * one, so a bulk button would destroy real annotation work behind a single confirm — the exact
 * thing the draft-lifecycle change was made to stop happening by accident.
 */
function OpenDraftLinks({ datasetId, units }: { datasetId: number; units: OpenDraftUnit[] }) {
  if (units.length === 0) return null;
  return (
    <p className="mt-2 flex flex-wrap items-center gap-x-2 gap-y-1">
      <span className="text-fg-muted">Open:</span>
      {units.map((unit) => (
        <Link
          key={unit.image_id}
          to={`/datasets/${datasetId}/annotate/${unit.sample_id}/${unit.image_id}`}
          className="rounded-control border border-line px-1.5 py-0.5 font-mono text-[11px] hover:border-signal hover:text-signal"
        >
          {unit.sample_key}
          {unit.channel !== null && unit.channel !== undefined && ` · ${unit.channel}`}
        </Link>
      ))}
    </p>
  );
}
