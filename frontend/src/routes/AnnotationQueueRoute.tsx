/** Dataset-first entry into manual annotation.
 *
 * This is deliberately a queue, not another asset browser. It answers “what can I work
 * on next?” and hands the chosen image to the full-height editor without interposing a
 * configuration form.
 */

import { ArrowRight, PenTool } from "lucide-react";
import { Link, useParams, useSearchParams } from "react-router";

import { imageUrl } from "../api/imageUrl";
import { DatasetSectionNav } from "../components/DatasetSectionNav";
import {
  Badge,
  Button,
  Empty,
  ErrorBox,
  PageHeader,
  ReadoutStrip,
  SkeletonRows,
} from "../components/ui";
import { useDataset, useSamples } from "../hooks/useCatalog";

const QUEUE_PAGE = 120;

export function AnnotationQueueRoute() {
  const datasetId = Number(useParams()["datasetId"]);
  const [searchParams, setSearchParams] = useSearchParams();
  const offset = Math.max(0, Number(searchParams.get("offset") ?? 0) || 0);
  const dataset = useDataset(datasetId);
  const samples = useSamples(datasetId, { limit: QUEUE_PAGE, offset });

  if (dataset.error) return <ErrorBox>{dataset.error.message}</ErrorBox>;
  if (dataset.isPending || !dataset.data) return <SkeletonRows rows={6} />;

  const images = (samples.data?.items ?? []).flatMap((sample) =>
    sample.images.map((image) => ({ sample, image })),
  );
  const first = images[0];
  const query = new URLSearchParams({ offset: String(offset) }).toString();

  return (
    <div className="flex min-h-0 flex-1 flex-col bg-ground">
      <header className="border-b border-line bg-ground px-6 py-4">
        <div className="mx-auto flex w-full max-w-[100rem] flex-col gap-3">
          <PageHeader
            title="Annotation queue"
            back={{ to: `/datasets/${datasetId}`, label: dataset.data.name }}
            actions={
              first ? (
                <Link
                  to={`/datasets/${datasetId}/annotate/${first.sample.id}/${first.image.id}?${query}`}
                >
                  <Button variant="primary" icon={<PenTool />}>
                    Start queue
                  </Button>
                </Link>
              ) : undefined
            }
            meta={
              <ReadoutStrip
                items={[
                  { label: "samples", value: samples.data?.total ?? dataset.data.samples },
                  { label: "images on page", value: images.length },
                  { value: "source-coordinate, versioned truth" },
                ]}
              />
            }
          />
          <DatasetSectionNav datasetId={datasetId} />
        </div>
      </header>

      <section className="mx-auto min-h-0 w-full max-w-[100rem] flex-1 overflow-y-auto overscroll-contain p-5">
        {samples.error && <ErrorBox>{samples.error.message}</ErrorBox>}
        {samples.isPending && !samples.data && <SkeletonRows rows={8} />}
        {samples.data && images.length === 0 && (
          <Empty>This dataset has no images to annotate.</Empty>
        )}
        {images.length > 0 && (
          <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-5 xl:grid-cols-6">
            {images.map(({ sample, image }, index) => (
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
                  <ArrowRight className="absolute right-2 bottom-2 size-4 text-fg opacity-0 transition-opacity group-hover:opacity-100" />
                </div>
                <div className="flex items-center justify-between gap-2 px-2.5 py-2">
                  <span className="min-w-0 truncate font-mono text-xs">
                    {sample.external_id}
                  </span>
                  <span className="flex shrink-0 items-center gap-1.5">
                    {image.channel && (
                      <span className="text-[10px] text-fg-subtle">{image.channel}</span>
                    )}
                    <Badge tone={sample.label}>{sample.label.slice(0, 3)}</Badge>
                  </span>
                </div>
              </Link>
            ))}
          </div>
        )}
      </section>

      {samples.data && samples.data.total > QUEUE_PAGE && (
        <footer className="border-t border-line bg-surface px-6 py-2.5">
          <div className="mx-auto flex max-w-[100rem] items-center justify-between">
            <span className="text-xs text-fg-muted">
              {offset + 1}–{Math.min(offset + QUEUE_PAGE, samples.data.total)} of {samples.data.total}
            </span>
            <span className="flex gap-2">
              <Button
                disabled={offset === 0}
                onClick={() => setSearchParams({ offset: String(Math.max(0, offset - QUEUE_PAGE)) })}
              >
                Previous
              </Button>
              <Button
                disabled={offset + QUEUE_PAGE >= samples.data.total}
                onClick={() => setSearchParams({ offset: String(offset + QUEUE_PAGE) })}
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
