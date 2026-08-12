/** The catalog: every imported dataset, with its composition. */

import { Check, ExternalLink, LibraryBig, Plus, Trash2 } from "lucide-react";
import { useQueryClient } from "@tanstack/react-query";
import { useCallback, useEffect, useState } from "react";
import { Link } from "react-router";

import {
  Button,
  Badge,
  Callout,
  ConfirmDialog,
  CountRun,
  Empty,
  ErrorBox,
  Input,
  PageHeader,
  Panel,
  SkeletonRows,
} from "../components/ui";
import { JobProgress } from "../components/JobProgress";
import { queryKeys } from "../api/queryKeys";
import {
  useDatasetDeletionPreview,
  useDatasets,
  useDeleteDataset,
  useReferencePacks,
  useRegisterReferencePacks,
} from "../hooks/useCatalog";
import { isTerminal, useJob } from "../hooks/useJob";

type Dataset = NonNullable<ReturnType<typeof useDatasets>["data"]>[number];

export function DatasetsRoute() {
  const datasets = useDatasets();
  const remove = useDeleteDataset();
  const [pendingDelete, setPendingDelete] = useState<Dataset | null>(null);
  const [confirmation, setConfirmation] = useState("");
  const deletionPreview = useDatasetDeletionPreview(pendingDelete?.id);

  return (
    <div className="flex flex-col gap-6">
      <PageHeader
        title="Datasets"
        actions={
          <Link to="/import">
            <Button variant="primary" icon={<Plus />}>
              Import
            </Button>
          </Link>
        }
      />

      {remove.error && <ErrorBox>{remove.error.message}</ErrorBox>}
      {datasets.error && <ErrorBox>{datasets.error.message}</ErrorBox>}
      {datasets.isPending && <SkeletonRows rows={4} />}

      <ReferenceDatasetShelf />

      {datasets.data?.length === 0 && (
        <Empty
          action={
            <Link to="/import">
              <Button variant="primary" icon={<Plus />}>
                Import a dataset
              </Button>
            </Link>
          }
        >
          Nothing imported yet. Point the importer at a directory of images to begin.
        </Empty>
      )}

      {datasets.data && datasets.data.length > 0 && (
        <ul className="grid gap-3 md:grid-cols-2">
          {datasets.data.map((dataset) => (
            <li key={dataset.id}>
              <Panel
                className="h-full"
                title={
                  <Link
                    to={`/datasets/${dataset.id}`}
                    className="transition-colors hover:text-signal"
                  >
                    {dataset.name}
                  </Link>
                }
                actions={
                  <Button
                    variant="ghost"
                    size="sm"
                    aria-label={`Delete ${dataset.name}`}
                    icon={<Trash2 />}
                    onClick={() => {
                      setConfirmation("");
                      setPendingDelete(dataset);
                    }}
                  />
                }
              >
                <div className="flex flex-col gap-2.5">
                  <CountRun
                    counts={[
                      ["normal", dataset.label_counts["normal"] ?? 0, "normal"],
                      ["defect", dataset.label_counts["defect"] ?? 0, "defect"],
                      ["unlabeled", dataset.label_counts["unlabeled"] ?? 0, "unlabeled"],
                    ]}
                  />
                  <p className="font-mono text-xs text-fg-muted">
                    {dataset.samples} samples · {dataset.images} images
                    {dataset.adapter ? ` · ${dataset.adapter}` : ""}
                  </p>
                  {/* Referenced in place, never copied (ADR-0001). */}
                  <p className="font-mono text-[11px] break-all text-fg-subtle">
                    {dataset.root_path}
                  </p>
                </div>
              </Panel>
            </li>
          ))}
        </ul>
      )}

      <ConfirmDialog
        open={pendingDelete !== null}
        onOpenChange={(open) => {
          if (!open) {
            setPendingDelete(null);
            setConfirmation("");
          }
        }}
        title="Delete this dataset?"
        description={
          pendingDelete && (
            <>
              <span className="font-medium text-fg">{pendingDelete.name}</span> will leave the
              catalog. Source images and source masks are never touched.
              {deletionPreview.isPending && (
                <span className="mt-3 block text-fg-subtle">Inspecting app-owned storage…</span>
              )}
              {deletionPreview.error && (
                <span className="mt-3 block text-defect">{deletionPreview.error.message}</span>
              )}
              {deletionPreview.data && (
                <span className="mt-3 block rounded-control border border-line bg-raised px-3 py-2">
                  <span className="block font-mono text-xs text-fg">
                    {deletionPreview.data.samples} samples · {deletionPreview.data.images} images
                    {" · "}
                    {deletionPreview.data.splits} splits · {deletionPreview.data.experiments}{" "}
                    experiments · {deletionPreview.data.jobs} jobs
                  </span>
                  <span className="mt-1 block font-mono text-xs text-fg-muted">
                    {deletionPreview.data.generated_files} generated files ·{" "}
                    {formatBytes(deletionPreview.data.generated_bytes)}
                  </span>
                  {deletionPreview.data.manual_labels > 0 && (
                    <span className="mt-1 block text-xs">
                      {deletionPreview.data.manual_labels} manual labels are part of the catalog
                      deletion.
                    </span>
                  )}
                  {deletionPreview.data.resident_loaded && (
                    <span className="mt-1 block text-xs">
                      The loaded inference worker will be evicted first.
                    </span>
                  )}
                  {deletionPreview.data.blocker && (
                    <span className="mt-1 block text-xs text-warn">
                      {deletionPreview.data.blocker}
                    </span>
                  )}
                </span>
              )}
              <label className="mt-3 block text-xs font-medium text-fg">
                Type <span className="font-mono">{pendingDelete.name}</span> to confirm
                <Input
                  className="mt-1 font-mono"
                  aria-label="Type dataset name to confirm"
                  value={confirmation}
                  autoComplete="off"
                  onChange={(event) => setConfirmation(event.target.value)}
                />
              </label>
            </>
          )
        }
        confirmLabel="Delete dataset"
        destructive
        loading={remove.isPending}
        disabled={
          !deletionPreview.data?.can_delete || confirmation !== pendingDelete?.name
        }
        onConfirm={() => {
          if (
            pendingDelete === null ||
            confirmation !== pendingDelete.name ||
            !deletionPreview.data?.can_delete
          )
            return;
          remove.mutate(pendingDelete.id, {
            onSettled: () => {
              setPendingDelete(null);
              setConfirmation("");
            },
          });
        }}
      />
    </div>
  );
}

function ReferenceDatasetShelf() {
  const catalog = useReferencePacks();
  const register = useRegisterReferencePacks();
  const [jobId, setJobId] = useState<number>();
  const clearJob = useCallback(() => setJobId(undefined), []);
  const available = catalog.data?.packs.filter((pack) => pack.status === "available") ?? [];

  return (
    <Panel
      title={
        <span className="flex items-center gap-2">
          <LibraryBig className="size-4 text-signal" />
          Reference datasets
        </span>
      }
      actions={
        catalog.data && catalog.data.pending_datasets > 0 ? (
          <Button
            variant="primary"
            size="sm"
            loading={register.isPending}
            disabled={jobId !== undefined}
            onClick={() =>
              register.mutate(
                { pack_keys: available.map((pack) => pack.key) },
                { onSuccess: (job) => setJobId(job.id) },
              )
            }
          >
            Register {catalog.data.pending_datasets}
          </Button>
        ) : undefined
      }
    >
      <p className="mb-4 max-w-3xl text-sm text-fg-muted">
        Public benchmarks found in the local datasets folder. Registration indexes files in
        place; it never copies or changes the source pack.
      </p>

      {catalog.error && <ErrorBox>{catalog.error.message}</ErrorBox>}
      {register.error && <ErrorBox>{register.error.message}</ErrorBox>}
      {catalog.isPending && <SkeletonRows rows={2} />}

      {catalog.data && (
        <ul className="grid gap-2 lg:grid-cols-2">
          {catalog.data.packs.map((pack) => {
            const registered = pack.datasets.filter(
              (dataset) => dataset.registered_dataset_id !== null,
            ).length;
            const tone =
              pack.status === "registered"
                ? "normal"
                : pack.status === "available"
                  ? "info"
                  : pack.status === "incomplete"
                    ? "warning"
                    : "neutral";
            return (
              <li
                key={pack.key}
                className="rounded-control border border-line bg-raised px-3.5 py-3"
              >
                <div className="flex items-start gap-3">
                  <div className="min-w-0 flex-1">
                    <div className="flex flex-wrap items-center gap-2">
                      <span className="text-sm font-medium text-fg">{pack.title}</span>
                      <Badge tone={tone}>{pack.status}</Badge>
                    </div>
                    <p className="mt-1 font-mono text-[11px] break-all text-fg-subtle">
                      {pack.root_path}
                    </p>
                  </div>
                  {pack.status === "registered" && (
                    <Check className="mt-0.5 size-4 shrink-0 text-normal" aria-hidden />
                  )}
                </div>
                <div className="mt-2 flex items-center justify-between gap-3 text-xs text-fg-muted">
                  <span>
                    {registered}/{pack.datasets.length} datasets registered
                  </span>
                  {(pack.status === "absent" || pack.status === "incomplete") && (
                    <a
                      href={pack.install_url}
                      target="_blank"
                      rel="noreferrer"
                      className="inline-flex items-center gap-1 text-signal hover:underline"
                    >
                      Get dataset <ExternalLink className="size-3" />
                    </a>
                  )}
                </div>
                {pack.status === "incomplete" && pack.missing.length > 0 && (
                  <p className="mt-2 text-xs text-warn">
                    Missing {pack.missing.length} required path
                    {pack.missing.length === 1 ? "" : "s"}.
                  </p>
                )}
              </li>
            );
          })}
        </ul>
      )}

      {jobId !== undefined && (
        <div className="mt-4 border-t border-line pt-4">
          <ReferenceRegistrationJob jobId={jobId} onFinished={clearJob} />
        </div>
      )}

      {catalog.data?.packs.every((pack) => pack.status === "absent") && (
        <Callout className="mt-3">
          No reference packs are installed. Place a supported pack under the local datasets
          folder, then return here; discovery is automatic.
        </Callout>
      )}
    </Panel>
  );
}

function ReferenceRegistrationJob({
  jobId,
  onFinished,
}: {
  jobId: number;
  onFinished: () => void;
}) {
  const queryClient = useQueryClient();
  const { job, lines, error } = useJob(jobId);

  useEffect(() => {
    if (!job || !isTerminal(job.status)) return;
    void queryClient.invalidateQueries({ queryKey: queryKeys.datasets() });
    void queryClient.invalidateQueries({ queryKey: queryKeys.referencePacks() });
    if (job.status === "succeeded") {
      const timer = window.setTimeout(onFinished, 1800);
      return () => window.clearTimeout(timer);
    }
  }, [job?.status, onFinished, queryClient]);

  return (
    <div className="flex flex-col gap-3">
      <JobProgress jobId={jobId} job={job} lines={lines} error={error} />
      {job && isTerminal(job.status) && job.status !== "succeeded" && (
        <Button className="self-end" size="sm" variant="ghost" onClick={onFinished}>
          Dismiss
        </Button>
      )}
    </div>
  );
}

function formatBytes(value: number): string {
  if (value < 1024) return `${value} B`;
  const units = ["KB", "MB", "GB", "TB"];
  let amount = value / 1024;
  let unit = units[0];
  for (const next of units.slice(1)) {
    if (amount < 1024) break;
    amount /= 1024;
    unit = next;
  }
  return `${amount.toFixed(amount >= 10 ? 1 : 2)} ${unit}`;
}
