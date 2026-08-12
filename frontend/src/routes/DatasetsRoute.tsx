/** The catalog: every imported dataset, with its composition. */

import { Plus, Trash2 } from "lucide-react";
import { useState } from "react";
import { Link } from "react-router";

import {
  Button,
  ConfirmDialog,
  CountRun,
  Empty,
  ErrorBox,
  Input,
  PageHeader,
  Panel,
  SkeletonRows,
} from "../components/ui";
import {
  useDatasetDeletionPreview,
  useDatasets,
  useDeleteDataset,
} from "../hooks/useCatalog";

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
