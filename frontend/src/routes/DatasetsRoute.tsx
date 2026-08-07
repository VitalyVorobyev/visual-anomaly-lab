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
  PageHeader,
  Panel,
  SkeletonRows,
} from "../components/ui";
import { useDatasets, useDeleteDataset } from "../hooks/useCatalog";

type Dataset = NonNullable<ReturnType<typeof useDatasets>["data"]>[number];

export function DatasetsRoute() {
  const datasets = useDatasets();
  const remove = useDeleteDataset();
  const [pendingDelete, setPendingDelete] = useState<Dataset | null>(null);

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
                    onClick={() => setPendingDelete(dataset)}
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

      {/* A dataset owns its samples, its splits and every experiment run against them, and
          this used to fire on one click of a red button. */}
      <ConfirmDialog
        open={pendingDelete !== null}
        onOpenChange={(open) => !open && setPendingDelete(null)}
        title="Delete this dataset?"
        description={
          pendingDelete && (
            <>
              <span className="font-medium text-fg">{pendingDelete.name}</span> and its{" "}
              {pendingDelete.samples} samples leave the catalog, along with every split and
              experiment built on them. The image files on disk are not touched.
            </>
          )
        }
        confirmLabel="Delete dataset"
        destructive
        loading={remove.isPending}
        onConfirm={() => {
          if (pendingDelete === null) return;
          remove.mutate(pendingDelete.id, { onSettled: () => setPendingDelete(null) });
        }}
      />
    </div>
  );
}
