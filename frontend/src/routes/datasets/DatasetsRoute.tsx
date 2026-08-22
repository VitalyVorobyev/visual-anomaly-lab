/**
 * The catalogue: every dataset, under what it belongs to.
 *
 * It was a flat two-column grid, so twelve VisA object classes sat beside three CanEnds
 * captures and a GKN blade set with nothing saying which of them were one benchmark — and
 * above them, a large panel that named the relationship in prose while remaining entirely
 * unclickable. The membership was in the payload the whole time:
 * `reference-packs` has always returned `packs[].datasets[].registered_dataset_id`.
 *
 * Grouping is now a field. `collection` is the stored override and the reference pack's
 * title is the derived default, resolved server-side into one effective value — so VisA's
 * twelve arrive grouped without anyone having filed them, and a dataset can be moved out
 * by typing a different name.
 */

import { FolderPlus, Plus } from "lucide-react";
import { useState } from "react";
import { Link } from "react-router";

import type { DatasetSummary } from "../../api/client";
import {
  Button,
  ConfirmDialog,
  Empty,
  ErrorBox,
  Input,
  PageHeader,
  SkeletonRows,
} from "@vitavision/lab-ui";
import {
  useDatasetDeletionPreview,
  useDatasets,
  useDeleteDataset,
} from "../../hooks/useCatalog";
import { CollectionDialog } from "./CollectionDialog";
import { DatasetEditDialog } from "./DatasetEditDialog";
import { DatasetGrid, DatasetGroup } from "./DatasetGroup";
import { groupDatasets } from "./grouping";
import { ReferencePackStrip } from "./ReferencePackStrip";
import { useCollapsedGroups } from "./useCollapsedGroups";

export function DatasetsRoute() {
  const datasets = useDatasets();
  const remove = useDeleteDataset();

  const [pendingDelete, setPendingDelete] = useState<DatasetSummary | null>(null);
  const [confirmation, setConfirmation] = useState("");
  const [editing, setEditing] = useState<DatasetSummary | null>(null);
  // `null` is closed; a string edits that collection and `""` creates one. A collection has
  // no id to hold on to — its name *is* its identity — so the name is the state.
  const [collecting, setCollecting] = useState<string | null>(null);
  const deletionPreview = useDatasetDeletionPreview(pendingDelete?.id);
  const { isCollapsed, toggle } = useCollapsedGroups();

  const all = datasets.data ?? [];
  const { ungrouped, groups } = groupDatasets(all);

  return (
    <div className="flex flex-col gap-6">
      <PageHeader
        title="Datasets"
        actions={
          <>
            {/* Only once there is something to file. On an empty catalogue the import
                prompt owns the screen and a group of nothing cannot exist anyway. */}
            {all.length > 0 && (
              <Button icon={<FolderPlus />} onClick={() => setCollecting("")}>
                New collection
              </Button>
            )}
            <Link to="/import">
              <Button variant="primary" icon={<Plus />}>
                Import
              </Button>
            </Link>
          </>
        }
      />

      {remove.error && <ErrorBox>{remove.error.message}</ErrorBox>}
      {datasets.error && <ErrorBox>{datasets.error.message}</ErrorBox>}
      {datasets.isPending && <SkeletonRows rows={4} />}

      <ReferencePackStrip />

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

      {/* Ungrouped first and bare: someone with three imports should never meet a heading
          for a group of everything. */}
      {ungrouped.length > 0 && (
        <DatasetGrid datasets={ungrouped} onEdit={setEditing} onDelete={openDelete} />
      )}

      {groups.map(([name, members]) => (
        <DatasetGroup
          key={name}
          name={name}
          datasets={members}
          open={!isCollapsed(name)}
          onOpenChange={() => toggle(name)}
          onEditCollection={setCollecting}
          onEdit={setEditing}
          onDelete={openDelete}
        />
      ))}

      {editing && (
        <DatasetEditDialog
          key={editing.id}
          dataset={editing}
          collections={groups.map(([name]) => name)}
          onClose={() => setEditing(null)}
        />
      )}

      {collecting !== null && (
        <CollectionDialog
          key={collecting}
          datasets={all}
          collection={collecting === "" ? undefined : collecting}
          onClose={() => setCollecting(null)}
        />
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
        disabled={!deletionPreview.data?.can_delete || confirmation !== pendingDelete?.name}
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

  function openDelete(dataset: DatasetSummary) {
    setConfirmation("");
    setPendingDelete(dataset);
  }
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
