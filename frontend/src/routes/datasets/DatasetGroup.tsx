/**
 * A collection: one heading over the datasets filed under it.
 *
 * The tree is deliberately one level deep. VisA is twelve object classes and GKN is one;
 * a user's own datasets group under whatever name they type. Nothing in that needs
 * nesting, and a catalogue you have to expand twice to see an entry is worse than a flat
 * list, not better.
 *
 * Ungrouped datasets do not render through this at all — they are cards with no heading
 * above them, so someone with three imports never sees group chrome for a group of one.
 */

import { ChevronRight, Pencil } from "lucide-react";

import type { DatasetSummary } from "../../api/client";
import { Button, cn, focusRing } from "@vitavision/lab-ui";
import { DatasetCard } from "./DatasetCard";

const GRID = "grid gap-3 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4";

export function DatasetGrid({
  datasets,
  onEdit,
  onDelete,
}: {
  datasets: DatasetSummary[];
  onEdit: (dataset: DatasetSummary) => void;
  onDelete: (dataset: DatasetSummary) => void;
}) {
  return (
    <ul className={GRID}>
      {datasets.map((dataset) => (
        <DatasetCard
          key={dataset.id}
          dataset={dataset}
          onEdit={onEdit}
          onDelete={onDelete}
        />
      ))}
    </ul>
  );
}

export function DatasetGroup({
  name,
  datasets,
  open,
  onOpenChange,
  onEditCollection,
  onEdit,
  onDelete,
}: {
  name: string;
  datasets: DatasetSummary[];
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onEditCollection: (name: string) => void;
  onEdit: (dataset: DatasetSummary) => void;
  onDelete: (dataset: DatasetSummary) => void;
}) {
  return (
    // A *named* group: the cards inside carry their own `group` for their action corner,
    // and an unnamed one here would reveal every card's buttons on any hover in the section.
    <section className="group/collection flex flex-col gap-3">
      {/* The toggle and the edit action are siblings, not nested: an action inside the
          heading button would fold the group on its way to opening the dialog. */}
      <div className="flex items-center gap-1">
        {/* A button rather than `<details>`: the open state is persisted and restored, and a
            native details element would fight the restore on every mount. */}
        <button
          type="button"
          aria-expanded={open}
          onClick={() => onOpenChange(!open)}
          className={cn(
            "flex min-w-0 flex-1 items-center gap-2 rounded-control py-1 text-left transition-colors hover:text-fg",
            focusRing,
          )}
        >
          <ChevronRight
            className={cn(
              "size-4 shrink-0 text-fg-subtle transition-transform",
              open && "rotate-90",
            )}
            aria-hidden
          />
          <h2 className="text-sm font-semibold tracking-tight text-fg">{name}</h2>
          <span className="font-mono text-xs text-fg-subtle tabular-nums">{datasets.length}</span>
          <span className="ml-3 h-px flex-1 bg-line" aria-hidden />
        </button>

        <span className="opacity-0 transition-opacity group-hover/collection:opacity-100 focus-within:opacity-100">
          <Button
            variant="ghost"
            size="sm"
            aria-label={`Edit collection ${name}`}
            icon={<Pencil />}
            onClick={() => onEditCollection(name)}
          />
        </span>
      </div>

      {open && <DatasetGrid datasets={datasets} onEdit={onEdit} onDelete={onDelete} />}
    </section>
  );
}
