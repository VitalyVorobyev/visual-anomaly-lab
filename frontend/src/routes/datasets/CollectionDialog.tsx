/**
 * Naming a collection and choosing what is in it — one step, because it has to be.
 *
 * A collection is not a row anywhere. It is a string on each dataset, and it exists for
 * exactly as long as some dataset names it. So "create a collection" cannot mean "make an
 * empty one and fill it later": an empty one would not survive a reload. The name and the
 * membership are therefore the same decision, and this dialog asks it once.
 *
 * The same shape answers renaming and re-filing, so `collection` selects the mode: absent
 * creates, present edits that group with its members already ticked. Saving with nothing
 * ticked is how a collection is dissolved — every member returns to its default and the
 * heading stops existing, which is the honest consequence of the model rather than a
 * separate destructive action.
 *
 * Unticking clears the *override*, not the group: a dataset that came from a reference pack
 * goes back to that pack, and one of your own becomes ungrouped. The client cannot tell
 * those two apart — the API returns the effective value, not the raw column — so the copy
 * says what the rule is instead of predicting each row's outcome.
 */

import { useState } from "react";

import type { DatasetSummary } from "../../api/client";
import { Button, Checkbox, Dialog, DialogClose, ErrorBox, Field, Input } from "@vitavision/lab-ui";
import { useMoveDatasets, type CollectionMove } from "../../hooks/useCatalog";
import { inCatalogueOrder } from "./grouping";

export function CollectionDialog({
  datasets,
  collection,
  onClose,
}: {
  datasets: DatasetSummary[];
  /** The collection being edited; absent creates a new one. */
  collection?: string;
  onClose: () => void;
}) {
  const move = useMoveDatasets();

  const rows = inCatalogueOrder(datasets);
  const [initial] = useState(
    () => new Set(collection ? membersOf(datasets, collection) : []),
  );
  const [name, setName] = useState(collection ?? "");
  const [members, setMembers] = useState<ReadonlySet<number>>(initial);

  const target = name.trim();
  const moves = movesFor(datasets, { target, members, initial });
  const existing = new Set(rows.map((row) => row.collection?.trim()).filter(Boolean));
  const joining = collection === undefined && existing.has(target);

  return (
    <Dialog
      open
      onOpenChange={(open) => {
        if (!open) onClose();
      }}
      title={collection === undefined ? "New collection" : `Edit ${collection}`}
      description={
        collection === undefined
          ? "Name the group, then tick what belongs in it."
          : "Rename it, or change which datasets are filed under it."
      }
      footer={
        <>
          <DialogClose asChild>
            <Button variant="ghost">Cancel</Button>
          </DialogClose>
          <Button
            variant="primary"
            loading={move.isPending}
            disabled={target === "" || moves.length === 0}
            onClick={() => move.mutate(moves, { onSuccess: onClose })}
          >
            {collection !== undefined ? "Save" : joining ? `Add to ${target}` : "Create"}
          </Button>
        </>
      }
    >
      <div className="mt-4 flex flex-col gap-4">
        <Field label="Name">
          <Input
            value={name}
            autoFocus
            autoComplete="off"
            placeholder="VisA, my captures, week 42…"
            onChange={(event) => setName(event.target.value)}
          />
        </Field>

        <div className="flex flex-col gap-2">
          <div className="flex items-baseline justify-between">
            <span className="text-xs font-medium text-fg">Datasets</span>
            <span className="font-mono text-xs text-fg-subtle tabular-nums">
              {members.size} selected
            </span>
          </div>

          <ul className="flex flex-col gap-2.5">
            {rows.map((dataset) => (
              <li key={dataset.id}>
                <Checkbox
                  checked={members.has(dataset.id)}
                  label={dataset.name}
                  description={dataset.collection ?? undefined}
                  onCheckedChange={(checked) =>
                    setMembers((current) => {
                      const next = new Set(current);
                      if (checked) next.add(dataset.id);
                      else next.delete(dataset.id);
                      return next;
                    })
                  }
                />
              </li>
            ))}
          </ul>

          {initial.size > 0 && (
            <p className="text-xs leading-relaxed text-fg-subtle">
              Unticking a dataset returns it to its default — the reference pack it came
              from, or no collection at all.
            </p>
          )}
        </div>

        {move.error && <ErrorBox>{move.error.message}</ErrorBox>}
      </div>
    </Dialog>
  );
}

/** The ids currently filed under a collection, by its effective name. */
function membersOf(datasets: DatasetSummary[], collection: string): number[] {
  return datasets
    .filter((dataset) => dataset.collection?.trim() === collection)
    .map((dataset) => dataset.id);
}

/**
 * What has to be written, and nothing more.
 *
 * A dataset already filed under the target name is left alone, so re-saving an untouched
 * group sends no requests at all — which is also what disables the button.
 */
export function movesFor(
  datasets: DatasetSummary[],
  {
    target,
    members,
    initial,
  }: { target: string; members: ReadonlySet<number>; initial: ReadonlySet<number> },
): CollectionMove[] {
  const moves: CollectionMove[] = [];
  for (const dataset of datasets) {
    const current = dataset.collection?.trim() ?? "";
    if (members.has(dataset.id)) {
      if (current !== target) moves.push({ datasetId: dataset.id, collection: target });
    } else if (initial.has(dataset.id)) {
      // Dropped from the group it was in: clear the override and let the default answer.
      moves.push({ datasetId: dataset.id, collection: "" });
    }
  }
  return moves;
}
