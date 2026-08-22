/**
 * What a dataset says about itself, and what it is filed under.
 *
 * Two fields, both free text, both clearable. `name` and `root_path` are deliberately not
 * here: they are the dataset's identity, unique in the schema, and the key a re-import
 * resolves against (handbook import.md).
 *
 * The collection field is a plain input with a `datalist` rather than a picker, because
 * the two things a reader wants from it are opposite. Joining an existing group should
 * take three keystrokes and a suggestion; starting a new one should take typing its name
 * and nothing else. A `<select>` plus a "new collection…" escape hatch would be worse at
 * both.
 *
 * Mounted only while it is open, and keyed on the dataset, so the fields are seeded from
 * props at construction. An effect that copied props into state would have to say what
 * happens when the dataset changes under a half-typed description; not existing is a
 * better answer than any of them.
 */

import { useId, useState } from "react";

import type { DatasetSummary } from "../../api/client";
import {
  Button,
  Dialog,
  DialogClose,
  ErrorBox,
  Field,
  Input,
  Textarea,
} from "@vitavision/lab-ui";
import { useUpdateDataset } from "../../hooks/useCatalog";

export function DatasetEditDialog({
  dataset,
  collections,
  onClose,
}: {
  dataset: DatasetSummary;
  /** Every collection already in use, so the second dataset joins the first by prefix. */
  collections: string[];
  onClose: () => void;
}) {
  const update = useUpdateDataset();
  const listId = useId();

  // Seeded from the *effective* values, so a reference dataset opens showing the text it
  // actually displays rather than an empty box that would look like it had none. Saving it
  // unchanged writes the pack's own words as an override, which is honest: the reader saw
  // them and kept them.
  const [description, setDescription] = useState(dataset.description ?? "");
  const [collection, setCollection] = useState(dataset.collection ?? "");

  return (
    <Dialog
      open
      onOpenChange={(open) => {
        if (!open) onClose();
      }}
      title={`Edit ${dataset.name}`}
      footer={
        <>
          <DialogClose asChild>
            <Button variant="ghost">Cancel</Button>
          </DialogClose>
          <Button
            variant="primary"
            loading={update.isPending}
            onClick={() =>
              update.mutate(
                // Blank means cleared, and the endpoint answers a blank with the value the
                // dataset's reference pack supplies.
                { datasetId: dataset.id, changes: { notes: description, collection } },
                { onSuccess: onClose },
              )
            }
          >
            Save
          </Button>
        </>
      }
    >
      <div className="mt-4 flex flex-col gap-4">
        <Field
          label="Description"
          description="A sentence or two. Leave it empty to restore the default."
        >
          <Textarea
            rows={3}
            className="font-sans"
            value={description}
            placeholder="What is in this dataset, and what counts as a defect in it?"
            onChange={(event) => setDescription(event.target.value)}
          />
        </Field>

        <Field label="Collection" description="Datasets sharing a name are grouped together.">
          <Input
            value={collection}
            list={listId}
            autoComplete="off"
            placeholder="Ungrouped"
            onChange={(event) => setCollection(event.target.value)}
          />
          <datalist id={listId}>
            {collections.map((name) => (
              <option key={name} value={name} />
            ))}
          </datalist>
        </Field>

        {update.error && <ErrorBox>{update.error.message}</ErrorBox>}
      </div>
    </Dialog>
  );
}
