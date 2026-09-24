/** The editor's two questions: which channels receive a copy, and whether to discard the draft. */

import { Copy } from "lucide-react";

import type { ImageSummary } from "../../api/client";
import type { DraftEnvelope } from "../../hooks/useAnnotations";
import { Button, Checkbox, ConfirmDialog, Dialog, ErrorBox } from "@vitavision/lab-ui";
import type { ChannelPanes } from "./useChannelPanes";
import { isConflict } from "./useDraftSession";

export function CopyRegionsDialog({
  panes,
  shapeCount,
  siblingDrafts,
  savePending,
}: {
  panes: ChannelPanes;
  shapeCount: number;
  siblingDrafts: Map<number, DraftEnvelope>;
  savePending: boolean;
}) {
  const { copy, currentImage, otherImages } = panes;
  return (
    <Dialog
      open={copy.open}
      onOpenChange={copy.setOpen}
      title="Copy regions to other channels"
      description={
        <>
          The {shapeCount} region
          {shapeCount === 1 ? "" : "s"} on {currentImage?.channel ?? "this channel"} are{" "}
          <em>added</em> to each channel you pick. Nothing already there is replaced, and each
          copy is editable on its own — the exposures are milliseconds apart, so a copy usually
          needs a nudge.
        </>
      }
      footer={
        <>
          <Button onClick={() => copy.setOpen(false)}>Cancel</Button>
          <Button
            variant="primary"
            icon={<Copy />}
            loading={copy.mutation.isPending || savePending}
            disabled={copy.targets.length === 0}
            onClick={() => void copy.run()}
          >
            Copy to {copy.targets.length} channel{copy.targets.length === 1 ? "" : "s"}
          </Button>
        </>
      }
    >
      <div className="mt-3 flex flex-col gap-2">
        {otherImages.map((image) => (
          <CopyTarget
            key={image.id}
            image={image}
            currentImage={currentImage}
            held={siblingDrafts.get(image.id)?.document.shapes.length}
            checked={copy.targets.includes(image.id)}
            onChecked={(checked) => copy.toggle(image.id, checked)}
          />
        ))}
        {copy.mutation.error && <ErrorBox>{copy.mutation.error.message}</ErrorBox>}
      </div>
    </Dialog>
  );
}

function CopyTarget({
  image,
  currentImage,
  held,
  checked,
  onChecked,
}: {
  image: ImageSummary;
  currentImage: ImageSummary | undefined;
  held: number | undefined;
  checked: boolean;
  onChecked: (checked: boolean) => void;
}) {
  const sized =
    currentImage !== undefined &&
    (image.width !== currentImage.width || image.height !== currentImage.height);
  return (
    <Checkbox
      checked={checked}
      disabled={sized}
      label={image.channel ?? "unassigned"}
      description={
        sized
          ? `${image.width} × ${image.height} — an annotation never leaves its source frame`
          : held === undefined
            ? "…"
            : `${held} region${held === 1 ? "" : "s"} here already`
      }
      onCheckedChange={onChecked}
    />
  );
}

export function DiscardDraftDialog({
  open,
  onOpenChange,
  perSample,
  error,
  pending,
  onDiscard,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  perSample: boolean;
  error: Error | null;
  pending: boolean;
  /** `force` is offered only after a 412 has said the draft moved under this window. */
  onDiscard: (force: boolean) => void;
}) {
  const conflict = isConflict(error);
  return (
    <ConfirmDialog
      open={open}
      onOpenChange={onOpenChange}
      destructive
      title="Discard this draft?"
      description={
        <>
          {perSample
            ? "Every channel of this part reopens on its newest completed truth, or on a blank canvas if it has none."
            : "This image reopens on its newest completed truth, or on a blank canvas if it has none."}{" "}
          Completed revisions are immutable and are not affected.
          {error && (
            <span className="mt-2 block text-defect">
              {error.message}
              {conflict && " Discarding now throws away whatever the other window saved."}
            </span>
          )}
        </>
      }
      confirmLabel={conflict ? "Discard anyway" : "Discard draft"}
      loading={pending}
      onConfirm={() => onDiscard(conflict)}
    />
  );
}
