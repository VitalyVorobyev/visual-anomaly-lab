/**
 * The channels of the part: which one is edited, which one is beside it, and moving regions
 * between them.
 *
 * Channel count is data here, never a layout: every rule below reads `sample.images` and
 * nothing else, so a two-channel part and a five-channel part go through the same code.
 */

import { useCallback, useState } from "react";
import { useNavigate } from "react-router";

import { resolveReference } from "../../api/annotationPanes";
import type { AnnotationDocument, SampleSummary } from "../../api/client";
import { type DraftEnvelope, useCopyRegions } from "../../hooks/useAnnotations";
import type { Flash } from "./useFlashMessage";
import type { Workspace } from "./useWorkspace";

export function useChannelPanes({
  datasetId,
  sample,
  imageId,
  perSample,
  queueOffset,
  workspace,
  present,
  siblingDrafts,
  dirty,
  persist,
  flash,
}: {
  datasetId: number;
  sample: SampleSummary;
  imageId: number;
  perSample: boolean;
  queueOffset: number;
  workspace: Workspace;
  /** The document being edited. */
  present: AnnotationDocument;
  /** What each channel already holds, so a copy is a decision and a reference pane is honest. */
  siblingDrafts: Map<number, DraftEnvelope>;
  dirty: boolean;
  persist: () => Promise<string>;
  flash: Flash;
}) {
  const navigate = useNavigate();
  const copyRegions = useCopyRegions(imageId);
  const [copyOpen, setCopyOpen] = useState(false);
  const [copyTargets, setCopyTargets] = useState<number[]>([]);

  const currentImage = sample.images.find((image) => image.id === imageId);
  const activeIndex = sample.images.findIndex((image) => image.id === imageId);
  const otherImages = sample.images.filter((image) => image.id !== imageId);
  const referenceIndex = resolveReference(
    sample.images.length,
    activeIndex,
    workspace.referenceIndex,
  );
  const reference = referenceIndex === null ? null : (sample.images[referenceIndex] ?? null);
  // A copy keeps its source-pixel coordinates, so a channel of another size cannot take one.
  const copyable = otherImages.filter(
    (image) =>
      currentImage === undefined ||
      (image.width === currentImage.width && image.height === currentImage.height),
  );

  /**
   * A pane draws the document that is truth *for that pane's image* — its own, never a
   * neighbour's.
   *
   * Under sample scope that is the edited document by construction: one completion writes the
   * same mask to every channel. Under image scope it is the reference channel's **own draft**,
   * prefetched with the rest of the part. Drawing the active channel's regions over a sibling
   * would claim truth that does not exist there; drawing nothing, which is what this did until
   * now, hid truth that does — a channel already annotated looked untouched, and the pane
   * beside the one being worked in appeared to be broken.
   *
   * Until that draft arrives the pane shows the bare photograph rather than a wrong overlay.
   */
  const referenceDocument: AnnotationDocument | null =
    reference === null
      ? null
      : perSample
        ? present
        : (siblingDrafts.get(reference.id)?.document ?? {
            ...present,
            base: "empty",
            image_width: reference.width,
            image_height: reference.height,
            shapes: [],
          });

  /**
   * Show a different channel of the same part.
   *
   * Under sample scope this is a pure display change: the document is the part's, so the
   * shapes stay on screen and visibly land — or fail to land — on the new illumination.
   * Under image scope each channel owns its own truth, so this is real navigation, and it
   * *saves first* rather than refusing. Disabling the tab while the draft was dirty was a
   * dead end that read as a broken control: the work was a keystroke away from being safe,
   * and the editor knew it. A 412 aborts the move and leaves the reload-or-keep choice on
   * screen, which is the one case where losing the edit is still possible.
   */
  const openChannel = useCallback(
    async (index: number): Promise<boolean> => {
      const image = sample.images[index];
      if (!image || image.id === imageId) return false;
      if (!perSample && dirty) {
        try {
          await persist();
        } catch {
          return false;
        }
      }
      navigate(`/datasets/${datasetId}/annotate/${sample.id}/${image.id}?offset=${queueOffset}`, {
        replace: true,
      });
      return true;
    },
    [
      datasetId,
      dirty,
      imageId,
      navigate,
      perSample,
      persist,
      queueOffset,
      sample.id,
      sample.images,
    ],
  );

  /**
   * Exchange the two panes, so the channel being looked at becomes the channel being edited.
   *
   * The reference is stored as a preference rather than derived, so the outgoing channel has
   * to be written into it: without that, a three-channel part would move the reference on to
   * the *next* channel and the pair on screen would change under the hand.
   */
  const swapPanes = useCallback(async () => {
    if (referenceIndex === null) return;
    const outgoing = activeIndex;
    if (await openChannel(referenceIndex)) workspace.setReferenceIndex(outgoing);
  }, [activeIndex, openChannel, referenceIndex, workspace]);

  /**
   * Open the copy dialog, pre-ticking only the channels that can actually receive the regions.
   * A sibling of another size is shown, disabled, with its dimensions — pre-ticking it would
   * arm a button whose only outcome is a 409.
   */
  const startCopy = useCallback(() => {
    setCopyTargets(copyable.map((image) => image.id));
    setCopyOpen(true);
  }, [copyable]);

  const setCopyDialogOpen = useCallback(
    (open: boolean) => {
      setCopyOpen(open);
      if (!open) copyRegions.reset();
    },
    [copyRegions],
  );

  const toggleCopyTarget = useCallback((imageId: number, checked: boolean) => {
    setCopyTargets((targets) =>
      checked ? [...targets, imageId] : targets.filter((target) => target !== imageId),
    );
  }, []);

  const copyToChannels = useCallback(async () => {
    if (copyTargets.length === 0) return;
    try {
      const currentEtag = await persist();
      const result = await copyRegions.mutateAsync({
        etag: currentEtag,
        targetImageIds: copyTargets,
      });
      const names = result.targets
        .map((item) => sample.images.find((image) => image.id === item.image_id)?.channel)
        .filter((channel): channel is string => Boolean(channel));
      setCopyOpen(false);
      setCopyTargets([]);
      flash(
        `${result.copied} region${result.copied === 1 ? "" : "s"} copied to ${
          names.length > 0 ? names.join(", ") : `${result.targets.length} channels`
        }`,
      );
    } catch {
      // The dialog stays open with the mutation's error under the list.
    }
  }, [copyRegions, copyTargets, flash, persist, sample.images]);

  return {
    currentImage,
    activeIndex,
    otherImages,
    referenceIndex,
    reference,
    referenceDocument,
    copyable,
    openChannel,
    swapPanes,
    copy: {
      open: copyOpen,
      setOpen: setCopyDialogOpen,
      targets: copyTargets,
      toggle: toggleCopyTarget,
      start: startCopy,
      run: copyToChannels,
      mutation: copyRegions,
    },
  };
}

export type ChannelPanes = ReturnType<typeof useChannelPanes>;
