/**
 * Where this document sits in the annotation queue, and how to leave it for a neighbour.
 *
 * The queue is paged (`QUEUE_PAGE` samples per page) and the pages either side are
 * prefetched, so stepping past either end of the current page lands on the next page's
 * first unit — or the previous page's last — rather than stopping at a boundary the reader
 * cannot see. Under sample scope a unit is a part; under image scope it is one photograph.
 */

import { useCallback, useMemo } from "react";
import { useNavigate } from "react-router";

import { queueUnits } from "../../api/annotationQueue";
import type { SampleSummary } from "../../api/client";

export const QUEUE_PAGE = 120;

export function useQueueNavigation({
  datasetId,
  sample,
  imageId,
  perSample,
  defaultChannel,
  queue,
  queueTotal,
  previousQueue,
  nextQueue,
  queueOffset,
}: {
  datasetId: number;
  sample: SampleSummary;
  imageId: number;
  perSample: boolean;
  /** The channel a part opens on under sample scope, so the queue agrees with the grid. */
  defaultChannel: string | null;
  queue: SampleSummary[];
  queueTotal: number;
  previousQueue: SampleSummary[];
  nextQueue: SampleSummary[];
  queueOffset: number;
}) {
  const navigate = useNavigate();
  const flatQueue = useMemo(
    () => queueUnits(queue, perSample, defaultChannel),
    [queue, perSample, defaultChannel],
  );
  const flatPrevious = useMemo(
    () => queueUnits(previousQueue, perSample, defaultChannel),
    [previousQueue, perSample, defaultChannel],
  );
  const flatNext = useMemo(
    () => queueUnits(nextQueue, perSample, defaultChannel),
    [nextQueue, perSample, defaultChannel],
  );
  const queueIndex = flatQueue.findIndex((item) =>
    perSample ? item.sample.id === sample.id : item.image.id === imageId,
  );

  const openQueueItem = useCallback(
    (index: number) => {
      let item = flatQueue[index];
      let nextOffset = queueOffset;
      if (index < 0 && queueOffset > 0) {
        item = flatPrevious.at(-1);
        nextOffset = Math.max(0, queueOffset - QUEUE_PAGE);
      } else if (index >= flatQueue.length && queueOffset + queue.length < queueTotal) {
        item = flatNext[0];
        nextOffset = queueOffset + QUEUE_PAGE;
      }
      if (!item) return;
      void navigate(
        `/datasets/${datasetId}/annotate/${item.sample.id}/${item.image.id}?offset=${nextOffset}`,
        { replace: true },
      );
    },
    [datasetId, flatNext, flatPrevious, flatQueue, navigate, queue.length, queueOffset, queueTotal],
  );

  const hasPrevious = !(queueIndex <= 0 && queueOffset === 0);
  const hasNext = !(
    queueIndex < 0 ||
    (queueIndex + 1 >= flatQueue.length && queueOffset + queue.length >= queueTotal)
  );

  return {
    /** Units on this page, for the header's "n of m". */
    length: flatQueue.length,
    index: queueIndex,
    offset: queueOffset,
    hasPrevious,
    hasNext,
    openNext: useCallback(() => openQueueItem(queueIndex + 1), [openQueueItem, queueIndex]),
    openPrevious: useCallback(() => openQueueItem(queueIndex - 1), [openQueueItem, queueIndex]),
  };
}

export type QueueNavigation = ReturnType<typeof useQueueNavigation>;
