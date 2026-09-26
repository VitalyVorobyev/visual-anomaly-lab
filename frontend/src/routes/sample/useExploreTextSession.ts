/**
 * Explore's Text mode: a phrase in, SAM 3's instances out.
 *
 * One request per submitted phrase, on whichever pane Explore targets. The answer is kept
 * until a newer one lands, so the overlay does not blink while the next phrase is asked, and
 * it is drawn only over the image that asked for it. Picking an instance draws its whole
 * mask alone — as picking a cluster does — and is what Send to editor carries.
 *
 * The checkpoint is a catalogued, licence-gated model asset. Until it is installed this mode
 * is the place to install it: the licence, the download job's progress, and a cancel.
 */

import { useCallback, useEffect, useRef, useState } from "react";

import type { ExploreTextCapability, ExploreTextResponse } from "../../api/client";
import { useCancelJob } from "../../hooks/useExperiments";
import { useExploreCapability, useExploreText } from "../../hooks/useExplore";
import { isTerminal, useJob } from "../../hooks/useJob";
import { useInstallModelAsset, useModelAssets } from "../../hooks/useModelAssets";

export function useExploreTextSession({
  target,
  sampleId,
}: {
  /** The image a phrase is asked about. */
  target: number | undefined;
  sampleId: number;
}) {
  const capability = useExploreCapability();
  const text: ExploreTextCapability | undefined = capability.data?.text;
  const modelAssets = useModelAssets();
  const asset = modelAssets.data?.assets.find((item) => item.key === text?.asset_key);
  const installAsset = useInstallModelAsset();
  const [assetJobId, setAssetJobId] = useState<number>();
  const followedAssetJobId = assetJobId ?? asset?.active_job?.id;
  const assetJob = useJob(followedAssetJobId);
  const cancelAssetJob = useCancelJob();
  const refreshedAssetJob = useRef<number | undefined>(undefined);

  const [phrase, setPhrase] = useState("");
  const [instance, setInstance] = useState<number | null>(null);
  const request = useExploreText();
  const { mutate, reset } = request;

  const [last, setLast] = useState<ExploreTextResponse | null>(null);
  // The images whose vision features the resident has encoded in this visit: what tells a
  // slow first phrase on an image from an instant second one.
  const [encoded, setEncoded] = useState<ReadonlySet<number>>(() => new Set());
  useEffect(() => {
    const data = request.data;
    if (!data) return;
    setLast(data);
    setInstance(null);
    setEncoded((current) => new Set(current).add(data.image_id));
  }, [request.data]);

  const asked = request.variables;
  const encoding = request.isPending && asked !== undefined && !encoded.has(asked.imageId);
  const answer = last && last.image_id === target ? last : undefined;

  const clear = useCallback(() => {
    setLast(null);
    setInstance(null);
    reset();
  }, [reset]);

  // A new sample is a new question: the old instances were never about it.
  useEffect(() => {
    clear();
  }, [sampleId, clear]);

  const trimmed = phrase.trim();
  const canAsk =
    text?.available === true &&
    target !== undefined &&
    trimmed.length > 0 &&
    trimmed.length <= (text.max_phrase_length ?? 80) &&
    !request.isPending;

  const ask = () => {
    if (!canAsk || target === undefined) return;
    mutate({ imageId: target, body: { phrase: trimmed, threshold: text.default_threshold ?? 0.5 } });
  };
  const retry = () => {
    if (request.variables) mutate(request.variables);
  };

  const install = () => {
    if (!asset) return;
    installAsset.mutate(asset.key, { onSuccess: (job) => setAssetJobId(job.id) });
  };

  // A finished download changes what the capability says; ask again rather than wait.
  useEffect(() => {
    if (!followedAssetJobId || !isTerminal(assetJob.job?.status)) return;
    if (refreshedAssetJob.current === followedAssetJobId) return;
    refreshedAssetJob.current = followedAssetJobId;
    void modelAssets.refetch();
    void capability.refetch();
  }, [assetJob.job?.status, capability, followedAssetJobId, modelAssets]);

  const picked = answer && instance !== null && instance <= answer.instances.length ? instance : null;

  return {
    capability: text,
    asset,
    modelAssets,
    installAsset,
    install,
    followedAssetJobId,
    assetJob,
    cancelAssetJob,
    phrase,
    setPhrase,
    canAsk,
    ask,
    retry,
    request,
    encoding,
    answer,
    instance: picked,
    setInstance,
    clear,
  };
}

export type ExploreTextSession = ReturnType<typeof useExploreTextSession>;
