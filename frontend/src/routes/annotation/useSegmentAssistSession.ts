/**
 * MobileSAM contour assist: the prompt being built, the suggestions it produced, and the
 * one-time checkpoint download that has to happen before any of it works.
 *
 * Suggestions are previewed, never committed: nothing reaches the document until one is
 * accepted, and accepting goes through `useDocumentCommands` like every other edit.
 */

import { useCallback, useEffect, useRef, useState } from "react";

import type { AssistBox, AssistPoint, BitmapShape, SegmentCandidate } from "../../api/client";
import type { CarriedCandidate } from "../../api/explore";
import { useSegmentAssist, useSegmentAssistCapability } from "../../hooks/useAnnotations";
import { useCancelJob } from "../../hooks/useExperiments";
import { isTerminal, useJob } from "../../hooks/useJob";
import { useInstallModelAsset, useModelAssets } from "../../hooks/useModelAssets";
import type { Flash } from "./useFlashMessage";

export type AssistMode = "point" | "box";

const ASSET_KEY = "mobile-sam-vit-t";
/** A prompt of more points than this is somebody clicking, not somebody refining. */
const MAX_ASSIST_POINTS = 32;

export function useSegmentAssistSession({
  imageId,
  labelKey,
  operation,
  flash,
  accept,
}: {
  imageId: number;
  labelKey: string;
  operation: "add" | "subtract";
  flash: Flash;
  /** Commit a suggestion to the document; `false` when it could not be converted. */
  accept: (shape: BitmapShape, asContour: boolean) => Promise<boolean>;
}) {
  const [mode, setModeState] = useState<AssistMode>("point");
  const [points, setPoints] = useState<AssistPoint[]>([]);
  const [box, setBoxState] = useState<AssistBox | null>(null);
  const [candidateIndex, setCandidateIndex] = useState(0);
  const [assetJobId, setAssetJobId] = useState<number>();
  const refreshedAssetJob = useRef<number | undefined>(undefined);

  const capability = useSegmentAssistCapability();
  const modelAssets = useModelAssets();
  const installAsset = useInstallModelAsset();
  const assist = useSegmentAssist(imageId);
  const asset = modelAssets.data?.assets.find((item) => item.key === ASSET_KEY);
  const followedAssetJobId = assetJobId ?? asset?.active_job?.id;
  const assetJob = useJob(followedAssetJobId);
  const cancelAssetJob = useCancelJob();
  /**
   * A mask carried from the sample viewer's Explore. It waits as the one candidate until it is
   * accepted, cleared, or replaced by a prompt of the editor's own; it needs no MobileSAM
   * checkpoint, because nothing here has to be inferred.
   */
  const [carried, setCarried] = useState<CarriedCandidate | null>(null);
  const candidates = assist.data?.candidates ?? [];
  const candidate: SegmentCandidate | null = assist.data
    ? (candidates[candidateIndex] ?? null)
    : carried
      ? { shape: carried.shape, area: carried.area, score: 0 }
      : null;

  const clear = useCallback(() => {
    setPoints([]);
    setBoxState(null);
    setCandidateIndex(0);
    setCarried(null);
    assist.reset();
  }, [assist]);

  const setMode = useCallback(
    (next: AssistMode) => {
      clear();
      setModeState(next);
    },
    [clear],
  );

  /** A new prompt invalidates the suggestions the old one produced. */
  const addPoint = useCallback(
    (point: AssistPoint) => {
      assist.reset();
      setCandidateIndex(0);
      setCarried(null);
      setPoints((current) => [...current, point].slice(-MAX_ASSIST_POINTS));
    },
    [assist],
  );

  const setBox = useCallback(
    (next: AssistBox | null) => {
      assist.reset();
      setCandidateIndex(0);
      setCarried(null);
      setBoxState(next);
    },
    [assist],
  );

  const request = async () => {
    try {
      const result = await assist.mutateAsync({
        points,
        box: box ?? undefined,
        label_key: labelKey,
        operation,
      });
      setCandidateIndex(0);
      flash(
        `${result.candidates.length} suggestion${result.candidates.length === 1 ? "" : "s"} · ${result.device.toUpperCase()} · ${result.warm ? "warm" : "loaded"}`,
      );
    } catch {
      // The mutation's error stays beside the controls that caused it.
    }
  };

  const acceptCandidate = async (asContour: boolean) => {
    if (!candidate) return;
    // A carried mask was cut before anyone chose a class for it, so it takes the class and
    // the operation the editor has selected now.
    const shape =
      !assist.data && carried ? { ...candidate.shape, label_key: labelKey, operation } : candidate.shape;
    if (!(await accept(shape, asContour))) return;
    clear();
    flash(asContour ? "Editable suggested contour accepted" : "Suggested mask accepted");
  };

  const install = () => {
    if (!asset) return;
    installAsset.mutate(asset.key, { onSuccess: (job) => setAssetJobId(job.id) });
  };

  useEffect(() => {
    if (!followedAssetJobId || !isTerminal(assetJob.job?.status)) return;
    if (refreshedAssetJob.current === followedAssetJobId) return;
    refreshedAssetJob.current = followedAssetJobId;
    void modelAssets.refetch();
    void capability.refetch();
    if (assetJob.job?.status === "succeeded") flash("MobileSAM is ready");
  }, [assetJob.job?.status, capability, flash, followedAssetJobId, modelAssets]);

  /** A box narrower than two pixels is a click, and MobileSAM answers it with noise. */
  const canRequest =
    !assist.isPending &&
    (points.length > 0 || (box !== null && box.x1 - box.x0 >= 2 && box.y1 - box.y0 >= 2));

  return {
    mode,
    setMode,
    points,
    box,
    addPoint,
    setBox,
    clear,
    request,
    canRequest,
    assist,
    candidates,
    candidate,
    candidateIndex,
    setCandidateIndex,
    acceptCandidate,
    carried,
    carry: setCarried,
    labelKey,
    asset,
    capability,
    modelAssets,
    installAsset,
    install,
    followedAssetJobId,
    assetJob,
    cancelAssetJob,
  };
}

export type SegmentAssistSession = ReturnType<typeof useSegmentAssistSession>;
