/**
 * Explore on the sample viewer: what a frozen encoder sees, asked by clicking.
 *
 * The purpose is intuition, not truth. Four questions share one prompt model:
 *
 * - **Similar** — a click is a patch the encoder should find again; shift-click one it
 *   should not. Every change of prompt is one request.
 * - **Clusters** — the image's own patches grouped by k-means; a click picks the group under
 *   it, answered on the client from the cells the response carried.
 * - **PCA** — the three leading directions of the features as false colour.
 * - **SAM** — MobileSAM through the editor's own contour-assist endpoint.
 *
 * The prompt belongs to one image. Clicking a different channel's pane moves Explore there
 * and starts a fresh prompt, so a multi-channel sample needs no special case: whichever pane
 * was clicked is the one asked about.
 */

import { useCallback, useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router";

import type { AssistPoint, ExploreRequest, ExploreResponse } from "../../api/client";
import {
  clusterAt,
  exploreMapUrl,
  type CarriedCandidate,
  type ImagePoint,
} from "../../api/explore";
import type { RasterLayer } from "../../components/viewer/SampleStage";
import { useAnnotationLabels, useSegmentAssist, useSegmentAssistCapability } from "../../hooks/useAnnotations";
import { useExploreCapability, useExploreRequest, useExploreShape } from "../../hooks/useExplore";

export type ExploreMode = "similar" | "clusters" | "pca" | "sam";

/** More points than this is somebody clicking, not somebody asking. */
const MAX_POINTS = 32;

interface Prompt {
  imageId: number | null;
  points: ImagePoint[];
  negatives: ImagePoint[];
  sam: AssistPoint[];
}

const EMPTY: Prompt = { imageId: null, points: [], negatives: [], sam: [] };

export function useExploreSession({
  datasetId,
  sampleId,
  shownImageId,
  imageIds,
}: {
  datasetId: number;
  sampleId: number;
  /** The image on screen when nothing has been clicked yet. */
  shownImageId: number | undefined;
  imageIds: readonly number[];
}) {
  const navigate = useNavigate();
  const capability = useExploreCapability();
  const samCapability = useSegmentAssistCapability();
  const labels = useAnnotationLabels(datasetId);

  const [on, setOn] = useState(false);
  const [mode, setModeState] = useState<ExploreMode>("similar");
  const [chosenBackbone, setBackbone] = useState<string | null>(null);
  const [clickedImage, setClickedImage] = useState<number | null>(null);
  const [stored, setPrompt] = useState<Prompt>(EMPTY);
  const [k, setK] = useState<number | null>(null);
  const [kDraft, setKDraft] = useState<number | null>(null);
  const [threshold, setThreshold] = useState(0.6);
  const [opacity, setOpacity] = useState(0.8);
  const [picked, setPicked] = useState<{ mapId: string; cluster: number } | null>(null);
  const [candidateIndex, setCandidateIndex] = useState(0);

  const target =
    clickedImage !== null && imageIds.includes(clickedImage) ? clickedImage : shownImageId;
  const prompt = stored.imageId === target ? stored : EMPTY;

  const backbones = useMemo(
    () => capability.data?.backbones.filter((entry) => entry.available) ?? [],
    [capability.data],
  );
  const backbone =
    backbones.find((entry) => entry.key === chosenBackbone) ??
    backbones.find((entry) => entry.key === capability.data?.default_backbone) ??
    backbones[0];
  const clusters = k ?? capability.data?.default_clusters ?? 6;

  const explore = useExploreRequest();
  const toShape = useExploreShape();
  const sam = useSegmentAssist(target ?? 0);
  const labelKey = labels.data?.[0]?.key ?? "defect";

  // One request per distinct question. The signature is what the question *is*; an effect
  // keyed on it asks once per change and never again for a re-render.
  const body: ExploreRequest | null =
    on && backbone && target !== undefined && mode !== "sam"
      ? mode === "similar"
        ? prompt.points.length > 0
          ? {
              mode,
              backbone: backbone.key,
              points: prompt.points,
              negatives: prompt.negatives,
              k: clusters,
              seed: 0,
            }
          : null
        : { mode, backbone: backbone.key, k: clusters, seed: 0 }
      : null;
  const signature = body && target !== undefined ? JSON.stringify([target, body]) : null;
  const { mutate: ask, reset: resetExplore } = explore;
  useEffect(() => {
    if (signature === null) return;
    const [imageId, request] = JSON.parse(signature) as [number, ExploreRequest];
    ask({ imageId, body: request });
  }, [signature, ask]);

  // A mutation forgets its data while the next one is pending; the overlay must not blink on
  // every click, so the last answer is kept until a newer one lands. The image-and-encoder
  // pairs already answered are what tell a slow first encode from an instant cached click.
  const [last, setLast] = useState<ExploreResponse | null>(null);
  const [encoded, setEncoded] = useState<ReadonlySet<string>>(() => new Set());
  useEffect(() => {
    const data = explore.data;
    if (!data) return;
    setLast(data);
    setEncoded((current) => new Set(current).add(`${data.image_id}:${data.backbone}`));
  }, [explore.data]);
  const asked = explore.variables;
  const encoding =
    explore.isPending &&
    asked !== undefined &&
    !encoded.has(`${asked.imageId}:${asked.body.backbone}`);

  // An answer is only drawn over the image, the mode and the encoder that asked for it.
  const answer: ExploreResponse | undefined =
    on && last && last.image_id === target && last.mode === mode && last.backbone === backbone?.key
      ? last
      : undefined;
  const cluster = picked && answer && picked.mapId === answer.map_id ? picked.cluster : null;
  const samAnswer = mode === "sam" && prompt.sam.length > 0 ? sam.data : undefined;
  const candidates = samAnswer && samAnswer.image_id === target ? samAnswer.candidates : [];
  const candidate = candidates[candidateIndex] ?? candidates[0] ?? null;

  const { mutate: askSam, reset: resetSam } = sam;
  const clear = useCallback(() => {
    setPrompt(EMPTY);
    setPicked(null);
    setCandidateIndex(0);
    setLast(null);
    resetExplore();
    resetSam();
  }, [resetExplore, resetSam]);

  // A new sample is a new prompt: the old points were never about it.
  useEffect(() => {
    setClickedImage(null);
    clear();
  }, [sampleId, clear]);

  const setMode = (next: ExploreMode) => {
    setModeState(next);
    setPicked(null);
    setCandidateIndex(0);
  };

  const pick = (imageId: number, point: ImagePoint, shift: boolean) => {
    if (!on) return;
    const base = imageId === target ? prompt : { ...EMPTY, imageId };
    if (imageId !== target) {
      setClickedImage(imageId);
      setPicked(null);
    }
    const at = { x: Math.round(point.x * 10) / 10, y: Math.round(point.y * 10) / 10 };
    if (mode === "similar") {
      setPrompt({
        ...base,
        imageId,
        points: shift ? base.points : [...base.points, at].slice(-MAX_POINTS),
        negatives: shift ? [...base.negatives, at].slice(-MAX_POINTS) : base.negatives,
      });
    } else if (mode === "clusters") {
      if (imageId !== target || !answer) return;
      const found = clusterAt(answer, point);
      setPicked(
        found === null || found === cluster ? null : { mapId: answer.map_id, cluster: found },
      );
    } else if (mode === "sam" && samCapability.data?.available) {
      const sampled: AssistPoint[] = [
        ...base.sam,
        { ...at, kind: shift ? ("negative" as const) : ("positive" as const) },
      ].slice(-MAX_POINTS);
      setPrompt({ ...base, imageId, sam: sampled });
      setCandidateIndex(0);
    }
  };

  // Every new SAM prompt is asked once, on the pane it was placed on — including one placed
  // on another channel's pane, which becomes the target in the same render.
  useEffect(() => {
    if (mode !== "sam" || !on || target === undefined) return;
    if (stored.imageId !== target || stored.sam.length === 0) return;
    if (sam.variables?.points === stored.sam || sam.isPending) return;
    askSam({ points: stored.sam, label_key: labelKey, operation: "add" });
  }, [mode, on, target, stored, sam.variables, sam.isPending, askSam, labelKey]);

  const layers: RasterLayer[] = [];
  if (on && answer) {
    if (answer.map_kind === "values") {
      layers.push({ key: "explore-heat", src: exploreMapUrl(answer.map_url), opacity });
      layers.push({
        key: "explore-mask",
        src: exploreMapUrl(answer.map_url, { threshold }),
        opacity,
      });
    } else if (answer.map_kind === "clusters") {
      layers.push({
        key: "explore-clusters",
        src: exploreMapUrl(answer.map_url, { clusters: answer.clusters ?? 0, cluster }),
        opacity,
      });
    } else {
      layers.push({ key: "explore-pca", src: exploreMapUrl(answer.map_url), opacity });
    }
  }

  const sendable =
    mode === "sam"
      ? candidate !== null
      : mode === "similar"
        ? answer !== undefined
        : mode === "clusters"
          ? cluster !== null
          : false;

  const send = async () => {
    if (target === undefined) return;
    let carried: CarriedCandidate | null = null;
    const encoder = backbone?.title ?? "the encoder";
    if (mode === "sam" && candidate) {
      carried = {
        imageId: target,
        shape: candidate.shape,
        area: candidate.area,
        source: `MobileSAM · quality ${candidate.score.toFixed(3)}`,
      };
    } else if (answer && (mode === "similar" || (mode === "clusters" && cluster !== null))) {
      try {
        const result = await toShape.mutateAsync({
          mapId: answer.map_id,
          body:
            mode === "similar"
              ? { threshold, label_key: labelKey }
              : { cluster: cluster ?? undefined, label_key: labelKey },
        });
        carried = {
          imageId: target,
          shape: result.shape,
          area: result.area,
          source:
            mode === "similar"
              ? `similarity ≥ ${threshold.toFixed(2)} · ${encoder}`
              : `cluster ${cluster} of ${answer.clusters ?? clusters} · ${encoder}`,
        };
      } catch {
        return; // The mutation's error is shown beside the button.
      }
    }
    if (!carried) return;
    void navigate(`/datasets/${datasetId}/annotate/${sampleId}/${target}`, {
      state: { exploreCandidate: carried },
    });
  };

  const retry = () => {
    if (explore.variables) ask(explore.variables);
  };
  const retrySam = () => {
    if (sam.variables) askSam(sam.variables);
  };

  return {
    retry,
    retrySam,
    capability,
    samCapability,
    on,
    setOn,
    mode,
    setMode,
    backbones,
    backbone,
    setBackbone,
    target,
    prompt,
    clear,
    pick,
    k: kDraft ?? clusters,
    setKDraft,
    commitK: (value: number) => {
      setKDraft(null);
      setK(value);
    },
    threshold,
    setThreshold,
    opacity,
    setOpacity,
    explore,
    encoding,
    answer,
    cluster,
    sam,
    candidates,
    candidate,
    candidateIndex,
    setCandidateIndex,
    layers,
    sendable,
    send,
    toShape,
  };
}

export type ExploreSession = ReturnType<typeof useExploreSession>;
