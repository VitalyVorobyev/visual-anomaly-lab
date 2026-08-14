/** Full-height manual annotation workbench.
 *
 * The image is the dominant surface. Tools stay in a narrow left rail, object properties
 * and the queue stay in a supporting right rail, and neither can create a second page
 * scrollbar. Keyboard actions mirror every operation needed for a labelling pass.
 */

import {
  ArrowLeft,
  ArrowLeftRight,
  ArrowRight,
  Brush,
  Check,
  CircleDot,
  Copy,
  Eraser,
  Maximize2,
  MousePointer2,
  Redo2,
  Save,
  Shapes,
  Trash2,
  Undo2,
  WandSparkles,
  ZoomIn,
  ZoomOut,
} from "lucide-react";
import { useCallback, useEffect, useMemo, useReducer, useRef, useState } from "react";
import { Link, useBeforeUnload, useNavigate, useParams, useSearchParams } from "react-router";

import {
  canonical,
  createHistory,
  historyReducer,
  nextShapeId,
  replaceShape,
  translateShape,
  withPolygonPoint,
  withShape,
  withoutShape,
} from "../api/annotationState";
import {
  bitmapStroke,
  paintStroke,
  strokeBounds,
  strokeTargets,
  traceBitmapShape,
} from "../api/annotationBitmap";
import { paneFrame, resolveReference, type PaneMode } from "../api/annotationPanes";
import { queueUnits } from "../api/annotationQueue";
import type {
  AnnotationDocument,
  AnnotationLabel,
  AnnotationPoint,
  AnnotationShape,
  AssistBox,
  AssistPoint,
  PolygonShape,
  SampleSummary,
} from "../api/client";
import { ApiError } from "../api/client";
import { ChannelTabs } from "../components/ChannelTabs";
import {
  AnnotationCanvas,
  INITIAL_CANVAS_VIEW,
  type AnnotationCanvasHandle,
  type CanvasView,
  type EditorTool,
} from "../components/annotation/AnnotationCanvas";
import {
  Badge,
  Button,
  Checkbox,
  ConfirmDialog,
  Dialog,
  Empty,
  ErrorBox,
  NumberInput,
  ProgressBar,
  SegmentedControl,
  Select,
  Slider,
  SkeletonRows,
  Tooltip,
  cn,
  focusRing,
} from "../components/ui";
import {
  type DraftEnvelope,
  type DraftTarget,
  useAnnotationLabels,
  useCompleteDraft,
  useCopyRegions,
  useDiscardDraft,
  useEditorDraft,
  useSaveDraft,
  useSiblingDrafts,
  useUpdateAnnotationLabel,
  useSegmentAssist,
  useSegmentAssistCapability,
} from "../hooks/useAnnotations";
import { useDataset, useSample, useSamples } from "../hooks/useCatalog";
import {
  MAX_BRUSH_SIZE,
  MIN_BRUSH_SIZE,
  useBrushSize,
} from "../hooks/useBrushSize";
import { useMaskOpacity } from "../hooks/useMaskOpacity";
import { useCancelJob } from "../hooks/useExperiments";
import { isTerminal, useJob } from "../hooks/useJob";
import { useInstallModelAsset, useModelAssets } from "../hooks/useModelAssets";

const QUEUE_PAGE = 120;

/**
 * Presentation state: how the reader is looking, as opposed to what they are looking at.
 *
 * It lives above the keyed editor because it belongs to the person, not to the document.
 * Everything used to sit inside `EditorReady`, which is keyed by the draft's target, so
 * changing the second channel of a side-by-side comparison remounted the editor and reset
 * the pane mode, the zoom and the active tool along with it — the reported "changing the
 * left channel jumps back to a single channel". Per-target state (history, selection, the
 * pending polygon) keeps the key and keeps resetting, which is the point of the key.
 */
interface Workspace {
  paneMode: PaneMode;
  setPaneMode: (mode: PaneMode) => void;
  /** A channel position, not an image id — see `resolveReference`. */
  referenceIndex: number | null;
  setReferenceIndex: (index: number | null) => void;
  overlayOpacity: number;
  setOverlayOpacity: (value: number) => void;
  maskOpacity: number;
  setMaskOpacity: (value: number) => void;
  tool: EditorTool;
  setTool: (tool: EditorTool) => void;
  brushSize: number;
  setBrushSize: (radius: number) => void;
  view: CanvasView;
  setView: (view: CanvasView) => void;
}

function useWorkspace(frame: string): Workspace {
  const [paneMode, setPaneMode] = useState<PaneMode>("single");
  const [referenceIndex, setReferenceIndex] = useState<number | null>(null);
  const [overlayOpacity, setOverlayOpacity] = useState(0.5);
  const [maskOpacity, setMaskOpacity] = useMaskOpacity();
  const [tool, setTool] = useState<EditorTool>("select");
  const [brushSize, setBrushSize] = useBrushSize();
  // The view is stamped with the frame it was expressed on and derived back out, so moving
  // to another part resets it during render rather than in an effect that would first paint
  // the previous part's zoom over the new photograph.
  const [viewMemo, setViewMemo] = useState({ frame, view: INITIAL_CANVAS_VIEW });
  const view = viewMemo.frame === frame ? viewMemo.view : INITIAL_CANVAS_VIEW;
  const setView = useCallback((next: CanvasView) => setViewMemo({ frame, view: next }), [frame]);

  return useMemo(
    () => ({
      paneMode,
      setPaneMode,
      referenceIndex,
      setReferenceIndex,
      overlayOpacity,
      setOverlayOpacity,
      maskOpacity,
      setMaskOpacity,
      tool,
      setTool,
      brushSize,
      setBrushSize,
      view,
      setView,
    }),
    [
      brushSize,
      maskOpacity,
      overlayOpacity,
      paneMode,
      referenceIndex,
      setMaskOpacity,
      setView,
      tool,
      view,
    ],
  );
}

export function AnnotationEditorRoute() {
  const params = useParams();
  const datasetId = Number(params["datasetId"]);
  const sampleId = Number(params["sampleId"]);
  const imageId = Number(params["imageId"]);
  const [searchParams] = useSearchParams();
  const [reloadGeneration, setReloadGeneration] = useState(0);
  const offset = Math.max(0, Number(searchParams.get("offset") ?? 0) || 0);

  const dataset = useDataset(datasetId);
  const sample = useSample(datasetId, sampleId);
  const queue = useSamples(datasetId, { limit: QUEUE_PAGE, offset });
  const previousQueue = useSamples(datasetId, {
    limit: QUEUE_PAGE,
    offset: Math.max(0, offset - QUEUE_PAGE),
  });
  const nextQueue = useSamples(datasetId, { limit: QUEUE_PAGE, offset: offset + QUEUE_PAGE });
  const labels = useAnnotationLabels(datasetId);

  const perSample = dataset.data?.annotation_scope === "sample";
  const target: DraftTarget | undefined = dataset.data
    ? perSample
      ? { scope: "sample", sampleId }
      : { scope: "image", imageId }
    : undefined;
  const draft = useEditorDraft(target);

  const images = useMemo(() => sample.data?.images ?? [], [sample.data]);
  const activeImage = images.find((image) => image.id === imageId);
  const workspace = useWorkspace(
    paneFrame(sampleId, activeImage?.width ?? 0, activeImage?.height ?? 0),
  );
  const siblingDrafts = useSiblingDrafts(
    useMemo(() => images.map((image) => image.id), [images]),
    // Sample scope has one document for the whole part, so there are no siblings to read.
    !perSample && dataset.data !== undefined,
  );

  const error = dataset.error ?? sample.error ?? queue.error ?? labels.error ?? draft.error;
  if (error) return <ErrorBox>{error.message}</ErrorBox>;
  if (
    dataset.isPending ||
    sample.isPending ||
    queue.isPending ||
    labels.isPending ||
    draft.isPending ||
    !dataset.data ||
    !sample.data ||
    !queue.data ||
    !labels.data ||
    !draft.data ||
    !target
  ) {
    return <SkeletonRows rows={8} />;
  }

  return (
    <EditorReady
      // Under sample scope the document belongs to the part, so switching channel changes
      // only which photograph is under it: remounting there would discard unsaved edits
      // and reopen the same draft.
      key={`${perSample ? `s${sampleId}` : imageId}-${reloadGeneration}`}
      datasetId={datasetId}
      imageId={imageId}
      target={target}
      datasetName={dataset.data.name}
      sample={sample.data}
      queue={queue.data.items}
      queueTotal={queue.data.total}
      previousQueue={previousQueue.data?.items ?? []}
      nextQueue={nextQueue.data?.items ?? []}
      queueOffset={offset}
      labels={labels.data}
      initial={draft.data}
      workspace={workspace}
      siblingDrafts={siblingDrafts}
      onReload={async () => {
        await draft.refetch();
        setReloadGeneration((generation) => generation + 1);
      }}
    />
  );
}

function EditorReady({
  datasetId,
  imageId,
  target,
  datasetName,
  sample,
  queue,
  queueTotal,
  previousQueue,
  nextQueue,
  queueOffset,
  labels,
  initial,
  workspace,
  siblingDrafts,
  onReload,
}: {
  datasetId: number;
  imageId: number;
  target: DraftTarget;
  datasetName: string;
  sample: SampleSummary;
  queue: SampleSummary[];
  queueTotal: number;
  previousQueue: SampleSummary[];
  nextQueue: SampleSummary[];
  queueOffset: number;
  labels: AnnotationLabel[];
  initial: DraftEnvelope;
  workspace: Workspace;
  /** What each channel already holds, so a copy is a decision and a reference pane is honest. */
  siblingDrafts: Map<number, DraftEnvelope>;
  onReload: () => Promise<void>;
}) {
  const navigate = useNavigate();
  const perSample = target.scope === "sample";
  const {
    paneMode,
    setPaneMode,
    overlayOpacity,
    setOverlayOpacity,
    maskOpacity,
    setMaskOpacity,
    tool,
    setTool,
    brushSize,
    setBrushSize,
    view,
    setView,
  } = workspace;
  const [history, dispatch] = useReducer(historyReducer, initial.document, createHistory);
  const [etag, setEtag] = useState(initial.etag);
  const [draftVersion, setDraftVersion] = useState(initial.version);
  const [savedDocument, setSavedDocument] = useState(initial.document);
  const [operation, setOperation] = useState<"add" | "subtract">("add");
  const [labelKey, setLabelKey] = useState(labels[0]?.key ?? "defect");
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [pendingPoints, setPendingPoints] = useState<AnnotationPoint[]>([]);
  const [message, setMessage] = useState<string | null>(null);
  const [traceError, setTraceError] = useState<string | null>(null);
  const [tracing, setTracing] = useState(false);
  const [assistMode, setAssistMode] = useState<"point" | "box">("point");
  const [assistPoints, setAssistPoints] = useState<AssistPoint[]>([]);
  const [assistBox, setAssistBox] = useState<AssistBox | null>(null);
  const [candidateIndex, setCandidateIndex] = useState(0);
  const [assetJobId, setAssetJobId] = useState<number>();
  const [confirmDiscard, setConfirmDiscard] = useState(false);
  const [copyOpen, setCopyOpen] = useState(false);
  const [copyTargets, setCopyTargets] = useState<number[]>([]);
  const canvasRef = useRef<AnnotationCanvasHandle>(null);
  const refreshedAssetJob = useRef<number | undefined>(undefined);

  const recolour = useUpdateAnnotationLabel(datasetId);
  const save = useSaveDraft(target);
  const discard = useDiscardDraft(target);
  const copyRegions = useCopyRegions(imageId);
  const complete = useCompleteDraft(
    target,
    useMemo(() => sample.images.map((image) => image.id), [sample.images]),
  );
  const capability = useSegmentAssistCapability();
  const modelAssets = useModelAssets();
  const installAsset = useInstallModelAsset();
  const assist = useSegmentAssist(imageId);
  const asset = modelAssets.data?.assets.find((item) => item.key === "mobile-sam-vit-t");
  const followedAssetJobId = assetJobId ?? asset?.active_job?.id;
  const assetJob = useJob(followedAssetJobId);
  const cancelAssetJob = useCancelJob();
  const dirty = canonical(history.present) !== canonical(savedDocument);

  const flatQueue = useMemo(() => queueUnits(queue, perSample), [queue, perSample]);
  const flatPrevious = useMemo(
    () => queueUnits(previousQueue, perSample),
    [previousQueue, perSample],
  );
  const flatNext = useMemo(() => queueUnits(nextQueue, perSample), [nextQueue, perSample]);
  const queueIndex = flatQueue.findIndex((item) =>
    perSample ? item.sample.id === sample.id : item.image.id === imageId,
  );
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
  const selected = history.present.shapes.find((shape) => shape.id === selectedId) ?? null;
  const candidates = assist.data?.candidates ?? [];
  const candidate = candidates[candidateIndex] ?? null;

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
        ? history.present
        : (siblingDrafts.get(reference.id)?.document ?? {
            ...history.present,
            base: "empty",
            image_width: reference.width,
            image_height: reference.height,
            shapes: [],
          });

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
      navigate(
        `/datasets/${datasetId}/annotate/${item.sample.id}/${item.image.id}?offset=${nextOffset}`,
        { replace: true },
      );
    },
    [datasetId, flatNext, flatPrevious, flatQueue, navigate, queue.length, queueOffset, queueTotal],
  );

  /**
   * Ensure a persisted draft exists and return the token that owns it.
   *
   * Two things it must get right. A clean document still has to be materialised when nothing
   * is persisted yet — completing an unedited seed is a real action, and accepting an imported
   * source mask as truth verbatim is the common case. And concurrent callers must share one
   * flight: the idle autosave and an explicit save would otherwise both start from the same
   * token and the loser would collect a 412 it caused itself.
   */
  const inFlight = useRef<Promise<string> | null>(null);
  const persist = useCallback((): Promise<string> => {
    if (inFlight.current) return inFlight.current;
    if (!dirty && etag !== null) return Promise.resolve(etag);
    const flight = save
      .mutateAsync({ document: history.present, etag })
      .then((saved) => {
        setEtag(saved.etag);
        setDraftVersion(saved.version);
        setSavedDocument(saved.document);
        setMessage("Draft saved");
        return saved.etag as string;
      })
      .finally(() => {
        inFlight.current = null;
      });
    inFlight.current = flight;
    return flight;
  }, [dirty, etag, history.present, save]);

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
      setMessage(
        `${result.copied} region${result.copied === 1 ? "" : "s"} copied to ${
          names.length > 0 ? names.join(", ") : `${result.targets.length} channels`
        }`,
      );
    } catch {
      // The dialog stays open with the mutation's error under the list.
    }
  }, [copyRegions, copyTargets, persist, sample.images]);

  const finishPolygon = useCallback(() => {
    if (pendingPoints.length < 3) return;
    const polygon: PolygonShape = {
      id: nextShapeId(),
      label_key: labelKey,
      kind: "polygon",
      operation,
      points: pendingPoints,
    };
    dispatch({ type: "commit", document: withShape(history.present, polygon) });
    setPendingPoints([]);
    setSelectedId(polygon.id);
    // The tool stays where it is. Most parts carry more than one defect, and dropping back to
    // Select after every ring meant pressing P again for each of them.
  }, [history.present, labelKey, operation, pendingPoints]);

  const moveShape = useCallback(
    (shapeId: string, dx: number, dy: number) => {
      dispatch({ type: "commit", document: translateShape(history.present, shapeId, dx, dy) });
    },
    [history.present],
  );

  /**
   * One brush or eraser gesture.
   *
   * The rule, stated in the inspector so it is never a guess: **a stroke extends the selected
   * region**, and with nothing selected the brush starts one. Every gesture used to mint a new
   * shape, so a defect painted in three strokes was three regions.
   *
   * **The eraser never creates.** It takes paint off the selected region, or — with nothing
   * selected — off every painted region it passes over. It used to append a `subtract` layer
   * instead, which is a region: the tool for removing things added one, and said so in the
   * region list. Cutting a hole through a *polygon* is still possible, but as an explicit
   * Subtract region in the New region panel rather than as the eraser's side effect.
   */
  const applyStroke = useCallback(
    async (points: AnnotationPoint[]) => {
      const geometry = {
        points,
        size: brushSize,
        imageWidth: history.present.image_width,
        imageHeight: history.present.image_height,
      };
      const target = selected?.kind === "bitmap" ? selected : null;

      if (tool === "eraser") {
        // The selection scopes the eraser exactly as it scopes the brush; without one, the
        // pointer itself is the scope.
        const targets = target
          ? [target]
          : strokeTargets(
              history.present.shapes,
              strokeBounds(points, brushSize, geometry.imageWidth, geometry.imageHeight),
            );
        if (targets.length === 0) {
          setMessage(
            selected?.kind === "polygon"
              ? "The eraser takes paint off painted regions. Reshape this polygon by its vertices, or delete it."
              : "Nothing painted here to erase.",
          );
          return;
        }
        const painted = await Promise.all(
          targets.map((shape) => paintStroke(shape, { ...geometry, erase: true })),
        );
        let next = history.present;
        let removed = 0;
        targets.forEach((shape, index) => {
          const result = painted[index];
          if (result === undefined) return;
          if (result === null) {
            next = withoutShape(next, shape.id);
            removed += 1;
            if (shape.id === selectedId) setSelectedId(null);
            return;
          }
          next = replaceShape(next, shape.id, [result]);
        });
        dispatch({ type: "commit", document: next });
        if (removed > 0) setMessage(removed === 1 ? "Region erased" : `${removed} regions erased`);
        return;
      }

      if (target) {
        const painted = await paintStroke(target, { ...geometry, erase: false });
        // A brush cannot empty a region, but `paintStroke` promises `null` for an empty result
        // and honouring it here keeps the one contract rather than two.
        if (painted === null) {
          dispatch({ type: "commit", document: withoutShape(history.present, target.id) });
          setSelectedId(null);
          return;
        }
        dispatch({
          type: "commit",
          document: replaceShape(history.present, target.id, [painted]),
        });
        return;
      }
      const shape = bitmapStroke({ ...geometry, labelKey, operation });
      if (!shape) return;
      dispatch({ type: "commit", document: withShape(history.present, shape) });
      setSelectedId(shape.id);
    },
    [brushSize, history.present, labelKey, operation, selected, selectedId, tool],
  );

  const completeCurrent = useCallback(async () => {
    try {
      const currentEtag = await persist();
      const revisions = await complete.mutateAsync(currentEtag);
      const first = revisions[0];
      setMessage(
        revisions.length > 1
          ? `Completed revision ${first?.revision_no} on ${revisions.length} channels`
          : `Completed revision ${first?.revision_no}`,
      );
      openQueueItem(queueIndex + 1);
    } catch {
      // The mutations expose their errors in the inspector; keep the current image open.
    }
  }, [complete, openQueueItem, persist, queueIndex]);

  const discardCurrent = useCallback(
    async (force: boolean) => {
      try {
        await discard.mutateAsync({ etag, force });
        setConfirmDiscard(false);
        setMessage("Draft discarded");
        await onReload();
      } catch {
        // A 412 keeps the dialog open and turns its button into the explicit force; the
        // mutation's error is rendered inside it.
      }
    },
    [discard, etag, onReload],
  );

  const removeSelected = useCallback(() => {
    if (!selectedId) return;
    dispatch({
      type: "commit",
      document: withoutShape(history.present, selectedId),
    });
    setSelectedId(null);
  }, [history.present, selectedId]);

  const updateSelected = (patch: Partial<Pick<AnnotationShape, "label_key" | "operation">>) => {
    if (!selectedId) return;
    dispatch({
      type: "commit",
      document: {
        ...history.present,
        shapes: history.present.shapes.map((shape) =>
          shape.id === selectedId ? { ...shape, ...patch } : shape,
        ),
      },
    });
  };

  const traceSelected = async () => {
    if (!selected || selected.kind !== "bitmap") return;
    setTracing(true);
    setTraceError(null);
    try {
      const polygons = await traceBitmapShape(selected);
      if (polygons.length === 0) throw new Error("No contour could be derived from this region.");
      dispatch({
        type: "commit",
        document: replaceShape(history.present, selected.id, polygons),
      });
      setSelectedId(polygons[0]?.id ?? null);
      setTool("select");
      setMessage(polygons.length === 1 ? "Editable contour created" : `${polygons.length} editable contours created`);
    } catch (error) {
      setTraceError(error instanceof Error ? error.message : "Contour tracing failed.");
    } finally {
      setTracing(false);
    }
  };

  const clearAssist = useCallback(() => {
    setAssistPoints([]);
    setAssistBox(null);
    setCandidateIndex(0);
    assist.reset();
  }, [assist]);

  const requestAssist = async () => {
    try {
      const result = await assist.mutateAsync({
        points: assistPoints,
        box: assistBox ?? undefined,
        label_key: labelKey,
        operation,
      });
      setCandidateIndex(0);
      setMessage(
        `${result.candidates.length} suggestion${result.candidates.length === 1 ? "" : "s"} · ${result.device.toUpperCase()} · ${result.warm ? "warm" : "loaded"}`,
      );
    } catch {
      // The mutation's error stays beside the controls that caused it.
    }
  };

  const acceptCandidate = async (asContour: boolean) => {
    if (!candidate) return;
    try {
      if (asContour) {
        const polygons = await traceBitmapShape(candidate.shape);
        if (polygons.length === 0) throw new Error("No editable contour could be derived.");
        dispatch({
          type: "commit",
          document: {
            ...history.present,
            shapes: [...history.present.shapes, ...polygons],
          },
        });
        setSelectedId(polygons[0]?.id ?? null);
      } else {
        dispatch({ type: "commit", document: withShape(history.present, candidate.shape) });
        setSelectedId(candidate.shape.id);
      }
      setTool("select");
      clearAssist();
      setMessage(asContour ? "Editable suggested contour accepted" : "Suggested mask accepted");
    } catch (error) {
      setTraceError(error instanceof Error ? error.message : "Suggestion conversion failed.");
    }
  };

  useEffect(() => {
    if (!followedAssetJobId || !isTerminal(assetJob.job?.status)) return;
    if (refreshedAssetJob.current === followedAssetJobId) return;
    refreshedAssetJob.current = followedAssetJobId;
    void modelAssets.refetch();
    void capability.refetch();
    if (assetJob.job?.status === "succeeded") setMessage("MobileSAM is ready");
  }, [assetJob.job?.status, capability, followedAssetJobId, modelAssets]);

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      const target = event.target;
      if (
        target instanceof HTMLElement &&
        target.closest("[data-annotation-canvas]") &&
        ["ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight", "Enter", " "].includes(event.key)
      ) {
        return;
      }
      if (
        target instanceof HTMLInputElement ||
        target instanceof HTMLTextAreaElement ||
        target instanceof HTMLSelectElement
      ) {
        return;
      }
      const key = event.key.toLowerCase();
      const command = event.metaKey || event.ctrlKey;
      if (command && key === "s") {
        event.preventDefault();
        void persist();
      } else if (command && key === "z") {
        event.preventDefault();
        dispatch({ type: event.shiftKey ? "redo" : "undo" });
      } else if (key === "v") {
        setTool("select");
      } else if (key === "p") {
        setTool("polygon");
      } else if (key === "b") {
        setTool("brush");
      } else if (key === "e") {
        setTool("eraser");
      } else if (key === "a") {
        setTool("assist");
      } else if (event.key === "Enter" && pendingPoints.length >= 3) {
        event.preventDefault();
        finishPolygon();
      } else if (event.key === "Escape") {
        // Cancel the current thing, and only that. Escape used to also drop back to Select,
        // which made "escape to start a fresh region" cost a second keystroke to get the
        // brush back — V is the way to Select, and it always was.
        if (pendingPoints.length > 0) {
          setPendingPoints([]);
        } else {
          setSelectedId(null);
          clearAssist();
        }
      } else if (
        (event.key === "Backspace" || event.key === "Delete") &&
        pendingPoints.length > 0
      ) {
        // Undo does not reach a ring that has not been committed yet, so the only way back
        // from a misplaced vertex used to be Escape and starting over.
        event.preventDefault();
        setPendingPoints((points) => points.slice(0, -1));
      } else if ((event.key === "Backspace" || event.key === "Delete") && selectedId) {
        event.preventDefault();
        removeSelected();
      } else if (event.key === "ArrowRight" && !dirty) {
        openQueueItem(queueIndex + 1);
      } else if (event.key === "ArrowLeft" && !dirty) {
        openQueueItem(queueIndex - 1);
      } else if (key === "j" && !dirty) {
        openQueueItem(queueIndex + 1);
      } else if (key === "k" && !dirty) {
        openQueueItem(queueIndex - 1);
      } else if (event.key === "[") {
        void openChannel(activeIndex - 1);
      } else if (event.key === "]") {
        void openChannel(activeIndex + 1);
      } else if (event.key === "," || event.key === "<") {
        // Not `[` and `]`, the conventional pair — those are channel navigation here and
        // have been longer. Shift jumps by ten so the whole range is a few keystrokes.
        setBrushSize(brushSize - (event.shiftKey ? 10 : 1));
      } else if (event.key === "." || event.key === ">") {
        setBrushSize(brushSize + (event.shiftKey ? 10 : 1));
      } else if (key === "c" && !complete.isPending) {
        void completeCurrent();
      } else if (key === "0") {
        canvasRef.current?.fit();
      } else if (key === "1") {
        canvasRef.current?.actualPixels();
      }
    };
    globalThis.addEventListener("keydown", onKey);
    return () => globalThis.removeEventListener("keydown", onKey);
  }, [activeIndex, clearAssist, complete.isPending, completeCurrent, dirty, finishPolygon, openChannel, openQueueItem, pendingPoints.length, persist, queueIndex, removeSelected, selectedId]);

  useEffect(() => {
    if (!message) return;
    const timer = globalThis.setTimeout(() => setMessage(null), 2200);
    return () => globalThis.clearTimeout(timer);
  }, [message]);

  const mutationError = save.error ?? complete.error;
  // 412 rather than a substring of the detail: the status is the contract, the prose is not.
  const isConflict = (error: Error | null | undefined) =>
    error instanceof ApiError && error.status === 412;

  useBeforeUnload(
    useCallback(
      (event: BeforeUnloadEvent) => {
        if (!dirty) return;
        event.preventDefault();
        event.returnValue = "";
      },
      [dirty],
    ),
  );

  useEffect(() => {
    if (
      !dirty ||
      pendingPoints.length > 0 ||
      save.isPending ||
      save.error ||
      complete.isPending
    ) return;
    const timer = globalThis.setTimeout(() => {
      void persist();
    }, 1200);
    return () => globalThis.clearTimeout(timer);
  }, [
    complete.isPending,
    dirty,
    history.present,
    pendingPoints.length,
    persist,
    save.error,
    save.isPending,
  ]);

  return (
    <div className="flex min-h-0 flex-1 flex-col overflow-hidden bg-ground">
      <header className="flex h-13 shrink-0 items-center gap-3 border-b border-line bg-surface px-3">
        <Link
          to={`/datasets/${datasetId}/annotate?offset=${queueOffset}`}
          className={cn("rounded-control p-1.5 text-fg-muted hover:bg-raised hover:text-fg", focusRing)}
          aria-label="Back to annotation queue"
        >
          <ArrowLeft className="size-4" />
        </Link>
        <div className="min-w-0">
          <div className="flex min-w-0 items-baseline gap-2">
            <span className="truncate text-sm font-semibold">{datasetName}</span>
            <span className="truncate font-mono text-xs text-fg-muted">
              {sample.external_id}{currentImage?.channel ? ` / ${currentImage.channel}` : ""}
            </span>
          </div>
          <div className="font-mono text-[10px] text-fg-subtle">
            {currentImage?.width} × {currentImage?.height} ·{" "}
            {perSample ? "sample" : "image"} {queueIndex + 1} of {flatQueue.length}
            {perSample && sample.images.length > 1 && ` · ${sample.images.length} channels share one annotation`}
          </div>
        </div>

        <div className="ml-auto flex items-center gap-2">
          <span className="hidden text-xs text-fg-muted sm:inline">
            {message ?? (dirty ? "Unsaved changes" : `Draft v${draftVersion}`)}
          </span>
          <Button
            variant="ghost"
            icon={<Trash2 />}
            disabled={etag === null || discard.isPending}
            title={
              etag === null
                ? "Nothing is saved yet — there is no draft to discard"
                : "Throw away this draft and reopen the newest completed truth"
            }
            onClick={() => setConfirmDiscard(true)}
          >
            Discard
          </Button>
          <Button
            icon={<Save />}
            disabled={!dirty || save.isPending}
            onClick={() => void persist()}
          >
            Save
          </Button>
          <Button
            variant="primary"
            icon={<Check />}
            disabled={save.isPending || complete.isPending || pendingPoints.length > 0}
            onClick={() => void completeCurrent()}
          >
            Complete
          </Button>
        </div>
      </header>

      {mutationError && (
        <div className="shrink-0 border-b border-line bg-surface px-3 py-2">
          <div className="flex items-center justify-between gap-3">
            <ErrorBox>{mutationError.message}</ErrorBox>
            {isConflict(mutationError) && (
              <Button onClick={() => void onReload()}>Reload server draft</Button>
            )}
          </div>
        </div>
      )}

      <div className="flex min-h-0 flex-1 overflow-hidden">
        <aside className="flex w-12 shrink-0 flex-col items-center gap-1 border-r border-line bg-surface py-2" aria-label="Annotation tools">
          <ToolButton icon={<MousePointer2 />} label="Select (V)" active={tool === "select"} onClick={() => setTool("select")} />
          <ToolButton icon={<Shapes />} label="Polygon (P)" active={tool === "polygon"} onClick={() => setTool("polygon")} />
          <ToolButton icon={<Brush />} label="Brush (B)" active={tool === "brush"} onClick={() => setTool("brush")} />
          <ToolButton icon={<Eraser />} label="Eraser (E)" active={tool === "eraser"} onClick={() => setTool("eraser")} />
          <ToolButton icon={<WandSparkles />} label="Contour assist (A)" active={tool === "assist"} onClick={() => setTool("assist")} />
          <span className="my-1 h-px w-6 bg-line" />
          <ToolButton icon={<Undo2 />} label="Undo (⌘Z)" disabled={history.past.length === 0} onClick={() => dispatch({ type: "undo" })} />
          <ToolButton icon={<Redo2 />} label="Redo (⇧⌘Z)" disabled={history.future.length === 0} onClick={() => dispatch({ type: "redo" })} />
          <span className="my-1 h-px w-6 bg-line" />
          <ToolButton icon={<ZoomIn />} label="Zoom in" onClick={() => setView({ ...view, zoom: Math.min(12, view.zoom * 1.25) })} />
          <ToolButton icon={<ZoomOut />} label="Zoom out" onClick={() => setView({ ...view, zoom: Math.max(0.25, view.zoom / 1.25) })} />
          <ToolButton icon={<Maximize2 />} label="Fit image (0)" onClick={() => canvasRef.current?.fit()} />
          <ToolButton icon={<span className="font-mono text-[9px] font-semibold">1:1</span>} label="Actual pixels (1)" onClick={() => canvasRef.current?.actualPixels()} />
          <span className="mt-auto px-1 text-center font-mono text-[9px] leading-3 text-fg-subtle">
            {view.zoom === 1 ? "Fit" : `${Math.round(view.zoom * 100)}% fit`}
          </span>
        </aside>

        <div className="flex min-h-0 min-w-0 flex-1 flex-col overflow-hidden">
          {sample.images.length > 1 && (
            <div className="flex h-11 shrink-0 items-center gap-3 border-b border-line bg-surface px-3">
              {/* The tabs are what gives way when the strip is over-subscribed, because they
                  are the one thing here that reads perfectly well half-scrolled. Everything to
                  the right is a control with a usable minimum size, and the blend slider in
                  particular collapsed to a few pixels of track once the view switch stopped
                  wrapping — a slider narrower than its own thumb is a rendering fault, not a
                  tight fit. */}
              <div className="min-w-0 flex-1 overflow-x-auto">
                <ChannelTabs
                  images={sample.images}
                  active={activeIndex}
                  onSelect={(index) => void openChannel(index)}
                />
              </div>
              <div className="flex shrink-0 items-center gap-2">
                {!perSample && (
                  <Button
                    icon={<Copy />}
                    disabled={history.present.shapes.length === 0 || copyable.length === 0}
                    title={
                      history.present.shapes.length === 0
                        ? "Draw a region first"
                        : copyable.length === 0
                          ? "No other channel of this part shares this source frame"
                          : "Put these regions on the other channels of this part"
                    }
                    onClick={() => {
                      // Only the channels that can actually receive it. A sibling of another
                      // size is shown, disabled, with its dimensions — pre-ticking it would
                      // arm a button whose only outcome is a 409.
                      setCopyTargets(copyable.map((image) => image.id));
                      setCopyOpen(true);
                    }}
                  >
                    Copy to…
                  </Button>
                )}
                {paneMode === "overlay" && (
                  <div className="flex w-56 shrink-0 items-center gap-2">
                    <span className="shrink-0 text-[11px] text-fg-muted">Blend</span>
                    <Slider
                      aria-label="Overlay opacity"
                      value={overlayOpacity}
                      min={0}
                      max={1}
                      step={0.02}
                      onValueChange={setOverlayOpacity}
                      readout={`${Math.round(overlayOpacity * 100)}%`}
                    />
                  </div>
                )}
                {paneMode !== "single" && referenceIndex !== null && (
                  // A fixed box, because `Select`'s trigger is `w-full` and would otherwise
                  // bid for the row against the slider beside it.
                  <div className="w-36 shrink-0">
                    <Select
                      aria-label="Second channel"
                      value={String(referenceIndex)}
                      options={sample.images.flatMap((image, index) =>
                        index === activeIndex
                          ? []
                          : [{ value: String(index), label: image.channel ?? "unassigned" }],
                      )}
                      onValueChange={(value) => workspace.setReferenceIndex(Number(value))}
                    />
                  </div>
                )}
                <SegmentedControl
                  aria-label="Channel view"
                  value={paneMode}
                  options={[
                    { value: "single", label: "One" },
                    { value: "compare", label: "Side by side" },
                    { value: "overlay", label: "Blend" },
                  ]}
                  onValueChange={(value) => setPaneMode(value as PaneMode)}
                />
              </div>
            </div>
          )}

          <div className="flex min-h-0 flex-1 overflow-hidden">
        <AnnotationCanvas
          ref={canvasRef}
          imageId={imageId}
          overlayImageId={paneMode === "overlay" ? reference?.id : undefined}
          overlayOpacity={overlayOpacity}
          maskOpacity={maskOpacity}
          label={`Annotation canvas — ${currentImage?.channel ?? "the sample"}`}
          document={history.present}
          labels={labels}
          selectedId={selectedId}
          tool={tool}
          pendingPoints={pendingPoints}
          brushSize={brushSize}
          assistMode={assistMode}
          assistPoints={assistPoints}
          assistBox={assistBox}
          assistShape={candidate?.shape ?? null}
          view={view}
          onView={setView}
          onSelect={setSelectedId}
          onPoint={(point) => setPendingPoints((points) => [...points, point])}
          onMovePoint={(shapeId, pointIndex, point) =>
            dispatch({
              type: "commit",
              document: withPolygonPoint(history.present, shapeId, pointIndex, point),
            })
          }
          onMoveShape={moveShape}
          onBrush={(points) => void applyStroke(points)}
          onFinishPolygon={finishPolygon}
          onAssistPoint={(point) => {
            assist.reset();
            setCandidateIndex(0);
            setAssistPoints((points) => [...points, point].slice(-32));
          }}
          onAssistBox={(box) => {
            assist.reset();
            setCandidateIndex(0);
            setAssistBox(box);
          }}
        />
            {paneMode === "compare" && reference && referenceDocument && (
              // A peer pane, not a second editor: it shares the one controlled `view`, so
              // panning or zooming either keeps both showing the same source pixels — which
              // is the entire point of putting two illuminations of one part side by side.
              //
              // Editing happens in one pane, always the left one, so there is never a question
              // of which document a stroke lands in. Wanting to draw on the right is answered
              // by making it the left: `Edit this channel` swaps the two.
              <div className="relative flex min-h-0 min-w-0 flex-1 border-l border-line">
                <div className="pointer-events-none absolute inset-x-0 top-0 z-10 flex items-start justify-between gap-2 p-2">
                  <span className="rounded-control border border-line bg-surface/90 px-2 py-1 text-[11px] text-fg-muted shadow-panel backdrop-blur-sm">
                    {reference.channel ?? "unassigned"}
                    {!perSample &&
                      ` · ${referenceDocument.shapes.length} region${referenceDocument.shapes.length === 1 ? "" : "s"}`}
                  </span>
                  {!perSample && (
                    <span className="pointer-events-auto">
                      <Button icon={<ArrowLeftRight />} onClick={() => void swapPanes()}>
                        Edit this channel
                      </Button>
                    </span>
                  )}
                </div>
                <AnnotationCanvas
                  imageId={reference.id}
                  maskOpacity={maskOpacity}
                  editable={false}
                  label={`Reference channel — ${reference.channel ?? "unassigned"}`}
                  document={referenceDocument}
                  labels={labels}
                  selectedId={null}
                  tool="select"
                  pendingPoints={[]}
                  brushSize={brushSize}
                  assistMode="point"
                  assistPoints={[]}
                  assistBox={null}
                  assistShape={null}
                  view={view}
                  onView={setView}
                  onSelect={() => undefined}
                  onPoint={() => undefined}
                  onMovePoint={() => undefined}
                  onMoveShape={() => undefined}
                  onBrush={() => undefined}
                  onFinishPolygon={() => undefined}
                  onAssistPoint={() => undefined}
                  onAssistBox={() => undefined}
                />
              </div>
            )}
          </div>
        </div>

        <aside className="flex w-72 shrink-0 flex-col border-l border-line bg-surface" aria-label="Annotation inspector">
          <section className="border-b border-line p-3">
            <h2 className="mb-2 text-xs font-semibold uppercase tracking-wide text-fg-muted">
              New region
            </h2>
            <div className="grid grid-cols-2 gap-2">
              <Select
                aria-label="New region label"
                value={labelKey}
                options={labels.map((label) => ({ value: label.key, label: label.name }))}
                onValueChange={setLabelKey}
              />
              <Select
                aria-label="New region operation"
                value={operation}
                options={[
                  { value: "add", label: "Add defect" },
                  { value: "subtract", label: "Subtract" },
                ]}
                onValueChange={(value) => setOperation(value as "add" | "subtract")}
              />
            </div>
            {pendingPoints.length > 0 && (
              // A readout, not a control. Closing is a click on the first vertex or a
              // double-click anywhere; a "Close" button in a side panel is neither where the
              // hand is nor what a polygon tool is expected to need.
              <p className="mt-2 rounded-control bg-raised px-2 py-1.5 text-xs text-fg-muted">
                {pendingPoints.length} vertex{pendingPoints.length === 1 ? "" : "es"} ·{" "}
                {pendingPoints.length < 3
                  ? "three closes a ring"
                  : "click the first vertex, double-click, or Enter"}{" "}
                · Backspace undoes one
              </p>
            )}
            {(tool === "brush" || tool === "eraser") && (
              <div className="mt-3 flex flex-col gap-2">
                <div>
                  <div className="mb-1 flex items-center justify-between text-xs text-fg-muted">
                    <span>Brush size</span>
                    <span className="font-mono">
                      {brushSize} px {brushSize === 1 ? "· one pixel" : ""}
                    </span>
                  </div>
                  {/* Slider *and* a number box: the useful values for correcting a mask are
                      at the very bottom of a 128-step track, where a drag cannot reliably
                      land on 1 rather than 2. The `,` and `.` keys do the same job with the
                      other hand still on the canvas. */}
                  <div className="flex items-center gap-2">
                    <Slider
                      aria-label="Brush size"
                      value={brushSize}
                      min={MIN_BRUSH_SIZE}
                      max={MAX_BRUSH_SIZE}
                      step={1}
                      onValueChange={setBrushSize}
                    />
                    <NumberInput
                      className="w-16 shrink-0"
                      aria-label="Brush size in pixels"
                      min={MIN_BRUSH_SIZE}
                      max={MAX_BRUSH_SIZE}
                      step={1}
                      value={brushSize}
                      onChange={(event) => setBrushSize(Number(event.target.value))}
                    />
                  </div>
                </div>
                {/* The rule, where the hand is, because it is the one thing about this tool
                    nobody can infer from looking at it. */}
                <p className="text-[11px] leading-4 text-fg-subtle">
                  {selected?.kind === "bitmap"
                    ? tool === "eraser"
                      ? "Erasing region " +
                        `${history.present.shapes.indexOf(selected) + 1} only. Escape to erase across all of them.`
                      : "Painting into region " +
                        `${history.present.shapes.indexOf(selected) + 1}. Escape starts a new one.`
                    : tool === "eraser"
                      ? "Nothing selected: this takes paint off whatever it passes over, and never adds a region."
                      : "Nothing selected: this starts a new region. Strokes after it extend that one."}
                </p>
              </div>
            )}
          </section>

          {tool === "assist" && (
            <section className="border-b border-line p-3">
              <div className="mb-2 flex items-center justify-between gap-2">
                <h2 className="text-xs font-semibold uppercase tracking-wide text-fg-muted">
                  Contour assist
                </h2>
                <Badge tone={asset?.status === "ready" ? "normal" : "neutral"}>
                  {asset?.status ?? "checking"}
                </Badge>
              </div>

              {modelAssets.isPending || capability.isPending ? (
                <SkeletonRows rows={2} />
              ) : followedAssetJobId && !isTerminal(assetJob.job?.status) ? (
                <div className="flex flex-col gap-2">
                  <div className="flex items-center justify-between gap-2 text-xs text-fg-muted">
                    <span>{assetJob.job?.message ?? "Downloading MobileSAM…"}</span>
                    <Button
                      size="sm"
                      variant="danger"
                      loading={cancelAssetJob.isPending}
                      onClick={() => cancelAssetJob.mutate(followedAssetJobId)}
                    >
                      Cancel
                    </Button>
                  </div>
                  <ProgressBar fraction={assetJob.job?.progress ?? 0} />
                </div>
              ) : asset?.status !== "ready" ? (
                <div className="flex flex-col gap-2 text-xs leading-5 text-fg-muted">
                  <p>
                    Download the verified 38.8 MiB TinyViT checkpoint once. It stays in
                    app-managed storage and is shared by every dataset.
                  </p>
                  {asset?.reason && <p className="text-warn">{asset.reason}</p>}
                  <a
                    href={asset?.license_url}
                    target="_blank"
                    rel="noreferrer"
                    className={cn("w-fit text-signal underline-offset-2 hover:underline", focusRing)}
                  >
                    Apache-2.0 licence
                  </a>
                  <Button
                    icon={<WandSparkles />}
                    loading={installAsset.isPending}
                    disabled={!asset}
                    onClick={() => {
                      if (!asset) return;
                      installAsset.mutate(asset.key, {
                        onSuccess: (job) => setAssetJobId(job.id),
                      });
                    }}
                  >
                    Accept licence & download
                  </Button>
                  {installAsset.error && <ErrorBox>{installAsset.error.message}</ErrorBox>}
                  {assetJob.job?.error && <ErrorBox>{assetJob.job.error}</ErrorBox>}
                </div>
              ) : capability.data && !capability.data.runtime_available ? (
                <ErrorBox>{capability.data.reason ?? "MobileSAM runtime is unavailable."}</ErrorBox>
              ) : (
                <div className="flex flex-col gap-3">
                  <SegmentedControl
                    aria-label="Assistance prompt"
                    value={assistMode}
                    options={[
                      { value: "point", label: "Points" },
                      { value: "box", label: "Box" },
                    ]}
                    onValueChange={(value) => {
                      clearAssist();
                      setAssistMode(value as "point" | "box");
                    }}
                  />
                  <p className="text-xs leading-5 text-fg-subtle">
                    {assistMode === "point"
                      ? "Click the defect. Shift-click marks background. Add points to refine."
                      : "Drag a tight box around the defect. Right-drag still pans."}
                  </p>
                  <div className="flex items-center justify-between rounded-control bg-raised px-2 py-1.5 text-xs">
                    <span>
                      {assistMode === "point"
                        ? `${assistPoints.length} point${assistPoints.length === 1 ? "" : "s"}`
                        : assistBox
                          ? `${Math.round(assistBox.x1 - assistBox.x0)} × ${Math.round(assistBox.y1 - assistBox.y0)} px`
                          : "No box"}
                    </span>
                    <Button size="sm" disabled={!assistPoints.length && !assistBox} onClick={clearAssist}>
                      Clear
                    </Button>
                  </div>
                  <Button
                    variant="primary"
                    icon={<WandSparkles />}
                    loading={assist.isPending}
                    disabled={
                      assist.isPending ||
                      (assistPoints.length === 0 &&
                        (!assistBox || assistBox.x1 - assistBox.x0 < 2 || assistBox.y1 - assistBox.y0 < 2))
                    }
                    onClick={() => void requestAssist()}
                  >
                    Suggest contours
                  </Button>
                  {assist.error && <ErrorBox>{assist.error.message}</ErrorBox>}
                  {candidates.length > 0 && (
                    <div className="flex flex-col gap-2">
                      <div className="grid grid-cols-3 gap-1" aria-label="Suggested masks">
                        {candidates.map((item, index) => (
                          <button
                            key={item.shape.id}
                            type="button"
                            onClick={() => setCandidateIndex(index)}
                            className={cn(
                              "rounded-control border px-1.5 py-1 text-left font-mono text-[10px]",
                              focusRing,
                              index === candidateIndex
                                ? "border-warn bg-warn/10 text-fg"
                                : "border-line text-fg-muted hover:border-line-strong",
                            )}
                          >
                            <span className="block">#{index + 1} · {item.score.toFixed(3)}</span>
                            <span className="block text-fg-subtle">{item.area.toLocaleString()} px</span>
                          </button>
                        ))}
                      </div>
                      <div className="grid grid-cols-2 gap-2">
                        <Button onClick={() => void acceptCandidate(false)}>Accept mask</Button>
                        <Button onClick={() => void acceptCandidate(true)}>Editable contour</Button>
                      </div>
                      <p className="text-[10px] leading-4 text-fg-subtle">
                        Quality is MobileSAM's own ranking score, not a calibrated probability.
                      </p>
                    </div>
                  )}
                </div>
              )}
            </section>
          )}

          <section className="min-h-0 flex-1 overflow-y-auto p-3">
            <div className="mb-2 flex items-center justify-between">
              <h2 className="text-xs font-semibold uppercase tracking-wide text-fg-muted">
                Regions
              </h2>
              <Badge tone="neutral">{history.present.shapes.length}</Badge>
            </div>

            {/* Appearance, beside the regions it governs. The right weight depends on the
                imagery: heavy enough to see over a bright specular surface is heavy enough to
                hide the texture of a dark field. */}
            <div className="mb-3 flex flex-col gap-2 rounded-control bg-raised/60 p-2">
              <div className="flex items-center gap-2">
                <span className="shrink-0 text-[11px] text-fg-muted">Mask</span>
                <Slider
                  aria-label="Mask opacity"
                  min={0.1}
                  max={1}
                  step={0.05}
                  value={maskOpacity}
                  onValueChange={setMaskOpacity}
                  readout={`${Math.round(maskOpacity * 100)}%`}
                />
              </div>
              <div className="flex flex-wrap items-center gap-1.5">
                {labels.map((label) => (
                  <Tooltip key={label.key} content={`Colour of ${label.name}`}>
                    <label
                      className={cn(
                        "flex cursor-pointer items-center gap-1.5 rounded-control px-1.5 py-1 text-[11px] text-fg-muted hover:bg-raised",
                        focusRing,
                      )}
                    >
                      <span
                        className="size-3 shrink-0 rounded-full border border-line-strong"
                        style={{ backgroundColor: label.color }}
                      />
                      <span className="max-w-24 truncate">{label.name}</span>
                      <input
                        type="color"
                        value={label.color}
                        aria-label={`Colour of ${label.name}`}
                        disabled={recolour.isPending}
                        // Committed on `change`, not on `input`: a colour picker streams every
                        // value the pointer passes over, and each one would be a PUT.
                        onChange={(event) =>
                          recolour.mutate({ ...label, color: event.target.value })
                        }
                        className="size-0 opacity-0"
                      />
                    </label>
                  </Tooltip>
                ))}
              </div>
            </div>
            {history.present.shapes.length === 0 ? (
              <Empty>No editable regions. Choose Polygon or press P.</Empty>
            ) : (
              <div className="flex flex-col gap-1">
                {history.present.shapes.map((shape, index) => (
                  <button
                    key={shape.id}
                    type="button"
                    onClick={() => setSelectedId(shape.id)}
                    className={cn(
                      "flex items-center gap-2 rounded-control px-2 py-1.5 text-left text-xs",
                      focusRing,
                      selectedId === shape.id ? "bg-raised text-fg" : "text-fg-muted hover:bg-raised/60",
                    )}
                  >
                    <CircleDot className="size-3.5" style={{ color: labels.find((label) => label.key === shape.label_key)?.color }} />
                    <span className="min-w-0 flex-1 truncate">
                      {index + 1}. {labels.find((label) => label.key === shape.label_key)?.name ?? shape.label_key}
                    </span>
                    <span className="font-mono text-[9px] text-fg-subtle">
                      {shape.operation === "subtract" ? "−" : "+"}{shape.kind === "polygon" ? `${shape.points.length}v` : "mask"}
                    </span>
                  </button>
                ))}
              </div>
            )}
          </section>

          <section className="min-h-32 border-t border-line p-3">
            <h2 className="mb-2 text-xs font-semibold uppercase tracking-wide text-fg-muted">
              Selection
            </h2>
            {!selected ? (
              <p className="text-xs leading-5 text-fg-subtle">
                Select a region to edit its class or operation, drag it with Select, or nudge it
                with the arrow keys.
              </p>
            ) : (
              <div className="flex flex-col gap-2">
                <Select
                  aria-label="Selected region label"
                  value={selected.label_key}
                  options={labels.map((label) => ({ value: label.key, label: label.name }))}
                  onValueChange={(value) => updateSelected({ label_key: value })}
                />
                <div className="flex gap-2">
                  <Select
                    aria-label="Selected region operation"
                    value={selected.operation}
                    options={[
                      { value: "add", label: "Add defect" },
                      { value: "subtract", label: "Subtract" },
                    ]}
                    onValueChange={(value) => updateSelected({ operation: value as "add" | "subtract" })}
                  />
                  <Button variant="danger" icon={<Trash2 />} onClick={removeSelected} aria-label="Delete selected region" />
                </div>
                <p className="text-[11px] leading-4 text-fg-subtle">
                  Drag to move · arrows nudge 1 px, Shift 10 px
                  {selected.kind === "polygon" ? " · drag a vertex to reshape" : " · brush extends it"}
                </p>
                {selected.kind === "bitmap" && (
                  <Button
                    icon={<WandSparkles />}
                    disabled={tracing}
                    onClick={() => void traceSelected()}
                  >
                    {tracing ? "Tracing…" : "Make editable contour"}
                  </Button>
                )}
                {traceError && <p className="text-xs text-defect">{traceError}</p>}
              </div>
            )}
          </section>

          <footer className="flex items-center justify-between border-t border-line px-3 py-2">
            <Button
              icon={<ArrowLeft />}
              disabled={(queueIndex <= 0 && queueOffset === 0) || dirty}
              onClick={() => openQueueItem(queueIndex - 1)}
              aria-label="Previous image"
            />
            <span className="text-center font-mono text-[10px] text-fg-subtle">
              J/K after save · C completes{sample.images.length > 1 ? " · [ ] channel" : ""}
            </span>
            <Button
              icon={<ArrowRight />}
              disabled={
                queueIndex < 0 ||
                (queueIndex + 1 >= flatQueue.length && queueOffset + queue.length >= queueTotal) ||
                dirty
              }
              onClick={() => openQueueItem(queueIndex + 1)}
              aria-label="Next image"
            />
          </footer>
        </aside>
      </div>

      <Dialog
        open={copyOpen}
        onOpenChange={(open) => {
          setCopyOpen(open);
          if (!open) copyRegions.reset();
        }}
        title="Copy regions to other channels"
        description={
          <>
            The {history.present.shapes.length} region
            {history.present.shapes.length === 1 ? "" : "s"} on{" "}
            {currentImage?.channel ?? "this channel"} are <em>added</em> to each channel you
            pick. Nothing already there is replaced, and each copy is editable on its own —
            the exposures are milliseconds apart, so a copy usually needs a nudge.
          </>
        }
        footer={
          <>
            <Button onClick={() => setCopyOpen(false)}>Cancel</Button>
            <Button
              variant="primary"
              icon={<Copy />}
              loading={copyRegions.isPending || save.isPending}
              disabled={copyTargets.length === 0}
              onClick={() => void copyToChannels()}
            >
              Copy to {copyTargets.length} channel{copyTargets.length === 1 ? "" : "s"}
            </Button>
          </>
        }
      >
        <div className="mt-3 flex flex-col gap-2">
          {otherImages.map((image) => {
            const held = siblingDrafts.get(image.id)?.document.shapes.length;
            const sized =
              currentImage !== undefined &&
              (image.width !== currentImage.width || image.height !== currentImage.height);
            return (
              <Checkbox
                key={image.id}
                checked={copyTargets.includes(image.id)}
                disabled={sized}
                label={image.channel ?? "unassigned"}
                description={
                  sized
                    ? `${image.width} × ${image.height} — an annotation never leaves its source frame`
                    : held === undefined
                      ? "…"
                      : `${held} region${held === 1 ? "" : "s"} here already`
                }
                onCheckedChange={(checked) =>
                  setCopyTargets((targets) =>
                    checked
                      ? [...targets, image.id]
                      : targets.filter((target) => target !== image.id),
                  )
                }
              />
            );
          })}
          {copyRegions.error && <ErrorBox>{copyRegions.error.message}</ErrorBox>}
        </div>
      </Dialog>

      <ConfirmDialog
        open={confirmDiscard}
        onOpenChange={setConfirmDiscard}
        destructive
        title="Discard this draft?"
        description={
          <>
            {perSample
              ? "Every channel of this part reopens on its newest completed truth, or on a blank canvas if it has none."
              : "This image reopens on its newest completed truth, or on a blank canvas if it has none."}{" "}
            Completed revisions are immutable and are not affected.
            {discard.error && (
              <span className="mt-2 block text-defect">
                {discard.error.message}
                {isConflict(discard.error) &&
                  " Discarding now throws away whatever the other window saved."}
              </span>
            )}
          </>
        }
        confirmLabel={isConflict(discard.error) ? "Discard anyway" : "Discard draft"}
        loading={discard.isPending}
        onConfirm={() => void discardCurrent(isConflict(discard.error))}
      />
    </div>
  );
}

function ToolButton({
  icon,
  label,
  active = false,
  disabled = false,
  onClick,
}: {
  icon: React.ReactNode;
  label: string;
  active?: boolean;
  disabled?: boolean;
  onClick: () => void;
}) {
  return (
    <Tooltip content={label}>
      <button
        type="button"
        aria-label={label}
        aria-pressed={active}
        disabled={disabled}
        onClick={onClick}
        className={cn(
          "grid size-9 place-items-center rounded-control text-fg-muted transition-colors disabled:opacity-30",
          active ? "bg-signal text-signal-fg" : "hover:bg-raised hover:text-fg",
          focusRing,
        )}
      >
        <span className="[&_svg]:size-4">{icon}</span>
      </button>
    </Tooltip>
  );
}
