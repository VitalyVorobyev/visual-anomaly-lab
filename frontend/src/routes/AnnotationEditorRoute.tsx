/** Full-height manual annotation workbench.
 *
 * The image is the dominant surface. Tools stay in a narrow left rail, object properties
 * and the queue stay in a supporting right rail, and neither can create a second page
 * scrollbar. Keyboard actions mirror every operation needed for a labelling pass.
 */

import {
  ArrowLeft,
  ArrowRight,
  Brush,
  Check,
  CircleDot,
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
  withPolygonPoint,
  withShape,
  withoutShape,
} from "../api/annotationState";
import { bitmapStroke, traceBitmapShape } from "../api/annotationBitmap";
import { queueUnits } from "../api/annotationQueue";
import type {
  AnnotationLabel,
  AnnotationPoint,
  AnnotationShape,
  AssistBox,
  AssistPoint,
  PolygonShape,
  SampleSummary,
} from "../api/client";
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
  Empty,
  ErrorBox,
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
  useEditorDraft,
  useSaveDraft,
  useSegmentAssist,
  useSegmentAssistCapability,
} from "../hooks/useAnnotations";
import { useDataset, useSample, useSamples } from "../hooks/useCatalog";
import { useCancelJob } from "../hooks/useExperiments";
import { isTerminal, useJob } from "../hooks/useJob";
import { useInstallModelAsset, useModelAssets } from "../hooks/useModelAssets";

const QUEUE_PAGE = 120;

/** How the whole channel column is shown at once. */
type PaneMode = "single" | "compare" | "overlay";

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
  onReload: () => Promise<void>;
}) {
  const navigate = useNavigate();
  const perSample = target.scope === "sample";
  const [history, dispatch] = useReducer(historyReducer, initial.document, createHistory);
  const [etag, setEtag] = useState(initial.etag);
  const [draftVersion, setDraftVersion] = useState(initial.version);
  const [savedDocument, setSavedDocument] = useState(initial.document);
  const [paneMode, setPaneMode] = useState<PaneMode>("single");
  const [referenceId, setReferenceId] = useState<number | null>(null);
  const [overlayOpacity, setOverlayOpacity] = useState(0.5);
  const [tool, setTool] = useState<EditorTool>("select");
  const [operation, setOperation] = useState<"add" | "subtract">("add");
  const [labelKey, setLabelKey] = useState(labels[0]?.key ?? "defect");
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [pendingPoints, setPendingPoints] = useState<AnnotationPoint[]>([]);
  const [view, setView] = useState<CanvasView>(INITIAL_CANVAS_VIEW);
  const [brushRadius, setBrushRadius] = useState(18);
  const [message, setMessage] = useState<string | null>(null);
  const [traceError, setTraceError] = useState<string | null>(null);
  const [tracing, setTracing] = useState(false);
  const [assistMode, setAssistMode] = useState<"point" | "box">("point");
  const [assistPoints, setAssistPoints] = useState<AssistPoint[]>([]);
  const [assistBox, setAssistBox] = useState<AssistBox | null>(null);
  const [candidateIndex, setCandidateIndex] = useState(0);
  const [assetJobId, setAssetJobId] = useState<number>();
  const canvasRef = useRef<AnnotationCanvasHandle>(null);
  const refreshedAssetJob = useRef<number | undefined>(undefined);

  const save = useSaveDraft(target);
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
  const reference =
    otherImages.find((image) => image.id === referenceId) ?? otherImages[0] ?? null;
  const selected = history.present.shapes.find((shape) => shape.id === selectedId) ?? null;
  const candidates = assist.data?.candidates ?? [];
  const candidate = candidates[candidateIndex] ?? null;

  /**
   * Shapes are drawn on a pane only when the document is truth *for that pane's image*.
   *
   * Under sample scope it is, by construction — one completion writes the same mask to
   * every channel. Under image scope the document belongs to the displayed photograph
   * alone, so drawing it over a sibling channel would show truth that does not exist
   * there. The reference pane then shows the bare photograph, which is still the thing
   * worth seeing: "is the defect visible in dark field at all?"
   */
  const referenceDocument = perSample
    ? history.present
    : { ...history.present, shapes: [] };

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
   * Show a different channel of the same part.
   *
   * Under sample scope this is a pure display change: the document is the part's, so the
   * shapes stay on screen and visibly land — or fail to land — on the new illumination.
   * Under image scope each channel owns its own truth, so this is real navigation and
   * takes the same dirty guard as moving through the queue.
   */
  const openChannel = useCallback(
    (index: number) => {
      const image = sample.images[index];
      if (!image || image.id === imageId) return;
      if (!perSample && dirty) return;
      navigate(
        `/datasets/${datasetId}/annotate/${sample.id}/${image.id}?offset=${queueOffset}`,
        { replace: true },
      );
    },
    [datasetId, dirty, imageId, navigate, perSample, queueOffset, sample.id, sample.images],
  );

  const persist = useCallback(async (): Promise<string> => {
    if (!dirty) return etag;
    const saved = await save.mutateAsync({ document: history.present, etag });
    setEtag(saved.etag);
    setDraftVersion(saved.version);
    setSavedDocument(saved.document);
    setMessage("Draft saved");
    return saved.etag;
  }, [dirty, etag, history.present, save]);

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
    setTool("select");
  }, [history.present, labelKey, operation, pendingPoints]);

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
        setPendingPoints([]);
        setSelectedId(null);
        clearAssist();
        setTool("select");
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
        openChannel(activeIndex - 1);
      } else if (event.key === "]") {
        openChannel(activeIndex + 1);
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
            {mutationError.message.includes("changed elsewhere") && (
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
            <div className="flex h-11 shrink-0 items-center gap-3 overflow-x-auto border-b border-line bg-surface px-3">
              <ChannelTabs
                images={sample.images}
                active={activeIndex}
                onSelect={openChannel}
                disabled={!perSample && dirty}
              />
              <div className="ml-auto flex shrink-0 items-center gap-2">
                {paneMode === "overlay" && (
                  <div className="flex w-40 items-center gap-2">
                    <span className="shrink-0 font-mono text-[10px] text-fg-subtle">
                      {Math.round(overlayOpacity * 100)}%
                    </span>
                    <Slider
                      aria-label="Overlay opacity"
                      value={overlayOpacity}
                      min={0}
                      max={1}
                      step={0.02}
                      onValueChange={setOverlayOpacity}
                    />
                  </div>
                )}
                {paneMode !== "single" && reference && (
                  <Select
                    aria-label="Second channel"
                    value={String(reference.id)}
                    options={otherImages.map((image) => ({
                      value: String(image.id),
                      label: image.channel ?? "unassigned",
                    }))}
                    onValueChange={(value) => setReferenceId(Number(value))}
                  />
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
          label={`Annotation canvas — ${currentImage?.channel ?? "the sample"}`}
          document={history.present}
          labels={labels}
          selectedId={selectedId}
          tool={tool}
          pendingPoints={pendingPoints}
          brushRadius={brushRadius}
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
          onBrush={(points) => {
            const shape = bitmapStroke({
              points,
              radius: brushRadius,
              imageWidth: history.present.image_width,
              imageHeight: history.present.image_height,
              labelKey,
              operation: tool === "eraser" ? "subtract" : operation,
            });
            if (!shape) return;
            dispatch({ type: "commit", document: withShape(history.present, shape) });
            setSelectedId(shape.id);
          }}
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
            {paneMode === "compare" && reference && (
              // A peer pane, not a second editor: it shares the one controlled `view`, so
              // panning or zooming either keeps both showing the same source pixels — which
              // is the entire point of putting two illuminations of one part side by side.
              <div className="flex min-h-0 min-w-0 flex-1 border-l border-line">
                <AnnotationCanvas
                  imageId={reference.id}
                  label={`Reference channel — ${reference.channel ?? "unassigned"}`}
                  document={referenceDocument}
                  labels={labels}
                  selectedId={null}
                  tool="select"
                  pendingPoints={[]}
                  brushRadius={brushRadius}
                  assistMode="point"
                  assistPoints={[]}
                  assistBox={null}
                  assistShape={null}
                  view={view}
                  onView={setView}
                  onSelect={() => undefined}
                  onPoint={() => undefined}
                  onMovePoint={() => undefined}
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
              <div className="mt-2 flex items-center justify-between rounded-control bg-raised px-2 py-1.5 text-xs">
                <span>{pendingPoints.length} vertices · Enter closes</span>
                <Button size="sm" disabled={pendingPoints.length < 3} onClick={finishPolygon}>
                  Close
                </Button>
              </div>
            )}
            {(tool === "brush" || tool === "eraser") && (
              <div className="mt-3">
                <div className="mb-1 flex items-center justify-between text-xs text-fg-muted">
                  <span>Brush size</span>
                  <span className="font-mono">{brushRadius}px</span>
                </div>
                <Slider
                  aria-label="Brush size"
                  value={brushRadius}
                  min={2}
                  max={96}
                  step={1}
                  onValueChange={setBrushRadius}
                />
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
              <p className="text-xs leading-5 text-fg-subtle">Select a region to edit its class or operation.</p>
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
