/** Full-height manual annotation workbench.
 *
 * The image is the dominant surface. Tools stay in a narrow left rail, object properties
 * and the queue stay in a supporting right rail, and neither can create a second page
 * scrollbar. Keyboard actions mirror every operation needed for a labelling pass.
 */

import {
  ArrowLeft,
  ArrowRight,
  Check,
  CircleDot,
  Hand,
  MousePointer2,
  Redo2,
  RotateCcw,
  Save,
  Shapes,
  Trash2,
  Undo2,
  ZoomIn,
  ZoomOut,
} from "lucide-react";
import { useCallback, useEffect, useMemo, useReducer, useState } from "react";
import { Link, useNavigate, useParams, useSearchParams } from "react-router";

import {
  canonical,
  createHistory,
  historyReducer,
  nextShapeId,
  withPolygonPoint,
  withShape,
  withoutShape,
} from "../api/annotationState";
import type {
  AnnotationLabel,
  AnnotationPoint,
  AnnotationShape,
  PolygonShape,
  SampleSummary,
} from "../api/client";
import {
  AnnotationCanvas,
  INITIAL_CANVAS_VIEW,
  type CanvasView,
  type EditorTool,
} from "../components/annotation/AnnotationCanvas";
import {
  Badge,
  Button,
  Empty,
  ErrorBox,
  Select,
  SkeletonRows,
  Tooltip,
  cn,
  focusRing,
} from "../components/ui";
import {
  type DraftEnvelope,
  useAnnotationDraft,
  useAnnotationLabels,
  useCompleteAnnotation,
  useSaveAnnotation,
} from "../hooks/useAnnotations";
import { useDataset, useSample, useSamples } from "../hooks/useCatalog";

const QUEUE_PAGE = 120;

export function AnnotationEditorRoute() {
  const params = useParams();
  const datasetId = Number(params["datasetId"]);
  const sampleId = Number(params["sampleId"]);
  const imageId = Number(params["imageId"]);
  const [searchParams] = useSearchParams();
  const offset = Math.max(0, Number(searchParams.get("offset") ?? 0) || 0);

  const dataset = useDataset(datasetId);
  const sample = useSample(datasetId, sampleId);
  const queue = useSamples(datasetId, { limit: QUEUE_PAGE, offset });
  const labels = useAnnotationLabels(datasetId);
  const draft = useAnnotationDraft(imageId);

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
    !draft.data
  ) {
    return <SkeletonRows rows={8} />;
  }

  return (
    <EditorReady
      key={imageId}
      datasetId={datasetId}
      imageId={imageId}
      datasetName={dataset.data.name}
      sample={sample.data}
      queue={queue.data.items}
      queueOffset={offset}
      labels={labels.data}
      initial={draft.data}
    />
  );
}

function EditorReady({
  datasetId,
  imageId,
  datasetName,
  sample,
  queue,
  queueOffset,
  labels,
  initial,
}: {
  datasetId: number;
  imageId: number;
  datasetName: string;
  sample: SampleSummary;
  queue: SampleSummary[];
  queueOffset: number;
  labels: AnnotationLabel[];
  initial: DraftEnvelope;
}) {
  const navigate = useNavigate();
  const [history, dispatch] = useReducer(historyReducer, initial.draft.document, createHistory);
  const [etag, setEtag] = useState(initial.etag);
  const [savedDocument, setSavedDocument] = useState(initial.draft.document);
  const [tool, setTool] = useState<EditorTool>("select");
  const [operation, setOperation] = useState<"add" | "subtract">("add");
  const [labelKey, setLabelKey] = useState(labels[0]?.key ?? "defect");
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [pendingPoints, setPendingPoints] = useState<AnnotationPoint[]>([]);
  const [view, setView] = useState<CanvasView>(INITIAL_CANVAS_VIEW);
  const [message, setMessage] = useState<string | null>(null);

  const save = useSaveAnnotation(imageId);
  const complete = useCompleteAnnotation(imageId);
  const dirty = canonical(history.present) !== canonical(savedDocument);

  const flatQueue = useMemo(
    () => queue.flatMap((item) => item.images.map((image) => ({ sample: item, image }))),
    [queue],
  );
  const queueIndex = flatQueue.findIndex((item) => item.image.id === imageId);
  const currentImage = sample.images.find((image) => image.id === imageId);
  const selected = history.present.shapes.find((shape) => shape.id === selectedId) ?? null;

  const openQueueItem = useCallback(
    (index: number) => {
      const item = flatQueue[index];
      if (!item) return;
      navigate(
        `/datasets/${datasetId}/annotate/${item.sample.id}/${item.image.id}?offset=${queueOffset}`,
        { replace: true },
      );
    },
    [datasetId, flatQueue, navigate, queueOffset],
  );

  const persist = useCallback(async (): Promise<string> => {
    if (!dirty) return etag;
    const saved = await save.mutateAsync({ document: history.present, etag });
    setEtag(saved.etag);
    setSavedDocument(saved.draft.document);
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
      const revision = await complete.mutateAsync(currentEtag);
      setMessage(`Completed revision ${revision.revision_no}`);
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

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      const target = event.target;
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
      } else if (key === "h") {
        setTool("hand");
      } else if (event.key === "Enter" && pendingPoints.length >= 3) {
        event.preventDefault();
        finishPolygon();
      } else if (event.key === "Escape") {
        setPendingPoints([]);
        setSelectedId(null);
        setTool("select");
      } else if ((event.key === "Backspace" || event.key === "Delete") && selectedId) {
        event.preventDefault();
        removeSelected();
      } else if (event.key === "ArrowRight" && !dirty) {
        openQueueItem(queueIndex + 1);
      } else if (event.key === "ArrowLeft" && !dirty) {
        openQueueItem(queueIndex - 1);
      } else if (key === "c" && !complete.isPending) {
        void completeCurrent();
      } else if (key === "0") {
        setView(INITIAL_CANVAS_VIEW);
      }
    };
    globalThis.addEventListener("keydown", onKey);
    return () => globalThis.removeEventListener("keydown", onKey);
  }, [complete.isPending, completeCurrent, dirty, finishPolygon, openQueueItem, pendingPoints.length, persist, queueIndex, removeSelected, selectedId]);

  useEffect(() => {
    if (!message) return;
    const timer = globalThis.setTimeout(() => setMessage(null), 2200);
    return () => globalThis.clearTimeout(timer);
  }, [message]);

  const mutationError = save.error ?? complete.error;

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
            {currentImage?.width} × {currentImage?.height} · image {queueIndex + 1} of {flatQueue.length}
          </div>
        </div>

        <div className="ml-auto flex items-center gap-2">
          <span className="hidden text-xs text-fg-muted sm:inline">
            {message ?? (dirty ? "Unsaved changes" : `Draft v${initial.draft.version}`)}
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
          <ErrorBox>{mutationError.message}</ErrorBox>
        </div>
      )}

      <div className="flex min-h-0 flex-1 overflow-hidden">
        <aside className="flex w-12 shrink-0 flex-col items-center gap-1 border-r border-line bg-surface py-2" aria-label="Annotation tools">
          <ToolButton icon={<MousePointer2 />} label="Select (V)" active={tool === "select"} onClick={() => setTool("select")} />
          <ToolButton icon={<Shapes />} label="Polygon (P)" active={tool === "polygon"} onClick={() => setTool("polygon")} />
          <ToolButton icon={<Hand />} label="Pan (H)" active={tool === "hand"} onClick={() => setTool("hand")} />
          <span className="my-1 h-px w-6 bg-line" />
          <ToolButton icon={<Undo2 />} label="Undo (⌘Z)" disabled={history.past.length === 0} onClick={() => dispatch({ type: "undo" })} />
          <ToolButton icon={<Redo2 />} label="Redo (⇧⌘Z)" disabled={history.future.length === 0} onClick={() => dispatch({ type: "redo" })} />
          <span className="my-1 h-px w-6 bg-line" />
          <ToolButton icon={<ZoomIn />} label="Zoom in" onClick={() => setView({ ...view, zoom: Math.min(12, view.zoom * 1.25) })} />
          <ToolButton icon={<ZoomOut />} label="Zoom out" onClick={() => setView({ ...view, zoom: Math.max(0.25, view.zoom / 1.25) })} />
          <ToolButton icon={<RotateCcw />} label="Fit image (0)" onClick={() => setView(INITIAL_CANVAS_VIEW)} />
          <span className="mt-auto font-mono text-[9px] text-fg-subtle">{Math.round(view.zoom * 100)}%</span>
        </aside>

        <AnnotationCanvas
          imageId={imageId}
          document={history.present}
          labels={labels}
          selectedId={selectedId}
          tool={tool}
          pendingPoints={pendingPoints}
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
        />

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
          </section>

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
              </div>
            )}
          </section>

          <footer className="flex items-center justify-between border-t border-line px-3 py-2">
            <Button
              icon={<ArrowLeft />}
              disabled={queueIndex <= 0 || dirty}
              onClick={() => openQueueItem(queueIndex - 1)}
              aria-label="Previous image"
            />
            <span className="text-center font-mono text-[10px] text-fg-subtle">
              ← → after save · C completes
            </span>
            <Button
              icon={<ArrowRight />}
              disabled={queueIndex < 0 || queueIndex + 1 >= flatQueue.length || dirty}
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
