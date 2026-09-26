/** Full-height manual annotation workbench.
 *
 * The image is the dominant surface. Tools stay in a narrow left rail, object properties
 * and the queue stay in a supporting right rail, and neither can create a second page
 * scrollbar. Keyboard actions mirror every operation needed for a labelling pass.
 *
 * This file is composition. Each seam is a hook in `routes/annotation/` — the draft session
 * (`useDraftSession`), queue traversal (`useQueueNavigation`), the channel panes
 * (`useChannelPanes`), edits to the document (`useDocumentCommands`), MobileSAM
 * (`useSegmentAssistSession`) and the keymap (`useEditorKeymap`) — and each panel is a
 * component beside them.
 */

import { useCallback, useMemo, useRef, useState } from "react";
import { useParams, useSearchParams } from "react-router";

import { paneFrame } from "../api/annotationPanes";
import { labelNote } from "../api/annotationLabelNote";
import type { AnnotationLabel, Label, SampleSummary } from "../api/client";
import {
  AnnotationCanvas,
  type AnnotationCanvasHandle,
} from "../components/annotation/AnnotationCanvas";
import { Button, ErrorBox, SkeletonRows } from "@vitavision/lab-ui";
import {
  type DraftEnvelope,
  type DraftTarget,
  useAnnotationLabels,
  useEditorDraft,
  useSiblingDrafts,
} from "../hooks/useAnnotations";
import { useDataset, useSample, useSamples, useSetLabel } from "../hooks/useCatalog";
import { AssistSection } from "./annotation/AssistSection";
import { ChannelToolbar } from "./annotation/ChannelToolbar";
import { CopyRegionsDialog, DiscardDraftDialog } from "./annotation/EditorDialogs";
import { EditorHeader } from "./annotation/EditorHeader";
import { QueueFooter } from "./annotation/QueueFooter";
import { ReferencePane } from "./annotation/ReferencePane";
import { RegionsSection } from "./annotation/RegionsSection";
import { SelectionSection } from "./annotation/SelectionSection";
import { ShortcutSheet } from "./annotation/ShortcutSheet";
import { ToolRail } from "./annotation/ToolRail";
import { ToolSection } from "./annotation/ToolSection";
import { useChannelPanes } from "./annotation/useChannelPanes";
import { useDocumentCommands } from "./annotation/useDocumentCommands";
import { isConflict, useAutosave, useDraftSession } from "./annotation/useDraftSession";
import { editorKeyActions, useEditorKeymap } from "./annotation/useEditorKeymap";
import { useFlashMessage } from "./annotation/useFlashMessage";
import { QUEUE_PAGE, useQueueNavigation } from "./annotation/useQueueNavigation";
import { useSegmentAssistSession } from "./annotation/useSegmentAssistSession";
import { type Workspace, useWorkspace } from "./annotation/useWorkspace";

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
      defaultChannel={dataset.data.default_channel}
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
  defaultChannel,
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
  /** The channel a part opens on under sample scope, so the queue agrees with the grid. */
  defaultChannel: string | null;
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
  const perSample = target.scope === "sample";
  const { tool, setTool, brushSize, setBrushSize, view, setView, regionsHidden } = workspace;
  const canvasRef = useRef<AnnotationCanvasHandle>(null);
  const [confirmDiscard, setConfirmDiscard] = useState(false);
  const [shortcutsOpen, setShortcutsOpen] = useState(false);
  const [message, flash] = useFlashMessage();

  const session = useDraftSession({
    target,
    initial,
    imageIds: useMemo(() => sample.images.map((image) => image.id), [sample.images]),
    flash,
  });
  const { history, dirty, persist } = session;
  const commands = useDocumentCommands({
    history,
    dispatch: session.dispatch,
    latest: session.latest,
    labels,
    tool,
    setTool,
    brushSize,
    flash,
  });
  useAutosave(session, commands.pendingPoints.length > 0);
  const queueNav = useQueueNavigation({
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
  });
  const panes = useChannelPanes({
    datasetId,
    sample,
    imageId,
    perSample,
    queueOffset,
    workspace,
    present: history.present,
    siblingDrafts,
    dirty,
    persist,
    flash,
  });
  const assist = useSegmentAssistSession({
    imageId,
    labelKey: commands.labelKey,
    operation: commands.operation,
    flash,
    accept: commands.acceptSuggestion,
  });

  /**
   * The part's verdict, edited where it is discovered to be wrong.
   *
   * **The label belongs to the sample, not the photograph** (ADR-0041), so this covers every
   * channel of the part however many times it was shot — which is why it sits beside the
   * sample's identity in the header rather than beside the channel strip.
   *
   * Deliberately *not* gated on `dirty`, unlike queue traversal. J/K are blocked with unsaved
   * work because they navigate away from it; this writes a different row and leaves the draft
   * alone. `useSetLabel` invalidates `["datasets", id]`, and drafts live under
   * `["annotations", ...]`, so the badge and the queue counts refresh while the open document,
   * its undo history and an unsaved stroke all survive.
   */
  const setLabel = useSetLabel(datasetId);
  const applyLabel = useCallback(
    (label: Label) => {
      if (setLabel.isPending || label === sample.label) return;
      setLabel.mutate({ sampleId: sample.id, label });
    },
    [setLabel, sample.id, sample.label],
  );

  const completeCurrent = useCallback(async () => {
    if (await session.completeDraft()) queueNav.openNext();
  }, [queueNav, session]);

  const discardCurrent = async (force: boolean) => {
    if (!(await session.discardDraft(force))) return;
    setConfirmDiscard(false);
    try {
      await onReload();
    } catch {
      // The refetch surfaces its own error through the draft query.
    }
  };

  useEditorKeymap(
    editorKeyActions({
      session,
      commands,
      queue: queueNav,
      panes,
      assist,
      workspace,
      canvas: canvasRef,
      applyLabel,
      complete: completeCurrent,
      openShortcuts: () => setShortcutsOpen(true),
    }),
  );

  // A label PATCH answers 404 or 500, never 412, so folding it in here cannot light up the
  // "Reload server draft" button below — that stays gated on an actual draft conflict.
  const mutationError = session.error ?? setLabel.error;
  const { reference, referenceDocument, currentImage } = panes;
  const { pendingPoints, selectedId } = commands;

  return (
    <div className="flex min-h-0 flex-1 flex-col overflow-hidden bg-ground">
      <EditorHeader
        datasetId={datasetId}
        datasetName={datasetName}
        sample={sample}
        currentImage={currentImage}
        perSample={perSample}
        queue={queueNav}
        labelPending={setLabel.isPending}
        onLabel={applyLabel}
        disagreement={labelNote(sample.label, history.present)}
        // No version until the first save creates the draft (a page view writes nothing), and
        // that state printed as "Draft vnull".
        status={
          message ??
          (dirty
            ? "Unsaved changes"
            : session.draftVersion === null || session.draftVersion === undefined
              ? "Not edited yet"
              : `Draft v${session.draftVersion}`)
        }
        canDiscard={session.etag !== null}
        discardPending={session.discard.isPending}
        onDiscard={() => setConfirmDiscard(true)}
        canSave={dirty && !session.save.isPending}
        onSave={() => void persist()}
        canComplete={
          !session.save.isPending && !session.complete.isPending && pendingPoints.length === 0
        }
        onComplete={() => void completeCurrent()}
      />

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
        <ToolRail
          tool={tool}
          onTool={setTool}
          canUndo={commands.canUndo}
          canRedo={commands.canRedo}
          onUndo={commands.undo}
          onRedo={commands.redo}
          onZoom={(factor) => canvasRef.current?.zoomBy(factor)}
          onFit={() => canvasRef.current?.fit()}
          onActualPixels={() => canvasRef.current?.actualPixels()}
          onShortcuts={() => setShortcutsOpen(true)}
        />

        <div className="flex min-h-0 min-w-0 flex-1 flex-col overflow-hidden">
          {sample.images.length > 1 && (
            <ChannelToolbar
              sample={sample}
              perSample={perSample}
              panes={panes}
              workspace={workspace}
              shapeCount={history.present.shapes.length}
            />
          )}

          <div className="flex min-h-0 flex-1 overflow-hidden">
            <AnnotationCanvas
              ref={canvasRef}
              imageId={imageId}
              overlayImageId={workspace.paneMode === "overlay" ? reference?.id : undefined}
              overlayOpacity={workspace.overlayOpacity}
              maskOpacity={workspace.maskOpacity}
              showRegions={!regionsHidden}
              label={`Annotation canvas — ${currentImage?.channel ?? "the sample"}`}
              document={history.present}
              labels={labels}
              selectedId={selectedId}
              tool={tool}
              pendingPoints={pendingPoints}
              brushSize={brushSize}
              assistMode={assist.mode}
              assistPoints={assist.points}
              assistBox={assist.box}
              assistShape={assist.candidate?.shape ?? null}
              view={view}
              onView={setView}
              onSelect={commands.setSelectedId}
              onPoint={commands.addPendingPoint}
              onMovePoint={commands.movePoint}
              onMoveShape={commands.moveShape}
              onBrush={(points) => void commands.applyStroke(points)}
              onFinishPolygon={commands.finishPolygon}
              onBox={commands.addBox}
              onAssistPoint={assist.addPoint}
              onAssistBox={assist.setBox}
            />
            {workspace.paneMode === "compare" && reference && referenceDocument && (
              <ReferencePane
                reference={reference}
                document={referenceDocument}
                labels={labels}
                perSample={perSample}
                maskOpacity={workspace.maskOpacity}
                showRegions={!regionsHidden}
                brushSize={brushSize}
                view={view}
                onView={setView}
                onSwap={() => void panes.swapPanes()}
              />
            )}
          </div>
        </div>

        <aside
          className="flex w-72 shrink-0 flex-col border-l border-line bg-surface"
          aria-label="Annotation inspector"
        >
          <ToolSection
            tool={tool}
            labels={labels}
            document={history.present}
            commands={commands}
            brushSize={brushSize}
            onBrushSize={setBrushSize}
          />
          {tool === "assist" && <AssistSection assist={assist} />}
          <RegionsSection
            datasetId={datasetId}
            shapes={history.present.shapes}
            labels={labels}
            selectedId={selectedId}
            onSelect={commands.setSelectedId}
            regionsHidden={regionsHidden}
            onRegionsHidden={workspace.setRegionsHidden}
            maskOpacity={workspace.maskOpacity}
            onMaskOpacity={workspace.setMaskOpacity}
          />
          <SelectionSection labels={labels} commands={commands} />
          <QueueFooter queue={queueNav} dirty={dirty} multiChannel={sample.images.length > 1} />
        </aside>
      </div>

      <CopyRegionsDialog
        panes={panes}
        shapeCount={history.present.shapes.length}
        siblingDrafts={siblingDrafts}
        savePending={session.save.isPending}
      />
      <DiscardDraftDialog
        open={confirmDiscard}
        onOpenChange={setConfirmDiscard}
        perSample={perSample}
        error={session.discard.error}
        pending={session.discard.isPending}
        onDiscard={(force) => void discardCurrent(force)}
      />
      <ShortcutSheet open={shortcutsOpen} onOpenChange={setShortcutsOpen} />
    </div>
  );
}
