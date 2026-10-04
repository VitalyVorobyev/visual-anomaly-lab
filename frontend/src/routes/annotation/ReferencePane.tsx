/**
 * The side-by-side reference pane: a peer, not a second editor.
 *
 * It shares the one controlled `view`, so panning or zooming either keeps both showing the
 * same source pixels — which is the entire point of putting two illuminations of one part
 * side by side. Its own measurements are not reported: the editor's pane holds the view, and a
 * second pane re-anchoring it to its own viewport as it mounts would move the one being edited.
 * Editing happens in one pane, always the left one, so there is never a question of which
 * document a stroke lands in. Wanting to draw on the right is answered by making it the left:
 * `Edit this channel` swaps the two.
 */

import { ArrowLeftRight } from "lucide-react";

import type { AnnotationDocument, AnnotationLabel, ImageSummary } from "../../api/client";
import type { StageView } from "@vitavision/stage2d";
import { Button } from "@vitavision/ui";

import { AnnotationStage } from "../../components/annotation/AnnotationStage";

const ignore = () => undefined;

export function ReferencePane({
  reference,
  document,
  labels,
  perSample,
  maskOpacity,
  showRegions,
  brushSize,
  view,
  onView,
  onSwap,
}: {
  reference: ImageSummary;
  /** The reference channel's own truth — see `useChannelPanes`. */
  document: AnnotationDocument;
  labels: AnnotationLabel[];
  perSample: boolean;
  maskOpacity: number;
  showRegions: boolean;
  brushSize: number;
  view: StageView | null;
  onView: (view: StageView) => void;
  onSwap: () => void;
}) {
  return (
    <div className="relative flex min-h-0 min-w-0 flex-1 border-l border-line">
      <div className="pointer-events-none absolute inset-x-0 top-0 z-10 flex items-start justify-between gap-2 p-2">
        <span className="rounded-control border border-line bg-surface/90 px-2 py-1 text-[11px] text-fg-muted shadow-panel backdrop-blur-sm">
          {reference.channel ?? "unassigned"}
          {!perSample &&
            ` · ${document.shapes.length} region${document.shapes.length === 1 ? "" : "s"}`}
        </span>
        {!perSample && (
          <span className="pointer-events-auto">
            <Button icon={<ArrowLeftRight />} onClick={onSwap}>
              Edit this channel
            </Button>
          </span>
        )}
      </div>
      <AnnotationStage
        imageId={reference.id}
        maskOpacity={maskOpacity}
        showRegions={showRegions}
        editable={false}
        label={`Reference channel — ${reference.channel ?? "unassigned"}`}
        document={document}
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
        onView={(next, change) => {
          if (change.cause !== "measure") onView(next);
        }}
        onSelect={ignore}
        onPoint={ignore}
        onReshapePolygon={ignore}
        onReshapeBox={ignore}
        onMoveShape={ignore}
        onBrush={ignore}
        onFinishPolygon={ignore}
        onBox={ignore}
        onAssistPoint={ignore}
        onAssistBox={ignore}
      />
    </div>
  );
}
