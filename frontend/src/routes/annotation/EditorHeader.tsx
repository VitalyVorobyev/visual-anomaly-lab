/** The editor's header: identity, the part's verdict, draft status and the three document actions. */

import { ArrowLeft, Check, Save, Trash2, TriangleAlert } from "lucide-react";
import { Link } from "react-router";

import type { ImageSummary, Label, SampleSummary } from "../../api/client";
import { Button, InfoHint, SegmentedControl, cn, focusRing } from "@vitavision/lab-ui";
import type { QueueNavigation } from "./useQueueNavigation";

/**
 * The sample verdict, as a control and as three keystrokes.
 *
 * The same letters the sample viewer uses (`SampleRoute.tsx`), because a reader moves between
 * the two screens and a labelling reflex should not have to be relearned.
 */
const LABEL_OPTIONS: { value: Label; label: string }[] = [
  { value: "normal", label: "normal" },
  { value: "defect", label: "defect" },
  { value: "unlabeled", label: "unlabeled" },
];

export function EditorHeader({
  datasetId,
  datasetName,
  sample,
  currentImage,
  perSample,
  queue,
  labelPending,
  onLabel,
  disagreement,
  status,
  canDiscard,
  discardPending,
  onDiscard,
  canSave,
  onSave,
  canComplete,
  onComplete,
}: {
  datasetId: number;
  datasetName: string;
  sample: SampleSummary;
  currentImage: ImageSummary | undefined;
  perSample: boolean;
  queue: QueueNavigation;
  labelPending: boolean;
  onLabel: (label: Label) => void;
  /** What to say when the label and the document above it disagree. `null` most of the time. */
  disagreement: string | null;
  status: string;
  /** `false` until the first save has created a draft there is to discard. */
  canDiscard: boolean;
  discardPending: boolean;
  onDiscard: () => void;
  canSave: boolean;
  onSave: () => void;
  canComplete: boolean;
  onComplete: () => void;
}) {
  return (
    <header className="flex h-13 shrink-0 items-center gap-3 border-b border-line bg-surface px-3">
      <Link
        to={`/datasets/${datasetId}/annotate?offset=${queue.offset}`}
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
          {perSample ? "sample" : "image"} {queue.index + 1} of {queue.length}
          {perSample && sample.images.length > 1 && ` · ${sample.images.length} channels share one annotation`}
          {/* `import` is the state that means nobody has checked this yet, so both halves are
              worth printing rather than only flagging the corrected ones. */}
          {` · ${sample.label_source === "manual" ? "hand-set" : "imported"} label`}
        </div>
      </div>

      <div className="ml-auto flex items-center gap-2">
        {/* The verdict for the whole part. `SegmentedControl` treats `""` as unset and
            highlights `defaultValue` in its place — so no `defaultValue` here: every label
            is a real value, `unlabeled` included, and `""` is not one of them. */}
        <span className="flex items-center gap-1">
          <SegmentedControl
            aria-label={
              sample.images.length > 1
                ? `Sample label — applies to all ${sample.images.length} channels of this part`
                : "Sample label"
            }
            value={sample.label}
            options={LABEL_OPTIONS}
            disabled={labelPending}
            onValueChange={(value) => onLabel(value as Label)}
          />
          {disagreement && (
            <InfoHint icon={TriangleAlert} label="The label and the drawn regions disagree">
              {disagreement}
            </InfoHint>
          )}
        </span>
        <span className="hidden text-xs text-fg-muted sm:inline">{status}</span>
        <Button
          variant="ghost"
          icon={<Trash2 />}
          disabled={!canDiscard || discardPending}
          title={
            canDiscard
              ? "Throw away this draft and reopen the newest completed truth"
              : "Nothing is saved yet — there is no draft to discard"
          }
          onClick={onDiscard}
        >
          Discard
        </Button>
        <Button icon={<Save />} disabled={!canSave} onClick={onSave}>
          Save
        </Button>
        <Button variant="primary" icon={<Check />} disabled={!canComplete} onClick={onComplete}>
          Complete
        </Button>
      </div>
    </header>
  );
}
