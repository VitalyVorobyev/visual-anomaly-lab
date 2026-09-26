/**
 * The sample viewer's Explore rail section: the mode, the encoder, the few numbers each mode
 * has, and what the last answer cost.
 *
 * Every state says something in words. The first request on an image names the encoder it is
 * loading, because that one is slow and every later click is not; a running job explains why
 * nothing can be asked; a missing `dl` extra says so instead of offering controls that would
 * all fail.
 */

import { Eraser, PenLine } from "lucide-react";

import {
  Button,
  Callout,
  ErrorBox,
  InfoHint,
  SegmentedControl,
  Select,
  SkeletonRows,
  Slider,
  Switch,
} from "@vitavision/lab-ui";

import { ApiError } from "../../api/client";
import { RailSection } from "../../components/viewer/RailSection";
import type { ExploreMode, ExploreSession } from "./useExploreSession";

const MODES: { value: ExploreMode; label: string }[] = [
  { value: "similar", label: "Similar" },
  { value: "clusters", label: "Clusters" },
  { value: "pca", label: "PCA" },
  { value: "sam", label: "SAM" },
];

const HINT: Record<ExploreMode, string> = {
  similar: "Click a patch to find what the encoder thinks is like it. Shift-click marks one it should not match.",
  clusters: "The image's own patches, grouped by k-means. Click a region to pick its cluster.",
  pca: "The three directions the features vary most, as false colour. The colours mean nothing between images.",
  sam: "Click an object for MobileSAM's masks. Shift-click marks background.",
};

export function ExploreSection({ session }: { session: ExploreSession }) {
  const { capability, on, mode } = session;

  return (
    <RailSection
      title="Explore"
      hint={on && mode !== "sam" && session.backbone ? session.backbone.title : ""}
    >
      {capability.isPending ? (
        <SkeletonRows rows={2} />
      ) : capability.error ? (
        <ErrorBox>{capability.error.message}</ErrorBox>
      ) : !capability.data?.available ? (
        <Callout tone="info" title="Explore is unavailable">
          {capability.data?.reason ??
            capability.data?.backbones.find((entry) => entry.reason)?.reason ??
            "No encoder can be loaded."}
        </Callout>
      ) : (
        <>
          <Switch
            checked={on}
            onCheckedChange={session.setOn}
            label="What the encoder sees"
            description="Clicks on the image ask a frozen encoder. For intuition, not a result."
          />
          {on && (
            <>
              <SegmentedControl
                aria-label="Explore mode"
                value={mode}
                options={MODES}
                onValueChange={(value) => session.setMode(value as ExploreMode)}
              />
              <p className="text-xs leading-5 text-fg-subtle">{HINT[mode]}</p>
              {mode === "sam" ? <SamControls session={session} /> : <EncoderControls session={session} />}
              <Slider
                aria-label="Overlay opacity"
                value={session.opacity}
                min={0.1}
                max={1}
                step={0.05}
                onValueChange={session.setOpacity}
                readout={`opacity ${Math.round(session.opacity * 100)}%`}
              />
              <SendToEditor session={session} />
            </>
          )}
        </>
      )}
    </RailSection>
  );
}

function EncoderControls({ session }: { session: ExploreSession }) {
  const { explore, answer, mode, prompt, backbone } = session;
  const pointCount = prompt.points.length + prompt.negatives.length;
  // The first request on an image loads the encoder and encodes it; say which encoder, so the
  // wait has a name. After that the grid is cached and every click is numpy.
  const { encoding } = session;

  return (
    <div className="flex flex-col gap-3">
      <div className="flex items-center gap-1.5">
        <Select
          aria-label="Encoder"
          className="min-w-0 flex-1"
          value={backbone?.key ?? ""}
          options={session.backbones.map((entry) => ({
            value: entry.key,
            label: entry.title,
            note: entry.gated ? "gated" : undefined,
          }))}
          onValueChange={session.setBackbone}
        />
        <InfoHint label="About the encoders">
          Patch features from the last two blocks, each L2-normalised — what `dino_memory` reads.
          The image is fitted into a 32×32-patch frame. Gated DINOv3 encoders appear once an
          approved HF_TOKEN is set.
        </InfoHint>
      </div>

      {mode === "clusters" && (
        <Slider
          aria-label="Number of clusters"
          value={session.k}
          min={session.capability.data?.min_clusters ?? 2}
          max={session.capability.data?.max_clusters ?? 12}
          step={1}
          onValueChange={session.setKDraft}
          onValueCommit={session.commitK}
          readout={`K = ${session.k}`}
        />
      )}
      {mode === "similar" && (
        <>
          <Slider
            aria-label="Similarity threshold"
            value={session.threshold}
            min={0.05}
            max={0.95}
            step={0.01}
            onValueChange={session.setThreshold}
            readout={`mask ≥ ${session.threshold.toFixed(2)}`}
          />
          <PromptRow
            count={pointCount}
            label={`${prompt.points.length} like · ${prompt.negatives.length} unlike`}
            onClear={session.clear}
          />
        </>
      )}
      {mode === "clusters" && (
        <p className="font-mono text-[11px] text-fg-muted">
          {session.cluster !== null
            ? `cluster ${session.cluster} of ${answer?.clusters ?? session.k} picked`
            : answer
              ? `${answer.clusters ?? session.k} clusters · click one to pick it`
              : " "}
        </p>
      )}

      {encoding ? (
        <p className="text-xs text-fg-muted" role="status">
          Encoding with {backbone?.title ?? "the encoder"}…
        </p>
      ) : explore.isPending ? (
        <p className="text-xs text-fg-muted" role="status">
          Asking…
        </p>
      ) : null}
      <RequestError error={explore.error} onRetry={session.retry} />
      {answer && !explore.isPending && (
        <p className="font-mono text-[11px] leading-4 text-fg-subtle tabular-nums">
          {answer.grid_rows}×{answer.grid_cols} patches · {answer.device.toUpperCase()} ·{" "}
          {answer.cached ? "cached" : `encoded ${Math.round(answer.encode_ms)} ms`} ·{" "}
          {Math.round(answer.elapsed_ms)} ms
        </p>
      )}
    </div>
  );
}

function SamControls({ session }: { session: ExploreSession }) {
  const { sam, samCapability, candidates, prompt } = session;
  if (samCapability.data && !samCapability.data.available) {
    return (
      <Callout tone="info" title="MobileSAM is not ready">
        {samCapability.data.reason ??
          "Download it once from the annotation editor's Contour assist tool."}
      </Callout>
    );
  }
  return (
    <div className="flex flex-col gap-3">
      <PromptRow
        count={prompt.sam.length}
        label={`${prompt.sam.filter((point) => point.kind === "positive").length} object · ${
          prompt.sam.filter((point) => point.kind === "negative").length
        } background`}
        onClear={session.clear}
      />
      {sam.isPending && (
        <p className="text-xs text-fg-muted" role="status">
          Asking MobileSAM…
        </p>
      )}
      <RequestError error={sam.error} onRetry={session.retrySam} />
      {candidates.length > 0 && (
        <SegmentedControl
          aria-label="MobileSAM candidate"
          value={String(session.candidateIndex)}
          options={candidates.map((item, index) => ({
            value: String(index),
            label: `#${index + 1} · ${item.score.toFixed(2)}`,
          }))}
          onValueChange={(value) => session.setCandidateIndex(Number(value))}
        />
      )}
    </div>
  );
}

function PromptRow({
  count,
  label,
  onClear,
}: {
  count: number;
  label: string;
  onClear: () => void;
}) {
  return (
    <div className="flex items-center justify-between gap-2 rounded-control bg-raised px-2 py-1.5 text-xs">
      <span className="tabular-nums">{count === 0 ? "No points yet" : label}</span>
      <Button size="sm" icon={<Eraser />} disabled={count === 0} onClick={onClear}>
        Clear
      </Button>
    </div>
  );
}

/** A 409 is not a failure: it means a job has the device, and says which one. */
function RequestError({ error, onRetry }: { error: Error | null; onRetry: () => void }) {
  if (!error) return null;
  if (error instanceof ApiError && error.status === 409) {
    return (
      <Callout
        tone="warning"
        title="Not while a job runs"
        actions={
          <Button size="sm" onClick={onRetry}>
            Try again
          </Button>
        }
      >
        <p>{error.message}</p>
        <p className="mt-1">
          Explore shares the accelerator with jobs, so it waits rather than competing for it.
        </p>
      </Callout>
    );
  }
  return <ErrorBox>{error.message}</ErrorBox>;
}

function SendToEditor({ session }: { session: ExploreSession }) {
  const why =
    session.mode === "pca"
      ? "False colour has no mask to send."
      : session.mode === "clusters"
        ? "Pick a cluster to send it."
        : session.mode === "sam"
          ? "Click an object to get a mask to send."
          : "Click a patch to get a mask to send.";
  return (
    <div className="flex flex-col gap-1.5">
      <Button
        icon={<PenLine />}
        disabled={!session.sendable}
        loading={session.toShape.isPending}
        onClick={() => void session.send()}
      >
        Send to editor
      </Button>
      <p className="text-[11px] leading-4 text-fg-subtle">
        {session.sendable
          ? "Opens the editor with this mask as a suggestion to accept or discard."
          : why}
      </p>
      {session.toShape.error && <ErrorBox>{session.toShape.error.message}</ErrorBox>}
    </div>
  );
}
