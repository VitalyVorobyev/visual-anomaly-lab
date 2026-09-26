/**
 * The sample viewer's Explore rail section: the mode, the encoder, the few numbers each mode
 * has, and what the last answer cost.
 *
 * Every state says something in words. The first request on an image names the encoder it is
 * loading, because that one is slow and every later click is not; a running job explains why
 * nothing can be asked; a missing `dl` extra says so instead of offering controls that would
 * all fail.
 */

import { Download, Eraser, PenLine, Search } from "lucide-react";

import {
  Button,
  Callout,
  ErrorBox,
  InfoHint,
  Input,
  ProgressBar,
  SegmentedControl,
  Select,
  SkeletonRows,
  Slider,
  Switch,
  ToggleChip,
  cn,
  focusRing,
} from "@vitavision/lab-ui";

import { ApiError } from "../../api/client";
import { clusterColour } from "../../api/explore";
import { RailSection } from "../../components/viewer/RailSection";
import { isTerminal } from "../../hooks/useJob";
import type { ExploreMode, ExploreSession } from "./useExploreSession";
import type { ExploreTextSession } from "./useExploreTextSession";
import { defined } from "../../api/defined";

const MODES: { value: ExploreMode; label: string }[] = [
  { value: "similar", label: "Similar" },
  { value: "clusters", label: "Clusters" },
  { value: "pca", label: "PCA" },
  { value: "sam", label: "SAM" },
  { value: "text", label: "Text" },
];

const HINT: Record<ExploreMode, string> = {
  similar: "Click a patch to find what the encoder thinks is like it. Shift-click marks one it should not match.",
  clusters:
    "The image's own patches, grouped by k-means; edges fall where two groups are equally likely. Click a region to pick its cluster.",
  pca: "The three directions the features vary most, as false colour. The colours mean nothing between images.",
  sam: "Click an object for MobileSAM's masks. Shift-click marks background.",
  text: "Name a thing — “candle”, “the cap” — for SAM 3's masks of every instance of it. It finds objects and parts well and defects by their name rarely: ask for the thing, not the flaw.",
};

/** The rail's subtitle: what is answering in this mode. */
function answeredBy(session: ExploreSession): string {
  if (!session.on) return "";
  if (session.mode === "text") return "SAM 3";
  if (session.mode === "sam") return "";
  return session.backbone?.title ?? "";
}

export function ExploreSection({ session }: { session: ExploreSession }) {
  const { capability, on, mode } = session;

  return (
    <RailSection title="Explore" hint={answeredBy(session)}>
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
                className="flex-wrap"
                value={mode}
                options={MODES}
                onValueChange={(value) => session.setMode(value as ExploreMode)}
              />
              <p className="text-xs leading-5 text-fg-subtle">{HINT[mode]}</p>
              {mode === "sam" ? (
                <SamControls session={session} />
              ) : mode === "text" ? (
                <TextControls text={session.text} />
              ) : (
                <EncoderControls session={session} />
              )}
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
            ...defined({ note: entry.gated ? "gated" : undefined }),
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
          {answer && answer.value_low != null && answer.value_high != null && (
            // The heatmap is stretched over this image's own range — its median to its top
            // percent — so it is legible on any image; the mask cut above is absolute cosine.
            <p className="font-mono text-[11px] leading-4 text-fg-muted tabular-nums">
              colour scaled to this image · median {answer.value_low.toFixed(2)} → top 1%{" "}
              {answer.value_high.toFixed(2)}
            </p>
          )}
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

function TextControls({ text }: { text: ExploreTextSession }) {
  const { capability, request, answer } = text;
  if (!capability) return null;
  if (!capability.available) {
    return capability.installable ? (
      <TextInstall text={text} />
    ) : (
      <Callout tone="info" title="SAM 3 is unavailable">
        {capability.reason ?? "SAM 3 cannot be loaded."}
      </Callout>
    );
  }
  return (
    <div className="flex flex-col gap-3">
      <form
        className="flex items-center gap-1.5"
        onSubmit={(event) => {
          event.preventDefault();
          text.ask();
        }}
      >
        <Input
          aria-label="Phrase"
          className="min-w-0 flex-1"
          placeholder="candle, the cap…"
          maxLength={capability.max_phrase_length}
          value={text.phrase}
          onChange={(event) => text.setPhrase(event.target.value)}
        />
        <Button type="submit" icon={<Search />} disabled={!text.canAsk} loading={request.isPending}>
          Find
        </Button>
      </form>

      {text.encoding ? (
        // The first phrase on an image runs SAM 3's image encoder; the first after SAM 3 was
        // idle also loads 3.4 GB of weights. Every later phrase on this image reuses both.
        <p className="text-xs leading-5 text-fg-muted" role="status">
          Encoding this image with SAM 3… If SAM 3 was not loaded, this first phrase also loads its
          weights and can take a minute.
        </p>
      ) : request.isPending ? (
        <p className="text-xs text-fg-muted" role="status">
          Asking SAM 3…
        </p>
      ) : null}
      <RequestError error={request.error} onRetry={text.retry} />

      {answer && answer.instances.length === 0 && (
        <p className="text-xs leading-5 text-fg-muted">
          Nothing scored ≥ {answer.threshold.toFixed(2)} for “{answer.phrase}”.
        </p>
      )}
      {answer && answer.instances.length > 0 && (
        <div className="flex flex-col gap-2">
          <p className="text-xs text-fg-muted">
            {answer.instances.length} instance{answer.instances.length === 1 ? "" : "s"} of “
            {answer.phrase}”{answer.dropped > 0 ? ` · ${answer.dropped} weaker not shown` : ""} · pick
            one to send it
          </p>
          <div className="flex flex-wrap gap-1.5" role="group" aria-label="Instances">
            {answer.instances.map((item) => (
              <ToggleChip
                key={item.index}
                checked={text.instance === item.index}
                onCheckedChange={(checked) => text.setInstance(checked ? item.index : null)}
                swatch={clusterColour(item.index)}
              >
                <span className="font-mono tabular-nums">
                  #{item.index} · {item.score.toFixed(2)}
                </span>
              </ToggleChip>
            ))}
          </div>
          <p className="text-[10px] leading-4 text-fg-subtle">
            Scores are SAM 3's own, not calibrated probabilities.
          </p>
        </div>
      )}
      {answer && !request.isPending && (
        <p className="font-mono text-[11px] leading-4 text-fg-subtle tabular-nums">
          {answer.device.toUpperCase()} ·{" "}
          {answer.cached ? "image cached" : `encoded ${Math.round(answer.encode_ms)} ms`} ·{" "}
          {Math.round(answer.elapsed_ms)} ms
        </p>
      )}
    </div>
  );
}

/** The one-time, licence-gated checkpoint download, followed as an ordinary job. */
function TextInstall({ text }: { text: ExploreTextSession }) {
  const { asset, capability, assetJob, followedAssetJobId, installAsset, cancelAssetJob } = text;
  if (followedAssetJobId && !isTerminal(assetJob.job?.status)) {
    return (
      <div className="flex flex-col gap-2">
        <div className="flex items-center justify-between gap-2 text-xs text-fg-muted">
          <span role="status">{assetJob.job?.message ?? "Downloading SAM 3…"}</span>
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
    );
  }
  const link = cn("w-fit text-signal underline-offset-2 hover:underline", focusRing);
  return (
    <div className="flex flex-col gap-2 text-xs leading-5 text-fg-muted">
      <p>{capability?.reason}</p>
      <p>
        It is verified file by file and kept in app-managed storage, shared by every dataset.
      </p>
      <div className="flex flex-wrap gap-x-3">
        {asset && (
          <a href={asset.license_url} target="_blank" rel="noreferrer" className={link}>
            {asset.license_name}
          </a>
        )}
        {capability?.access_url && (
          <a href={capability.access_url} target="_blank" rel="noreferrer" className={link}>
            Request access
          </a>
        )}
      </div>
      <Button icon={<Download />} loading={installAsset.isPending} disabled={!asset} onClick={text.install}>
        Accept licence & download
      </Button>
      {installAsset.error && <ErrorBox>{installAsset.error.message}</ErrorBox>}
      {assetJob.job?.error && <ErrorBox>{assetJob.job.error}</ErrorBox>}
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
          : session.mode === "text"
            ? "Find a phrase and pick an instance to send it."
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
