/**
 * Contour assist in the inspector: the one-time checkpoint download, then the prompt, the
 * request and the ranked suggestions.
 */

import { WandSparkles } from "lucide-react";

import {
  Badge,
  Button,
  ErrorBox,
  ProgressBar,
  SegmentedControl,
  SkeletonRows,
  cn,
  focusRing,
} from "@vitavision/lab-ui";
import { isTerminal } from "../../hooks/useJob";
import type { AssistMode, SegmentAssistSession } from "./useSegmentAssistSession";

export function AssistSection({ assist }: { assist: SegmentAssistSession }) {
  const {
    asset,
    capability,
    modelAssets,
    followedAssetJobId,
    assetJob,
    cancelAssetJob,
    installAsset,
    mode,
    points,
    box,
    candidates,
    candidateIndex,
  } = assist;

  return (
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
            Download the verified 38.8 MiB TinyViT checkpoint once. It stays in app-managed
            storage and is shared by every dataset.
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
            onClick={assist.install}
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
            value={mode}
            options={[
              { value: "point", label: "Points" },
              { value: "box", label: "Box" },
            ]}
            onValueChange={(value) => assist.setMode(value as AssistMode)}
          />
          <p className="text-xs leading-5 text-fg-subtle">
            {mode === "point"
              ? "Click the defect. Shift-click marks background. Add points to refine."
              : "Drag a tight box around the defect. Right-drag still pans."}
          </p>
          <div className="flex items-center justify-between rounded-control bg-raised px-2 py-1.5 text-xs">
            <span>
              {mode === "point"
                ? `${points.length} point${points.length === 1 ? "" : "s"}`
                : box
                  ? `${Math.round(box.x1 - box.x0)} × ${Math.round(box.y1 - box.y0)} px`
                  : "No box"}
            </span>
            <Button size="sm" disabled={!points.length && !box} onClick={assist.clear}>
              Clear
            </Button>
          </div>
          <Button
            variant="primary"
            icon={<WandSparkles />}
            loading={assist.assist.isPending}
            disabled={!assist.canRequest}
            onClick={() => void assist.request()}
          >
            Suggest contours
          </Button>
          {assist.assist.error && <ErrorBox>{assist.assist.error.message}</ErrorBox>}
          {candidates.length > 0 && (
            <div className="flex flex-col gap-2">
              <div className="grid grid-cols-3 gap-1" aria-label="Suggested masks">
                {candidates.map((item, index) => (
                  <button
                    key={item.shape.id}
                    type="button"
                    onClick={() => assist.setCandidateIndex(index)}
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
                <Button onClick={() => void assist.acceptCandidate(false)}>Accept mask</Button>
                <Button onClick={() => void assist.acceptCandidate(true)}>Editable contour</Button>
              </div>
              <p className="text-[10px] leading-4 text-fg-subtle">
                Quality is MobileSAM's own ranking score, not a calibrated probability.
              </p>
            </div>
          )}
        </div>
      )}
    </section>
  );
}
