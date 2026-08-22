/**
 * A running job: progress, a live console, and a cancel button.
 *
 * Shared because every long operation in the application looks like this — the import
 * scan and the thumbnail pre-warm today, training and inference from M3 — and the whole
 * point of putting import on the job system was to avoid a second progress mechanism.
 */

import { useMutation, useQueryClient } from "@tanstack/react-query";

import { api } from "../api/client";
import type { JobDetail, JobStatus } from "../api/client";
import { queryKeys } from "../api/queryKeys";
import { isTerminal } from "../hooks/useJob";
import { Badge, Button, cn, Disclosure, ErrorBox, ProgressBar, SkeletonRows, type Tone } from "@vitavision/lab-ui";

const STATUS_TONE: Record<JobStatus, Tone> = {
  queued: "neutral",
  running: "info",
  succeeded: "normal",
  failed: "defect",
  cancelled: "unlabeled",
};

/** How much of the log is worth seeing without asking for the rest. */
const TAIL_LINES = 12;

export function JobProgress({
  jobId,
  job,
  lines,
  error,
}: {
  jobId: number;
  job: JobDetail | undefined;
  lines: string[];
  error: Error | null;
}) {
  const queryClient = useQueryClient();

  const cancel = useMutation({
    mutationFn: async () => {
      await api.POST("/api/jobs/{job_id}/cancel", { params: { path: { job_id: jobId } } });
    },
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: queryKeys.job(jobId) });
    },
  });

  if (error) return <ErrorBox>{error.message}</ErrorBox>;
  if (!job) return <SkeletonRows rows={2} />;

  const finished = isTerminal(job.status);

  return (
    <div className="flex flex-col gap-3">
      <div className="flex items-center gap-2">
        <Badge tone={STATUS_TONE[job.status]}>{job.status}</Badge>
        <span className="min-w-0 truncate text-sm text-fg-muted">{job.message ?? ""}</span>
        {!finished && (
          <Button
            className="ml-auto"
            size="sm"
            variant="danger"
            loading={cancel.isPending}
            onClick={() => cancel.mutate()}
          >
            Cancel
          </Button>
        )}
      </div>

      <ProgressBar fraction={job.progress} />

      {job.error && <ErrorBox>{job.error}</ErrorBox>}

      {/*
        The tail is clipped, not scrolled. This sits inside a page that already scrolls, and
        a scroller inside a scroller on the same axis is the thing that puts two bars on
        screen at once. Clipping to the last few lines *is* following the tail — it needs no
        ref and no scroll effect — and the disclosure puts every line back in the page's own
        flow, where the one scroller that exists can reach them.
      */}
      {lines.length > 0 && (
        <>
          <LogPre lines={lines.slice(-TAIL_LINES)} className="overflow-hidden" />
          {lines.length > TAIL_LINES && (
            <Disclosure summary="Full log" count={lines.length}>
              <LogPre lines={lines} />
            </Disclosure>
          )}
        </>
      )}
    </div>
  );
}

/**
 * Held on a near-black field regardless of theme: this is a terminal, and a log tail that
 * changes colour with the app reads as chrome rather than as output.
 */
function LogPre({ lines, className }: { lines: string[]; className?: string }) {
  return (
    <pre
      className={cn(
        "rounded-control border border-line bg-[#08090a] p-3 font-mono text-xs whitespace-pre-wrap text-[#c9d1d9]",
        className,
      )}
    >
      {lines.join("\n")}
    </pre>
  );
}
