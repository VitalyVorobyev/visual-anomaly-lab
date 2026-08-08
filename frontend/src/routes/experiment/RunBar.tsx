/**
 * Start a run, stop a run, and see that one is running — from any tab.
 *
 * These controls used to live in a `Runs` panel on the Overview tab, which put them in the
 * one place they were least wanted. Starting a scoring run is something you decide while
 * looking at the Samples gallery; knowing a training job is still alive matters while
 * reading the Benchmark charts. A control that is only reachable from the front page of a
 * screen is a control you navigate away from your work to press.
 *
 * So it sits in the page chrome, above the tabs, beside the status badge — and the Overview
 * tab is free to be about what the experiment *found*. The live chart and the console are
 * the Training tab; this bar carries only enough to know whether to go and look.
 */

import { Button, ErrorBox, ProgressBar, StatusDot } from "../../components/ui";
import type { JobSummary } from "../../api/client";
import { isTerminal } from "../../hooks/useJob";
import { useCancelJob, useStartRun } from "../../hooks/useExperiments";
import { jobTone } from "./OverviewTab";

export function RunBar({
  experimentId,
  jobs,
  hasTrained,
  onFollow,
  onViewLog,
}: {
  experimentId: number;
  jobs: JobSummary[];
  hasTrained: boolean;
  onFollow: (jobId: number) => void;
  /** Take the reader to the tab where the chart and the console are. */
  onViewLog: () => void;
}) {
  const start = useStartRun(experimentId);
  const cancel = useCancelJob(experimentId);

  // `jobs` arrives newest first, so the first unfinished one is the live one. The queue
  // runs a single job at a time (ADR-0009), so there is never more than one.
  const live = jobs.find((job) => !isTerminal(job.status));
  const busy = live !== undefined || start.isPending;

  const run = (kind: "train" | "infer") =>
    start.mutate({ kind }, { onSuccess: (job) => onFollow(job.id) });

  return (
    <div className="flex flex-col gap-2 rounded-lg border border-line bg-surface px-4 py-3">
      <div className="flex flex-wrap items-center gap-2">
        <Button
          variant="primary"
          disabled={busy}
          loading={start.isPending}
          onClick={() => run("train")}
        >
          Train
        </Button>
        <Button
          disabled={busy || !hasTrained}
          title={hasTrained ? undefined : "Nothing has been trained yet."}
          onClick={() => run("infer")}
        >
          Score &amp; evaluate
        </Button>

        {live !== undefined && (
          <>
            <StatusDot tone={jobTone(live.status)}>
              <span className="font-mono">
                #{live.id} {live.kind}
              </span>
            </StatusDot>
            <button
              type="button"
              className="text-xs text-fg-muted hover:text-fg hover:underline"
              onClick={onViewLog}
            >
              view log
            </button>
            <Button
              className="ml-auto"
              loading={cancel.isPending}
              onClick={() => cancel.mutate(live.id)}
            >
              Cancel
            </Button>
          </>
        )}
      </div>

      {/* Only while something is live. A progress bar frozen at 100% after a run finished
          reads as a run that is still going. */}
      {live !== undefined && (
        <ProgressBar fraction={live.progress ?? 0} label={live.message ?? live.status} />
      )}

      {start.error && <ErrorBox>{start.error.message}</ErrorBox>}
      {cancel.error && <ErrorBox>{cancel.error.message}</ErrorBox>}
    </div>
  );
}
