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

import { useState } from "react";

import {
  Button,
  ConfirmDialog,
  ErrorBox,
  NumberInput,
  ProgressBar,
  StatusDot,
} from "@vitavision/lab-ui";
import type { ExperimentDetail, JobDetail, JobSummary } from "../../api/client";
import { isTerminal } from "../../hooks/useJob";
import { useCancelJob, useStartExport, useStartRun } from "../../hooks/useExperiments";
import { jobTone } from "./OverviewTab";

export function RunBar({
  experimentId,
  detail,
  jobs,
  liveJob,
  hasTrained,
  onFollow,
  onViewLog,
}: {
  experimentId: number;
  detail: ExperimentDetail;
  jobs: JobSummary[];
  /**
   * The followed job's own row, which is the only copy that moves during a run.
   *
   * `jobs` comes from `ExperimentDetail`, and a `progress` frame deliberately does not
   * invalidate that query — at four frames a second it would be a poll wearing an event's
   * clothes. So the bar drew a number that was refreshed on window focus and on terminal
   * frames, and stood still for everything in between: the top of the screen read
   * `step 421/8000` while the job card below it, reading `["jobs", id]`, read `step 461`.
   * Whoever simplifies this back to `jobs` alone will reintroduce exactly that.
   *
   * Optional because it genuinely is: there is no row until the first snapshot lands, and
   * the summary carries the same fields, one refresh behind.
   */
  liveJob?: JobDetail | undefined;
  hasTrained: boolean;
  onFollow: (jobId: number) => void;
  /** Take the reader to the tab where the chart and the console are. */
  onViewLog: () => void;
}) {
  const start = useStartRun(experimentId);
  const startExport = useStartExport(experimentId);
  const cancel = useCancelJob(experimentId);
  const [confirmRetrain, setConfirmRetrain] = useState(false);

  const state = detail.training_state ?? null;
  const trained = state !== null;
  /*
   * Defaults to the configured per-run budget, so "another 4000" is one click. `max_steps`
   * is the frozen config's field; a method without one starts the box empty rather than
   * inventing a number.
   */
  const budget = Number(detail.config?.["max_steps"]);
  const [steps, setSteps] = useState<number>(Number.isFinite(budget) ? budget : 0);

  const canContinue = detail.supports_resume && trained;
  const continueReason = !detail.supports_resume
    ? "This method has no notion of a training step, so there is nothing to continue."
    : "Nothing has been trained yet.";

  // `jobs` arrives newest first, so the first unfinished one is the live one. The queue
  // runs a single job at a time (ADR-0009), so there is never more than one.
  const live = jobs.find((job) => !isTerminal(job.status));
  // Identity from the experiment payload, progress from the live row — and only when the
  // two are the same job, so a stale subscription cannot label another run's numbers.
  const reading = liveJob !== undefined && liveJob.id === live?.id ? liveJob : live;
  const busy = live !== undefined || start.isPending || startExport.isPending;
  const canExportOnnx = detail.portable_formats.includes("onnx");

  const run = (kind: "train" | "infer", additionalSteps?: number) =>
    start.mutate({ kind, additionalSteps }, { onSuccess: (job) => onFollow(job.id) });

  return (
    <div className="flex flex-col gap-2 rounded-lg border border-line bg-surface px-4 py-3">
      <div className="flex flex-wrap items-center gap-2">
        {/* On a trained experiment this is no longer the obvious action — continuing is.
            It also stops looking like a button that repeats the same run for no reason,
            which is exactly how it read before. */}
        <Button
          variant={trained ? "secondary" : "primary"}
          disabled={busy}
          loading={start.isPending && !trained}
          onClick={() => (trained ? setConfirmRetrain(true) : run("train"))}
        >
          {trained ? "Retrain from scratch" : "Train"}
        </Button>

        {detail.supports_resume && (
          <span className="flex items-center gap-1.5">
            <Button
              variant={trained ? "primary" : "secondary"}
              disabled={busy || !canContinue || steps < 1}
              title={canContinue ? undefined : continueReason}
              onClick={() => run("train", steps)}
            >
              Continue
            </Button>
            <NumberInput
              className="w-24"
              aria-label="Additional steps"
              min={1}
              max={200000}
              value={steps || ""}
              disabled={!canContinue}
              onChange={(event) => setSteps(Number(event.target.value))}
            />
            <span className="text-xs text-fg-muted">more steps</span>
          </span>
        )}
        <Button
          disabled={busy || !hasTrained}
          title={hasTrained ? undefined : "Nothing has been trained yet."}
          onClick={() => run("infer")}
        >
          Score &amp; evaluate
        </Button>
        <Button
          variant="secondary"
          disabled={busy || !hasTrained || !canExportOnnx}
          loading={startExport.isPending}
          title={
            !canExportOnnx
              ? "This method has no numerically verified ONNX exporter yet."
              : !hasTrained
                ? "Nothing has been trained yet."
                : "Create a checksummed ONNX bundle with parity fixtures."
          }
          onClick={() =>
            startExport.mutate(undefined, { onSuccess: (job) => onFollow(job.id) })
          }
        >
          {canExportOnnx ? "Export ONNX" : "Export unavailable"}
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
      {live !== undefined && reading !== undefined && (
        <ProgressBar fraction={reading.progress ?? 0} label={reading.message ?? reading.status} />
      )}

      {/* The consequence is surprising enough that it is printed before the run rather
          than discovered afterwards in the learning-rate chart (handbook jobs.md). */}
      {canContinue && state !== null && steps > 0 && (
        <p className="text-xs text-fg-muted">
          {state.completed_steps} steps completed over {state.runs} run
          {state.runs === 1 ? "" : "s"}. Continuing to {state.completed_steps + steps} moves the
          learning-rate drop to step {Math.floor(0.95 * (state.completed_steps + steps))}, so the
          rate returns to its base value at the resume point. The optimizer, the schedule, the
          step counter and the training image order resume exactly; the penalty-set order
          restarts.
        </p>
      )}

      {start.error && <ErrorBox>{start.error.message}</ErrorBox>}
      {startExport.error && <ErrorBox>{startExport.error.message}</ErrorBox>}
      {cancel.error && <ErrorBox>{cancel.error.message}</ErrorBox>}

      <ConfirmDialog
        open={confirmRetrain}
        onOpenChange={setConfirmRetrain}
        title="Retrain from scratch?"
        description={
          <>
            This discards the current checkpoint
            {state ? ` and the ${state.completed_steps} steps it has completed` : ""}, and starts
            again from the pretrained teacher. The stored scores and anomaly maps stay until you
            score again, so the results screens would describe a model that no longer exists.
          </>
        }
        confirmLabel="Retrain"
        destructive
        loading={start.isPending}
        onConfirm={() => {
          setConfirmRetrain(false);
          run("train");
        }}
      />
    </div>
  );
}
