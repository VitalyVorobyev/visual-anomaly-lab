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
import { jobStatusTone } from "../../api/statusTone";

const NOT_TRAINED = "Nothing has been trained yet.";

export function RunBar({
  experimentId,
  detail,
  jobs,
  liveJob,
  onFollow,
  onViewLog,
  onViewFiles,
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
  onFollow: (jobId: number) => void;
  /** Take the reader to the tab where the chart and the console are. */
  onViewLog: () => void;
  /** Take the reader to the tab that lists the run's files, where a bundle lands. */
  onViewFiles?: () => void;
}) {
  const start = useStartRun(experimentId);
  const startExport = useStartExport(experimentId);
  const cancel = useCancelJob(experimentId);
  const [confirmRetrain, setConfirmRetrain] = useState(false);

  /*
   * Trained means the experiment's own status says so — the only thing the scoring and
   * export endpoints accept. It used to be read off `training_state`, a sidecar only a
   * *resumable* method writes, so a trained PatchCore or pixel_reference run still offered
   * a primary "Train" that retrained without asking; and a failed train job, counted as
   * "has trained", enabled scoring a model that was never saved.
   */
  const trained = detail.status === "trained";
  const state = detail.training_state ?? null;
  /*
   * Defaults to the configured per-run budget, so "another 4000" is one click. `max_steps`
   * is the frozen config's field; a method without one starts the box empty rather than
   * inventing a number.
   */
  const budget = Number(detail.config?.["max_steps"]);
  const [steps, setSteps] = useState<number>(Number.isFinite(budget) ? budget : 0);

  const canContinue = detail.supports_resume && trained && state !== null;
  const continueReason = !detail.supports_resume
    ? "This method has no notion of a training step, so there is nothing to continue."
    : NOT_TRAINED;

  // `jobs` arrives newest first, so the first unfinished one is the live one. The queue
  // runs a single job at a time (ADR-0009), so there is never more than one.
  const live = jobs.find((job) => !isTerminal(job.status));
  // Identity from the experiment payload, progress from the live row — and only when the
  // two are the same job, so a stale subscription cannot label another run's numbers.
  const reading = liveJob !== undefined && liveJob.id === live?.id ? liveJob : live;
  const busy = live !== undefined || start.isPending || startExport.isPending;
  const canExportOnnx = detail.portable_formats.includes("onnx");
  // The newest job, when it is a finished export: the bundle exists, and the only place
  // that says where is another tab — so say so here, where the button was pressed.
  const exported = live === undefined && jobs[0]?.kind === "export" && jobs[0].status === "succeeded";

  const run = (kind: "train" | "infer", additionalSteps?: number, thenScore = false) =>
    start.mutate({ kind, additionalSteps, thenScore }, { onSuccess: (job) => onFollow(job.id) });

  return (
    <div className="flex flex-col gap-2 rounded-lg border border-line bg-surface px-4 py-3">
      <div className="flex flex-wrap items-center gap-2">
        {/* On a trained experiment this is no longer the obvious action — continuing is.
            It also stops looking like a button that repeats the same run for no reason,
            which is exactly how it read before. */}
        {/* A first run is one press: train, and score once it succeeds. A new experiment
            used to land on a draft that needed Train, a wait, then Score & evaluate. */}
        <Button
          variant={trained ? "secondary" : "primary"}
          disabled={busy}
          loading={start.isPending && !trained}
          onClick={() => (trained ? setConfirmRetrain(true) : run("train", undefined, true))}
        >
          {trained ? "Retrain from scratch" : "Train & score"}
        </Button>
        {!trained && (
          <Button variant="ghost" disabled={busy} onClick={() => run("train")}>
            Train only
          </Button>
        )}

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
          disabled={busy || !trained}
          title={trained ? undefined : NOT_TRAINED}
          onClick={() => run("infer")}
        >
          Score &amp; evaluate
        </Button>
        {canExportOnnx ? (
          <Button
            variant="secondary"
            disabled={busy || !trained}
            loading={startExport.isPending}
            title={
              !trained ? NOT_TRAINED : "Create a checksummed ONNX bundle with parity fixtures."
            }
            onClick={() =>
              startExport.mutate(undefined, { onSuccess: (job) => onFollow(job.id) })
            }
          >
            Export ONNX
          </Button>
        ) : (
          // Said, not greyed out: a dead button whose reason lives in a tooltip is a question
          // the screen leaves the reader to ask.
          <span className="text-xs text-fg-muted">No verified ONNX export for this method</span>
        )}

        {live !== undefined && (
          <>
            <StatusDot tone={jobStatusTone(live.status)}>
              <span className="font-mono">
                #{live.id} {live.kind}
              </span>
            </StatusDot>
            <button
              type="button"
              className="text-xs text-fg-muted hover:text-fg hover:underline focus-visible:outline-2 focus-visible:outline-signal"
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

      {/* What a continuation will do, before it is started. Deliberately in terms every
          resumable method shares — checkpoint, optimizer, step counter. How a particular
          method's schedule treats the added steps is that method's business, and naming
          one method's schedule here is how this bar came to describe EfficientAD for
          every run that could resume (ADR-0007; handbook jobs.md). */}
      {canContinue && state !== null && steps > 0 && (
        <p className="text-xs text-fg-muted">
          {state.completed_steps} steps completed over {state.runs} run
          {state.runs === 1 ? "" : "s"}. Continuing trains {steps} more, to{" "}
          {state.completed_steps + steps} in total, from the stored checkpoint and optimizer
          state rather than from the beginning.
        </p>
      )}

      {exported && (
        <p className="text-xs text-fg-muted">
          ONNX bundle written.{" "}
          {onViewFiles && (
            <button
              type="button"
              className="text-signal underline-offset-2 hover:underline focus-visible:outline-2 focus-visible:outline-signal"
              onClick={onViewFiles}
            >
              Open Jobs &amp; files
            </button>
          )}
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
            This discards the fitted model
            {state ? ` and the ${state.completed_steps} steps it has completed` : ""}, and fits
            it again from the beginning. The stored scores and anomaly maps stay until you score
            again, so the results screens would describe a model that no longer exists.
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
