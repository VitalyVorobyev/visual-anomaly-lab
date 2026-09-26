/**
 * `/datasets/:id/run` — the guided run, the front door to training.
 *
 * Five steps, each one decision made in front of images, each with a default already chosen:
 * the goal (from the dataset's truth), the look (the full frame, or a saved region profile,
 * previewed at the run's size), the split (the task's first preset, drawn only when the run
 * starts), the method (the registry's recommendation, its `x-primary` options in front) and
 * the run itself (a summary, a name, one press). Pressing Next through all five starts a
 * sensible run; "Review & run" jumps to the last step from any other. The Prepare and Splits
 * tabs and the full create form stay as the expert surface.
 *
 * The step is in the URL (`?step=`), so the browser's Back walks the steps; the choices are in
 * `sessionStorage` per dataset (`api/guidedRun.ts`), so a detour to Prepare — which returns
 * here with the profile it saved (`?profile=`) — comes back to the same run. The layout is
 * `CanvasLayout`: the band and the footer stay put, and the step between them is the one
 * region that scrolls.
 */

import { Button, ButtonLink, cn, ErrorBox, focusRing } from "@vitavision/lab-ui";
import { ArrowLeft, ArrowRight } from "lucide-react";
import { useEffect } from "react";
import { Link, useNavigate, useParams, useSearchParams } from "react-router";

import {
  clearGuidedRun,
  furthest,
  guidedRunKey,
  readStep,
  STEP_TITLE,
  STEPS,
  stepIndex,
  type Step,
} from "../api/guidedRun";
import { useHotkeys } from "../hooks/useHotkeys";
import { useLaunchRun } from "../hooks/useLaunchRun";
import { GoalStep } from "./run/GoalStep";
import { LookStep } from "./run/LookStep";
import { MethodStep } from "./run/MethodStep";
import { RunStep } from "./run/RunStep";
import { SplitStep } from "./run/SplitStep";
import { choiceKey, TASK_TITLE, useGuidedRun, type GuidedRun } from "./run/useGuidedRun";

export function GuidedRunRoute() {
  const datasetId = Number(useParams()["datasetId"]);
  const [params, setParams] = useSearchParams();
  const step = readStep(params.get("step"));
  const run = useGuidedRun(datasetId);
  const launch = useLaunchRun();
  const navigate = useNavigate();
  const { update } = run;

  // Back from Prepare with the profile it saved: that is the look, and the URL forgets it.
  const returnedProfile = params.get("profile");
  useEffect(() => {
    if (returnedProfile === null) return;
    const id = Number(returnedProfile);
    if (Number.isFinite(id)) update({ profileId: id });
    const next = new URLSearchParams(params);
    next.delete("profile");
    setParams(next, { replace: true });
  }, [returnedProfile]);

  // How far the reader has been: the rail offers every step up to it.
  const reached = run.state.reached;
  useEffect(() => {
    if (furthest(step, reached) !== reached) update({ reached: step });
  }, [step, reached, update]);

  const go = (target: Step) => setParams({ step: target });
  const index = stepIndex(step);
  const back = index > 0 ? STEPS[index - 1] : undefined;
  const next = index < STEPS.length - 1 ? STEPS[index + 1] : undefined;

  useHotkeys((event) => {
    if (event.key !== "Enter") return;
    // A focused button or link answers Enter itself; a second answer here would act twice.
    if (event.target instanceof Element && event.target.closest("button, a, [role='button']")) {
      return;
    }
    if (event.shiftKey && back) {
      event.preventDefault();
      go(back);
    } else if (!event.shiftKey && next) {
      event.preventDefault();
      go(next);
    }
  });

  const start = () => {
    const option = run.split;
    if (!option || !run.method || run.task === undefined || run.missing.length > 0) return;
    const key = choiceKey(option);
    const reused = run.state.created?.key === key ? run.state.created.splitId : undefined;
    const split =
      reused !== undefined
        ? { id: reused }
        : option.choice.kind === "existing"
          ? { id: option.choice.id }
          : { params: option.params! };
    launch.mutate(
      {
        split,
        run: true,
        onSplit: (splitId) => update({ created: { key, splitId } }),
        experiment: {
          name: run.name,
          dataset_id: datasetId,
          // Omitted, the dataset's "Full frame".
          ...(run.profile && !run.isFullFrame ? { region_profile_id: run.profile.id } : {}),
          model_type: run.method.key,
          task: run.task,
          target_label: run.targeted ? (run.targetClass ?? null) : null,
          config: run.methodOptions,
          preprocessing: {},
          evaluation: {},
          channels: [],
        },
      },
      {
        onSuccess: (experiment) => {
          clearGuidedRun(guidedRunKey(datasetId));
          void navigate(`/experiments/${experiment.id}`);
        },
      },
    );
  };

  return (
    <div className="flex min-h-0 flex-1 flex-col gap-3">
      <header className="flex shrink-0 flex-col gap-3">
        <div className="flex flex-wrap items-end justify-between gap-x-4 gap-y-1">
          <div className="flex min-w-0 flex-col gap-0.5">
            <Link
              to={`/datasets/${datasetId}`}
              className={cn(
                "w-fit truncate rounded-sm text-xs text-fg-muted transition-colors hover:text-signal",
                focusRing,
              )}
            >
              ← {run.dataset.data?.name ?? "Dataset"}
            </Link>
            <h1 className="text-xl font-semibold tracking-tight text-fg">Start a run</h1>
          </div>
          <ButtonLink to={`/datasets/${datasetId}/experiments/new`} size="sm" variant="ghost">
            Open the full form
          </ButtonLink>
        </div>
        <StepRail run={run} step={step} reached={reached} onGo={go} />
      </header>

      <div
        data-scroll="step"
        className="min-h-0 flex-1 overflow-y-auto overscroll-contain [scrollbar-gutter:stable]"
      >
        <div className="flex w-full max-w-6xl flex-col gap-4 pb-4">
          {run.readError && <ErrorBox>{run.readError.message}</ErrorBox>}
          {step === "goal" && <GoalStep run={run} datasetId={datasetId} />}
          {step === "look" && <LookStep run={run} datasetId={datasetId} />}
          {step === "split" && <SplitStep run={run} datasetId={datasetId} />}
          {step === "method" && <MethodStep run={run} />}
          {step === "run" && (
            <RunStep
              run={run}
              onEdit={go}
              onStart={start}
              starting={launch.isPending}
              error={launch.error}
            />
          )}
        </div>
      </div>

      <footer className="flex shrink-0 items-center justify-between gap-3 border-t border-line pt-3">
        <div className="flex min-w-0 items-center gap-3">
          {back && (
            <Button variant="secondary" icon={<ArrowLeft />} onClick={() => go(back)}>
              Back
            </Button>
          )}
          <span className="hidden text-xs text-fg-subtle md:inline">
            Enter continues · Shift+Enter goes back
          </span>
        </div>
        <div className="flex items-center gap-2">
          {step !== "run" && step !== "method" && (
            <Button variant="ghost" onClick={() => go("run")}>
              Review &amp; run
            </Button>
          )}
          {next && (
            <Button variant="primary" icon={<ArrowRight />} onClick={() => go(next)}>
              Next: {STEP_TITLE[next]}
            </Button>
          )}
        </div>
      </footer>
    </div>
  );
}

/**
 * Where the run stands: every step with what it is set to, the current one marked, and
 * every step up to the furthest reached a button back to it. A step not yet reached is
 * text, not a disabled control — there is nothing to explain about it but its order.
 */
function StepRail({
  run,
  step,
  reached,
  onGo,
}: {
  run: GuidedRun;
  step: Step;
  reached: Step;
  onGo: (step: Step) => void;
}) {
  const summary: Record<Step, string | undefined> = {
    goal: run.task ? TASK_TITLE[run.task] : undefined,
    look: run.profile?.name,
    split: run.split?.label,
    method: run.method?.title,
    run: run.missing.length === 0 ? "ready" : undefined,
  };
  return (
    <nav aria-label="Run steps">
      <ol className="grid grid-cols-5 gap-1.5">
        {STEPS.map((entry, position) => {
          const current = entry === step;
          const open = position <= stepIndex(reached);
          const body = (
            <>
              <span className="flex items-baseline gap-1.5">
                <span className="font-mono text-[11px] tabular-nums">{position + 1}</span>
                <span className="text-sm font-medium">{STEP_TITLE[entry]}</span>
              </span>
              <span className="block truncate text-xs text-fg-muted">
                {summary[entry] ?? "—"}
              </span>
            </>
          );
          const frame = cn(
            "flex min-w-0 flex-col gap-0.5 rounded-control border-t-2 px-2 pt-1.5 pb-1 text-left",
            current
              ? "border-signal text-fg"
              : open
                ? "border-line-strong text-fg-muted"
                : "border-line text-fg-subtle",
          );
          return (
            <li key={entry} className="min-w-0">
              {open && !current ? (
                <button
                  type="button"
                  onClick={() => onGo(entry)}
                  className={cn(frame, "w-full transition-colors hover:border-signal hover:text-fg", focusRing)}
                >
                  {body}
                </button>
              ) : (
                <div className={frame} aria-current={current ? "step" : undefined}>
                  {body}
                </div>
              )}
            </li>
          );
        })}
      </ol>
    </nav>
  );
}
