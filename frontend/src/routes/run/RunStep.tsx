/**
 * Step 5, the run: every choice on one screen, a name, and one press.
 *
 * Start creates the split if it is still a preset, creates the experiment and queues Train &
 * score (`useLaunchRun`), then lands on the run page, which follows the job. Each summary
 * line links back to the step that decided it.
 */

import { Button, ErrorBox, Field, Input, Skeleton, cn, focusRing } from "@vitavision/lab-ui";
import { Play } from "lucide-react";
import type { ReactNode } from "react";

import type { Step } from "../../api/guidedRun";
import { StepIntro } from "./ChoiceCard";
import { TASK_TITLE, type GuidedRun } from "./useGuidedRun";

export function RunStep({
  run,
  onEdit,
  onStart,
  starting,
  error,
}: {
  run: GuidedRun;
  onEdit: (step: Step) => void;
  onStart: () => void;
  starting: boolean;
  error: Error | null;
}) {
  const settled = !run.loading && run.readError === null;
  const ready = settled && run.missing.length === 0;
  const target = run.targetClass
    ? (run.classes.find((entry) => entry.key === run.targetClass)?.name ?? run.targetClass)
    : undefined;

  return (
    <div className="flex max-w-3xl flex-col gap-5">
      <StepIntro title="Ready to run">
        One press draws the split if it is new, creates the experiment and queues Train &amp;
        score. The run page follows it from there.
      </StepIntro>

      {run.loading ? (
        <Skeleton className="h-52" />
      ) : (
        <dl className="grid grid-cols-[8rem_minmax(0,1fr)_auto] items-baseline gap-x-4 gap-y-2.5 rounded-panel border border-line bg-surface p-4 text-sm">
          <Row term="Dataset">{run.dataset.data?.name ?? "—"}</Row>
          <Row term="Task" step="goal" onEdit={onEdit}>
            {run.task ? TASK_TITLE[run.task] : "—"}
            {target && <span className="text-fg-muted"> · {target}</span>}
          </Row>
          <Row term="Look" step="look" onEdit={onEdit}>
            {run.profile ? `${run.profile.name} · r${run.profile.revision_no}` : "Full frame"}
          </Row>
          <Row term="Split" step="split" onEdit={onEdit}>
            {run.split ? (
              <>
                {run.split.label}
                <span className="text-fg-muted">
                  {" "}
                  · {run.split.choice.kind === "preset" ? "drawn now" : "existing"}
                  {run.split.composition &&
                    ` · ${run.split.composition
                      .filter((row) => row.total > 0)
                      .map((row) => `${row.subset} ${row.total}`)
                      .join(", ")}`}
                </span>
              </>
            ) : (
              "—"
            )}
          </Row>
          <Row term="Method" step="method" onEdit={onEdit}>
            {run.method ? (
              <>
                {run.method.title}{" "}
                <span className="font-mono text-xs text-fg-subtle">{run.method.key}</span>
              </>
            ) : (
              "—"
            )}
          </Row>
          <Row term="Input size" step="method" onEdit={onEdit}>
            <span className="font-mono">{run.size.caption}</span>
          </Row>
        </dl>
      )}

      <Field label="Name" description="Left empty, the run is named after its method and dataset." className="max-w-md">
        <Input
          aria-label="Name"
          value={run.state.name}
          placeholder={run.suggestedName || "Name this run"}
          onChange={(event) => run.update({ name: event.target.value })}
        />
      </Field>

      {error && <ErrorBox>{error.message}</ErrorBox>}

      <div className="flex flex-wrap items-center gap-3">
        <Button
          variant="primary"
          icon={<Play />}
          loading={starting}
          disabled={!ready}
          onClick={onStart}
        >
          Start run
        </Button>
        {settled && !ready && (
          <p className="text-xs text-warn">Still needs {joinWords(run.missing)}.</p>
        )}
      </div>
    </div>
  );
}

function Row({
  term,
  step,
  onEdit,
  children,
}: {
  term: string;
  step?: Step;
  onEdit?: (step: Step) => void;
  children: ReactNode;
}) {
  return (
    <>
      <dt className="text-fg-muted">{term}</dt>
      <dd className="min-w-0 truncate text-fg">{children}</dd>
      <dd>
        {step && onEdit ? (
          <button
            type="button"
            onClick={() => onEdit(step)}
            className={cn("rounded-sm text-xs text-signal hover:underline", focusRing)}
            aria-label={`Change ${term.toLowerCase()}`}
          >
            Change
          </button>
        ) : null}
      </dd>
    </>
  );
}

function joinWords(words: string[]): string {
  if (words.length <= 1) return words.join("");
  return `${words.slice(0, -1).join(", ")} and ${words[words.length - 1]}`;
}
