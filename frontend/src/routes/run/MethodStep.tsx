/**
 * Step 4, the method: the task's methods in the registry's order of standing, the one it
 * recommends chosen, and the decisions its schema marks `x-primary` in front of the rest.
 *
 * The cards are the create form's (`MethodCard`), and the options are its `SchemaForm`: lab-ui
 * shows a field marked `x-primary` and folds the others under Advanced, so the form asks only
 * what a person decides. An untouched field is sent as nothing and Python's default applies.
 * The options sit beside the cards on a wide window, held in view while the list scrolls, and
 * say the input size this configuration resolves to.
 */

import { Callout, SchemaForm, SkeletonRows } from "@vitavision/lab-ui";

import { isRecommended } from "../../api/methodChoice";
import { MethodCard } from "../../components/MethodCard";
import { StepIntro } from "./ChoiceCard";
import type { GuidedRun } from "./useGuidedRun";

export function MethodStep({ run }: { run: GuidedRun }) {
  const task = run.task;
  const method = run.method;

  return (
    <div className="flex flex-col gap-5">
      <StepIntro title="Which method?">
        In the registry&apos;s order: its recommendation for this task first, the floor — the
        method every other is measured against — last.
      </StepIntro>

      {run.unavailableRecommendation && (
        <Callout
          tone="info"
          title={`The recommended method, ${run.unavailableRecommendation.title}, cannot run here`}
        >
          <span className="block">
            {run.unavailableRecommendation.availability.reason ??
              "Its dependencies are not installed."}
          </span>
          <span className="block">
            The methods this installation can run come first, and the first of them is chosen.
          </span>
        </Callout>
      )}

      <div className="grid items-start gap-5 lg:grid-cols-[minmax(0,1fr)_24rem]">
        <div className="flex flex-col gap-3">
          {run.catalog.isPending && <SkeletonRows rows={3} />}
          {task !== undefined &&
            run.methodsForTask.map((entry) => (
              <MethodCard
                key={entry.key}
                method={entry}
                recommended={isRecommended(entry, task)}
                selected={entry.key === method?.key}
                onSelect={() => run.update({ methodKey: entry.key, configValues: {} })}
              />
            ))}
        </div>

        {method && (
          <section
            aria-labelledby="method-options"
            className="flex flex-col gap-3 rounded-panel border border-line bg-surface p-4 lg:sticky lg:top-0"
          >
            <div className="flex flex-col gap-0.5">
              <h3 id="method-options" className="text-sm font-medium text-fg">
                {method.title} · options
              </h3>
              <span className="text-xs text-fg-muted">
                Input size <span className="font-mono text-fg">{run.size.caption}</span>
              </span>
            </div>
            {run.configFields.length === 0 ? (
              <p className="text-sm text-fg-muted">This method has nothing to decide.</p>
            ) : (
              <SchemaForm
                fields={run.configFields}
                values={run.configValues}
                onChange={(values) => run.update({ methodKey: method.key, configValues: values })}
              />
            )}
            {run.configProblems.length > 0 && (
              <Callout tone="warning" title="Some options need a second look">
                {run.configProblems.join(", ")}
              </Callout>
            )}
          </section>
        )}
      </div>
    </div>
  );
}
