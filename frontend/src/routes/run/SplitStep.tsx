/**
 * Step 3, the split: which samples train and which are scored, seen before it exists.
 *
 * The task's presets come first (`GET /api/datasets/{id}/split-presets`), the first of them
 * chosen, each with the composition its dry run produced — the Splits tab's own bar
 * (`SplitComposition`) — and a few samples of every subset from the dry run's `examples`. A
 * split the dataset already has that serves the task can be chosen instead. Nothing is
 * created here: a preset becomes a split only when the run starts.
 */

import { Badge, Callout, Skeleton } from "@vitavision/lab-ui";
import { Link } from "react-router";

import { SplitComposition } from "../../components/SplitComposition";
import { ChoiceCard, StepIntro } from "./ChoiceCard";
import { SampleThumbById } from "./Thumbs";
import { sameChoice, type GuidedRun, type SplitOption } from "./useGuidedRun";

export function SplitStep({ run, datasetId }: { run: GuidedRun; datasetId: number }) {
  const loading = run.loading;
  const presetCount = run.splitOptions.filter((option) => option.choice.kind === "preset").length;

  return (
    <div className="flex flex-col gap-5">
      <StepIntro title="Which samples train, and which are scored?">
        A preset is drawn when the run starts; an existing split is reused as it is.
      </StepIntro>

      {loading && (
        <div className="flex flex-col gap-3">
          <Skeleton className="h-40" />
          <Skeleton className="h-24" />
        </div>
      )}
      {!loading && !run.readError && run.splitOptions.length === 0 && (
        <Callout title="Nothing to split by yet">
          No preset fits this task on this dataset.{" "}
          <Link
            className="text-signal underline underline-offset-2"
            to={`/datasets/${datasetId}/splits`}
          >
            Draw one by hand on Splits
          </Link>
          .
        </Callout>
      )}

      <div className="flex flex-col gap-3">
        {run.splitOptions.map((option, index) => (
          <ChoiceCard
            key={option.choice.kind === "preset" ? `p:${option.choice.key}` : `s:${option.choice.id}`}
            name="split"
            value={option.choice.kind === "preset" ? option.choice.key : String(option.choice.id)}
            selected={sameChoice(option.choice, run.choice)}
            onSelect={() => run.update({ split: option.choice })}
            title={option.label}
            badges={
              <>
                {index === 0 && option.choice.kind === "preset" && <Badge tone="info">suggested</Badge>}
                <Badge tone="neutral">{option.choice.kind === "preset" ? "new" : "existing"}</Badge>
              </>
            }
            description={option.meaning}
          >
            <OptionBody
              option={option}
              run={run}
              datasetId={datasetId}
              expanded={sameChoice(option.choice, run.choice)}
            />
          </ChoiceCard>
        ))}
      </div>
      {presetCount > 0 && (
        <p className="text-xs text-fg-subtle">
          Other fractions, seeds or strategies are on{" "}
          <Link className="text-signal underline underline-offset-2" to={`/datasets/${datasetId}/splits`}>
            Splits
          </Link>
          ; a split made there is offered here.
        </p>
      )}
    </div>
  );
}

function OptionBody({
  option,
  run,
  datasetId,
  expanded,
}: {
  option: SplitOption;
  run: GuidedRun;
  datasetId: number;
  expanded: boolean;
}) {
  if (option.error) return <p className="text-xs text-warn">{option.error}</p>;
  if (option.pending || option.composition === undefined) return <Skeleton className="h-12" />;
  const outline =
    option.mode.kind === "class"
      ? ({ kind: "class", key: option.mode.focus } as const)
      : run.task === "anomaly"
        ? ({ kind: "defect" } as const)
        : undefined;
  return (
    <div className="flex flex-col gap-3">
      <SplitComposition
        composition={option.composition}
        mode={option.mode}
        classes={run.classInfo}
        normalsOnlyTrain={run.task === "anomaly"}
      />
      {expanded && (
        <div className="grid gap-3 sm:grid-cols-3">
          {option.composition
            .filter((row) => row.total > 0)
            .map((row) => (
              <div key={row.subset} className="flex min-w-0 flex-col gap-1.5">
                <span className="text-xs font-medium text-fg-muted">
                  {row.subset} <span className="font-mono text-fg-subtle">· {row.total}</span>
                </span>
                <ul aria-label={`Examples from ${row.subset}`} className="flex flex-wrap gap-1.5">
                  {row.examples.map((sampleId) => (
                    <li key={sampleId}>
                      <SampleThumbById
                        datasetId={datasetId}
                        sampleId={sampleId}
                        defaultChannel={run.dataset.data?.default_channel}
                        outline={outline}
                        size="sm"
                      />
                    </li>
                  ))}
                </ul>
              </div>
            ))}
        </div>
      )}
      {expanded && (
        <span className="font-mono text-xs text-fg-subtle">{option.name}</span>
      )}
    </div>
  );
}
