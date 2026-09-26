/**
 * Step 1, the goal: what the run is asked to do, chosen in front of what the truth looks like.
 *
 * Every task some method runs and the dataset's truth can serve is a card, with its meaning
 * and a strip of the samples that would teach it — the defects for anomaly detection, the
 * target class for few-shot segmentation, one sample of each class for segmentation and
 * detection. A task that still needs an annotated class says so and links to annotation
 * rather than being offered. Few-shot adds the class, the most frequent by default.
 */

import { Badge, Field, Select, Skeleton } from "@vitavision/lab-ui";
import { Fragment } from "react";
import { Link } from "react-router";

import type { Task } from "../../api/client";
import { useSamples } from "../../hooks/useCatalog";
import { ChoiceCard, StepIntro } from "./ChoiceCard";
import { SampleThumb, ThumbStrip } from "./Thumbs";
import { TASK_MEANING, TASK_TITLE, type GuidedRun } from "./useGuidedRun";

const STRIP = 6;

export function GoalStep({ run, datasetId }: { run: GuidedRun; datasetId: number }) {
  if (run.dataset.isPending || run.catalog.isPending) {
    return (
      <div className="flex flex-col gap-3">
        <Skeleton className="h-6 w-64" />
        <Skeleton className="h-36" />
        <Skeleton className="h-36" />
      </div>
    );
  }

  // The suggestion first: it is the answer most readers came for, and on a dataset of
  // classes it would otherwise sit below the fold under tasks it does not favour.
  const ordered = [
    ...run.offered.filter((task) => task === run.suggested),
    ...run.offered.filter((task) => task !== run.suggested),
  ];

  return (
    <div className="flex flex-col gap-5">
      <StepIntro title="What should the run do?">
        The suggestion follows this dataset&apos;s truth. The pictures are what each choice
        learns from.
      </StepIntro>

      {!run.readError && run.offered.length === 0 && (
        <p className="text-sm text-fg-muted">
          No task fits this dataset yet.{" "}
          <Link className="text-signal underline underline-offset-2" to={`/datasets/${datasetId}`}>
            Label samples normal or defect
          </Link>{" "}
          or{" "}
          <Link
            className="text-signal underline underline-offset-2"
            to={`/datasets/${datasetId}/annotate`}
          >
            annotate a class
          </Link>
          .
        </p>
      )}

      <div className="flex flex-col gap-3">
        {ordered.map((task) =>
          run.needsAnnotation(task) ? (
            <article
              key={task}
              aria-label={TASK_TITLE[task]}
              className="flex flex-col gap-1.5 rounded-panel border border-dashed border-line p-3.5"
            >
              <span className="text-sm font-semibold tracking-tight text-fg-muted">
                {TASK_TITLE[task]}
              </span>
              <span className="text-xs text-fg-muted">
                {TASK_MEANING[task]} It needs an annotated class first —{" "}
                <Link
                  className="text-signal underline underline-offset-2"
                  to={`/datasets/${datasetId}/annotate`}
                >
                  annotate one
                </Link>
                .
              </span>
            </article>
          ) : (
            <Fragment key={task}>
              <ChoiceCard
                name="task"
                value={task}
                selected={task === run.task}
                onSelect={() =>
                  // Another task is another run: its split, method and options start over.
                  run.update({ task, split: undefined, methodKey: undefined, configValues: {} })
                }
                title={TASK_TITLE[task]}
                badges={task === run.suggested ? <Badge tone="info">suggested</Badge> : undefined}
                description={TASK_MEANING[task]}
              >
                <TaskPictures task={task} run={run} datasetId={datasetId} />
              </ChoiceCard>
              {/* Beside the card it belongs to, not inside it: the card is one radio. */}
              {task === run.task && run.targeted && run.classes.length > 0 && (
                <section aria-label="Target class" className="-mt-1 pl-3.5">
                  <Field
                    as="group"
                    label="Which class"
                    description="The run segments this class in every image; the most frequent is first. The pictures above follow it."
                    className="max-w-md"
                  >
                    <Select
                      aria-label="Target class"
                      value={run.targetClass ?? ""}
                      onValueChange={(value) => run.update({ targetClass: value, split: undefined })}
                      options={run.classes.map((entry) => ({
                        value: entry.key,
                        label: entry.name,
                        note: `${entry.samples} samples`,
                      }))}
                    />
                  </Field>
                </section>
              )}
            </Fragment>
          ),
        )}
      </div>
    </div>
  );
}

/** What a task learns from, as pictures. */
function TaskPictures({ task, run, datasetId }: { task: Task; run: GuidedRun; datasetId: number }) {
  if (task === "anomaly") return <DefectStrip run={run} datasetId={datasetId} />;
  if (task === "few_shot_segmentation") {
    const key = run.targetClass ?? run.classes[0]?.key;
    return key ? <ClassStrip run={run} datasetId={datasetId} classKey={key} /> : null;
  }
  return <EachClassStrip run={run} datasetId={datasetId} />;
}

function DefectStrip({ run, datasetId }: { run: GuidedRun; datasetId: number }) {
  const samples = useSamples(datasetId, { label: "defect", limit: STRIP });
  const items = samples.data?.items ?? [];
  return (
    <ThumbStrip
      label="Defect samples"
      pending={samples.isPending}
      empty={
        samples.error
          ? samples.error.message
          : items.length === 0
            ? "No sample is labelled defect yet: the run learns from normals and is scored on whatever is labelled."
            : null
      }
    >
      {items.map((sample) => (
        <SampleThumb
          key={sample.id}
          sample={sample}
          defaultChannel={run.dataset.data?.default_channel}
          outline={{ kind: "defect" }}
        />
      ))}
    </ThumbStrip>
  );
}

function ClassStrip({
  run,
  datasetId,
  classKey,
}: {
  run: GuidedRun;
  datasetId: number;
  classKey: string;
}) {
  const samples = useSamples(datasetId, { classKey, presence: "present", limit: STRIP });
  const items = samples.data?.items ?? [];
  return (
    <ThumbStrip
      label={`Samples of ${classKey}`}
      pending={samples.isPending}
      empty={samples.error ? samples.error.message : items.length === 0 ? "No sample shows it." : null}
    >
      {items.map((sample) => (
        <SampleThumb
          key={sample.id}
          sample={sample}
          defaultChannel={run.dataset.data?.default_channel}
          outline={{ kind: "class", key: classKey }}
        />
      ))}
    </ThumbStrip>
  );
}

/** One sample of each class, named — what a run that learns all of them sees. */
function EachClassStrip({ run, datasetId }: { run: GuidedRun; datasetId: number }) {
  const shown = run.classes.slice(0, STRIP);
  if (shown.length === 0) return null;
  return (
    <ul aria-label="One sample of each class" className="flex flex-wrap gap-2">
      {shown.map((entry) => (
        <li key={entry.key}>
          <OneOfClass run={run} datasetId={datasetId} classKey={entry.key} name={entry.name} />
        </li>
      ))}
      {run.classes.length > shown.length && (
        <li className="self-center text-xs text-fg-subtle">
          and {run.classes.length - shown.length} more classes
        </li>
      )}
    </ul>
  );
}

function OneOfClass({
  run,
  datasetId,
  classKey,
  name,
}: {
  run: GuidedRun;
  datasetId: number;
  classKey: string;
  name: string;
}) {
  const samples = useSamples(datasetId, { classKey, presence: "present", limit: 1 });
  const sample = samples.data?.items[0];
  if (samples.isPending) return <Skeleton className="h-24 w-28" />;
  if (!sample) return null;
  return (
    <SampleThumb
      sample={sample}
      defaultChannel={run.dataset.data?.default_channel}
      outline={{ kind: "class", key: classKey }}
      caption={name}
    />
  );
}
