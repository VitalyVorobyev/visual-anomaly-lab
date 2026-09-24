/**
 * One line beside "New experiment" saying whether pressing it can lead anywhere yet — per task.
 *
 * Every run needs a built region profile; an anomaly run needs a split, and a few-shot run a
 * class with references and a split of them (ADR-0040). The band says it before the create
 * form does, in the order the steps have to be done, and each step is the link that does it.
 * With one task it reads as a checklist. With two, the shared first step comes first, and
 * then each task says whether it is ready or what it needs next. Held to one line with no
 * wrap, because the band's height is fixed by construction.
 */

import { Check, CircleDashed } from "lucide-react";
import { Link } from "react-router";

import type { Task } from "../../api/client";
import { cn, focusRing } from "@vitavision/lab-ui";

import {
  useDatasetReadiness,
  type ReadinessStep,
  type TaskReadiness,
} from "../../hooks/useDatasetReadiness";

const STEP: Record<ReadinessStep, { label: string; path: string }> = {
  prepare: { label: "Build a region profile", path: "prepare" },
  split: { label: "Make a split", path: "splits" },
  annotate: { label: "Annotate a class", path: "annotate" },
  references: { label: "Choose references", path: "splits" },
};

const TASK_NAME: Partial<Record<Task, string>> = {
  anomaly: "Anomaly",
  few_shot_segmentation: "Few-shot",
};

const LINK = cn("rounded-sm transition-colors hover:text-signal", focusRing);
const TODO = cn(LINK, "flex items-center gap-1 font-medium text-warn");

export function DatasetReadiness({ datasetId }: { datasetId: number }) {
  const readiness = useDatasetReadiness(datasetId);
  if (!readiness.known || readiness.tasks.length === 0) return null;

  const runs = (
    <Link to={`/datasets/${datasetId}/experiments`} className={LINK}>
      {readiness.runs === 1 ? "1 run" : `${readiness.runs} runs`}
    </Link>
  );
  const [only] = readiness.tasks;
  const shared = readiness.tasks.every((entry) => entry.missing[0] === "prepare");

  if (readiness.tasks.length === 1 || shared) {
    const missing =
      readiness.tasks.length === 1 && only ? only.missing : (["prepare"] as ReadinessStep[]);
    if (missing.length === 0) {
      return (
        <p
          aria-label="Readiness"
          className="flex items-center gap-1.5 whitespace-nowrap text-xs text-fg-muted"
        >
          <Check aria-hidden className="size-3.5 text-normal" />
          Ready to train
          <span aria-hidden className="text-fg-subtle">·</span>
          {runs}
        </p>
      );
    }
    return (
      <nav
        aria-label="Before training"
        className="flex items-center gap-2 whitespace-nowrap text-xs text-fg-muted"
      >
        <span>Before training:</span>
        {missing.map((step, index) => (
          <Link key={step} to={`/datasets/${datasetId}/${STEP[step].path}`} className={TODO}>
            <CircleDashed aria-hidden className="size-3.5" />
            {index + 1}. {STEP[step].label}
          </Link>
        ))}
      </nav>
    );
  }

  return (
    <nav
      aria-label="Readiness by task"
      className="flex items-center gap-2 whitespace-nowrap text-xs text-fg-muted"
    >
      {readiness.tasks.map((entry) => (
        <TaskState key={entry.task} entry={entry} datasetId={datasetId} />
      ))}
      <span aria-hidden className="text-fg-subtle">·</span>
      {runs}
    </nav>
  );
}

function TaskState({ entry, datasetId }: { entry: TaskReadiness; datasetId: number }) {
  const name = TASK_NAME[entry.task] ?? entry.task;
  const [next] = entry.missing;
  if (next === undefined) {
    return (
      <span className="flex items-center gap-1">
        <Check aria-hidden className="size-3.5 text-normal" />
        {name}
      </span>
    );
  }
  return (
    <span className="flex items-center gap-1">
      {name}:
      <Link to={`/datasets/${datasetId}/${STEP[next].path}`} className={TODO}>
        <CircleDashed aria-hidden className="size-3.5" />
        {STEP[next].label}
      </Link>
    </span>
  );
}
