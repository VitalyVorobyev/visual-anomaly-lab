/**
 * What a dataset's truth says, in the words of its kind (ADR-0041).
 *
 * An anomaly dataset is counted in verdicts — `12 normal · 9 defect`. A dataset of classes is
 * counted in classes — `20 classes`, with each class and the samples that show it one hover
 * away. A dataset holding both says both. Neither is an empty run, not `0 defect`.
 */

import { Tags } from "lucide-react";

import type { DatasetSummary } from "../api/client";
import { hasClasses, hasLabels } from "../api/truth";
import { CountRun, InfoHint } from "@vitavision/lab-ui";

type Truthful = Pick<DatasetSummary, "truth" | "label_counts" | "class_counts">;

/** `K classes`, or nothing for a dataset without class truth. */
export function classCountText(dataset: Truthful): string | null {
  if (!hasClasses(dataset.truth)) return null;
  const count = dataset.class_counts.length;
  return `${count} ${count === 1 ? "class" : "classes"}`;
}

/** The verdict run for a dataset with labels; nothing for one without. */
export function LabelRun({ dataset }: { dataset: Truthful }) {
  if (!hasLabels(dataset.truth)) return null;
  return (
    <CountRun
      counts={[
        ["normal", dataset.label_counts["normal"] ?? 0, "normal"],
        ["defect", dataset.label_counts["defect"] ?? 0, "defect"],
        ["unlabeled", dataset.label_counts["unlabeled"] ?? 0, "unlabeled"],
      ]}
    />
  );
}

/** The classes and their sample counts, behind a mark, for a dataset with class truth. */
export function ClassHint({ dataset }: { dataset: Truthful }) {
  if (!hasClasses(dataset.truth)) return null;
  return (
    <InfoHint icon={Tags} label="Classes and the samples that show them">
      <ClassList dataset={dataset} />
    </InfoHint>
  );
}

/**
 * The catalogue card's one line of counts: samples, then what its truth is counted in.
 * Plain text rather than badges — the card is a picture and a sentence, and this is a
 * footnote to them.
 */
export function truthLine(dataset: Truthful & Pick<DatasetSummary, "samples">): string {
  const parts = [`${dataset.samples} ${dataset.samples === 1 ? "sample" : "samples"}`];
  const classes = classCountText(dataset);
  if (classes) parts.push(classes);
  if (hasLabels(dataset.truth)) {
    for (const label of ["normal", "defect"] as const) {
      const count = dataset.label_counts[label] ?? 0;
      if (count > 0) parts.push(`${count} ${label}`);
    }
  }
  return parts.join(" · ");
}

export function ClassList({ dataset }: { dataset: Truthful }) {
  return (
    <ul aria-label="Classes" className="flex flex-col gap-1">
      {dataset.class_counts.map((entry) => (
        <li key={entry.key} className="flex items-center gap-2">
          <span
            aria-hidden
            className="size-2.5 shrink-0 rounded-full border border-line-strong"
            // The class's own colour, as the editor paints it: data, not a design decision.
            style={{ backgroundColor: entry.color }}
          />
          <span className="min-w-0 flex-1 truncate">{entry.name}</span>
          <span className="font-mono tabular-nums text-fg-subtle">{entry.samples}</span>
        </li>
      ))}
    </ul>
  );
}
