/**
 * What was different about how these runs were set up.
 *
 * **Preprocessing first, and loudly**, because it is the one difference that changes what a
 * metric means rather than merely how it was produced: a comparison between two methods
 * only means something if they were shown the same pixels, and a difference in AUROC
 * between two resolutions is partly a measurement of the resize. Every plugin loads its
 * pixels through one function precisely so this cannot happen by accident — so when it
 * happens on purpose, the screen has to say so.
 *
 * Method configuration is last and folded away. Two methods have nothing in common there,
 * so "differs" is the expected state and carries no information; it is here to be read, not
 * to be alarmed by.
 */

import type { ComparedRun } from "../../api/client";
import { Disclosure, Panel, cn } from "@vitavision/lab-ui";

type Block = "preprocessing" | "evaluation" | "config";

export function ConfigDiff({ runs }: { runs: ComparedRun[] }) {
  return (
    <Panel title="Configuration">
      <p className="mb-4 text-xs text-fg-muted">
        Frozen into each experiment when it was created, so this is what the runs{" "}
        <em>used</em> — not what the form defaults to now.
      </p>

      <div className="flex flex-col gap-6">
        <Block
          runs={runs}
          block="preprocessing"
          title="Preprocessing"
          note="The pixels each method was shown. A difference here is a difference in the question, not only in the answer."
          highlight
        />
        <Block
          runs={runs}
          block="evaluation"
          title="Evaluation"
          note="How the stored scores were read into metrics. The scores are unaffected."
          highlight
        />
        <Disclosure summary="Method configuration">
          <Block
            runs={runs}
            block="config"
            title=""
            note="Two methods share no options, so differences here are expected rather than suspicious."
          />
        </Disclosure>
      </div>
    </Panel>
  );
}

function Block({
  runs,
  block,
  title,
  note,
  highlight = false,
}: {
  runs: ComparedRun[];
  block: Block;
  title: string;
  note: string;
  /** Mark differing rows. Off for method config, where every row differs and none matters. */
  highlight?: boolean;
}) {
  const values = runs.map((run) => (run[block] ?? {}) as Record<string, unknown>);
  const keys = [...new Set(values.flatMap((entry) => Object.keys(entry)))].sort();

  if (keys.length === 0) {
    return (
      <div className="flex flex-col gap-1.5">
        {title && <h3 className="text-xs font-semibold text-fg">{title}</h3>}
        <p className="text-xs text-fg-muted">Every option left at its default in every run.</p>
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-1.5">
      {title && <h3 className="text-xs font-semibold text-fg">{title}</h3>}
      <p className="text-xs text-fg-muted">{note}</p>
      <div className="w-full overflow-x-auto">
        <table className="w-full text-left text-sm">
          <thead>
            <tr className="border-b border-line">
              <th className="py-1.5 pr-4 text-xs font-medium text-fg-muted">Option</th>
              {runs.map((run) => (
                <th key={run.id} className="min-w-32 px-3 py-1.5 text-xs font-semibold text-fg">
                  <span className="block truncate">{run.name}</span>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {keys.map((key) => {
              /* Absence is "this run has no such option", not "its value stringified to
                 nothing" — a method that never had the key left it at a Python default. */
              const cells = values.map((entry) =>
                key in entry ? JSON.stringify(entry[key]) : undefined,
              );
              const differs = new Set(cells).size > 1;
              return (
                <tr key={key} className="border-b border-line/60 last:border-0">
                  <th
                    className={cn(
                      "py-1 pr-4 text-left font-mono text-xs font-normal",
                      highlight && differs ? "text-warn" : "text-fg-muted",
                    )}
                  >
                    {key}
                  </th>
                  {cells.map((cell, index) => (
                    <td
                      key={index}
                      className={cn(
                        "px-3 py-1 font-mono text-xs tabular-nums",
                        highlight && differs ? "text-warn" : "text-fg",
                      )}
                    >
                      {cell ?? <span className="text-fg-subtle">—</span>}
                    </td>
                  ))}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
