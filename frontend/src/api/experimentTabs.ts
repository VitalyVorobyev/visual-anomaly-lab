/**
 * Which view of an experiment is open.
 *
 * Lives here rather than in `ExperimentRoute` because `ResultsState` carries it, and the
 * results state is what the gallery hands to the sample page and gets back on the way out.
 * A route importing from `api/` is the direction this codebase already goes; the reverse
 * would be a cycle.
 *
 * `overview` is the absence of the parameter, matching how the tab strip writes it: an
 * untouched view has a clean URL.
 */

export const TAB_IDS = [
  "overview",
  "samples",
  "training",
  "benchmark",
  "architecture",
  "inspector",
  "jobs",
] as const;

export type TabId = (typeof TAB_IDS)[number];

/** A hand-edited URL naming a tab that does not exist lands on Overview, not on nothing. */
export function parseTab(raw: string | null): TabId {
  return TAB_IDS.includes(raw as TabId) ? (raw as TabId) : "overview";
}
