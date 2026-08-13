/**
 * The catalogue's reading order, in one place because two screens have to agree on it.
 *
 * The page renders it and the collection dialog lists datasets in it, so a group you are
 * editing is in the same order as the group behind the dialog.
 */

import type { DatasetSummary } from "../../api/client";

/**
 * Split the catalogue into the ungrouped and the groups, both in a stable order.
 *
 * Collections sort by name and datasets keep the order the API gave them (`ORDER BY id`,
 * i.e. import order), so nothing moves between visits and a registration appends rather
 * than reshuffles.
 */
export function groupDatasets(datasets: DatasetSummary[]): {
  ungrouped: DatasetSummary[];
  groups: [string, DatasetSummary[]][];
} {
  const ungrouped: DatasetSummary[] = [];
  const grouped = new Map<string, DatasetSummary[]>();

  for (const dataset of datasets) {
    const name = dataset.collection?.trim();
    if (!name) {
      ungrouped.push(dataset);
      continue;
    }
    const members = grouped.get(name);
    if (members) members.push(dataset);
    else grouped.set(name, [dataset]);
  }

  return {
    ungrouped,
    groups: [...grouped.entries()].sort(([a], [b]) => a.localeCompare(b)),
  };
}

/** The same order, flattened: ungrouped first, then each collection's members. */
export function inCatalogueOrder(datasets: DatasetSummary[]): DatasetSummary[] {
  const { ungrouped, groups } = groupDatasets(datasets);
  return [...ungrouped, ...groups.flatMap(([, members]) => members)];
}
