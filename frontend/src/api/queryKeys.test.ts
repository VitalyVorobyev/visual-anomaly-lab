/**
 * The property that makes invalidation work.
 *
 * TanStack Query invalidates by key *prefix*. Everything about one dataset — its detail,
 * its sample pages, an open sample, its splits — is nested under `dataset(id)`, so
 * relabelling a sample refreshes the grid, the open sample and the label counts with one
 * invalidation. Get this wrong and nothing errors: a stale row simply stays on screen.
 */

import { describe, expect, it } from "vitest";

import { queryKeys } from "./queryKeys";

function isPrefixOf(prefix: readonly unknown[], key: readonly unknown[]): boolean {
  return prefix.every((part, index) => JSON.stringify(part) === JSON.stringify(key[index]));
}

describe("queryKeys", () => {
  it("nests everything about a dataset under that dataset's key", () => {
    const dataset = queryKeys.dataset(7);

    expect(isPrefixOf(dataset, queryKeys.samples(7))).toBe(true);
    expect(isPrefixOf(dataset, queryKeys.samples(7, { label: "defect" }))).toBe(true);
    expect(isPrefixOf(dataset, queryKeys.sample(7, 42))).toBe(true);
    expect(isPrefixOf(dataset, queryKeys.splits(7))).toBe(true);
  });

  it("does not reach into a different dataset", () => {
    expect(isPrefixOf(queryKeys.dataset(7), queryKeys.sample(8, 42))).toBe(false);
    expect(isPrefixOf(queryKeys.dataset(7), queryKeys.splits(8))).toBe(false);
  });

  it("keeps the dataset list a prefix of every dataset", () => {
    expect(isPrefixOf(queryKeys.datasets(), queryKeys.dataset(7))).toBe(true);
  });

  it("distinguishes sample pages by their filters", () => {
    const all = queryKeys.samples(7);
    const defects = queryKeys.samples(7, { label: "defect" });
    const page2 = queryKeys.samples(7, { label: "defect", offset: 200 });

    expect(JSON.stringify(all)).not.toBe(JSON.stringify(defects));
    expect(JSON.stringify(defects)).not.toBe(JSON.stringify(page2));
  });

  it("keeps jobs out of the dataset tree", () => {
    // A job outlives the screen that started it and is not owned by a dataset, so
    // invalidating a dataset must not discard a running job's snapshot.
    expect(isPrefixOf(queryKeys.datasets(), queryKeys.job(3))).toBe(false);
  });
});
