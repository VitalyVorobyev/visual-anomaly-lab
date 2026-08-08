import { describe, expect, it } from "vitest";

import {
  DEFAULT_RECALL_TARGET,
  EMPTY_COMPARE,
  MAX_RUNS,
  cutFor,
  readCompareState,
  refusalReason,
  toggleRun,
  writeCompareState,
} from "./compareState";
import { DEFAULT_CUT } from "./resultsState";

function read(params: string) {
  return readCompareState(new URLSearchParams(params));
}

describe("reading a comparison out of a URL", () => {
  it("is empty with no parameters", () => {
    expect(read("")).toEqual(EMPTY_COMPARE);
  });

  it("keeps the runs in the order the link named them", () => {
    // Order is the column order, and a table that reorders itself on every click is one
    // whose comparisons have to be re-found each time.
    expect(read("ids=7,3,5").ids).toEqual([7, 3, 5]);
  });

  it("carries the operating point, because it decides every confusion matrix", () => {
    const state = read("at=recall&rt=0.8&subset=val&view=samples&d=1&outcome=fp");
    expect(state.at).toBe("recall");
    expect(state.recallTarget).toBe(0.8);
    expect(state.subset).toBe("val");
    expect(state.view).toBe("samples");
    expect(state.disagreeOnly).toBe(true);
    expect(state.outcome).toBe("fp");
  });

  it("drops a duplicate id a hand-edited URL introduced", () => {
    // The server answers 422 for a repeated run; the reader did not cause that and should
    // not see it.
    expect(read("ids=4,4,9").ids).toEqual([4, 9]);
  });

  it("drops anything that is not a run id", () => {
    expect(read("ids=3,abc,-1,0,7").ids).toEqual([3, 7]);
  });

  it("never returns more runs than the server will accept", () => {
    expect(read("ids=1,2,3,4,5,6,7,8").ids).toHaveLength(MAX_RUNS);
  });

  it("falls back to the defaults for values outside their range", () => {
    const state = read("at=nonsense&rt=4&view=elsewhere&cut=3");
    expect(state.at).toBe("f1");
    expect(state.recallTarget).toBe(DEFAULT_RECALL_TARGET);
    expect(state.view).toBe("metrics");
    expect(state.cut).toBe(DEFAULT_CUT);
  });
});

describe("writing a comparison into a URL", () => {
  it("writes nothing for an untouched screen", () => {
    expect(writeCompareState(EMPTY_COMPARE).toString()).toBe("");
  });

  it("round-trips everything that changes a number", () => {
    const state = {
      ...EMPTY_COMPARE,
      ids: [2, 5],
      subset: "test" as const,
      at: "recall" as const,
      recallTarget: 0.9,
      view: "curves" as const,
      region: true,
      cut: 0.55,
    };
    expect(readCompareState(writeCompareState(state))).toEqual(state);
  });
});

describe("choosing which runs to compare", () => {
  it("appends rather than sorts, and removes on a second press", () => {
    expect(toggleRun([3, 1], 2)).toEqual([3, 1, 2]);
    expect(toggleRun([3, 1, 2], 1)).toEqual([3, 2]);
  });

  it("allows anything while nothing is selected", () => {
    expect(refusalReason({ id: 1, dataset_id: 1, split_id: 1 }, undefined, [])).toBeNull();
  });

  it("refuses a run on another dataset", () => {
    const reason = refusalReason(
      { id: 2, dataset_id: 9, split_id: 1 },
      { dataset_id: 1, split_id: 1 },
      [1],
    );
    expect(reason).toContain("different dataset");
  });

  it("refuses another split of the same dataset", () => {
    // Same data, different question: the numbers would be computed over different samples.
    const reason = refusalReason(
      { id: 2, dataset_id: 1, split_id: 7 },
      { dataset_id: 1, split_id: 1 },
      [1],
    );
    expect(reason).toContain("different split");
  });

  it("never refuses a run that is already selected", () => {
    // Otherwise the last column could not be unticked once the cap was reached.
    const selected = [1, 2, 3, 4, 5, 6];
    expect(
      refusalReason({ id: 3, dataset_id: 1, split_id: 1 }, { dataset_id: 1, split_id: 1 }, selected),
    ).toBeNull();
  });

  it("stops at the cap the server enforces", () => {
    const selected = [1, 2, 3, 4, 5, 6];
    const reason = refusalReason(
      { id: 7, dataset_id: 1, split_id: 1 },
      { dataset_id: 1, split_id: 1 },
      selected,
    );
    expect(reason).toContain(String(MAX_RUNS));
  });
});

describe("the shared cut", () => {
  it("resolves to a different value in every run, which is the point", () => {
    const state = { ...EMPTY_COMPARE, cut: 0.5 };
    expect(cutFor(state, { low: 0, high: 8 })).toBe(4);
    expect(cutFor(state, { low: 1, high: 1.4 })).toBeCloseTo(1.2);
  });

  it("has no value for a run that recorded no range", () => {
    // Never another run's range: that would cut one method's map at a threshold derived
    // from a different method's score distribution.
    expect(cutFor(EMPTY_COMPARE, null)).toBeNull();
    expect(cutFor(EMPTY_COMPARE, undefined)).toBeNull();
  });
});
