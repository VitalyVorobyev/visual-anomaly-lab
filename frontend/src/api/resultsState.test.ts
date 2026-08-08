import { describe, expect, it } from "vitest";

import {
  DEFAULT_CUT,
  EMPTY_RESULTS,
  cutValue,
  readResultsState,
  writeResultsState,
} from "./resultsState";

function roundTrip(params: string) {
  return readResultsState(new URLSearchParams(params));
}

describe("reading the results view out of a URL", () => {
  it("defaults to the heatmap and the ground truth, with no filter", () => {
    expect(roundTrip("")).toEqual(EMPTY_RESULTS);
  });

  it("keeps the filter and the layers a link carried", () => {
    const state = roundTrip("subset=test&outcome=fn&seg=1&map=0&cut=0.8&t=0.42");
    expect(state.subset).toBe("test");
    expect(state.outcome).toBe("fn");
    expect(state.region).toBe(true);
    expect(state.heatmap).toBe(false);
    expect(state.cut).toBe(0.8);
    expect(state.threshold).toBe(0.42);
  });

  it("refuses an outcome the server never emits", () => {
    // A hand-edited URL must not put an unknown value into a filter.
    expect(roundTrip("outcome=maybe").outcome).toBeUndefined();
    expect(roundTrip("subset=holdout").subset).toBeUndefined();
  });

  it("refuses a cut outside the range it is a fraction of", () => {
    expect(roundTrip("cut=1.5").cut).toBe(DEFAULT_CUT);
    expect(roundTrip("cut=-1").cut).toBe(DEFAULT_CUT);
    expect(roundTrip("cut=abc").cut).toBe(DEFAULT_CUT);
  });

  it("leaves the threshold unset rather than inventing one", () => {
    // Unset means "whatever the server suggested", and the server also supplies the
    // sentence explaining that choice. A fabricated default would read as a recommendation.
    expect(roundTrip("").threshold).toBeUndefined();
    expect(roundTrip("t=nonsense").threshold).toBeUndefined();
  });
});

describe("writing it back", () => {
  it("writes nothing for an untouched view", () => {
    expect(writeResultsState(EMPTY_RESULTS).toString()).toBe("");
  });

  it("round-trips everything it writes", () => {
    const state = {
      ...EMPTY_RESULTS,
      subset: "val" as const,
      outcome: "fp" as const,
      threshold: 1.25,
      sort: "score-asc" as const,
      heatmap: false,
      region: true,
      truth: false,
      cut: 0.3,
    };
    expect(readResultsState(writeResultsState(state))).toEqual(state);
  });
});

describe("the segmentation cut", () => {
  it("is a fraction of the run's range, not of the image on screen", () => {
    // The same fraction has to be the same absolute cut on every image of the run, or two
    // samples' predicted regions are drawn at different thresholds and cannot be compared.
    expect(cutValue({ ...EMPTY_RESULTS, cut: 0.5 }, { low: 0, high: 2 })).toBe(1);
    expect(cutValue({ ...EMPTY_RESULTS, cut: 0.25 }, { low: 1, high: 5 })).toBe(2);
  });

  it("has no value before a run has recorded a range", () => {
    expect(cutValue(EMPTY_RESULTS, null)).toBeNull();
    expect(cutValue(EMPTY_RESULTS, undefined)).toBeNull();
  });
});
