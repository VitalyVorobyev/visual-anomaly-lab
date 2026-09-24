import { describe, expect, it } from "vitest";

import {
  DEFAULT_CUT,
  EMPTY_RESULTS,
  cutValue,
  readResultsState,
  resolveSubset,
  resultsDefaults,
  writeResultsState,
} from "./resultsState";

function roundTrip(params: string) {
  return readResultsState(new URLSearchParams(params), "anomaly");
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

  it("carries the peak marker, which is off until asked for", () => {
    // Off by default, unlike the heatmap: it answers *why* a verdict says what it does, and
    // a marker on every sample would compete with the layers that answer the first question.
    expect(roundTrip("").peak).toBe(false);
    expect(roundTrip("pk=1").peak).toBe(true);
    expect(roundTrip("pk=nonsense").peak).toBe(false);
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

  it("carries the tab a link came from, defaulting to overview", () => {
    expect(roundTrip("tab=samples").tab).toBe("samples");
    expect(roundTrip("").tab).toBe("overview");
  });

  it("refuses a tab that does not exist", () => {
    expect(roundTrip("tab=nonsense").tab).toBe("overview");
  });
});

describe("writing it back", () => {
  it("writes nothing for an untouched view", () => {
    expect(writeResultsState(EMPTY_RESULTS, "anomaly").toString()).toBe("");
  });

  it("round-trips everything it writes", () => {
    const state = {
      ...EMPTY_RESULTS,
      tab: "samples" as const,
      subset: "val" as const,
      outcome: "fp" as const,
      threshold: 1.25,
      sort: "score-asc" as const,
      heatmap: false,
      region: true,
      truth: false,
      peak: true,
      cut: 0.3,
    };
    expect(readResultsState(writeResultsState(state, "anomaly"), "anomaly")).toEqual(state);
  });

  it("brings a reader back to the tab they left", () => {
    /**
     * The bug this fixes: `writeResultsState` built the gallery's tile link, the sample
     * page's back link *and* its prev/next, and knew nothing about tabs — so all three
     * dropped `tab=samples` and every route out of a sample landed on Overview. Carrying
     * the tab here fixes all three at once, and any future link built the same way is
     * correct without anyone remembering to re-attach it.
     */
    const fromGallery = readResultsState(
      new URLSearchParams("tab=samples&subset=test"),
      "anomaly",
    );
    expect(writeResultsState(fromGallery, "anomaly").get("tab")).toBe("samples");
  });

  it("writes no tab for overview, matching how the tab strip clears it", () => {
    expect(writeResultsState({ ...EMPTY_RESULTS, tab: "overview" }, "anomaly").has("tab")).toBe(false);
  });
});

describe("the defaults are the task's", () => {
  it("opens a supervised segmentation run on its label map, not its heatmap", () => {
    const state = readResultsState(new URLSearchParams(""), "semantic_segmentation");
    expect(state.region).toBe(true);
    expect(state.heatmap).toBe(false);
    expect(state.truth).toBe(true);
    expect(state.peak).toBe(false);
  });

  it("keeps a few-shot run on its foreground map, as an anomaly run is", () => {
    expect(resultsDefaults("few_shot_segmentation")).toEqual(EMPTY_RESULTS);
  });

  it("reads an unknown task as an anomaly run until the experiment has loaded", () => {
    expect(resultsDefaults(undefined)).toEqual(EMPTY_RESULTS);
  });

  it.each(["anomaly", "few_shot_segmentation", "semantic_segmentation"] as const)(
    "writes nothing for an untouched %s view",
    (task) => {
      expect(writeResultsState(resultsDefaults(task), task).toString()).toBe("");
    },
  );

  it.each(["anomaly", "semantic_segmentation"] as const)(
    "round-trips every layer combination under the %s defaults",
    (task) => {
      for (const bits of Array.from({ length: 16 }, (_, index) => index)) {
        const state = {
          ...resultsDefaults(task),
          heatmap: (bits & 1) !== 0,
          region: (bits & 2) !== 0,
          truth: (bits & 4) !== 0,
          peak: (bits & 8) !== 0,
        };
        expect(readResultsState(writeResultsState(state, task), task)).toEqual(state);
      }
    },
  );

  it("writes a layer only where it departs from its own task's default", () => {
    // Heatmap on is the anomaly default and a departure on a semantic run, and the reverse
    // for the prediction — so the same state is a different URL under each.
    const state = { ...EMPTY_RESULTS, heatmap: true, region: true };
    expect(writeResultsState(state, "anomaly").toString()).toBe("seg=1");
    expect(writeResultsState(state, "semantic_segmentation").toString()).toBe("map=1");
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

describe("resolving the subset every tab reads", () => {
  it("falls back to the last scored subset rather than to all of them", () => {
    // No subset on the server means every scored subset ranked together — a different
    // population from the one Overview's confusion matrix is computed over.
    expect(resolveSubset(EMPTY_RESULTS, ["val", "test"]).subset).toBe("test");
  });

  it("keeps a subset the URL names when the run scored it", () => {
    expect(resolveSubset(roundTrip("subset=val"), ["val", "test"]).subset).toBe("val");
  });

  it("replaces a subset the run never scored", () => {
    expect(resolveSubset(roundTrip("subset=train"), ["test"]).subset).toBe("test");
  });

  it("stays unset before anything is scored", () => {
    expect(resolveSubset(EMPTY_RESULTS, []).subset).toBeUndefined();
  });
});
