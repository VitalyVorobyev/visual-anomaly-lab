import { describe, expect, it } from "vitest";

import {
  activeFilterCount,
  EMPTY_EXPERIMENT_CATALOG,
  type ExperimentCatalogState,
  readExperimentCatalogState,
  toExperimentListQuery,
  writeExperimentCatalogState,
} from "./experimentState";

const read = (query: string) => readExperimentCatalogState(new URLSearchParams(query));

describe("experiment catalogue state", () => {
  it("round-trips every supported filter", () => {
    const state: ExperimentCatalogState = {
      modelTypes: ["patchcore_anomalib", "fss_dino"],
      status: "trained",
      query: "candle",
      createdFrom: "2026-09-01",
      createdTo: "2026-09-24",
      sort: "method",
    };
    expect(read(writeExperimentCatalogState(state).toString())).toEqual(state);
  });

  it("keeps the default catalogue URL clean", () => {
    expect(writeExperimentCatalogState(EMPTY_EXPERIMENT_CATALOG).toString()).toBe("");
  });

  it("rejects unknown enums from hand-edited URLs", () => {
    expect(read("status=ghost&sort=fastest&from=yesterday")).toEqual(EMPTY_EXPERIMENT_CATALOG);
  });

  it("adds dataset scope to the API query without writing it into local filters", () => {
    expect(toExperimentListQuery({ ...EMPTY_EXPERIMENT_CATALOG, query: "  run  " }, 7)).toEqual({
      datasetId: 7,
      modelTypes: [],
      status: undefined,
      query: "run",
      createdFrom: undefined,
      createdTo: undefined,
      sort: "newest",
    });
  });

  it("reads repeated methods as a set and counts each narrowing filter once", () => {
    const state = read("method=fss_dino&method=proto_seg&method=fss_dino&from=2026-09-01");
    expect(state.modelTypes).toEqual(["fss_dino", "proto_seg"]);
    expect(activeFilterCount(state)).toBe(2);
  });
});
