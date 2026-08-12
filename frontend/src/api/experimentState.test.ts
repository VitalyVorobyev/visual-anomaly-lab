import { describe, expect, it } from "vitest";

import {
  EMPTY_EXPERIMENT_CATALOG,
  readExperimentCatalogState,
  toExperimentListQuery,
  writeExperimentCatalogState,
} from "./experimentState";

const read = (query: string) => readExperimentCatalogState(new URLSearchParams(query));

describe("experiment catalogue state", () => {
  it("round-trips every supported filter", () => {
    const state = {
      modelType: "patchcore_anomalib",
      status: "trained",
      query: "candle",
      sort: "oldest",
    } as const;
    expect(read(writeExperimentCatalogState(state).toString())).toEqual(state);
  });

  it("keeps the default catalogue URL clean", () => {
    expect(writeExperimentCatalogState(EMPTY_EXPERIMENT_CATALOG).toString()).toBe("");
  });

  it("rejects unknown enums from hand-edited URLs", () => {
    expect(read("status=ghost&sort=fastest")).toEqual(EMPTY_EXPERIMENT_CATALOG);
  });

  it("adds dataset scope to the API query without writing it into local filters", () => {
    expect(toExperimentListQuery({ ...EMPTY_EXPERIMENT_CATALOG, query: "  run  " }, 7)).toEqual({
      datasetId: 7,
      modelType: undefined,
      status: undefined,
      query: "run",
      sort: "newest",
    });
  });
});
