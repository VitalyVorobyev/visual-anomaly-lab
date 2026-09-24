import { describe, expect, it } from "vitest";

import { readPrepareState, writePrepareState } from "./prepareState";

const read = (query: string) => readPrepareState(new URLSearchParams(query));

describe("the Prepare tab's URL state", () => {
  it("round-trips a followed build", () => {
    const state = { profile: 4, job: 91, jobProfile: 4, mode: "build" as const };
    expect(readPrepareState(writePrepareState(state))).toEqual(state);
  });

  it("is empty with no parameters", () => {
    expect(read("")).toEqual({
      profile: undefined,
      job: undefined,
      jobProfile: undefined,
      mode: undefined,
    });
  });

  it("drops a job whose mode is missing or unknown, and a mode with no job", () => {
    expect(read("profile=4&job=91").job).toBeUndefined();
    expect(read("profile=4&job=91&mode=train").job).toBeUndefined();
    expect(read("profile=4&mode=build").mode).toBeUndefined();
  });

  it("ignores ids that are not positive integers", () => {
    expect(read("profile=-1").profile).toBeUndefined();
    expect(read("profile=1.5").profile).toBeUndefined();
  });
});
