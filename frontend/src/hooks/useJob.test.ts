/**
 * Turning a job's raw log into something a person reads.
 *
 * The log file is the raw worker stream by design (ADR-0009) — every byte, so a failed
 * run is diagnosable afterwards. A scan of nine hundred images writes nine hundred
 * progress events into it, and showing those verbatim buries the four lines that matter.
 */

import { describe, expect, it } from "vitest";

import { formatLogLine, formatLogTail, isTerminal } from "./useJob";

describe("formatLogLine", () => {
  it("drops progress, which drives the bar rather than the console", () => {
    expect(formatLogLine('{"ev":"progress","fraction":0.5,"message":"400 of 800"}')).toBeNull();
  });

  it("renders the events worth reading", () => {
    expect(formatLogLine('{"ev":"log","level":"warning","message":"13 samples differ"}')).toBe(
      "[warning] 13 samples differ",
    );
    expect(formatLogLine('{"ev":"metric","name":"loss","value":0.5}')).toBe("[metric] loss = 0.5");
    expect(formatLogLine('{"ev":"error","type":"RuntimeError","message":"boom"}')).toBe(
      "[error] RuntimeError: boom",
    );
  });

  it("shows anything that is not an event as it was written", () => {
    // Library chatter and native crash messages are exactly what a post-mortem needs.
    expect(formatLogLine("libc++abi: terminating")).toBe("libc++abi: terminating");
    expect(formatLogLine("{not json")).toBe("{not json");
    expect(formatLogLine("   ")).toBeNull();
  });
});

describe("formatLogTail", () => {
  it("keeps a scan's console to what happened, not how far it got", () => {
    const raw = [
      '{"ev":"log","level":"info","message":"scanning /data"}',
      ...Array.from({ length: 500 }, (_, index) => `{"ev":"progress","fraction":${index / 500}}`),
      '{"ev":"log","level":"warning","message":"13 samples differ"}',
      '{"ev":"done","result":{"samples":189}}',
    ];

    expect(formatLogTail(raw)).toEqual([
      "[info] scanning /data",
      "[warning] 13 samples differ",
      "[done]",
    ]);
  });
});

describe("isTerminal", () => {
  it("knows which states a job never leaves", () => {
    expect(isTerminal("succeeded")).toBe(true);
    expect(isTerminal("failed")).toBe(true);
    expect(isTerminal("cancelled")).toBe(true);
    expect(isTerminal("queued")).toBe(false);
    expect(isTerminal("running")).toBe(false);
    expect(isTerminal(undefined)).toBe(false);
  });
});
