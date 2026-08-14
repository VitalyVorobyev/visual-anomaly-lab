/**
 * Turning a job's raw log into something a person reads.
 *
 * The log file is the raw worker stream by design (ADR-0009) — every byte, so a failed
 * run is diagnosable afterwards. A scan of nine hundred images writes nine hundred
 * progress events into it, and showing those verbatim buries the four lines that matter.
 */

import { describe, expect, it } from "vitest";

import type { JobMetrics } from "../api/client";
import { queryKeys } from "../api/queryKeys";
import {
  bothSnapshotsSettled,
  formatLogLine,
  formatLogTail,
  invalidatedBy,
  isTerminal,
  mergeSeries,
  reconnectDelay,
  toSeries,
} from "./useJob";

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

describe("bothSnapshotsSettled", () => {
  it("waits for the metric history, not just for the job", () => {
    /**
     * Found by reloading during a real training run. The job snapshot is the smaller
     * request and lands first; subscribing there froze the metric baseline empty, and the
     * chart redrew from the moment the page opened. It still moved, which is exactly why
     * it read as working — and it meant a reload threw the whole run away.
     */
    expect(bothSnapshotsSettled(true, true)).toBe(false);
    expect(bothSnapshotsSettled(true, false)).toBe(true);
  });

  it("never subscribes before the job snapshot", () => {
    expect(bothSnapshotsSettled(false, false)).toBe(false);
    expect(bothSnapshotsSettled(false, true)).toBe(false);
  });

  it("settles rather than succeeds, so a failed metrics read still gets a console", () => {
    // `isPending` goes false on error too — the console is live and the chart is empty,
    // which is the honest outcome rather than a screen that never connects.
    expect(bothSnapshotsSettled(true, false)).toBe(true);
  });
});

describe("invalidatedBy", () => {
  /**
   * The regression guard for M4.6's opening bug. Every frame used to invalidate the job row
   * and nothing else, so a run that finished while its own screen was open never refreshed
   * the experiment: `scored_subsets` stayed empty and the Benchmark tab stayed disabled,
   * the run list still read `queued`, and navigating away and back appeared to fix it.
   */
  it("refreshes the experiment when the run ends, not only the job", () => {
    for (const ev of ["done", "error", "end"] as const) {
      expect(invalidatedBy(ev, 10, 2)).toEqual([
        { queryKey: queryKeys.job(10), exact: false },
        { queryKey: queryKeys.experiment(2), exact: false },
        { queryKey: queryKeys.experimentLists(), exact: false },
      ]);
    }
  });

  it("leaves the experiment alone on progress", () => {
    // Progress lands up to four times a second; refetching the experiment at that rate
    // would be a poll wearing an event's clothes.
    expect(invalidatedBy("progress", 10, 2)).toEqual([
      { queryKey: queryKeys.job(10), exact: true },
    ]);
  });

  it("does not drag the metric snapshot along on progress", () => {
    /**
     * `["jobs", 10]` is a *prefix* of `["jobs", 10, "metrics"]`, and `invalidateQueries`
     * matches by prefix. So progress used to refetch the metric history too — four times a
     * second, each one re-reading and re-parsing the whole job log file, and each one
     * violating the freeze rule the snapshot depends on (ADR-0020).
     */
    const [progress] = invalidatedBy("progress", 10, 2);
    expect(progress?.exact).toBe(true);
    expect(queryKeys.jobMetrics(10).slice(0, 2)).toEqual([...queryKeys.job(10)]);
  });

  it("refreshes only the job for a run that belongs to no experiment", () => {
    // Import, verify and prewarm jobs have no experiment to refresh.
    expect(invalidatedBy("done", 4, undefined)).toEqual([
      { queryKey: queryKeys.job(4), exact: false },
    ]);
  });

  it("refreshes nothing for frames that change no server state", () => {
    expect(invalidatedBy("log", 10, 2)).toEqual([]);
    expect(invalidatedBy("metric", 10, 2)).toEqual([]);
  });
});

describe("reconnectDelay", () => {
  /**
   * A dropped socket used to stay dropped: no `onclose`, no `end` frame to explain it, so
   * the console froze at whatever step it had reached beside a badge still reading
   * `running`, and only a remount recovered.
   *
   * It backs off because the sidecar being *down* is indistinguishable from one dropped
   * socket, and a fixed retry would be a one-second poll for as long as the tab is open.
   */
  it("reopens promptly, then backs off to a ceiling", () => {
    expect(reconnectDelay(0)).toBe(1000);
    expect(reconnectDelay(1)).toBe(2000);
    expect(reconnectDelay(4)).toBe(15000);
    expect(reconnectDelay(40)).toBe(15000);
  });
});

/**
 * The metric store follows the console's freeze rule (§6). These pin the part that goes
 * wrong quietly: a snapshot re-read live already contains the points the socket delivered,
 * so the two halves are merged rather than concatenated blindly.
 */
function metrics(series: JobMetrics["series"]): JobMetrics {
  return { job_id: 1, series } as JobMetrics;
}

describe("toSeries", () => {
  it("is empty before the snapshot lands, rather than undefined", () => {
    expect(toSeries(undefined)).toEqual([]);
  });

  it("carries the run's truncation note through to the chart", () => {
    const [first] = toSeries(
      metrics([{ name: "loss_st", points: [{ step: 0, value: 1 }], total: 5000, dropped: 4999 }]),
    );
    expect(first?.total).toBe(5000);
    expect(first?.dropped).toBe(4999);
  });

  it("gives a stepless metric a step rather than an undefined x", () => {
    const [first] = toSeries(
      metrics([{ name: "reference_images", points: [{ step: null, value: 8 }], total: 1, dropped: 0 }]),
    );
    expect(first?.points).toEqual([{ step: 0, value: 8 }]);
  });
});

describe("mergeSeries", () => {
  const baseline = metrics([
    { name: "loss_st", points: [{ step: 0, value: 1 }], total: 1, dropped: 0 },
  ]);

  it("appends what the socket delivered to what the snapshot held", () => {
    const merged = mergeSeries(baseline, new Map([["loss_st", [{ step: 20, value: 0.5 }]]]));

    expect(merged).toHaveLength(1);
    expect(merged[0]?.points).toEqual([
      { step: 0, value: 1 },
      { step: 20, value: 0.5 },
    ]);
    expect(merged[0]?.total).toBe(2);
  });

  it("charts a name that has only ever appeared on the socket", () => {
    // The first seconds of a run: the metric exists on the wire before any snapshot has
    // been taken that contains it.
    const merged = mergeSeries(baseline, new Map([["loss_ae", [{ step: 0, value: 9 }]]]));
    expect(merged.map((entry) => entry.name)).toEqual(["loss_ae", "loss_st"]);
  });

  it("leaves the snapshot alone when nothing has streamed yet", () => {
    expect(mergeSeries(baseline, new Map())).toEqual([
      { name: "loss_st", points: [{ step: 0, value: 1 }], total: 1, dropped: 0 },
    ]);
  });

  it("orders series by name so the legend does not reshuffle between renders", () => {
    const merged = mergeSeries(
      metrics([
        { name: "loss_stae", points: [], total: 0, dropped: 0 },
        { name: "learning_rate", points: [], total: 0, dropped: 0 },
        { name: "loss_ae", points: [], total: 0, dropped: 0 },
      ]),
      new Map(),
    );
    expect(merged.map((entry) => entry.name)).toEqual(["learning_rate", "loss_ae", "loss_stae"]);
  });
});
