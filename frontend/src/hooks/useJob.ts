/**
 * Following one job: **snapshot, then subscribe** (§6).
 *
 * `GET /api/jobs/{id}` returns the current status, progress and a log tail; only then is
 * the WebSocket opened. First load, a reconnection after the socket drops, and a screen
 * opened halfway through a running job are therefore the same code path — there is no
 * replay buffer in the server and no missed-event reconciliation here.
 *
 * The stream ends with a parent-generated `end` frame carrying the terminal status.
 * Without it a client cannot tell a finished job from a dropped socket, so that frame is
 * what triggers the final refetch rather than a timer.
 */

import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useRef, useState } from "react";

import { websocketUrl } from "../api/baseUrl";
import { api, unwrap } from "../api/client";
import type { JobDetail, JobMetrics, JobStatus } from "../api/client";
import { queryKeys } from "../api/queryKeys";

/** A worker frame, or the parent's closing frame. */
export interface JobEvent {
  ev: "progress" | "log" | "metric" | "done" | "error" | "end";
  fraction?: number;
  message?: string;
  level?: string;
  name?: string;
  value?: number;
  /** Present on `metric`. The backend has always sent it; this client used to drop it. */
  step?: number;
  status?: JobStatus;
  type?: string;
}

/** One scalar series as a chart wants it: named, ordered, with its own truncation note. */
export interface Series {
  name: string;
  points: { step: number; value: number }[];
  /** How many points the run emitted, which exceeds `points.length` when downsampled. */
  total: number;
  dropped: number;
}

export const TERMINAL: readonly JobStatus[] = ["succeeded", "failed", "cancelled"];

/** How long a dropped socket waits before it is reopened, doubling up to a ceiling. */
export function reconnectDelay(retries: number): number {
  return Math.min(1000 * 2 ** retries, 15000);
}

export function isTerminal(status: JobStatus | undefined): boolean {
  return status !== undefined && TERMINAL.includes(status);
}

export async function fetchJob(jobId: number): Promise<JobDetail> {
  return unwrap(
    await api.GET("/api/jobs/{job_id}", { params: { path: { job_id: jobId } } }),
    "the job",
  );
}

export async function fetchJobMetrics(jobId: number): Promise<JobMetrics> {
  return unwrap(
    await api.GET("/api/jobs/{job_id}/metrics", { params: { path: { job_id: jobId } } }),
    "the job's metrics",
  );
}

export interface UseJobResult {
  job: JobDetail | undefined;
  /** Log lines from the snapshot, then everything the socket has delivered since. */
  lines: string[];
  /** Scalar series from the snapshot, then everything the socket has delivered since. */
  series: Series[];
  error: Error | null;
  isPending: boolean;
}

/**
 * A console that started from a snapshot and has been following the socket since.
 *
 * `baseline` is frozen the moment the socket is opened. It has to be: every event also
 * refreshes the snapshot, whose `log_tail` grows to include the very lines the socket
 * just delivered, so reading the tail live would print each event twice.
 */
interface Console {
  baseline: string[];
  streamed: string[];
}

/**
 * The scalar series, under the same freeze rule as the console.
 *
 * `baseline` is the metric snapshot taken when the socket opened; `streamed` is what has
 * arrived since. They are kept apart for the reason the log tail is: every event
 * invalidates the snapshot, and a snapshot re-read live would already contain the points
 * the socket just delivered, drawing each one twice.
 */
interface Metrics {
  baseline: JobMetrics;
  streamed: Map<string, { step: number; value: number }[]>;
}

/**
 * Whether **both** snapshots have settled, which is when the socket may be opened.
 *
 * "Snapshot, then subscribe" (§6) is now two requests, not one, and gating on the job's
 * alone is a bug that looks like working software. The job snapshot is the smaller
 * request and lands first; subscribing there opens the socket while the metric history is
 * still on the wire, so the baseline freezes empty and the chart redraws from the moment
 * the page opened. It still *moves*, which is exactly why it reads as fine — and it means
 * a reload during a two-hour run silently throws the run away.
 *
 * Settled, not successful: a metrics read that failed must still give the console a live
 * socket. The chart is then empty, which is the honest outcome for that case.
 */
export function bothSnapshotsSettled(hasJob: boolean, metricsPending: boolean): boolean {
  return hasJob && !metricsPending;
}

export interface UseJobOptions {
  /**
   * The experiment this job belongs to, when it has one.
   *
   * A job's own row is not the only thing its terminal frame changes: a `train` run flips
   * `Experiment.status`, an `infer` run fills `scored_subsets` and writes the metric sets,
   * and both append to the job history the screen is listing. None of that was refetched,
   * so a run that finished while its own screen was open left the Benchmark tab disabled
   * and the run list showing `queued` until the user navigated away and came back.
   */
  experimentId?: number | undefined;
}

/**
 * What one event makes stale.
 *
 * A named function rather than two conditions inline, for the reason `bothSnapshotsSettled`
 * is one: getting this wrong produces a screen that looks like it is working. Until M4.6
 * every frame invalidated the job row and nothing else, so a run that finished while its
 * own screen was open left `ExperimentDetail` untouched — `scored_subsets` stayed empty and
 * the Benchmark tab stayed disabled, the job list still read `queued`, and navigating away
 * and back "fixed" it. Nothing was broken except what the client had bothered to re-ask.
 *
 * `progress` is deliberately narrow. It arrives up to four times a second (the queue
 * persists it at a 0.25 s interval) and refetching the whole experiment at that rate would
 * be a poll wearing an event's clothes; the job row alone carries what the bar draws.
 *
 * It is narrow in the *other* axis too, and that is why `exact` exists. `queryKeys` is
 * hierarchical and `invalidateQueries` matches by prefix, so `["jobs", id]` also names
 * `["jobs", id, "metrics"]` — and that snapshot is read by parsing the entire job log
 * file. Four times a second, for the length of a training run. It also contradicts the
 * freeze rule this file is built on: the metric baseline must not be re-read live, because
 * it comes back already holding the points the socket just delivered (handbook jobs.md).
 *
 * A terminal frame keeps the prefix behaviour. There the metric series *should* be
 * re-read, and the experiment key is meant to cover its results, threshold report, curves,
 * per-sample images and diagnostics index.
 */
export function invalidatedBy(
  ev: JobEvent["ev"],
  jobId: number,
  experimentId: number | undefined,
): readonly { queryKey: readonly unknown[]; exact: boolean }[] {
  if (ev === "progress") return [{ queryKey: queryKeys.job(jobId), exact: true }];
  if (ev !== "done" && ev !== "error" && ev !== "end") return [];
  const job = [{ queryKey: queryKeys.job(jobId), exact: false }];
  if (experimentId === undefined) return job;
  return [
    ...job,
    { queryKey: queryKeys.experiment(experimentId), exact: false },
    { queryKey: queryKeys.experimentLists(), exact: false },
  ];
}

export function useJob(jobId: number | undefined, options: UseJobOptions = {}): UseJobResult {
  const { experimentId } = options;
  const queryClient = useQueryClient();
  const [live, setLive] = useState<Console | null>(null);
  const [metrics, setMetrics] = useState<Metrics | null>(null);
  /** Bumped to reopen a dropped socket; see the reconnection comment in the effect. */
  const [attempt, setAttempt] = useState(0);
  /** Consecutive failed reconnections, for the backoff. Reset by a successful open. */
  const retries = useRef(0);

  const snapshot = useQuery({
    queryKey: queryKeys.job(jobId ?? -1),
    queryFn: () => fetchJob(jobId as number),
    enabled: jobId !== undefined,
    retry: false,
  });

  // The history behind the chart. Without it, reloading during a two-hour training run
  // would throw the run away — `log_tail` is 200 raw lines of a stream that also carries
  // progress and library output, which for a long run is its final seconds (handbook jobs.md).
  const storedMetrics = useQuery({
    queryKey: queryKeys.jobMetrics(jobId ?? -1),
    queryFn: () => fetchJobMetrics(jobId as number),
    enabled: jobId !== undefined,
    retry: false,
  });

  const status = snapshot.data?.status;
  const finished = isTerminal(status);
  const snapshotted = bothSnapshotsSettled(
    snapshot.data !== undefined,
    storedMetrics.isPending,
  );

  // Read inside the effect without making the effect depend on every refetch.
  const tailRef = useRef<string[]>([]);
  tailRef.current = snapshot.data?.log_tail ?? [];
  const metricsRef = useRef<JobMetrics | undefined>(undefined);
  metricsRef.current = storedMetrics.data;

  useEffect(() => {
    // **Snapshot, then subscribe** (§6), in that order. Waiting for the snapshot is what
    // gives the console a defined starting point; opening the socket first would leave
    // the tail and the stream overlapping by an unknown amount.
    //
    // Nothing to follow, or the job was already over when we looked: a socket for a
    // finished job would connect and immediately close.
    if (jobId === undefined || !snapshotted || finished) return;

    setLive({ baseline: formatLogTail(tailRef.current), streamed: [] });
    setMetrics({
      baseline: metricsRef.current ?? { job_id: jobId, series: [] },
      streamed: new Map(),
    });
    const socket = new WebSocket(websocketUrl(`/ws/jobs/${jobId}`));
    /** Whether the stream ended on purpose, which is the one close not worth reopening. */
    let ended = false;

    const invalidate = (ev: JobEvent["ev"]) => {
      for (const { queryKey, exact } of invalidatedBy(ev, jobId, experimentId)) {
        void queryClient.invalidateQueries({ queryKey, exact });
      }
    };

    socket.onmessage = (event: MessageEvent<string>) => {
      let parsed: JobEvent;
      try {
        parsed = JSON.parse(event.data) as JobEvent;
      } catch {
        // The stream is JSON lines by contract; anything else is not ours to interpret.
        return;
      }

      if (parsed.ev === "end") {
        // Refetch rather than trusting the frame: the snapshot carries the result payload
        // and the terminal status, which the stream does not.
        ended = true;
        invalidate(parsed.ev);
        return;
      }

      if (parsed.ev === "metric" && typeof parsed.name === "string") {
        const point = { step: parsed.step ?? 0, value: parsed.value ?? 0 };
        const name = parsed.name;
        setMetrics((current) => {
          if (current === null) return current;
          const streamed = new Map(current.streamed);
          streamed.set(name, [...(streamed.get(name) ?? []), point]);
          return { ...current, streamed };
        });
      }

      const line = describe(parsed);
      if (line !== null) {
        setLive((current) =>
          current === null ? current : { ...current, streamed: [...current.streamed, line] },
        );
      }
      invalidate(parsed.ev);
    };

    /*
     * A socket that drops is reopened, because until now one that dropped stayed dropped.
     *
     * There is no `end` frame in that case — the queue only sends one when the job really
     * finished — so the console simply stopped, at whatever step it had reached, next to a
     * badge still reading `running`. Navigating away and back fixed it, which is the same
     * signature as a screen that never asked again. A sidecar restart, a proxy idle
     * timeout and a subscriber-buffer overflow all produce it.
     *
     * Recovery is the cycle this file already describes rather than a second mechanism:
     * bumping `attempt` re-runs the effect, which takes a fresh snapshot and subscribes to
     * it. First load, a late join and a reconnection stay one code path, and the snapshot
     * is what closes the gap the drop left.
     *
     * The delay backs off, because the sidecar being *down* looks exactly like one dropped
     * socket and a fixed retry would be a one-second poll for as long as the tab is open.
     * The count lives in a ref rather than in `attempt`: resetting state on a successful
     * open would re-run this effect and close the socket it had just opened.
     */
    let unmounting = false;
    let timer: ReturnType<typeof setTimeout> | undefined;
    socket.onopen = () => {
      retries.current = 0;
    };
    socket.onclose = () => {
      // `end` already said the run is over, and the refetch it triggered will set
      // `finished` and keep this effect from running again. Reopening in the meantime
      // would connect to a finished job just to be closed again.
      if (unmounting || ended) return;
      // Refresh the snapshot first, so the effect's next run reads a tail that covers the
      // gap rather than the one that was current when the socket died.
      invalidate("end");
      const delay = reconnectDelay(retries.current);
      retries.current += 1;
      timer = setTimeout(() => setAttempt((count) => count + 1), delay);
    };

    return () => {
      unmounting = true;
      if (timer !== undefined) clearTimeout(timer);
      socket.close();
    };
  }, [jobId, snapshotted, finished, queryClient, experimentId, attempt]);

  return {
    job: snapshot.data,
    // A job that was already finished when this screen opened has no stream to follow, so
    // its console is the log file — the raw worker output, every byte of it, which is what
    // makes a failed run diagnosable afterwards (ADR-0009).
    lines: live === null
      ? formatLogTail(snapshot.data?.log_tail ?? [])
      : [...live.baseline, ...live.streamed],
    series:
      metrics === null ? toSeries(storedMetrics.data) : mergeSeries(metrics.baseline, metrics.streamed),
    error: snapshot.error,
    isPending: snapshot.isPending && jobId !== undefined,
  };
}

/** The stored series alone — what a finished run, or one not yet subscribed to, shows. */
export function toSeries(stored: JobMetrics | undefined): Series[] {
  return (stored?.series ?? []).map((entry) => ({
    name: entry.name,
    points: entry.points.map((point) => ({ step: point.step ?? 0, value: point.value })),
    total: entry.total,
    dropped: entry.dropped,
  }));
}

/**
 * The frozen snapshot, plus whatever the socket has delivered since.
 *
 * A name that only ever appeared on the socket still gets a series, which is what makes a
 * chart move within the first seconds of a run rather than after the first refetch.
 */
export function mergeSeries(
  baseline: JobMetrics,
  streamed: Map<string, { step: number; value: number }[]>,
): Series[] {
  const merged = new Map<string, Series>();
  for (const entry of baseline.series) {
    merged.set(entry.name, {
      name: entry.name,
      points: entry.points.map((point) => ({ step: point.step ?? 0, value: point.value })),
      total: entry.total,
      dropped: entry.dropped,
    });
  }
  for (const [name, points] of streamed) {
    const existing = merged.get(name);
    if (existing) {
      existing.points = [...existing.points, ...points];
      existing.total += points.length;
    } else {
      merged.set(name, { name, points: [...points], total: points.length, dropped: 0 });
    }
  }
  return [...merged.values()].sort((left, right) => left.name.localeCompare(right.name));
}

export function formatLogTail(raw: string[]): string[] {
  const formatted: string[] = [];
  for (const line of raw) {
    const rendered = formatLogLine(line);
    if (rendered !== null) formatted.push(rendered);
  }
  return formatted;
}

/**
 * One raw log line as the console should show it.
 *
 * A line that is not an event is shown as-is: library chatter and native crash messages
 * are exactly what someone reading a failed job's log needs to see.
 */
export function formatLogLine(line: string): string | null {
  const trimmed = line.trim();
  if (trimmed === "") return null;
  if (!trimmed.startsWith("{")) return trimmed;
  try {
    return describe(JSON.parse(trimmed) as JobEvent);
  } catch {
    return trimmed;
  }
}

/** Render one event as a console line, or `null` if it is not worth showing. */
function describe(event: JobEvent): string | null {
  switch (event.ev) {
    case "log":
      return `[${event.level ?? "info"}] ${event.message ?? ""}`;
    case "metric":
      return `[metric] ${event.name ?? "?"} = ${event.value ?? "?"}`;
    case "error":
      return `[error] ${event.type ?? "Error"}: ${event.message ?? ""}`;
    case "done":
      return "[done]";
    default:
      // Progress drives the bar, not the console; echoing it would bury the log.
      return null;
  }
}
