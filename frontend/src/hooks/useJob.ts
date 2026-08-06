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
import type { JobDetail, JobStatus } from "../api/client";
import { queryKeys } from "../api/queryKeys";

/** A worker frame, or the parent's closing frame. */
export interface JobEvent {
  ev: "progress" | "log" | "metric" | "done" | "error" | "end";
  fraction?: number;
  message?: string;
  level?: string;
  name?: string;
  value?: number;
  status?: JobStatus;
  type?: string;
}

export const TERMINAL: readonly JobStatus[] = ["succeeded", "failed", "cancelled"];

export function isTerminal(status: JobStatus | undefined): boolean {
  return status !== undefined && TERMINAL.includes(status);
}

export async function fetchJob(jobId: number): Promise<JobDetail> {
  return unwrap(
    await api.GET("/api/jobs/{job_id}", { params: { path: { job_id: jobId } } }),
    "the job",
  );
}

export interface UseJobResult {
  job: JobDetail | undefined;
  /** Log lines from the snapshot, then everything the socket has delivered since. */
  lines: string[];
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

export function useJob(jobId: number | undefined): UseJobResult {
  const queryClient = useQueryClient();
  const [live, setLive] = useState<Console | null>(null);

  const snapshot = useQuery({
    queryKey: queryKeys.job(jobId ?? -1),
    queryFn: () => fetchJob(jobId as number),
    enabled: jobId !== undefined,
    retry: false,
  });

  const status = snapshot.data?.status;
  const finished = isTerminal(status);
  const snapshotted = snapshot.data !== undefined;

  // Read inside the effect without making the effect depend on every refetch.
  const tailRef = useRef<string[]>([]);
  tailRef.current = snapshot.data?.log_tail ?? [];

  useEffect(() => {
    // **Snapshot, then subscribe** (§6), in that order. Waiting for the snapshot is what
    // gives the console a defined starting point; opening the socket first would leave
    // the tail and the stream overlapping by an unknown amount.
    //
    // Nothing to follow, or the job was already over when we looked: a socket for a
    // finished job would connect and immediately close.
    if (jobId === undefined || !snapshotted || finished) return;

    setLive({ baseline: formatLogTail(tailRef.current), streamed: [] });
    const socket = new WebSocket(websocketUrl(`/ws/jobs/${jobId}`));

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
        void queryClient.invalidateQueries({ queryKey: queryKeys.job(jobId) });
        return;
      }

      const line = describe(parsed);
      if (line !== null) {
        setLive((current) =>
          current === null ? current : { ...current, streamed: [...current.streamed, line] },
        );
      }
      if (parsed.ev === "progress" || parsed.ev === "done" || parsed.ev === "error") {
        void queryClient.invalidateQueries({ queryKey: queryKeys.job(jobId) });
      }
    };

    return () => socket.close();
  }, [jobId, snapshotted, finished, queryClient]);

  return {
    job: snapshot.data,
    // A job that was already finished when this screen opened has no stream to follow, so
    // its console is the log file — the raw worker output, every byte of it, which is what
    // makes a failed run diagnosable afterwards (ADR-0009).
    lines: live === null
      ? formatLogTail(snapshot.data?.log_tail ?? [])
      : [...live.baseline, ...live.streamed],
    error: snapshot.error,
    isPending: snapshot.isPending && jobId !== undefined,
  };
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
