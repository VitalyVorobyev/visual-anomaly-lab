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

export function useJob(jobId: number | undefined): UseJobResult {
  const queryClient = useQueryClient();
  const [streamed, setStreamed] = useState<string[]>([]);
  const socketRef = useRef<WebSocket | null>(null);

  const snapshot = useQuery({
    queryKey: queryKeys.job(jobId ?? -1),
    queryFn: () => fetchJob(jobId as number),
    enabled: jobId !== undefined,
    retry: false,
  });

  const status = snapshot.data?.status;
  const finished = isTerminal(status);

  useEffect(() => {
    // Nothing to follow, or the job was already over when we looked. Opening a socket for
    // a finished job would just add a connection that immediately closes.
    if (jobId === undefined || finished) return;

    setStreamed([]);
    const socket = new WebSocket(websocketUrl(`/ws/jobs/${jobId}`));
    socketRef.current = socket;

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
        // and the final log tail, which the stream does not.
        void queryClient.invalidateQueries({ queryKey: queryKeys.job(jobId) });
        return;
      }

      const line = describe(parsed);
      if (line !== null) setStreamed((previous) => [...previous, line]);
      if (parsed.ev === "progress" || parsed.ev === "done" || parsed.ev === "error") {
        void queryClient.invalidateQueries({ queryKey: queryKeys.job(jobId) });
      }
    };

    return () => {
      socketRef.current = null;
      socket.close();
    };
  }, [jobId, finished, queryClient]);

  return {
    job: snapshot.data,
    lines: [...(snapshot.data?.log_tail ?? []), ...streamed],
    error: snapshot.error,
    isPending: snapshot.isPending && jobId !== undefined,
  };
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
