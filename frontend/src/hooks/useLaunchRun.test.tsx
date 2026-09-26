/**
 * One press, three requests, in order: the split if it is still a preset, the experiment,
 * and Train & score. Answered through the API client's own middleware (see
 * `useAnnotations.test.ts` for why not `globalThis.fetch`), so the real mutation runs.
 */

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, renderHook, waitFor } from "@testing-library/react";
import { createElement, type ReactNode } from "react";
import { afterEach, describe, expect, it } from "vitest";

import { api } from "../api/client";
import { useLaunchRun, type LaunchRequest } from "./useLaunchRun";

interface Call {
  method: string;
  path: string;
  body: Record<string, unknown> | null;
}

let release: (() => void) | undefined;
afterEach(() => release?.());

function answer(failOn?: string): Call[] {
  const calls: Call[] = [];
  const middleware = {
    async onRequest({ request }: { request: Request }) {
      const path = new URL(request.url).pathname;
      const text = await request.clone().text();
      calls.push({ method: request.method, path, body: text ? JSON.parse(text) : null });
      if (path === failOn) {
        return new Response(JSON.stringify({ detail: "refused" }), {
          status: 422,
          headers: { "content-type": "application/json" },
        });
      }
      const body = path === "/api/splits" ? { id: 21 } : path === "/api/experiments" ? { id: 34 } : { id: 55 };
      return new Response(JSON.stringify(body), {
        status: 200,
        headers: { "content-type": "application/json" },
      });
    },
  };
  api.use(middleware);
  release = () => api.eject(middleware);
  return calls;
}

function wrapper({ children }: { children: ReactNode }) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false, gcTime: 0 } } });
  return createElement(QueryClientProvider, { client }, children);
}

const EXPERIMENT: LaunchRequest["experiment"] = {
  name: "run",
  dataset_id: 7,
  model_type: "pixel_reference",
  task: "anomaly",
  target_label: null,
  config: {},
  preprocessing: {},
  evaluation: {},
  channels: [],
};

describe("launching a run", () => {
  it("draws a preset, creates the experiment on it and queues Train & score", async () => {
    const calls = answer();
    const created: number[] = [];
    const { result } = renderHook(() => useLaunchRun(), { wrapper });

    await act(async () => {
      await result.current.mutateAsync({
        split: { params: { strategy: "normal_only_train" } as never },
        experiment: EXPERIMENT,
        run: true,
        onSplit: (id) => created.push(id),
      });
    });

    expect(calls.map((call) => `${call.method} ${call.path}`)).toEqual([
      "POST /api/splits",
      "POST /api/experiments",
      "POST /api/experiments/34/train",
    ]);
    expect(calls[1]?.body?.["split_id"]).toBe(21);
    expect(calls[2]?.body?.["then_score"]).toBe(true);
    expect(created).toEqual([21]);
  });

  it("reuses an existing split, and stops at a draft when asked to", async () => {
    const calls = answer();
    const { result } = renderHook(() => useLaunchRun(), { wrapper });

    await act(async () => {
      await result.current.mutateAsync({ split: { id: 3 }, experiment: EXPERIMENT, run: false });
    });

    expect(calls.map((call) => `${call.method} ${call.path}`)).toEqual(["POST /api/experiments"]);
    expect(calls[0]?.body?.["split_id"]).toBe(3);
  });

  it("says which stage failed, after handing back the split it made", async () => {
    answer("/api/experiments");
    const created: number[] = [];
    const { result } = renderHook(() => useLaunchRun(), { wrapper });

    act(() => {
      result.current.mutate({
        split: { params: { strategy: "normal_only_train" } as never },
        experiment: EXPERIMENT,
        run: true,
        onSplit: (id) => created.push(id),
      });
    });

    await waitFor(() => expect(result.current.error).not.toBeNull());
    expect(created).toEqual([21]);
  });
});
