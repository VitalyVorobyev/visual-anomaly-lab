/**
 * Which request the first save makes.
 *
 * This is the part of the draft lifecycle that is easy to get subtly wrong and impossible to
 * see from the screen: a draft must be *created* by the first save and *updated* by every save
 * after it, and the create has to carry `If-None-Match: *` so it is create-only. Without that
 * header the route would be an upsert, and an upsert is how a second window holding the same
 * seed silently overwrites the first — it would be handed a currently-valid token for a
 * document it never read.
 */

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { renderHook, waitFor } from "@testing-library/react";
import { createElement, type ReactNode } from "react";
import { describe, expect, it } from "vitest";

import { api } from "../api/client";
import type { AnnotationDocumentInput } from "../api/client";
import { useSaveDraft } from "./useAnnotations";

const DOCUMENT: AnnotationDocumentInput = {
  schema_version: 1,
  image_width: 8,
  image_height: 8,
  base: "empty",
  shapes: [],
};

interface Call {
  url: string;
  method: string;
  headers: Record<string, string>;
}

/**
 * Intercept through the client's own middleware rather than by replacing `globalThis.fetch`.
 *
 * `openapi-fetch` captures the global at client-creation time, so a later reassignment is
 * simply never seen — and the global it captured is `test-setup.ts`'s never-settling stub.
 * Returning a `Response` from `onRequest` short-circuits before any network call.
 */
function captureRequests(status: number): { calls: Call[]; release: () => void } {
  const calls: Call[] = [];
  const middleware = {
    onRequest({ request }: { request: Request }) {
      calls.push({
        url: request.url,
        method: request.method,
        // Lower-cased on the way in: this DOM implementation preserves the casing the caller
        // used, so a lookup by the canonical name would silently miss.
        headers: Object.fromEntries(
          [...request.headers.entries()].map(([name, value]) => [name.toLowerCase(), value]),
        ),
      });
      return new Response(
        JSON.stringify({ image_id: 7, document: DOCUMENT, version: 1, updated_at: "now" }),
        {
          status,
          headers: { "content-type": "application/json", etag: '"annotation-draft-7-v1"' },
        },
      );
    },
  };
  api.use(middleware);
  return { calls, release: () => api.eject(middleware) };
}

function wrapper({ children }: { children: ReactNode }) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false, gcTime: 0 } } });
  return createElement(QueryClientProvider, { client }, children);
}

describe("useSaveDraft", () => {
  it("creates with a create-only precondition when nothing is persisted yet", async () => {
    const { calls, release } = captureRequests(201);
    const { result } = renderHook(() => useSaveDraft({ scope: "image", imageId: 7 }), { wrapper });

    result.current.mutate({ document: DOCUMENT, etag: null });
    await waitFor(() => expect(calls).toHaveLength(1));
    release();

    expect(calls[0]?.method).toBe("POST");
    expect(calls[0]?.url).toContain("/api/images/7/annotations/draft");
    expect(calls[0]?.headers["if-none-match"]).toBe("*");
    expect(calls[0]?.headers["if-match"]).toBeUndefined();
  });

  it("updates against the token it holds once the draft exists", async () => {
    const { calls, release } = captureRequests(200);
    const { result } = renderHook(() => useSaveDraft({ scope: "image", imageId: 7 }), { wrapper });

    result.current.mutate({ document: DOCUMENT, etag: '"annotation-draft-7-v1"' });
    await waitFor(() => expect(calls).toHaveLength(1));
    release();

    expect(calls[0]?.method).toBe("PUT");
    expect(calls[0]?.headers["if-match"]).toBe('"annotation-draft-7-v1"');
    expect(calls[0]?.headers["if-none-match"]).toBeUndefined();
  });

  it("uses the sample route and its own namespace under sample scope", async () => {
    const { calls, release } = captureRequests(201);
    const { result } = renderHook(() => useSaveDraft({ scope: "sample", sampleId: 4 }), {
      wrapper,
    });

    result.current.mutate({ document: DOCUMENT, etag: null });
    await waitFor(() => expect(calls).toHaveLength(1));
    release();

    expect(calls[0]?.url).toContain("/api/samples/4/annotations/draft");
    expect(calls[0]?.headers["if-none-match"]).toBe("*");
  });
});
