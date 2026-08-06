import { describe, expect, it } from "vitest";

import { DEFAULT_API_BASE_URL, resolveApiBaseUrl, websocketUrl } from "./baseUrl";

describe("resolveApiBaseUrl", () => {
  it("prefers the URL injected by the Tauri shell", () => {
    expect(
      resolveApiBaseUrl({ injected: "http://127.0.0.1:54321", env: "http://localhost:8000" }),
    ).toBe("http://127.0.0.1:54321");
  });

  it("falls back to the build-time environment override", () => {
    expect(resolveApiBaseUrl({ env: "http://localhost:9000" })).toBe("http://localhost:9000");
  });

  it("falls back to the documented development port", () => {
    expect(resolveApiBaseUrl({})).toBe(DEFAULT_API_BASE_URL);
  });

  it("ignores an empty injected value rather than producing a relative URL", () => {
    expect(resolveApiBaseUrl({ injected: "   ", env: "http://localhost:9000" })).toBe(
      "http://localhost:9000",
    );
  });

  it("strips trailing slashes, which would otherwise double up on every path", () => {
    expect(resolveApiBaseUrl({ injected: "http://127.0.0.1:8000//" })).toBe("http://127.0.0.1:8000");
  });
});

describe("websocketUrl", () => {
  it("swaps http for ws", () => {
    expect(websocketUrl("/ws/echo", "http://127.0.0.1:8000")).toBe("ws://127.0.0.1:8000/ws/echo");
  });

  it("swaps https for wss", () => {
    expect(websocketUrl("/ws/echo", "https://example.test")).toBe("wss://example.test/ws/echo");
  });

  it("keeps the port the shell handed over", () => {
    expect(websocketUrl("/ws/echo", "http://127.0.0.1:54321")).toBe(
      "ws://127.0.0.1:54321/ws/echo",
    );
  });
});
