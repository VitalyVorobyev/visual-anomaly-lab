import { afterEach, describe, expect, it } from "vitest";

import { shellStartupError } from "./shell";

afterEach(() => {
  delete window.__ANOMALY_LAB__;
});

describe("shellStartupError", () => {
  it("reads the failure the shell injects when the backend never started", () => {
    window.__ANOMALY_LAB__ = {
      startupError: { message: "`uv` was not found.", detail: "Searched:\n  /usr/bin/uv" },
    };

    expect(shellStartupError()).toEqual({
      message: "`uv` was not found.",
      detail: "Searched:\n  /usr/bin/uv",
    });
  });

  it("is null for a shell that started its backend — the ordinary case", () => {
    window.__ANOMALY_LAB__ = { apiBaseUrl: "http://127.0.0.1:54321" };

    expect(shellStartupError()).toBeNull();
  });

  it("is null in a plain browser, where nothing is injected at all", () => {
    expect(shellStartupError()).toBeNull();
  });

  it("ignores a malformed injection rather than painting an empty panel", () => {
    // @ts-expect-error deliberately the wrong shape, which is what a stale shell would send.
    window.__ANOMALY_LAB__ = { startupError: { detail: "no message" } };

    expect(shellStartupError()).toBeNull();
  });
});
