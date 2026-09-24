import { describe, expect, it } from "vitest";

import { ApiError } from "./client";
import { shouldRetry } from "./retry";

describe("the read retry policy", () => {
  it("never asks again after a request error", () => {
    expect(shouldRetry(0, new ApiError("no such experiment", 404))).toBe(false);
    expect(shouldRetry(0, new ApiError("refused", 422))).toBe(false);
  });

  it("asks once more after a server or network failure, and then shows it", () => {
    expect(shouldRetry(0, new ApiError("boom", 500))).toBe(true);
    expect(shouldRetry(0, new TypeError("Failed to fetch"))).toBe(true);
    expect(shouldRetry(1, new ApiError("boom", 503))).toBe(false);
  });
});
