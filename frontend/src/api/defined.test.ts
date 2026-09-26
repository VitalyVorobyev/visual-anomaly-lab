import { describe, expect, it } from "vitest";

import { defined } from "./defined";

describe("defined", () => {
  it("leaves out undefined values and keeps every other falsy one", () => {
    expect(defined({ a: undefined, b: null, c: 0, d: "", e: false, f: 1 })).toEqual({
      b: null,
      c: 0,
      d: "",
      e: false,
      f: 1,
    });
    expect("a" in defined({ a: undefined })).toBe(false);
  });
});
