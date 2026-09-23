import { describe, expect, it } from "vitest";

import { formatBytes } from "./format";

describe("formatting a byte count", () => {
  it("keeps small counts exact", () => {
    expect(formatBytes(0)).toBe("0 B");
    expect(formatBytes(1023)).toBe("1023 B");
  });

  it("names binary units, because it divides by 1024", () => {
    expect(formatBytes(1024)).toBe("1.0 KiB");
    expect(formatBytes(64.7 * 1024 ** 2)).toBe("64.7 MiB");
    expect(formatBytes(3 * 1024 ** 3)).toBe("3.0 GiB");
  });

  it("drops the decimal once it stops carrying information", () => {
    expect(formatBytes(512 * 1024 ** 2)).toBe("512 MiB");
  });
});
