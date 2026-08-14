import { describe, expect, it } from "vitest";

import { paneFrame, resolveReference } from "./annotationPanes";

describe("resolveReference", () => {
  it("has nothing to show beside a single-channel sample", () => {
    expect(resolveReference(1, 0, null)).toBeNull();
    expect(resolveReference(0, -1, null)).toBeNull();
  });

  it("takes the next channel when nothing was chosen", () => {
    expect(resolveReference(3, 0, null)).toBe(1);
    expect(resolveReference(3, 1, null)).toBe(2);
    expect(resolveReference(3, 2, null)).toBe(0);
  });

  it("keeps the reader's choice while it is still another channel", () => {
    expect(resolveReference(3, 0, 2)).toBe(2);
    expect(resolveReference(3, 1, 2)).toBe(2);
  });

  it("moves off the preference rather than showing one channel twice", () => {
    // The two-channel case, which is the one that actually bites: with `dark` preferred,
    // switching the main pane to `dark` would otherwise put `dark` beside `dark`.
    expect(resolveReference(2, 1, 1)).toBe(0);
    expect(resolveReference(2, 0, 0)).toBe(1);
  });

  it("ignores a preference that no longer names a channel", () => {
    // A sample with fewer channels than the last one; the stored index is simply gone.
    expect(resolveReference(2, 0, 5)).toBe(1);
    expect(resolveReference(2, 0, -1)).toBe(1);
  });
});

describe("paneFrame", () => {
  it("is shared by the channels of one part when they share a source frame", () => {
    expect(paneFrame(7, 1280, 1024)).toBe(paneFrame(7, 1280, 1024));
  });

  it("changes with the part, and with a channel that is not the same size", () => {
    expect(paneFrame(7, 1280, 1024)).not.toBe(paneFrame(8, 1280, 1024));
    expect(paneFrame(7, 1280, 1024)).not.toBe(paneFrame(7, 1280, 1000));
  });
});
