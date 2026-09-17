/**
 * Two decisions that are made in a URL and are wrong in a way nothing on screen reveals.
 *
 * A tier chosen from the wrong number still draws a picture — just a blurred or a needlessly
 * enormous one. A mask fetched in the wrong frame still draws an outline — just not around
 * the thing it is claiming to outline.
 */

import { describe, expect, it } from "vitest";

import { apiBaseUrl } from "./client";
import { maskUrl, tierFor } from "./imageUrl";

describe("tierFor", () => {
  it("asks for the lossless tier only once a screen pixel is worth less than an image pixel", () => {
    // `scale` is CSS pixels per image pixel, so the cut is a statement about the picture
    // rather than about the column it happens to be in.
    expect(tierFor({ scale: 0.49, tx: 0, ty: 0 })).toBe("preview");
    expect(tierFor({ scale: 0.5, tx: 0, ty: 0 })).toBe("preview");
    expect(tierFor({ scale: 0.51, tx: 0, ty: 0 })).toBe("full");
    expect(tierFor({ scale: 1, tx: 0, ty: 0 })).toBe("full");
    expect(tierFor({ scale: 8, tx: 0, ty: 0 })).toBe("full");
  });

  it("treats the unresolved opening view as the cheap tier", () => {
    // `null` is ImageStage's "open at a sensible view", which is not known until the
    // viewport has been measured. Guessing `full` there would move a megabyte per pane on
    // every arrival, for one frame, to be replaced.
    expect(tierFor(null)).toBe("preview");
  });
});

describe("maskUrl", () => {
  it("is the source frame with no experiment named", () => {
    expect(maskUrl(42)).toBe(`${apiBaseUrl}/api/images/42/mask`);
  });

  it("names the run whose pinned build defines the prepared frame", () => {
    const url = new URL(maskUrl(42, { experimentId: 7 }), "http://placeholder");

    expect(url.pathname).toBe("/api/images/42/mask");
    expect(url.searchParams.get("frame")).toBe("prepared");
    expect(url.searchParams.get("experiment_id")).toBe("7");
  });
});
