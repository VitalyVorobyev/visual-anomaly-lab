/**
 * Which photograph stands for a part, when a dataset has said which.
 *
 * The rule is small; what it has to guarantee is that no screen can end up showing nothing.
 * A stored channel name is validated when it is written and outlives the import that wrote
 * it, so the read side has to survive a name that no longer matches anything.
 */

import { describe, expect, it } from "vitest";

import { preferredImageIndex } from "./defaultChannel";

const IMAGES = [{ channel: "bright" }, { channel: "dark" }, { channel: "dome" }];

describe("the image a part is shown as", () => {
  it("is the first one when the dataset names no default", () => {
    expect(preferredImageIndex(IMAGES, null)).toBe(0);
    expect(preferredImageIndex(IMAGES, undefined)).toBe(0);
    expect(preferredImageIndex(IMAGES, "")).toBe(0);
  });

  it("is the named channel when the part has it", () => {
    expect(preferredImageIndex(IMAGES, "dome")).toBe(2);
    expect(preferredImageIndex(IMAGES, "bright")).toBe(0);
  });

  it("falls back to the first when the part was not photographed that way", () => {
    // The two-channel capture group in the reference data, which must not go blank.
    expect(preferredImageIndex([{ channel: "bright" }, { channel: "dark" }], "dome")).toBe(0);
  });

  it("falls back to the first when a re-import renamed the channel away", () => {
    expect(preferredImageIndex(IMAGES, "domefield")).toBe(0);
  });

  it("never answers -1, even with nothing to choose from", () => {
    expect(preferredImageIndex([], "dome")).toBe(0);
  });

  it("matches an unassigned image by nothing, rather than by null", () => {
    // An image outside the channel dictionary carries `channel: null`, and a default is a
    // non-empty name, so the two can never meet.
    expect(preferredImageIndex([{ channel: null }, { channel: "dome" }], "dome")).toBe(1);
  });
});
