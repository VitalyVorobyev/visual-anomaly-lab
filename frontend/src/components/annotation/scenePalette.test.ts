/**
 * The Konva scene has to follow the theme like everything else.
 *
 * It cannot use Tailwind classes, so for a long time it simply did not follow it at all: fifteen
 * literals copied out of the *dark* palette, which meant the light theme rendered a dark-theme
 * canvas over a light page. Reading the tokens is what puts this surface back under ADR-0021.
 */

import { describe, expect, it } from "vitest";

import { withAlpha } from "./scenePalette";

describe("withAlpha", () => {
  it("appends the alpha byte a canvas fill needs", () => {
    expect(withAlpha("#3bc9db", 1)).toBe("#3bc9dbff");
    expect(withAlpha("#3bc9db", 0)).toBe("#3bc9db00");
    expect(withAlpha("#3bc9db", 0.5)).toBe("#3bc9db80");
  });

  it("clamps rather than emitting a colour string the canvas would ignore", () => {
    expect(withAlpha("#000000", 2)).toBe("#000000ff");
    expect(withAlpha("#000000", -1)).toBe("#00000000");
  });

  it("leaves a non-hex token alone", () => {
    // A themed value is free to become `oklch(...)`, and blindly appending two hex digits to it
    // produces something the canvas discards silently — an invisible shape rather than a
    // visibly wrong one, which is the worse failure.
    expect(withAlpha("oklch(0.7 0.1 200)", 0.5)).toBe("oklch(0.7 0.1 200)");
    expect(withAlpha("#abc", 0.5)).toBe("#abc");
  });
});
