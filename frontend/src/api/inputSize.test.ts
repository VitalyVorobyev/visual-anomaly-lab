import { describe, expect, it } from "vitest";

import type { RegionProfileRevision } from "./client";
import { fullFrameProfile, inputSizeState, snapToMultiple } from "./inputSize";

const DINO = { model_type: "dino_memory", width: 448, height: 448, multiple: 14 };

describe("a run's input size", () => {
  it("empty is the method's own, and says so", () => {
    const state = inputSizeState("", "", DINO);
    expect(state.named).toBeUndefined();
    expect(state.error).toBeUndefined();
    expect(state.caption).toBe("448 × 448 · from dino_memory");
  });

  it("typed, sends both sides and says the rule it snaps to", () => {
    const state = inputSizeState("392", "392", DINO);
    expect(state.named).toEqual({ width: 392, height: 392 });
    expect(state.caption).toContain("Snaps to 14");
  });

  it("refuses one side alone, a size out of range and a size the patch does not divide", () => {
    expect(inputSizeState("448", "", DINO).error).toMatch(/both sides/);
    expect(inputSizeState("4", "4", DINO).error).toMatch(/between 8 and 2048/);
    expect(inputSizeState("450", "448", DINO).error).toMatch(/multiple of 14/);
  });

  it("snaps to the nearest multiple within bounds, and leaves empty alone", () => {
    expect(snapToMultiple("450", 14)).toBe("448");
    expect(snapToMultiple("3", 14)).toBe("14");
    expect(snapToMultiple("5000", 16)).toBe("2048");
    expect(snapToMultiple("", 14)).toBe("");
    expect(snapToMultiple("257", 1)).toBe("257");
  });
});

describe("the default region profile", () => {
  const profile = (id: number, name: string, revision_no: number, extractor_type = "identity") =>
    ({ id, name, revision_no, extractor_type }) as RegionProfileRevision;

  it("is the newest Full frame identity revision", () => {
    expect(
      fullFrameProfile([
        profile(1, "Full frame", 1),
        profile(2, "Dominant object", 1, "foreground_threshold"),
        profile(3, "Full frame", 2),
      ])?.id,
    ).toBe(3);
    expect(fullFrameProfile([profile(2, "Dominant object", 1, "foreground_threshold")])).toBe(
      undefined,
    );
  });
});
