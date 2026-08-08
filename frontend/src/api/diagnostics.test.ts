import { describe, expect, it } from "vitest";

import type { DiagnosticEntry, DiagnosticIndex } from "./client";
import {
  budgetNote,
  diagnosedImageIds,
  diagnosticPayloadUrl,
  gridFrameCount,
  imageScoped,
  isArrayKind,
  isOnDemand,
  modelScoped,
  ofKinds,
  missingNote,
  onDemandNote,
} from "./diagnostics";

function entry(overrides: Partial<DiagnosticEntry> = {}): DiagnosticEntry {
  return {
    key: "m",
    title: "M",
    kind: "map",
    scope: "model",
    image_id: null,
    path: "m.npy",
    payload: null,
    shape: null,
    description: null,
    ...overrides,
  } as DiagnosticEntry;
}

function index(overrides: Partial<DiagnosticIndex> = {}): DiagnosticIndex {
  return {
    version: 1,
    entries: [],
    ranges: {},
    image_budget: null,
    truncated_images: 0,
    ...overrides,
  } as DiagnosticIndex;
}

describe("diagnosticPayloadUrl", () => {
  it("names a key rather than a path, so no request can name a file", () => {
    const url = diagnosticPayloadUrl(7, entry({ key: "teacher_magnitude" }));
    expect(url).toContain("/api/experiments/7/diagnostics/payload");
    expect(url).toContain("key=teacher_magnitude");
    expect(url).not.toContain(".npy");
  });

  it("scopes to an image only when the entry is scoped to one", () => {
    expect(diagnosticPayloadUrl(1, entry())).not.toContain("image_id");
    expect(diagnosticPayloadUrl(1, entry({ scope: "image", image_id: 42 }))).toContain(
      "image_id=42",
    );
  });

  it("carries a grid frame, and omits the default", () => {
    expect(diagnosticPayloadUrl(1, entry({ kind: "grid" }), 0)).not.toContain("frame");
    expect(diagnosticPayloadUrl(1, entry({ kind: "grid" }), 3)).toContain("frame=3");
  });

  it("escapes a key rather than splicing it into the query", () => {
    expect(diagnosticPayloadUrl(1, entry({ key: "a b&c=d" }))).toContain("key=a+b%26c%3Dd");
  });
});

describe("isArrayKind", () => {
  it("separates the fetched payloads from the inline ones by kind, never by key", () => {
    expect(isArrayKind(entry({ kind: "map" }))).toBe(true);
    expect(isArrayKind(entry({ kind: "image" }))).toBe(true);
    expect(isArrayKind(entry({ kind: "grid" }))).toBe(true);
    expect(isArrayKind(entry({ kind: "graph" }))).toBe(false);
    expect(isArrayKind(entry({ kind: "table" }))).toBe(false);
  });
});

describe("scoping", () => {
  const populated = index({
    entries: [
      entry({ key: "arch", kind: "graph", scope: "model" }),
      entry({ key: "err", scope: "image", image_id: 5 }),
      entry({ key: "other", scope: "image", image_id: 9 }),
      entry({ key: "err2", scope: "image", image_id: 5 }),
    ],
  });

  it("splits model-scoped from image-scoped", () => {
    expect(modelScoped(populated).map((found) => found.key)).toEqual(["arch"]);
    expect(imageScoped(populated, 5).map((found) => found.key)).toEqual(["err", "err2"]);
    expect(imageScoped(populated, 404)).toEqual([]);
  });

  it("lists the images that got diagnostics, deduplicated and ordered", () => {
    expect(diagnosedImageIds(populated)).toEqual([5, 9]);
  });

  it("treats a missing index as empty rather than throwing", () => {
    expect(modelScoped(undefined)).toEqual([]);
    expect(imageScoped(undefined, 1)).toEqual([]);
    expect(diagnosedImageIds(undefined)).toEqual([]);
  });
});

describe("ofKinds", () => {
  it("narrows by kind", () => {
    const entries = [
      entry({ key: "a", kind: "map" }),
      entry({ key: "b", kind: "grid" }),
      entry({ key: "c", kind: "graph" }),
    ];
    expect(ofKinds(entries, ["map", "grid"]).map((found) => found.key)).toEqual(["a", "b"]);
  });
});

describe("gridFrameCount", () => {
  it("reads the cell count off the recorded shape", () => {
    expect(gridFrameCount(entry({ kind: "grid", shape: [16, 32, 32] }))).toEqual(16);
  });

  it("is zero when the shape was never recorded", () => {
    expect(gridFrameCount(entry({ kind: "grid", shape: null }))).toEqual(0);
  });
});

describe("budgetNote", () => {
  it("says nothing when nothing was dropped", () => {
    expect(budgetNote(index({ image_budget: 12, truncated_images: 0 }))).toBeNull();
    expect(budgetNote(index({ image_budget: null, truncated_images: 0 }))).toBeNull();
  });

  it("reports what was dropped, because a blank panel would read as 'none exist'", () => {
    const note = budgetNote(
      index({
        image_budget: 12,
        truncated_images: 88,
        entries: [entry({ scope: "image", image_id: 1 })],
      }),
    );
    expect(note).toContain("88");
    expect(note).toContain("12");
  });
});


describe("origin", () => {
  const sample = index({
    image_budget: 64,
    truncated_images: 200,
    entries: [
      entry({ scope: "image", image_id: 1, origin: "run" }),
      entry({ scope: "image", image_id: 2, origin: "run" }),
      entry({ scope: "image", image_id: 90, origin: "on_demand" }),
      entry({ scope: "image", image_id: 91, origin: "on_demand" }),
    ],
  });

  it("counts only the run's own images against the run's budget", () => {
    // The failure this prevents is arithmetic that reads as a broken cap: browse enough
    // images and an unfiltered count climbs past the budget it claims to be under.
    expect(budgetNote(sample)).toContain("recorded for 2 image(s)");
    expect(budgetNote(sample)).toContain("budget of 64");
  });

  it("reports what was asked for as its own fact", () => {
    expect(onDemandNote(sample)).toContain("2 image(s) diagnosed on demand");
    expect(onDemandNote(index())).toBeNull();
  });

  it("narrows the image list to one producer, and to both by default", () => {
    expect(diagnosedImageIds(sample, "run")).toEqual([1, 2]);
    expect(diagnosedImageIds(sample, "on_demand")).toEqual([90, 91]);
    expect(diagnosedImageIds(sample)).toEqual([1, 2, 90, 91]);
  });

  it("treats an entry written before origins existed as the run's", () => {
    // Additive, so the index version did not move; a missing origin has to read as `run`
    // or an old index would badge every pane as something somebody asked for.
    expect(isOnDemand(entry({ scope: "image", image_id: 3, origin: "run" }))).toBe(false);
    expect(isOnDemand(entry({ scope: "image", image_id: 3, origin: "on_demand" }))).toBe(true);
  });
});


describe("missingNote", () => {
  it("prefers the budget, which is the most specific explanation available", () => {
    const withBudget = index({
      image_budget: 64,
      truncated_images: 300,
      entries: [entry({ scope: "image", image_id: 1, origin: "run" })],
    });
    expect(missingNote(withBudget)).toContain("budget of 64");
  });

  it("says the run sampled other images when there is no budget recorded", () => {
    // The case seen in the running application: the last flush was a training run, so the
    // index carries no budget, and the panel claimed the method records nothing — beside a
    // button offering to record something.
    const noBudget = index({
      entries: [
        entry({ scope: "image", image_id: 1, origin: "run" }),
        entry({ scope: "image", image_id: 2, origin: "run" }),
      ],
    });
    expect(missingNote(noBudget)).toContain("2 other image(s)");
  });

  it("falls back to the method only when nothing image-scoped exists at all", () => {
    expect(missingNote(index())).toContain("ctx.emit_diagnostic");
  });
});
