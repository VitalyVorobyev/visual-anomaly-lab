import { describe, expect, it } from "vitest";

import {
  clearGuidedRun,
  EMPTY_GUIDED_RUN,
  furthest,
  readGuidedRun,
  readStep,
  suggestedTask,
  writeGuidedRun,
  type GuidedRunState,
} from "./guidedRun";
import type { Task } from "./client";

const ALL: Task[] = ["anomaly", "few_shot_segmentation", "semantic_segmentation", "object_detection"];

function memory(): Storage {
  const store = new Map<string, string>();
  return {
    get length() {
      return store.size;
    },
    clear: () => store.clear(),
    getItem: (key) => store.get(key) ?? null,
    key: (index) => [...store.keys()][index] ?? null,
    removeItem: (key) => void store.delete(key),
    setItem: (key, value) => void store.set(key, value),
  };
}

describe("the suggested task", () => {
  it("is anomaly detection for a dataset with verdicts", () => {
    expect(suggestedTask(["labels"], null, ALL)).toBe("anomaly");
    expect(suggestedTask(["labels", "classes"], "regions", ALL)).toBe("anomaly");
  });

  it("is few-shot segmentation for classes drawn as regions", () => {
    expect(suggestedTask(["classes"], "regions", ALL.slice(1))).toBe("few_shot_segmentation");
  });

  it("is detection for classes drawn only as boxes", () => {
    expect(suggestedTask(["classes"], "boxes", ALL.slice(1))).toBe("object_detection");
  });

  it("is anomaly detection for a dataset with no truth yet, which labelling makes one", () => {
    expect(suggestedTask([], null, ALL)).toBe("anomaly");
  });

  it("is only ever a task that is offered", () => {
    expect(suggestedTask(["classes"], "boxes", ["few_shot_segmentation"])).toBe(
      "few_shot_segmentation",
    );
    expect(suggestedTask(["labels"], null, [])).toBeUndefined();
  });
});

describe("the step", () => {
  it("reads an unknown step as the first", () => {
    expect(readStep("split")).toBe("split");
    expect(readStep("nonsense")).toBe("goal");
    expect(readStep(null)).toBe("goal");
  });

  it("knows how far the reader has been", () => {
    expect(furthest("look", "method")).toBe("method");
    expect(furthest("run", "goal")).toBe("run");
  });
});

describe("the stored choices", () => {
  const STATE: GuidedRunState = {
    task: "few_shot_segmentation",
    targetClass: "screw",
    profileId: 4,
    split: { kind: "preset", key: "few_shot_1" },
    methodKey: "proto_seg",
    configValues: { shots: "3" },
    name: "kept",
    reached: "method",
    created: { key: "preset:{}", splitId: 9 },
  };

  it("round-trip through storage and clear", () => {
    const storage = memory();
    writeGuidedRun("k", STATE, storage);
    expect(readGuidedRun("k", storage)).toEqual(STATE);
    clearGuidedRun("k", storage);
    expect(readGuidedRun("k", storage)).toEqual(EMPTY_GUIDED_RUN);
  });

  it("drop anything malformed instead of trusting it", () => {
    const storage = memory();
    storage.setItem("k", "{not json");
    expect(readGuidedRun("k", storage)).toEqual(EMPTY_GUIDED_RUN);
    storage.setItem(
      "k",
      JSON.stringify({ task: "painting", split: { kind: "existing", id: "3" }, reached: "far" }),
    );
    const read = readGuidedRun("k", storage);
    expect(read.task).toBeUndefined();
    expect(read.split).toBeUndefined();
    expect(read.reached).toBe("goal");
  });
});
