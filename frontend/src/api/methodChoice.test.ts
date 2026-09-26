import { describe, expect, it } from "vitest";

import type { ModelDescription } from "./client";
import { defaultMethod, orderMethods, schemaForTask } from "./methodChoice";

function method(
  key: string,
  status: ModelDescription["status"],
  recommended_for: ModelDescription["recommended_for"] = [],
): ModelDescription {
  return { key, status, recommended_for } as ModelDescription;
}

describe("the order methods are offered in", () => {
  const floor = method("floor", "floor");
  const trial = method("trial", "experimental");
  const kept = method("kept", "supported");
  const chosen = method("chosen", "supported", ["anomaly"]);

  it("puts the task's default first, the floor last, and keeps registry order within a rank", () => {
    const order = orderMethods([floor, trial, kept, chosen], "anomaly").map((entry) => entry.key);
    expect(order).toEqual(["chosen", "kept", "trial", "floor"]);
  });

  it("recommends for its own task only", () => {
    expect(defaultMethod([floor, trial, chosen], "object_detection")?.key).toBe("chosen");
    expect(defaultMethod([floor, trial], "object_detection")?.key).toBe("trial");
    expect(defaultMethod([], "anomaly")).toBeUndefined();
  });
});

describe("options scoped to a task", () => {
  const schema = {
    type: "object",
    properties: {
      aggregation: { type: "string" },
      pixel_bins: { type: "integer", "x-tasks": ["anomaly"] },
    },
  };

  it("keeps an unmarked field for every task and a marked one for its own", () => {
    expect(Object.keys(schemaForTask(schema, "anomaly").properties)).toEqual([
      "aggregation",
      "pixel_bins",
    ]);
    expect(Object.keys(schemaForTask(schema, "semantic_segmentation").properties)).toEqual([
      "aggregation",
    ]);
  });
});
