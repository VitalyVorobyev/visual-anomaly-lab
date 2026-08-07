import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { SchemaForm } from "./SchemaForm";
import { describeFields, initialValues } from "../api/schemaForm";

/**
 * Adapter options and model hyperparameters have opposite shapes, and the same rule has
 * to serve both. An adapter's "which folder holds the good images" has no sensible
 * default, so it must be asked. A model's `smoothing_sigma` has one, so it must not be
 * the first thing on the screen. What follows pins the two ends of that.
 */

const ADAPTER_SCHEMA = {
  properties: {
    normal_dirs: { type: "array", items: { type: "string" }, default: [], description: "Where." },
    extensions: {
      type: "array",
      items: { type: "string" },
      default: [".png", ".jpg"],
      description: "Which files.",
    },
  },
  required: [],
};

const MODEL_SCHEMA = {
  properties: {
    smoothing_sigma: { type: "number", default: 4.0, description: "Blur, in pixels." },
    score_percentile: { type: "number", default: 99.5, description: "Which percentile." },
  },
  required: [],
};

function renderSchema(schema: Parameters<typeof describeFields>[0]) {
  const fields = describeFields(schema);
  render(<SchemaForm fields={fields} values={initialValues(fields)} onChange={() => {}} />);
}

describe("the generated options form", () => {
  it("asks the question that has no default, and folds away the one that has", () => {
    renderSchema(ADAPTER_SCHEMA);

    expect(screen.getByLabelText("Normal dirs")).toBeTruthy();
    expect(screen.getByText("Advanced")).toBeTruthy();
  });

  it("shows every option outright when none of them has to be decided", () => {
    // The alternative is a box containing nothing but a closed disclosure, which reads as
    // a dead end rather than as "there is nothing here you must decide". A disclosure you
    // always have to open is a disclosure that should not exist.
    renderSchema(MODEL_SCHEMA);

    expect(document.querySelector("details")).toBeNull();
    expect(screen.getByLabelText("Smoothing sigma")).toBeTruthy();
    expect(screen.getByLabelText("Score percentile")).toBeTruthy();
  });

  it("says so plainly when there is nothing to configure at all", () => {
    renderSchema({ properties: {}, required: [] });
    expect(screen.getByText(/takes no options/)).toBeTruthy();
  });

  it("shows the backend's default as a placeholder rather than as the value", () => {
    // An empty box means "unset, the backend decides"; pre-filling it would claim the
    // default as the operator's own choice and send it back on every request.
    renderSchema(MODEL_SCHEMA);

    const input = screen.getByLabelText("Smoothing sigma") as HTMLInputElement;
    expect(input.value).toBe("");
    expect(input.placeholder).toBe("4");
  });
});
