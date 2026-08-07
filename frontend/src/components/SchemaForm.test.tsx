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
    expect(screen.getByText(/1 more, already set to a working default/)).toBeTruthy();
  });

  it("opens the disclosure when every option already has a default", () => {
    // Otherwise a model's configuration panel is an empty box with a closed disclosure
    // in it, which reads as a dead end rather than as "there is nothing you must decide".
    renderSchema(MODEL_SCHEMA);

    const disclosure = document.querySelector("details");
    expect(disclosure?.open).toBe(true);
    expect(screen.getByText(/2 options, every one already set to a working default/)).toBeTruthy();
    expect(screen.getByLabelText("Smoothing sigma")).toBeTruthy();
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
