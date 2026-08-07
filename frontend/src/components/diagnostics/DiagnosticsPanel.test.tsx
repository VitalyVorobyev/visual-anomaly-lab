/**
 * The ADR-0018 guarantee, pinned.
 *
 * These are the tests M6 relies on: if the panel ever starts branching on `key` or on a
 * method name, one of them fails here rather than silently drawing the wrong picture for
 * `efficientad_custom` months later.
 */

import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import type { DiagnosticEntry } from "../../api/client";
import { DiagnosticsPanel } from "./DiagnosticsPanel";

function entry(overrides: Partial<DiagnosticEntry>): DiagnosticEntry {
  return {
    key: "k",
    title: "Title",
    kind: "map",
    scope: "model",
    image_id: null,
    path: "k.npy",
    payload: null,
    shape: null,
    description: null,
    ...overrides,
  } as DiagnosticEntry;
}

describe("DiagnosticsPanel", () => {
  it("renders the same key differently when the kind differs", () => {
    /**
     * `pixel_reference` emits `reference_median` as an `image` on a three-channel config
     * and as a `map` on a greyscale one, from one line of Python. Anything keyed on the
     * name would draw one of these two wrong.
     */
    const { container } = render(
      <DiagnosticsPanel
        experimentId={1}
        entries={[
          entry({ key: "reference_median", title: "As image", kind: "image" }),
          entry({ key: "reference_median", title: "As map", kind: "map", image_id: 3 }),
        ]}
      />,
    );

    const images = Array.from(container.querySelectorAll("img"));
    expect(images).toHaveLength(2);
    expect(images[0]?.getAttribute("src")).not.toContain("image_id");
    expect(images[1]?.getAttribute("src")).toContain("image_id=3");
  });

  it("draws a kind it has never heard of as a named placeholder, not a crash", () => {
    render(
      <DiagnosticsPanel
        experimentId={1}
        entries={[entry({ key: "future", title: "From M6", kind: "volume" as never })]}
      />,
    );

    expect(screen.getByText("volume")).toBeTruthy();
    expect(screen.getByText("From M6")).toBeTruthy();
  });

  it("renders a graph payload as boxes with the shapes the model recorded", () => {
    render(
      <DiagnosticsPanel
        experimentId={1}
        entries={[
          entry({
            key: "architecture",
            title: "Model architecture",
            kind: "graph",
            path: null,
            payload: {
              nodes: [
                {
                  id: "teacher",
                  label: "teacher",
                  type: "PDN",
                  parameters: 1234567,
                  input_shape: [1, 3, 256, 256],
                  output_shape: [1, 384, 64, 64],
                },
                { id: "student", label: "student", type: "PDN", parameters: 500 },
              ],
              edges: [{ source: "teacher", target: "student", label: "distillation" }],
              total_parameters: 1235067,
            },
          }),
        ]}
      />,
    );

    expect(screen.getByText("(1, 3, 256, 256)")).toBeTruthy();
    expect(screen.getByText("(1, 384, 64, 64)")).toBeTruthy();
    expect(screen.getByText("1.23 M")).toBeTruthy();
    expect(screen.getByText("distillation")).toBeTruthy();
  });

  it("orders graph nodes by the recorded edges rather than by array position", () => {
    render(
      <DiagnosticsPanel
        experimentId={1}
        entries={[
          entry({
            key: "architecture",
            title: "Arch",
            kind: "graph",
            path: null,
            payload: {
              nodes: [{ id: "b", label: "second" }, { id: "a", label: "first" }],
              edges: [{ source: "a", target: "b" }],
            },
          }),
        ]}
      />,
    );

    const labels = screen.getAllByText(/first|second/).map((node) => node.textContent);
    expect(labels).toEqual(["first", "second"]);
  });

  it("renders a table payload as rows", () => {
    render(
      <DiagnosticsPanel
        experimentId={1}
        entries={[
          entry({
            key: "score_normalization",
            title: "Score normalization",
            kind: "table",
            path: null,
            payload: {
              columns: ["quantity", "value"],
              rows: [
                ["qa_st", "0.001"],
                ["fitted on", "90 held-out normals"],
              ],
            },
          }),
        ]}
      />,
    );

    expect(screen.getByText("qa_st")).toBeTruthy();
    expect(screen.getByText("90 held-out normals")).toBeTruthy();
  });

  it("shows the plugin's own description verbatim, since the contract is weakly typed", () => {
    render(
      <DiagnosticsPanel
        experimentId={1}
        entries={[entry({ description: "Where the student failed to reproduce the teacher." })]}
      />,
    );

    expect(screen.getByText("Where the student failed to reproduce the teacher.")).toBeTruthy();
  });

  it("renders a grid as one request per cell, capped with the cap disclosed", () => {
    const { container } = render(
      <DiagnosticsPanel
        experimentId={1}
        entries={[
          entry({ key: "teacher_features_grid", kind: "grid", shape: [64, 32, 32] }),
        ]}
      />,
    );

    expect(container.querySelectorAll("img")).toHaveLength(16);
    expect(screen.getByText("Showing 16 of 64 — show all")).toBeTruthy();
  });

  it("renders an empty index as nothing rather than as an error", () => {
    const { container } = render(<DiagnosticsPanel experimentId={1} entries={[]} />);
    expect(container.querySelectorAll("img")).toHaveLength(0);
  });
});
