import { render, screen, fireEvent } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { ModuleTree, hasHierarchy } from "./ModuleTree";
import type { ModuleNode } from "./ModuleTree";

function node(id: string, overrides: Partial<ModuleNode> = {}): ModuleNode {
  const depth = id.split(".").length - 1;
  return {
    id,
    label: id.split(".").at(-1),
    type: "Conv2d",
    parameters: 100,
    parameters_own: 100,
    parent: depth === 0 ? null : id.split(".").slice(0, -1).join("."),
    depth,
    order: 0,
    executed: true,
    calls: 1,
    leaf: true,
    input_shape: [1, 3, 8, 8],
    output_shape: [1, 8, 4, 4],
    ...overrides,
  };
}

const TREE: ModuleNode[] = [
  node("teacher", { leaf: false, parameters: 300, parameters_own: 0, type: "PDN" }),
  node("teacher.conv1", { order: 0 }),
  node("teacher.conv2", { order: 1 }),
  node("student", { leaf: false, parameters: 100, parameters_own: 0, type: "PDN" }),
  node("student.conv1", { order: 0 }),
];

describe("hasHierarchy", () => {
  it("is false for the flat payload M4 recorded", () => {
    // Which is what makes the card view still render an older index unchanged.
    expect(hasHierarchy([{ id: "teacher" }, { id: "student" }])).toBe(false);
  });

  it("is true once nodes name a parent", () => {
    expect(hasHierarchy(TREE)).toBe(true);
  });
});

describe("ModuleTree", () => {
  it("opens the branches and their layers, not every leaf", () => {
    render(<ModuleTree nodes={TREE} />);

    expect(screen.getByText("teacher")).toBeTruthy();
    expect(screen.getByText("conv2")).toBeTruthy();
  });

  it("collapses a subtree behind its caret", () => {
    render(<ModuleTree nodes={TREE} />);
    fireEvent.click(screen.getByLabelText("Collapse teacher"));

    expect(screen.queryByText("conv2")).toBeNull();
    // The other branch is untouched.
    expect(screen.getByText("student")).toBeTruthy();
  });

  it("keeps a match's ancestors when filtering", () => {
    /**
     * A matched leaf shown at the root with no path to it reads as a different model
     * rather than as a filtered one.
     */
    render(<ModuleTree nodes={TREE} />);
    fireEvent.change(screen.getByLabelText("Filter modules"), {
      target: { value: "teacher.conv2" },
    });

    expect(screen.getByText("conv2")).toBeTruthy();
    expect(screen.getByText("teacher")).toBeTruthy();
    expect(screen.queryByText("student")).toBeNull();
  });

  it("filters on the type as well as the name", () => {
    // "show me every AvgPool2d" is the question a filter over a backbone gets asked.
    render(
      <ModuleTree
        nodes={[
          node("teacher", { leaf: false, type: "PDN" }),
          node("teacher.pool", { type: "AvgPool2d" }),
          node("teacher.conv", { type: "Conv2d" }),
        ]}
      />,
    );
    fireEvent.change(screen.getByLabelText("Filter modules"), { target: { value: "avgpool" } });

    expect(screen.getByText("pool")).toBeTruthy();
    expect(screen.queryByText("conv")).toBeNull();
  });

  it("shows both parameter counts where they differ", () => {
    render(<ModuleTree nodes={TREE} />);
    // A container declares nothing itself but holds 300 — one number alone would not add up.
    expect(screen.getByText(/300/)).toBeTruthy();
  });

  it("marks a module the pass never reached rather than blanking it", () => {
    render(
      <ModuleTree
        nodes={[
          node("teacher", { leaf: false }),
          node("teacher.unused", {
            executed: false,
            calls: 0,
            input_shape: undefined,
            output_shape: undefined,
          }),
        ]}
      />,
    );

    expect(screen.getByText("not called")).toBeTruthy();
  });

  it("reports a module that ran more than once", () => {
    render(<ModuleTree nodes={[node("block", { calls: 4 })]} />);
    expect(screen.getByText("×4")).toBeTruthy();
  });

  it("says what the payload dropped rather than implying it is whole", () => {
    render(<ModuleTree nodes={TREE} truncated={12} maxNodes={1500} />);
    expect(screen.getByText(/12 deeper module\(s\) were not recorded/)).toBeTruthy();
  });

  it("says nothing about truncation when nothing was truncated", () => {
    render(<ModuleTree nodes={TREE} truncated={0} />);
    expect(screen.queryByText(/were not recorded/)).toBeNull();
  });
});
