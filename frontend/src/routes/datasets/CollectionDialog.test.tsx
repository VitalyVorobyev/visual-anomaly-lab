/**
 * What the dialog writes, which is the part that can be wrong invisibly.
 *
 * `movesFor` is the whole decision — who is touched, with what, and who is deliberately
 * left alone — so it is tested directly rather than through five renders. The rendered
 * cases cover the two things a reader can act on and a unit test cannot see: that the
 * button says whether it is creating or joining, and that it refuses to write nothing.
 */

import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import type { DatasetSummary } from "../../api/client";
import { withProviders } from "../../test-harness";
import { CollectionDialog, movesFor } from "./CollectionDialog";

function dataset(over: Partial<DatasetSummary> & { id: number; name: string }): DatasetSummary {
  return {
    root_path: `/roots/${over.name}`,
    adapter: "csv_table",
    created_at: "2026-08-13T09:00:00.000Z",
    notes: null,
    samples: 1100,
    images: 1100,
    label_counts: { normal: 1000, defect: 100, unlabeled: 0 },
    collection: null,
    description: null,
    cover_image_id: 42,
    ...over,
  } as DatasetSummary;
}

const CANDLE = dataset({ id: 1, name: "candle", collection: "VisA" });
const CASHEW = dataset({ id: 2, name: "cashew", collection: "VisA" });
const BRIGHT = dataset({ id: 3, name: "CanEndsBright" });
const DARK = dataset({ id: 4, name: "CanEndsDark" });
const ALL = [CANDLE, CASHEW, BRIGHT, DARK];

function renderDialog(collection?: string) {
  return render(
    withProviders(<CollectionDialog datasets={ALL} collection={collection} onClose={() => {}} />),
  );
}

const row = (name: string) => screen.getByRole("checkbox", { name: new RegExp(name) });
const save = () => screen.getByRole("button", { name: /Create|Save|Add to/ });

describe("what a collection edit writes", () => {
  it("files the ticked datasets and touches nobody else", () => {
    expect(
      movesFor(ALL, { target: "CanEnds", members: new Set([3, 4]), initial: new Set() }),
    ).toEqual([
      { datasetId: 3, collection: "CanEnds" },
      { datasetId: 4, collection: "CanEnds" },
    ]);
  });

  it("renames a collection by rewriting every member", () => {
    expect(
      movesFor(ALL, { target: "VisA 1.0", members: new Set([1, 2]), initial: new Set([1, 2]) }),
    ).toEqual([
      { datasetId: 1, collection: "VisA 1.0" },
      { datasetId: 2, collection: "VisA 1.0" },
    ]);
  });

  it("clears the override of a dataset dropped from the group", () => {
    expect(
      movesFor(ALL, { target: "VisA", members: new Set([1]), initial: new Set([1, 2]) }),
    ).toEqual([{ datasetId: 2, collection: "" }]);
  });

  it("writes nothing when a group is saved unchanged", () => {
    expect(
      movesFor(ALL, { target: "VisA", members: new Set([1, 2]), initial: new Set([1, 2]) }),
    ).toEqual([]);
  });
});

describe("the collection dialog", () => {
  it("refuses to create a collection that would hold nothing", () => {
    renderDialog();

    // A name alone is not a collection: it is stored on the datasets filed under it, so a
    // group of nobody would not survive the next reload.
    fireEvent.change(screen.getByRole("textbox"), { target: { value: "CanEnds" } });
    expect(save().hasAttribute("disabled")).toBe(true);

    fireEvent.click(row("CanEndsBright"));
    expect(save().hasAttribute("disabled")).toBe(false);
  });

  it("says it is joining when the name is one that already exists", () => {
    renderDialog();

    fireEvent.change(screen.getByRole("textbox"), { target: { value: "CanEnds" } });
    expect(save().textContent).toContain("Create");

    fireEvent.change(screen.getByRole("textbox"), { target: { value: "VisA" } });
    expect(save().textContent).toContain("Add to VisA");
  });

  it("opens an existing collection with its members already ticked", () => {
    renderDialog("VisA");

    expect(row("candle").getAttribute("aria-checked")).toBe("true");
    expect(row("CanEndsBright").getAttribute("aria-checked")).toBe("false");
    // Nothing has changed yet, so there is nothing to save.
    expect(save().hasAttribute("disabled")).toBe(true);
  });
});
