/**
 * The catalogue's two claims: it groups, and it is quiet.
 *
 * Grouping is the one that would fail silently — the effective `collection` is computed on
 * the server from a stored override *or* the reference pack a dataset came from, and
 * nothing on screen distinguishes the two, so a regression that lost the derivation would
 * look exactly like a user who had not filed anything yet.
 */

import { fireEvent, render, screen, within } from "@testing-library/react";
import { MemoryRouter } from "react-router";
import { beforeEach, describe, expect, it } from "vitest";

import type { DatasetSummary } from "../../api/client";
import { queryKeys } from "../../api/queryKeys";
import { withProviders } from "../../test-harness";
import { DatasetsRoute } from "./DatasetsRoute";
import { groupDatasets } from "./grouping";

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

const VISA = ["candle", "capsules", "cashew"].map((name, index) =>
  dataset({ id: index + 1, name, collection: "VisA", description: `The ${name} class.` }),
);
const MINE = dataset({ id: 90, name: "CanEndsBright" });

function renderCatalogue(datasets: DatasetSummary[]) {
  return render(
    withProviders(
      <MemoryRouter initialEntries={["/"]}>
        <DatasetsRoute />
      </MemoryRouter>,
      [
        [queryKeys.datasets(), datasets],
        // Everything registered, so the strip has nothing to offer and renders nothing.
        [
          queryKeys.referencePacks(),
          { packs: [], available_datasets: 0, pending_datasets: 0 },
        ],
      ],
    ),
  );
}

/**
 * The fold toggle, addressed through its heading rather than by name: the edit action
 * beside it also carries the collection's name, and both are buttons.
 */
function foldToggle(name: string): HTMLElement {
  return screen.getByRole("heading", { level: 2, name }).closest("button")!;
}

describe("the datasets catalogue", () => {
  // The fold is persisted on purpose, so it has to be reset between tests that share it.
  beforeEach(() => globalThis.localStorage.clear());

  it("gathers a reference pack's classes under one heading", () => {
    renderCatalogue([...VISA, MINE]);

    const heading = foldToggle("VisA");
    expect(heading.textContent).toContain("3");

    // The group's own cards, not the whole page: `CanEndsBright` is ungrouped and must
    // not be swept in with them.
    const group = heading.closest("section")!;
    const names = within(group)
      .getAllByRole("heading", { level: 3 })
      .map((node) => node.textContent);
    expect(names).toEqual(["candle", "capsules", "cashew"]);
  });

  it("gives an ungrouped dataset no heading at all", () => {
    const { container } = renderCatalogue([MINE]);

    expect(screen.getByRole("heading", { level: 3, name: "CanEndsBright" })).toBeTruthy();
    // No group chrome for a group of everything: the collapse toggle is the only thing
    // that carries `aria-expanded`, so its absence is the absence of a heading row.
    expect(container.querySelectorAll("[aria-expanded]")).toHaveLength(0);
    expect(screen.queryByRole("heading", { level: 2 })).toBeNull();
  });

  it("folds a collection away and keeps its heading", () => {
    renderCatalogue([...VISA, MINE]);

    const heading = foldToggle("VisA");
    expect(heading.getAttribute("aria-expanded")).toBe("true");

    fireEvent.click(heading);

    expect(heading.getAttribute("aria-expanded")).toBe("false");
    expect(screen.queryByRole("heading", { level: 3, name: "candle" })).toBeNull();
    // The ungrouped card is not inside the fold and must survive it.
    expect(screen.getByRole("heading", { level: 3, name: "CanEndsBright" })).toBeTruthy();
  });

  it("says what a dataset is, not how many rows it has", () => {
    renderCatalogue([VISA[0]!]);

    const card = screen.getByRole("heading", { level: 3, name: "candle" }).closest("li")!;

    expect(card.textContent).toContain("The candle class.");
    // Everything the card used to spend three lines on. The counts live in the band, one
    // click away, where they are being used rather than skimmed.
    expect(card.textContent).not.toContain("samples");
    expect(card.textContent).not.toContain("csv_table");
    expect(card.textContent).not.toContain("/roots/");
    expect(card.textContent).not.toContain("1000");
  });

  it("offers to make a collection, but only once there is something to file", () => {
    renderCatalogue([MINE]);
    expect(screen.getByRole("button", { name: "New collection" })).toBeTruthy();

    fireEvent.click(screen.getByRole("button", { name: "New collection" }));
    expect(screen.getByRole("dialog").textContent).toContain("Name the group");
  });

  it("hides the collection action while the catalogue is empty", () => {
    renderCatalogue([]);
    expect(screen.queryByRole("button", { name: "New collection" })).toBeNull();
  });

  it("edits a collection from its heading without folding it", () => {
    renderCatalogue([...VISA, MINE]);
    const toggle = foldToggle("VisA");

    fireEvent.click(screen.getByRole("button", { name: "Edit collection VisA" }));

    expect(screen.getByRole("dialog").textContent).toContain("Edit VisA");
    // The action sits beside the toggle rather than inside it, so reaching it does not
    // collapse the group on the way.
    expect(toggle.getAttribute("aria-expanded")).toBe("true");
  });

  it("shows a cover, and a placeholder for a dataset that has no images", () => {
    renderCatalogue([VISA[0]!, dataset({ id: 91, name: "empty", cover_image_id: null })]);

    const covered = screen.getByRole("heading", { level: 3, name: "candle" }).closest("li")!;
    expect(covered.querySelector("img")?.getAttribute("src")).toContain("/api/images/42/thumb");

    const bare = screen.getByRole("heading", { level: 3, name: "empty" }).closest("li")!;
    expect(bare.querySelector("img")).toBeNull();
  });
});

describe("grouping", () => {
  it("sorts collections by name and leaves datasets in the order the API gave them", () => {
    const { ungrouped, groups } = groupDatasets([
      dataset({ id: 1, name: "zebra", collection: "Zoo" }),
      dataset({ id: 2, name: "loose" }),
      dataset({ id: 3, name: "candle", collection: "VisA" }),
      dataset({ id: 4, name: "aardvark", collection: "Zoo" }),
    ]);

    expect(ungrouped.map((item) => item.name)).toEqual(["loose"]);
    expect(groups.map(([name]) => name)).toEqual(["VisA", "Zoo"]);
    // Import order inside a group, so a registration appends rather than reshuffles.
    expect(groups[1]![1].map((item) => item.name)).toEqual(["zebra", "aardvark"]);
  });

  it("treats a blank collection as ungrouped", () => {
    const { ungrouped, groups } = groupDatasets([dataset({ id: 1, name: "x", collection: "  " })]);

    expect(ungrouped).toHaveLength(1);
    expect(groups).toHaveLength(0);
  });
});
