import { render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router";
import { describe, expect, it } from "vitest";

import { queryKeys } from "../../api/queryKeys";
import { withProviders } from "../../test-harness";
import { CLASS_PALETTE, ClassManager, classKeyFor, nextClassColour } from "./ClassManager";

describe("classKeyFor", () => {
  it("derives the stable key a region stores from a name", () => {
    expect(classKeyFor("Surface scratch", [])).toBe("surface_scratch");
    expect(classKeyFor("  Crack / chip!", [])).toBe("crack_chip");
    expect(classKeyFor("3D dent", [])).toBe("class_3d_dent");
    expect(classKeyFor("Ünïcode näme", [])).toBe("unicode_name");
    expect(classKeyFor("###", [])).toBe("class");
  });

  it("never collides with a key already taken", () => {
    expect(classKeyFor("Defect", ["defect"])).toBe("defect_2");
    expect(classKeyFor("Defect", ["defect", "defect_2"])).toBe("defect_3");
  });

  it("stays inside the backend's key pattern", () => {
    for (const name of ["Ünïcode näme", "a".repeat(200), "x-y_z 9"]) {
      expect(classKeyFor(name, [])).toMatch(/^[a-z][a-z0-9_-]{0,63}$/);
    }
  });
});

describe("nextClassColour", () => {
  it("takes the first palette colour no class uses yet", () => {
    expect(nextClassColour([])).toBe(CLASS_PALETTE[0]);
    expect(nextClassColour([{ color: CLASS_PALETTE[0].toUpperCase() }])).toBe(CLASS_PALETTE[1]);
  });
});

describe("ClassManager", () => {
  it("lists every class with its key and colour", () => {
    render(
      withProviders(
        <MemoryRouter>
          <ClassManager datasetId={7} />
        </MemoryRouter>, [
        [
          queryKeys.annotationLabels(7),
          [
            { id: 1, dataset_id: 7, key: "defect", name: "Defect", color: "#e03131", position: 0, created_at: "" },
            { id: 2, dataset_id: 7, key: "scratch", name: "Scratch", color: "#1c7ed6", position: 1, created_at: "" },
          ],
        ],
      ]),
    );
    screen.getByText("Classes").click();
    expect(screen.getByRole<HTMLInputElement>("textbox", { name: "Name of scratch" }).value).toBe(
      "Scratch",
    );
    expect(screen.getByLabelText<HTMLInputElement>("Colour of Defect").value).toBe("#e03131");
    expect(screen.getByRole("button", { name: "Move Defect up" })).toHaveProperty("disabled", true);
    expect(screen.getByRole("button", { name: "Move Scratch down" })).toHaveProperty("disabled", true);
  });

  it("links each class to its reference studio", () => {
    render(
      withProviders(
        <MemoryRouter>
          <ClassManager datasetId={7} />
        </MemoryRouter>,
        [
          [
            queryKeys.annotationLabels(7),
            [{ id: 1, dataset_id: 7, key: "scratch", name: "Scratch", color: "#00aa00", position: 0 }],
          ],
        ],
      ),
    );
    expect(
      screen.getByRole("link", { name: "Choose references" }).getAttribute("href"),
    ).toBe("/datasets/7/studio/scratch");
  });
});
