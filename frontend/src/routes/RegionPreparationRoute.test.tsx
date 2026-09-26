/**
 * Removing a saved preset, and being told when it cannot be removed.
 *
 * A profile revision is immutable and experiments pin it with `ON DELETE RESTRICT`, so the
 * interesting case is not the deletion — it is the refusal. A greyed-out button that does
 * not say *which* runs are holding the profile leaves the operator with nothing to do next,
 * which is why the preview names them and the dialog prints what it says.
 */

import { fireEvent, render, screen, within } from "@testing-library/react";
import { MemoryRouter, Route, Routes } from "react-router";
import { describe, expect, it } from "vitest";

import type { RegionProfileDeletionPreview, RegionProfileRevision } from "../api/client";
import { queryKeys } from "../api/queryKeys";
import { RegionPreparationRoute } from "./RegionPreparationRoute";
import { withProviders } from "../test-harness";

const DATASET_ID = 7;

const profile = {
  id: 12,
  dataset_id: DATASET_ID,
  name: "model input",
  revision_no: 3,
  extractor_type: "mobile_sam",
  extractor_config: {},
  padding_fraction: 0.05,
  resample: "bilinear",
  created_at: "2026-01-01T00:00:00Z",
  sample_alignment: "per_image",
} as RegionProfileRevision;

function deletionPreview(
  overrides: Partial<RegionProfileDeletionPreview> = {},
): RegionProfileDeletionPreview {
  return {
    profile_id: profile.id,
    name: profile.name,
    revision_no: profile.revision_no,
    experiments: [],
    generated_files: 24,
    generated_bytes: 4096,
    active_jobs: [],
    storage_location_safe: true,
    can_delete: true,
    blocker: null,
    ...overrides,
  } as RegionProfileDeletionPreview;
}

function mount(preview: RegionProfileDeletionPreview) {
  return render(
    withProviders(
      <MemoryRouter initialEntries={[`/datasets/${DATASET_ID}/prepare`]}>
        <Routes>
          <Route path="datasets/:datasetId/prepare" element={<RegionPreparationRoute />} />
        </Routes>
      </MemoryRouter>,
      [
        [queryKeys.regionProfiles(DATASET_ID), [profile]],
        [queryKeys.regionProfileDeletion(profile.id), preview],
      ],
    ),
  );
}

describe("deleting a saved profile revision", () => {
  it("offers a delete beside revise, and says what it would reclaim", () => {
    mount(deletionPreview());

    fireEvent.click(screen.getByRole("button", { name: "Delete model input revision 3" }));

    // Scoped to the dialog: the panel heading names the same revision, and a match there
    // would pass without the dialog ever having opened.
    const dialog = within(screen.getByRole("dialog"));
    expect(dialog.getByText(/model input · revision 3/)).toBeTruthy();
    expect(dialog.getByText(/24 prepared files/)).toBeTruthy();
    expect(
      screen.getByRole("button", { name: "Delete revision" }).hasAttribute("disabled"),
    ).toBe(false);
  });

  it("refuses, and names the experiments holding the profile", () => {
    mount(
      deletionPreview({
        experiments: [{ experiment_id: 4, name: "efficientad baseline" }],
        can_delete: false,
        blocker: "1 experiment still use this input (#4 efficientad baseline). Delete them first.",
      }),
    );

    fireEvent.click(screen.getByRole("button", { name: "Delete model input revision 3" }));

    const dialog = within(screen.getByRole("dialog"));
    expect(dialog.getByText(/#4 efficientad baseline/)).toBeTruthy();
    expect(
      screen.getByRole("button", { name: "Delete revision" }).hasAttribute("disabled"),
    ).toBe(true);
  });
});

describe("the size a profile is prepared at", () => {
  function mountSelected(seed: [readonly unknown[], unknown][] = []) {
    return render(
      withProviders(
        <MemoryRouter initialEntries={[`/datasets/${DATASET_ID}/prepare?profile=${profile.id}`]}>
          <Routes>
            <Route path="datasets/:datasetId/prepare" element={<RegionPreparationRoute />} />
          </Routes>
        </MemoryRouter>,
        [[queryKeys.regionProfiles(DATASET_ID), [profile]], ...seed],
      ),
    );
  }

  it("is not part of the profile: preview and build name it, 448 unless changed", () => {
    mountSelected();

    expect((screen.getByLabelText("Preview width") as HTMLInputElement).value).toBe("448");
    expect((screen.getByLabelText("Preview height") as HTMLInputElement).value).toBe("448");
    expect(screen.getByText(/Preview and Build all prepare at 448×448/)).toBeTruthy();
    // The revision form asks where to look, never how large.
    expect(screen.queryByText("Width")).toBeNull();
    expect(screen.getByRole("button", { name: "Build all" }).hasAttribute("disabled")).toBe(false);
  });

  it("says when that size is built, lists every built size, and does not build it twice", () => {
    const report = { profile_id: 12, dataset_id: 7, width: 448, height: 448, total: 4, succeeded: 4, failed: 0, preview_entries: [], storage_bytes: 10 };
    mountSelected([
      [queryKeys.regionBuild(profile.id, 448, 448), report],
      [queryKeys.regionBuilds(profile.id), [{ ...report, width: 256, height: 256 }, report]],
    ]);

    expect(screen.getByText(/This size is built/)).toBeTruthy();
    expect(screen.getByRole("button", { name: "256×256" })).toBeTruthy();
    expect(screen.getByRole("button", { name: "Build all" }).hasAttribute("disabled")).toBe(true);

    fireEvent.click(screen.getByRole("button", { name: "256×256" }));
    expect((screen.getByLabelText("Preview width") as HTMLInputElement).value).toBe("256");
  });
});

describe("sharing one crop across a sample", () => {
  it("starts per image, and revising a shared profile carries the choice over", () => {
    render(
      withProviders(
        <MemoryRouter initialEntries={[`/datasets/${DATASET_ID}/prepare?profile=${profile.id}`]}>
          <Routes>
            <Route path="datasets/:datasetId/prepare" element={<RegionPreparationRoute />} />
          </Routes>
        </MemoryRouter>,
        [[queryKeys.regionProfiles(DATASET_ID), [{ ...profile, sample_alignment: "union" }]]],
      ),
    );
    const shared = () => screen.getByRole("radio", { name: "Shared" }) as HTMLInputElement;

    expect(shared().checked).toBe(false);
    expect(screen.getByText(/shared crop/)).toBeTruthy();

    fireEvent.click(screen.getByRole("button", { name: "Revise" }));

    expect(shared().checked).toBe(true);
  });
});
