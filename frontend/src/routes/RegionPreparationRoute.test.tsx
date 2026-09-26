/**
 * The Prepare screen: a saved revision loads into the form, the form is previewed live, and
 * only a form that differs from what is saved offers to be saved.
 *
 * Removing a revision is here too, and there the interesting case is not the deletion — it
 * is the refusal. A profile revision is immutable and experiments pin it with `ON DELETE
 * RESTRICT`; a greyed-out button that does not say *which* runs hold it leaves the operator
 * with nothing to do next, which is why the preview names them and the dialog prints it.
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
  };
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

const extractors = [
  {
    key: "mobile_sam",
    title: "MobileSAM automatic region",
    summary: "Masks.",
    availability: { available: true, reason: null },
    required_assets: [],
    config_schema: { properties: {} },
  },
];

function mountSelected(seed: [readonly unknown[], unknown][] = [], saved: RegionProfileRevision = profile) {
  return render(
    withProviders(
      <MemoryRouter initialEntries={[`/datasets/${DATASET_ID}/prepare?profile=${saved.id}`]}>
        <Routes>
          <Route path="datasets/:datasetId/prepare" element={<RegionPreparationRoute />} />
        </Routes>
      </MemoryRouter>,
      [[queryKeys.regionProfiles(DATASET_ID), [saved]], ...seed],
    ),
  );
}

describe("the size a profile is prepared at", () => {
  it("is not part of the profile: preview, check and build name it, 448 unless changed", () => {
    mountSelected();

    expect(screen.getByLabelText<HTMLInputElement>("Preview width").value).toBe("448");
    expect(screen.getByLabelText<HTMLInputElement>("Preview height").value).toBe("448");
    expect(screen.getByText(/The preview, Check 24 and Build all prepare at 448×448/)).toBeTruthy();
    expect(screen.getByText("Check 24 · 448×448")).toBeTruthy();
    // The profile form asks where to look, never how large.
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
    expect(screen.getByLabelText<HTMLInputElement>("Preview width").value).toBe("256");
  });
});

describe("opening a saved revision", () => {
  it("loads it into the form, shared crop included", () => {
    mountSelected([], { ...profile, sample_alignment: "union" });

    expect(screen.getByRole<HTMLInputElement>("radio", { name: "Shared" }).checked).toBe(true);
    expect(screen.getByText(/Showing r3 as saved · shared crop/)).toBeTruthy();
  });

  it("offers Save only once the form differs, under a name that says what it is", () => {
    mountSelected([[queryKeys.regionExtractors(), extractors]]);

    expect(screen.queryByRole("button", { name: "Save profile" })).toBeNull();

    fireEvent.change(screen.getByLabelText("Padding"), { target: { value: "0.1" } });

    expect(screen.getByRole("button", { name: "Save profile" })).toBeTruthy();
    expect(screen.getByText("Edited · not saved")).toBeTruthy();
    expect(screen.getByLabelText("Profile name").getAttribute("placeholder")).toBe(
      "MobileSAM automatic region · pad 10%",
    );
    // Build all prepares what is saved, and says so rather than greying out in silence.
    expect(screen.getByRole("button", { name: "Build all" }).hasAttribute("disabled")).toBe(true);
    expect(screen.getByText(/save these changes first/)).toBeTruthy();

    fireEvent.change(screen.getByLabelText("Padding"), { target: { value: "0.05" } });
    expect(screen.queryByRole("button", { name: "Save profile" })).toBeNull();
  });
});

describe("the live stage", () => {
  const strip = {
    total: 30,
    images: [1, 2, 3].map((index) => ({
      image_id: 100 + index,
      sample_id: index,
      group_key: "good",
      external_id: `00${index}`,
      channel: null,
      width: 64,
      height: 48,
    })),
  };

  it("steps through images spread over the dataset with the arrow keys", () => {
    mountSelected([[queryKeys.regionPreviewImages(DATASET_ID, "per_image"), strip]]);

    expect(screen.getByText("good/001")).toBeTruthy();
    expect(screen.getByText("1 / 3 of 30")).toBeTruthy();

    fireEvent.keyDown(window, { key: "ArrowRight" });
    expect(screen.getByText("good/002")).toBeTruthy();

    fireEvent.keyDown(window, { key: "ArrowLeft" });
    fireEvent.keyDown(window, { key: "ArrowLeft" });
    expect(screen.getByText("good/003")).toBeTruthy();

    fireEvent.click(screen.getByRole("button", { name: "good/002" }));
    expect(screen.getByText("2 / 3 of 30")).toBeTruthy();
  });
});

