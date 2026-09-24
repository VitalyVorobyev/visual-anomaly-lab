/**
 * The create form answers what it can and says what it cannot, beside the field.
 *
 * Seeded with settled queries, so these are about the form's decisions — auto-selecting a
 * lone choice, suggesting a name, restoring a draft, placing errors — not about loading.
 */

import { fireEvent, render, screen } from "@testing-library/react";
import { MemoryRouter, Route, Routes } from "react-router";
import { afterEach, describe, expect, it } from "vitest";

import { draftKey } from "../api/experimentDraft";
import { queryKeys } from "../api/queryKeys";
import { withProviders } from "../test-harness";
import { DatasetCreateExperimentRoute } from "./ExperimentsRoute";

const METHOD = {
  key: "pixel_reference",
  title: "Pixel reference",
  summary: "A median image.",
  capabilities: {
    requires_training: true,
    produces_anomaly_map: true,
    produces_diagnostics: false,
    channel_aware: false,
    dataset_specific: false,
    supports_resume: false,
    portable_formats: [],
    preferred_device: "cpu",
  },
  availability: { available: true, reason: null },
  config_schema: { type: "object", properties: {} },
};

const SPLIT = {
  id: 3,
  dataset_id: 7,
  name: "published",
  strategy: "imported",
  seed: 0,
  created_at: "2026-09-01T00:00:00Z",
  composition: [],
  params: {},
};

const PROFILE = {
  id: 11,
  dataset_id: 7,
  name: "full frame",
  revision_no: 1,
  extractor_type: "identity",
  extractor_config: {},
  prepared_width: 256,
  prepared_height: 256,
  padding_fraction: 0,
  resample: "bilinear",
  failure_policy: "fail",
  seed: 17,
  created_at: "2026-09-01T00:00:00Z",
};

function seed({ splits = [SPLIT] }: { splits?: unknown[] } = {}): [readonly unknown[], unknown][] {
  return [
    [
      queryKeys.modelTypes(),
      {
        methods: [METHOD],
        preprocessing_schema: { type: "object", properties: {} },
        evaluation_schema: { type: "object", properties: {} },
      },
    ],
    [queryKeys.datasets(), [{ id: 7, name: "candle" }]],
    [queryKeys.dataset(7), { id: 7, name: "candle", channels: [] }],
    [queryKeys.splits(7), splits],
    [queryKeys.regionProfiles(7), [PROFILE]],
    [queryKeys.regionBuild(11), { profile_id: 11, dataset_id: 7, total: 10, succeeded: 10, failed: 0 }],
  ];
}

function renderForm(splits?: unknown[]) {
  return render(
    withProviders(
      <MemoryRouter initialEntries={["/datasets/7/experiments/new"]}>
        <Routes>
          <Route path="datasets/:datasetId/experiments/new" element={<DatasetCreateExperimentRoute />} />
        </Routes>
      </MemoryRouter>,
      seed(splits === undefined ? {} : { splits }),
    ),
  );
}

afterEach(() => sessionStorage.clear());

describe("the create-experiment form", () => {
  it("selects the only split and the only prepared input on its own", () => {
    renderForm();
    expect(screen.getByRole("combobox", { name: "Split" }).textContent).toContain("published");
    expect(screen.getByRole("combobox", { name: "Region profile" }).textContent).toContain(
      "full frame",
    );
  });

  it("suggests a name instead of demanding one", () => {
    renderForm();
    expect(screen.getByRole("textbox", { name: "Name" }).getAttribute("placeholder")).toBe(
      "Pixel reference on candle",
    );
    expect(screen.queryByText(/Still needs/)).toBeNull();
  });

  it("puts what is missing beside the field once Create is pressed", () => {
    renderForm([]);
    expect(screen.queryByText("Choose which samples train and which are scored.")).toBeNull();
    fireEvent.click(screen.getByRole("button", { name: "Create experiment" }));
    expect(screen.getByText("Choose which samples train and which are scored.")).toBeTruthy();
  });

  it("brings back what was typed before following a prerequisite link", () => {
    sessionStorage.setItem(
      draftKey(7),
      JSON.stringify({ name: "kept from before", channels: [], configValues: {} }),
    );
    renderForm();
    expect((screen.getByRole("textbox", { name: "Name" }) as HTMLInputElement).value).toBe(
      "kept from before",
    );
  });
});
