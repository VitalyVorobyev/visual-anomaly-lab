/**
 * The create form follows the dataset's truth (ADR-0041): a dataset of classes alone is not
 * offered anomaly detection, and the form opens on a task it can serve.
 */

import { render, screen } from "@testing-library/react";
import { MemoryRouter, Route, Routes } from "react-router";
import { afterEach, describe, expect, it } from "vitest";

import { queryKeys } from "../api/queryKeys";
import { withProviders } from "../test-harness";
import { DatasetCreateExperimentRoute } from "./ExperimentsRoute";

function method(key: string, title: string, tasks: string[]) {
  return {
    key,
    title,
    summary: "",
    capabilities: {
      tasks,
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
}

const METHODS = [
  method("pixel_reference", "Pixel reference", ["anomaly"]),
  method("color_prototype", "Colour prototype", ["few_shot_segmentation"]),
  method("color_classifier", "Colour classifier", ["semantic_segmentation"]),
];

function renderForm(truth: string[]) {
  return render(
    withProviders(
      <MemoryRouter initialEntries={["/datasets/7/experiments/new"]}>
        <Routes>
          <Route
            path="datasets/:datasetId/experiments/new"
            element={<DatasetCreateExperimentRoute />}
          />
        </Routes>
      </MemoryRouter>,
      [
        [
          queryKeys.modelTypes(),
          {
            methods: METHODS,
            preprocessing_schema: { type: "object", properties: {} },
            evaluation_schema: { type: "object", properties: {} },
          },
        ],
        [queryKeys.datasets(), [{ id: 7, name: "FSS-1000 panel" }]],
        [queryKeys.dataset(7), { id: 7, name: "FSS-1000 panel", channels: [], truth }],
        [queryKeys.splits(7), []],
        [queryKeys.regionProfiles(7), []],
        [queryKeys.annotationLabels(7), []],
      ],
    ),
  );
}

afterEach(() => sessionStorage.clear());

describe("the create form on a dataset's truth", () => {
  it("offers a dataset of classes the tasks that read classes, and opens on the first", () => {
    renderForm(["classes"]);
    expect(screen.queryByRole("radio", { name: "Anomaly detection" })).toBeNull();
    const fewShot = screen.getByRole("radio", { name: "Few-shot segmentation" });
    expect((fewShot as HTMLInputElement).checked).toBe(true);
  });

  it("offers anomaly detection once the dataset has labels", () => {
    renderForm(["labels", "classes"]);
    const anomaly = screen.getByRole("radio", { name: "Anomaly detection" });
    expect((anomaly as HTMLInputElement).checked).toBe(true);
  });
});
