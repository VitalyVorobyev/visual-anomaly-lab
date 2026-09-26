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
    tasks: ["anomaly"],
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
  name: "Full frame",
  revision_no: 1,
  extractor_type: "identity",
  extractor_config: {},
  padding_fraction: 0,
  resample: "bilinear",
  created_at: "2026-09-01T00:00:00Z",
};

const OTHER_PROFILE = {
  ...PROFILE,
  id: 12,
  name: "Dominant object",
  extractor_type: "foreground_threshold",
};

function seed({
  splits = [SPLIT],
  methods = [METHOD],
  profiles = [PROFILE],
}: { splits?: unknown[]; methods?: unknown[]; profiles?: unknown[] } = {}): [
  readonly unknown[],
  unknown,
][] {
  return [
    [
      queryKeys.modelTypes(),
      {
        methods,
        preprocessing_schema: { type: "object", properties: {} },
        evaluation_schema: { type: "object", properties: {} },
      },
    ],
    [queryKeys.datasets(), [{ id: 7, name: "candle" }]],
    [queryKeys.dataset(7), { id: 7, name: "candle", channels: [] }],
    [queryKeys.splits(7), splits],
    [queryKeys.regionProfiles(7), profiles],
    [
      queryKeys.inputSize("pixel_reference", {}),
      { model_type: "pixel_reference", width: 256, height: 256, multiple: 1 },
    ],
    [
      queryKeys.inputSize("dino_memory", {}),
      { model_type: "dino_memory", width: 448, height: 448, multiple: 14 },
    ],
    [
      queryKeys.annotationLabels(7),
      [
        { id: 1, dataset_id: 7, key: "defect", name: "Defect", color: "#c026d3", position: 0 },
        { id: 2, dataset_id: 7, key: "scratch", name: "Scratch", color: "#00aa00", position: 1 },
      ],
    ],
  ];
}

function renderForm(splits?: unknown[], methods?: unknown[], profiles?: unknown[]) {
  return render(
    withProviders(
      <MemoryRouter initialEntries={["/datasets/7/experiments/new"]}>
        <Routes>
          <Route path="datasets/:datasetId/experiments/new" element={<DatasetCreateExperimentRoute />} />
        </Routes>
      </MemoryRouter>,
      seed({
        ...(splits === undefined ? {} : { splits }),
        ...(methods === undefined ? {} : { methods }),
        ...(profiles === undefined ? {} : { profiles }),
      }),
    ),
  );
}

afterEach(() => sessionStorage.clear());

describe("the create-experiment form", () => {
  it("selects the only split, and looks at the full frame unless told otherwise", () => {
    renderForm(undefined, undefined, [OTHER_PROFILE, PROFILE]);
    expect(screen.getByRole("combobox", { name: "Split" }).textContent).toContain("published");
    expect(screen.getByRole("combobox", { name: "Region profile" }).textContent).toContain(
      "Full frame",
    );
    // No build is asked for: the run prepares its profile when it trains.
    expect(screen.queryByText(/not built/)).toBeNull();
  });

  it("shows the method's own size beside empty size fields, and snaps a typed one", () => {
    const memory = {
      ...METHOD,
      key: "dino_memory",
      title: "DINO memory",
    };
    renderForm(undefined, [memory]);
    expect(screen.getByText("448 × 448 · from dino_memory")).toBeTruthy();
    const width = screen.getByRole("spinbutton", { name: "Input width" }) as HTMLInputElement;
    expect(width.value).toBe("");
    expect(width.getAttribute("placeholder")).toBe("448");

    fireEvent.change(width, { target: { value: "450" } });
    fireEvent.blur(width);
    expect(width.value).toBe("448");
    expect(screen.getByText(/Snaps to 14/)).toBeTruthy();
    // One side alone is not a size.
    fireEvent.click(screen.getByRole("button", { name: "Create experiment" }));
    expect(screen.getByText(/Give both sides/)).toBeTruthy();
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

  it("asks for a task only when there is more than one, and shows only its methods", () => {
    renderForm();
    expect(screen.queryByRole("radiogroup", { name: "Task" })).toBeNull();
    expect(screen.getAllByText("Pixel reference").length).toBeGreaterThan(0);

    const detector = {
      ...METHOD,
      key: "box_finder",
      title: "Box finder",
      capabilities: { ...METHOD.capabilities, tasks: ["object_detection"] },
    };
    renderForm(undefined, [METHOD, detector]);
    expect(screen.getAllByText("Pixel reference").length).toBeGreaterThan(0);
    expect(screen.queryByText("Box finder")).toBeNull();
    expect(screen.getAllByText("Object detection").length).toBeGreaterThan(0);
  });

  it("asks a few-shot run for its class, and takes it from a split drawn for one", () => {
    const floor = {
      ...METHOD,
      key: "color_prototype",
      title: "Colour prototype",
      capabilities: { ...METHOD.capabilities, tasks: ["few_shot_segmentation"] },
    };
    const drawn = { ...SPLIT, strategy: "few_shot", params: { label_key: "scratch", shots: 3 } };
    renderForm([drawn], [METHOD, floor]);
    expect(screen.queryByRole("combobox", { name: "Target class" })).toBeNull();

    fireEvent.click(screen.getByRole("radio", { name: "Few-shot segmentation" }));
    expect(screen.getByRole("combobox", { name: "Target class" }).textContent).toContain(
      "Scratch",
    );
  });

  it("asks for the task first, and offers only the splits that task trains on", () => {
    const floor = {
      ...METHOD,
      key: "color_prototype",
      title: "Colour prototype",
      capabilities: { ...METHOD.capabilities, tasks: ["few_shot_segmentation"] },
    };
    const references = {
      ...SPLIT,
      id: 4,
      name: "five shots",
      strategy: "few_shot",
      params: { label_key: "scratch", shots: 5 },
    };
    renderForm([SPLIT, references], [METHOD, floor]);
    // The task is step 1, before the inputs it decides.
    expect(screen.getByText("Task")).toBeTruthy();
    expect(screen.getByRole("combobox", { name: "Split" }).textContent).toContain("published");

    fireEvent.click(screen.getByRole("radio", { name: "Few-shot segmentation" }));
    const split = screen.getByRole("combobox", { name: "Split" });
    expect(split.textContent).toContain("five shots");
    expect(screen.getAllByText("Colour prototype").length).toBeGreaterThan(0);
    expect(screen.queryByText("Pixel reference")).toBeNull();
  });

  it("offers a segmentation run only a split of annotated samples, and links to drawing one", () => {
    const floor = {
      ...METHOD,
      key: "color_classifier",
      title: "Colour classifier",
      capabilities: { ...METHOD.capabilities, tasks: ["semantic_segmentation"] },
    };
    renderForm([SPLIT], [METHOD, floor]);
    fireEvent.click(screen.getByRole("radio", { name: "Segmentation" }));
    const link = screen.getByRole("link", { name: "Draw one by class" });
    expect(link.getAttribute("href")).toBe("/datasets/7/splits?strategy=class_stratified");
  });

  it("offers a detection run the same split a segmentation run takes, and links to drawing one", () => {
    const floor = {
      ...METHOD,
      key: "box_floor",
      title: "Box floor",
      capabilities: { ...METHOD.capabilities, tasks: ["object_detection"] },
    };
    const { unmount } = renderForm([SPLIT], [METHOD, floor]);
    fireEvent.click(screen.getByRole("radio", { name: "Object detection" }));
    expect(screen.getByText(/No split of annotated samples yet/)).toBeTruthy();
    const link = screen.getByRole("link", { name: "Draw one by class" });
    expect(link.getAttribute("href")).toBe("/datasets/7/splits?strategy=class_stratified");
    unmount();

    const drawn = { ...SPLIT, id: 4, name: "by class", strategy: "class_stratified" };
    renderForm([SPLIT, drawn], [METHOD, floor]);
    fireEvent.click(screen.getByRole("radio", { name: "Object detection" }));
    const split = screen.getByRole("combobox", { name: "Split" });
    expect(split.textContent).toContain("by class");
    expect(screen.queryByRole("link", { name: "Draw one by class" })).toBeNull();
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
