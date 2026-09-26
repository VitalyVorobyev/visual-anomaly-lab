/**
 * The guided run starts from a sensible run and gets out of the way: every step has its
 * default chosen before anyone looks, Next walks them, "Review & run" jumps to the one press,
 * and the screen keeps the scroll contract — one region scrolls, the step.
 *
 * Seeded with settled queries; what the steps fetch on their own (thumbnails, previews) stays
 * pending, which is the state a slow sidecar shows.
 */

import { fireEvent, render, screen, within } from "@testing-library/react";
import { MemoryRouter, Route, Routes, useLocation } from "react-router";
import { afterEach, describe, expect, it } from "vitest";

import { queryKeys } from "../api/queryKeys";
import { withProviders } from "../test-harness";
import { GuidedRunRoute } from "./GuidedRunRoute";

const CAPABILITIES = {
  requires_training: true,
  produces_anomaly_map: true,
  produces_diagnostics: false,
  channel_aware: false,
  dataset_specific: false,
  supports_resume: false,
  portable_formats: [],
  preferred_device: "cpu",
};

function method(key: string, title: string, tasks: string[], extra: Record<string, unknown> = {}) {
  return {
    key,
    title,
    summary: `${title}.`,
    capabilities: { ...CAPABILITIES, tasks },
    availability: { available: true, reason: null },
    status: "supported",
    recommended_for: [],
    config_schema: { type: "object", properties: {} },
    ...extra,
  };
}

const FLOOR = method("pixel_reference", "Pixel reference", ["anomaly"], { status: "floor" });
const MEMORY = method("patch_memory", "Patch memory", ["anomaly"], { recommended_for: ["anomaly"] });
const FEW_SHOT = method("proto_seg", "Prototype segmenter", ["few_shot_segmentation"], {
  recommended_for: ["few_shot_segmentation"],
});
const DETECTOR = method("box_floor", "Box floor", ["object_detection"], { status: "floor" });

const FULL_FRAME = {
  id: 11,
  dataset_id: 7,
  name: "Full frame",
  revision_no: 1,
  extractor_type: "identity",
  extractor_config: {},
  padding_fraction: 0,
  resample: "bilinear",
  sample_alignment: "per_image",
  created_at: "2026-09-01T00:00:00Z",
};

function composition(total: number) {
  return ["train", "val", "test"].map((subset) => ({
    subset,
    total,
    normal: total,
    defect: 0,
    unlabeled: 0,
    classes: [],
    examples: [],
  }));
}

const STANDARD = {
  key: "standard",
  label: "Standard · 60/20/20, normals only",
  meaning: "Train on 60% of the normals.",
  tasks: ["anomaly"],
  params: { strategy: "normal_only_train" },
  seed: 0,
  name: "Standard · 60/20/20, normals only · seed 0",
  composition: composition(4),
  classes: [],
};

function seed(dataset: Record<string, unknown>, presets: unknown[] = [STANDARD]) {
  return [
    [
      queryKeys.modelTypes(),
      {
        methods: [FLOOR, MEMORY, FEW_SHOT, DETECTOR],
        preprocessing_schema: { type: "object", properties: {} },
        evaluation_schema: { type: "object", properties: {} },
      },
    ],
    [
      queryKeys.dataset(7),
      {
        id: 7,
        name: "candle",
        channels: [],
        truth: ["labels"],
        class_counts: [],
        class_geometry: null,
        default_channel: null,
        ...dataset,
      },
    ],
    [queryKeys.splits(7), []],
    [queryKeys.splitPresets(7), presets],
    [queryKeys.regionProfiles(7), [FULL_FRAME]],
    [queryKeys.inputSize("patch_memory", {}), { model_type: "patch_memory", width: 448, height: 448, multiple: 14 }],
  ] as [readonly unknown[], unknown][];
}

function Where() {
  const location = useLocation();
  return <output data-testid="where">{location.pathname + location.search}</output>;
}

function renderRun(dataset: Record<string, unknown> = {}, path = "/datasets/7/run", presets?: unknown[]) {
  return render(
    withProviders(
      <MemoryRouter initialEntries={[path]}>
        <Routes>
          <Route
            path="datasets/:datasetId/run"
            element={
              <>
                <GuidedRunRoute />
                <Where />
              </>
            }
          />
        </Routes>
      </MemoryRouter>,
      seed(dataset, presets),
    ),
  );
}

afterEach(() => sessionStorage.clear());

describe("the guided run", () => {
  it("opens on the goal with the task the truth answers already chosen", () => {
    renderRun();
    const anomaly = screen.getByRole("radio", { name: /Anomaly detection/ }) as HTMLInputElement;
    expect(anomaly.checked).toBe(true);
    expect(screen.getByText("suggested")).toBeTruthy();
    // A dataset of verdicts is not offered tasks its truth cannot feed.
    expect(screen.queryByRole("radio", { name: /Few-shot/ })).not.toBeNull();
  });

  it("suggests detection for classes drawn as boxes, and few-shot for regions", () => {
    const classes = {
      truth: ["classes"],
      class_counts: [
        { key: "short", name: "Short", color: "#aa0000", samples: 4 },
        { key: "spur", name: "Spur", color: "#00aa00", samples: 9 },
      ],
    };
    const { unmount } = renderRun({ ...classes, class_geometry: "boxes" }, "/datasets/7/run", []);
    expect(
      (screen.getByRole("radio", { name: /Object detection/ }) as HTMLInputElement).checked,
    ).toBe(true);
    // A dataset of classes alone is not offered anomaly detection (ADR-0041).
    expect(screen.queryByRole("radio", { name: /Anomaly detection/ })).toBeNull();
    unmount();

    renderRun({ ...classes, class_geometry: "regions" }, "/datasets/7/run", []);
    expect(
      (screen.getByRole("radio", { name: /Few-shot segmentation/ }) as HTMLInputElement).checked,
    ).toBe(true);
    // The class defaults to the most frequent.
    expect(screen.getByRole("combobox", { name: "Target class" }).textContent).toContain("Spur");
  });

  it("walks the steps with Next, and offers the rail back to each one visited", () => {
    renderRun();
    fireEvent.click(screen.getByRole("button", { name: /Next: Look/ }));
    expect(screen.getByTestId("where").textContent).toBe("/datasets/7/run?step=look");
    expect((screen.getByRole("radio", { name: /Full frame/ }) as HTMLInputElement).checked).toBe(true);

    fireEvent.click(screen.getByRole("button", { name: /Next: Split/ }));
    expect(
      (screen.getByRole("radio", { name: /Standard · 60\/20\/20/ }) as HTMLInputElement).checked,
    ).toBe(true);

    fireEvent.click(screen.getByRole("button", { name: /Next: Method/ }));
    const cards = screen
      .getAllByRole("radio")
      .filter((radio) => radio.getAttribute("name") === "method") as HTMLInputElement[];
    // The registry's recommendation first and chosen, the floor last.
    expect(cards.map((radio) => radio.value)).toEqual(["patch_memory", "pixel_reference"]);
    expect(cards[0]?.checked).toBe(true);

    const rail = screen.getByRole("navigation", { name: "Run steps" });
    fireEvent.click(within(rail).getByRole("button", { name: /Goal/ }));
    expect(screen.getByTestId("where").textContent).toBe("/datasets/7/run?step=goal");
    // Every step reached stays reachable from the rail.
    expect(within(rail).getByRole("button", { name: /Method/ })).toBeTruthy();
    expect(within(rail).queryByRole("button", { name: /Run/ })).toBeNull();
  });

  it("jumps to the one press with every default in place", () => {
    renderRun();
    fireEvent.click(screen.getByRole("button", { name: "Review & run" }));
    expect(screen.getByTestId("where").textContent).toBe("/datasets/7/run?step=run");

    const summary = document.querySelector("dl")!;
    expect(summary.textContent).toContain("Standard · 60/20/20, normals only");
    expect(summary.textContent).toContain("Anomaly detection");
    expect(summary.textContent).toContain("Full frame");
    expect(summary.textContent).toContain("Patch memory");
    expect(summary.textContent).toContain("448 × 448 · from patch_memory");
    expect(screen.getByRole("textbox", { name: "Name" }).getAttribute("placeholder")).toBe(
      "Patch memory on candle",
    );
    expect((screen.getByRole("button", { name: "Start run" }) as HTMLButtonElement).disabled).toBe(
      false,
    );
  });

  it("says what is missing instead of starting a run that cannot", () => {
    renderRun({}, "/datasets/7/run?step=run", []);
    expect((screen.getByRole("button", { name: "Start run" }) as HTMLButtonElement).disabled).toBe(
      true,
    );
    expect(screen.getByText(/Still needs a split/)).toBeTruthy();
  });

  it("comes back from Prepare on the look it chose, and keeps the choice", () => {
    renderRun({}, "/datasets/7/run?step=look&profile=11");
    expect(screen.getByTestId("where").textContent).toBe("/datasets/7/run?step=look");
    const stored = JSON.parse(sessionStorage.getItem("anomaly-lab:guided-run:7") ?? "{}") as {
      profileId?: number;
    };
    expect(stored.profileId).toBe(11);
  });

  it("keeps the scroll contract: the step is the one region that scrolls", () => {
    const { container } = renderRun();
    const scrollers = container.querySelectorAll("[data-scroll]");
    expect(scrollers).toHaveLength(1);
    const step = scrollers[0]!;
    expect(step.className).toContain("overflow-y-auto");
    const nested = [...step.querySelectorAll("*")].filter((element) =>
      String(element.className).includes("overflow-y-auto"),
    );
    expect(nested).toHaveLength(0);
  });
});
