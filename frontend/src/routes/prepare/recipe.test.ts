/**
 * The form against what is saved: a revision stores every option, the form sends only what
 * was touched, and "differs" has to mean the same thing on both sides of that.
 */

import { describeFields, type OptionsSchema } from "@vitavision/lab-ui";
import { describe, expect, it } from "vitest";

import type { RegionPreparationEntry } from "../../api/client";
import { failuresFirst, recipeDiffers, recipeOf, suggestedName, type SavedRecipe } from "./recipe";

const schema: OptionsSchema = {
  properties: {
    min_contrast: { type: "integer", default: 12, minimum: 1, maximum: 255 },
    border_fraction: { type: "number", default: 0.08 },
  },
};
const fields = describeFields(schema);

const saved: SavedRecipe = {
  extractor_type: "foreground_threshold",
  extractor_config: { min_contrast: 12, border_fraction: 0.08 },
  padding_fraction: 0.05,
  resample: "bilinear",
  sample_alignment: "per_image",
};

function form(values: Record<string, string>, padding = "0.05") {
  return recipeOf({
    extractorKey: "foreground_threshold",
    configFields: fields,
    configValues: values,
    padding,
    resample: "bilinear",
    alignment: "per_image",
  });
}

describe("recipeDiffers", () => {
  it("reads an untouched option at its default, so an empty form matches a saved default", () => {
    expect(recipeDiffers(form({}), fields, saved)).toBe(false);
    expect(recipeDiffers(form({ min_contrast: "12" }), fields, saved)).toBe(false);
  });

  it("sees an option, the padding or the extractor change", () => {
    expect(recipeDiffers(form({ min_contrast: "18" }), fields, saved)).toBe(true);
    expect(recipeDiffers(form({}, "0.1"), fields, saved)).toBe(true);
    expect(recipeDiffers({ ...form({}), extractor_type: "identity" }, fields, saved)).toBe(true);
  });

  it("differs from nothing saved at all", () => {
    expect(recipeDiffers(form({}), fields, undefined)).toBe(true);
  });
});

describe("suggestedName", () => {
  it("names where it looks and the padding, and says when the crop is shared", () => {
    expect(suggestedName(form({}), "Foreground threshold")).toBe("Foreground threshold · pad 5%");
    expect(
      suggestedName({ ...form({}, "0.125"), sample_alignment: "union" }, "Centred crop"),
    ).toBe("Centred crop · pad 12.5% · shared");
  });
});

describe("failuresFirst", () => {
  it("puts what failed where the reader looks first, keeping each group's order", () => {
    const entry = (image_id: number, status: "succeeded" | "failed") =>
      ({ image_id, status }) as RegionPreparationEntry;
    const ordered = failuresFirst([entry(1, "succeeded"), entry(2, "failed"), entry(3, "succeeded"), entry(4, "failed")]);
    expect(ordered.map((item) => item.image_id)).toEqual([2, 4, 1, 3]);
  });
});
