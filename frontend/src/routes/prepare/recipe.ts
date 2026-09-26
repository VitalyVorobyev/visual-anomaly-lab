/**
 * The Prepare form as a recipe: what it would save, whether that differs from the saved
 * revision it started from, and the name it would be saved under.
 *
 * A recipe has no size. The preview size is chosen beside it and travels with each request,
 * because a profile says where to look and a run says how large (ADR-0033).
 */

import { toOptions, type FieldSpec, type RawValues } from "@vitavision/lab-ui";

import type {
  RegionPreparationEntry,
  RegionProfileRevision,
  RegionRecipe,
  SampleAlignment,
  SpatialResample,
} from "../../api/client";

/** A recipe as the backend stores it: every option present. A revision, or what a check ran. */
export type SavedRecipe = Pick<
  RegionProfileRevision,
  "extractor_type" | "extractor_config" | "padding_fraction" | "resample" | "sample_alignment"
>;

export interface RecipeForm {
  extractorKey: string;
  configFields: FieldSpec[];
  configValues: RawValues;
  padding: string;
  resample: SpatialResample;
  alignment: SampleAlignment;
}

/** What the form would send. An untouched option is left out, so Python's default applies. */
export function recipeOf(form: RecipeForm): RegionRecipe {
  return {
    extractor_type: form.extractorKey,
    extractor_config: toOptions(form.configFields, form.configValues),
    padding_fraction: Number(form.padding),
    resample: form.resample,
    sample_alignment: form.alignment,
  };
}

/**
 * Whether the form describes something other than `saved`.
 *
 * A saved revision stores every option, defaults filled in; the form sends only what was
 * touched. So each option is compared at its *effective* value — what was typed, else the
 * schema's own default — and an option the schema does not know is ignored on both sides.
 */
export function recipeDiffers(
  recipe: RegionRecipe,
  fields: FieldSpec[],
  saved: SavedRecipe | undefined,
): boolean {
  if (saved === undefined) return true;
  if (
    recipe.extractor_type !== saved.extractor_type ||
    recipe.padding_fraction !== saved.padding_fraction ||
    recipe.resample !== saved.resample ||
    recipe.sample_alignment !== saved.sample_alignment
  ) {
    return true;
  }
  const options = recipe.extractor_config ?? {};
  return fields.some((field) => {
    const mine = field.name in options ? options[field.name] : field.fallback;
    const theirs = field.name in saved.extractor_config ? saved.extractor_config[field.name] : field.fallback;
    return JSON.stringify(mine ?? null) !== JSON.stringify(theirs ?? null);
  });
}

/** "Foreground threshold · pad 5%": where it looks, and the one number most often tuned. */
export function suggestedName(recipe: RegionRecipe, extractorTitle: string): string {
  const pad = Math.round((recipe.padding_fraction ?? 0) * 1000) / 10;
  const shared = recipe.sample_alignment === "union" ? " · shared" : "";
  return `${extractorTitle} · pad ${pad}%${shared}`;
}

/** A saved revision as form state, every option as the string a control holds. */
export function formValuesOf(profile: RegionProfileRevision): RawValues {
  return Object.fromEntries(
    Object.entries(profile.extractor_config).map(([key, value]) => [
      key,
      typeof value === "boolean" ? value : typeof value === "string" ? value : JSON.stringify(value),
    ]),
  );
}

/** A check's entries with the failures first: what the reader came to find. */
export function failuresFirst(entries: RegionPreparationEntry[]): RegionPreparationEntry[] {
  return [
    ...entries.filter((entry) => entry.status === "failed"),
    ...entries.filter((entry) => entry.status !== "failed"),
  ];
}
