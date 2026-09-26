/**
 * One full-window screenshot per main route, light and dark, against the seeded backend.
 *
 * Ids are the seed's (`scripts/e2e-seed.py`): dataset 1, sample 1 (a blemished disc),
 * image 1, experiment 1. The clock is fixed so relative times ("3 days ago") read the same
 * on every run.
 */

import { expect, test } from "@playwright/test";

const ROUTES: [name: string, path: string][] = [
  ["datasets", "/"],
  ["import", "/import"],
  ["experiments", "/experiments"],
  ["experiments-new", "/experiments/new"],
  ["compare", "/compare"],
  ["health", "/health"],
  ["not-found", "/no-such-route"],
  ["dataset", "/datasets/1"],
  ["dataset-annotate", "/datasets/1/annotate"],
  ["dataset-prepare", "/datasets/1/prepare"],
  ["dataset-splits", "/datasets/1/splits"],
  ["dataset-experiments", "/datasets/1/experiments"],
  ["dataset-run", "/datasets/1/run"],
  ["sample", "/datasets/1/samples/1"],
  ["annotation-editor", "/datasets/1/annotate/1/1"],
  ["experiment", "/experiments/1"],
  ["experiment-sample", "/experiments/1/samples/1"],
];

for (const theme of ["light", "dark"] as const) {
  for (const [name, path] of ROUTES) {
    test(`${name} (${theme})`, async ({ page }) => {
      await page.clock.setFixedTime(new Date("2030-01-01T12:00:00Z"));
      await page.addInitScript((value) => localStorage.setItem("anomaly-lab-theme", value), theme);
      await page.goto(`/#${path}`);
      await page.waitForLoadState("networkidle");
      await expect(page.getByText(/^Loading/)).toHaveCount(0);
      await expect(page).toHaveScreenshot(`${name}-${theme}.png`, { timeout: 15_000 });
    });
  }
}
