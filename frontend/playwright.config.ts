/**
 * The screenshot suite: every main route of the app, against a real, seeded backend.
 *
 *     bun run test:screens                 # compare against the local baseline
 *     bun run test:screens --update-snapshots
 *
 * The baseline lives in `e2e/.state/screenshots` and is **not committed**: it is captured
 * on one machine before a change and compared on the same machine after it (a toolchain
 * upgrade must not move a pixel), and the repository's own rule caps a committed PNG at
 * 256 KB, which a full-window capture exceeds. `global-setup.ts` explains the backend.
 */

import { defineConfig, devices } from "@playwright/test";

import { API_PORT } from "./e2e/global-setup";

const PORT = 5174;

export default defineConfig({
  testDir: "./e2e",
  globalSetup: "./e2e/global-setup.ts",
  snapshotPathTemplate: "{testDir}/.state/screenshots/{arg}{ext}",
  fullyParallel: false,
  workers: 1,
  reporter: [["list"]],
  expect: { toHaveScreenshot: { maxDiffPixelRatio: 0.001, animations: "disabled" } },
  use: {
    ...devices["Desktop Chrome"],
    viewport: { width: 1440, height: 900 },
    baseURL: `http://127.0.0.1:${PORT}`,
  },
  webServer: {
    command: `bunx vite --port ${PORT} --strictPort --host 127.0.0.1`,
    url: `http://127.0.0.1:${PORT}`,
    reuseExistingServer: false,
    env: { VITE_API_BASE_URL: `http://127.0.0.1:${API_PORT}` },
  },
});
