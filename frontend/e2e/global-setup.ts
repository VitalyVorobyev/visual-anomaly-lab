/**
 * Start a throwaway backend for the screenshot suite, seeded once and then reused.
 *
 * The data directory is `e2e/.state/data` (gitignored). The first run seeds it through
 * `scripts/e2e-seed.py`; every later run reuses it, so the ids, names and timestamps a
 * baseline captured are the ones the run compared against it sees. Delete `e2e/.state`
 * to start over, and re-capture the baseline when you do.
 */

import { spawn, spawnSync, type ChildProcess } from "node:child_process";
import { existsSync, mkdirSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const HERE = dirname(fileURLToPath(import.meta.url));
const REPO = resolve(HERE, "../..");
export const STATE = join(HERE, ".state");
export const API_PORT = 8765;
const API = `http://127.0.0.1:${API_PORT}`;

async function healthy(): Promise<boolean> {
  try {
    return (await fetch(`${API}/api/health`)).ok;
  } catch {
    return false;
  }
}

export default async function globalSetup(): Promise<() => void> {
  const data = join(STATE, "data");
  const seeded = existsSync(join(data, "app.sqlite3"));
  mkdirSync(data, { recursive: true });

  const backend: ChildProcess = spawn(
    "uv",
    ["run", "--directory", "backend", "uvicorn", "anomaly_lab.api.app:create_app", "--factory", "--port", String(API_PORT)],
    {
      cwd: REPO,
      env: {
        ...process.env,
        ANOMALY_LAB_DATA_DIR: data,
        // No public reference packs: the catalogue is exactly what the seed wrote.
        ANOMALY_LAB_REFERENCE_DATASETS_DIR: join(STATE, "no-reference-datasets"),
      },
      stdio: "ignore",
    },
  );
  for (let i = 0; i < 120 && !(await healthy()); i++) await new Promise((r) => setTimeout(r, 500));
  if (!(await healthy())) {
    backend.kill();
    throw new Error(`the backend did not answer on ${API}`);
  }

  if (!seeded) {
    const seed = spawnSync(
      "uv",
      ["run", "--directory", "backend", "python", "../scripts/e2e-seed.py", join(STATE, "tree"), API],
      { cwd: REPO, stdio: "inherit" },
    );
    if (seed.status !== 0) {
      backend.kill();
      throw new Error("seeding the backend failed");
    }
  }

  return () => {
    backend.kill();
  };
}
