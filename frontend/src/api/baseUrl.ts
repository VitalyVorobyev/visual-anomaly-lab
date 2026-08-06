/**
 * Where the backend lives.
 *
 * The same bundle runs in three places and must find the sidecar in each:
 *
 *  1. **Tauri shell** — the shell injects `window.__ANOMALY_LAB__` before the app loads,
 *     because the sidecar's port is ephemeral and only known at runtime.
 *  2. **`vite dev` in a browser** — `VITE_API_BASE_URL` if set.
 *  3. **Fallback** — the port the documented `uv run` development command uses.
 *
 * Note that the shell hands the URL over as an injected global rather than through
 * `@tauri-apps/api`. That is deliberate: nothing under `src/` imports a Tauri API, so the
 * UI stays a pure HTTP/WebSocket client and the browser path remains first-class (§2).
 */

// The shape of the injected global is declared once, in `shell.ts`, alongside the other
// capabilities the shell hands over.
import "./shell";

export const DEFAULT_API_BASE_URL = "http://127.0.0.1:8000";

export interface BaseUrlSources {
  /** Injected by the Tauri shell once the sidecar has announced its port. */
  injected?: string | undefined;
  /** Build-time override for browser development. */
  env?: string | undefined;
}

function readSources(): BaseUrlSources {
  return {
    injected: globalThis.window?.__ANOMALY_LAB__?.apiBaseUrl,
    env: import.meta.env["VITE_API_BASE_URL"] as string | undefined,
  };
}

/** Resolve the backend base URL, without a trailing slash. */
export function resolveApiBaseUrl(sources: BaseUrlSources = readSources()): string {
  const candidate = [sources.injected, sources.env].find(
    (value) => typeof value === "string" && value.trim() !== "",
  );
  return (candidate ?? DEFAULT_API_BASE_URL).trim().replace(/\/+$/, "");
}

/** Turn an HTTP base URL into the WebSocket URL for `path`. */
export function websocketUrl(path: string, baseUrl: string = resolveApiBaseUrl()): string {
  const url = new URL(path, `${baseUrl}/`);
  url.protocol = url.protocol === "https:" ? "wss:" : "ws:";
  return url.toString();
}
