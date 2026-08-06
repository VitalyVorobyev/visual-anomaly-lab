/**
 * Capabilities the desktop shell injects, and how the browser lives without them.
 *
 * ADR-0012 settled that nothing under `src/` imports a Tauri API — that is what keeps the
 * browser path first-class rather than a degraded mode. The shell therefore *injects*
 * what it can offer onto `window.__ANOMALY_LAB__` before the page loads, and this module
 * feature-detects it.
 *
 * The import flow is where it matters. A directory picker has to return an absolute path
 * the **backend** can open, and a browser's file input cannot give one: it yields a
 * `File` and a bare filename, deliberately, for the same security reason the backend
 * refuses to serve files by client-supplied path. So the desktop app offers a native
 * picker and the browser offers a text field, and neither is a special case of the other.
 */

declare global {
  interface Window {
    __ANOMALY_LAB__?: {
      apiBaseUrl?: string;
      /** Native folder chooser. Resolves to `null` when the user cancels. */
      pickDirectory?: () => Promise<string | null>;
    };
  }
}

export function hasDirectoryPicker(): boolean {
  return typeof globalThis.window?.__ANOMALY_LAB__?.pickDirectory === "function";
}

export async function pickDirectory(): Promise<string | null> {
  const pick = globalThis.window?.__ANOMALY_LAB__?.pickDirectory;
  if (typeof pick !== "function") return null;
  return pick();
}
