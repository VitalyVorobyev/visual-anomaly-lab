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
      /** Show a path in the OS file manager. Rejects for a path outside `data/`. */
      revealPath?: (path: string) => Promise<void>;
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

/**
 * Whether this host can show a path in a file manager.
 *
 * The second capability, and the shape ADR-0014 predicted: the desktop app opens Finder,
 * the browser shows the path as selectable text. Neither is a degraded version of the
 * other, and the browser must never see a disabled button whose only explanation is that
 * it is not the desktop app.
 */
export function hasRevealPath(): boolean {
  return typeof globalThis.window?.__ANOMALY_LAB__?.revealPath === "function";
}

export async function revealPath(path: string): Promise<void> {
  const reveal = globalThis.window?.__ANOMALY_LAB__?.revealPath;
  if (typeof reveal !== "function") return;
  await reveal(path);
}
