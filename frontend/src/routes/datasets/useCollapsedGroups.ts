/**
 * Which collections the reader has folded away, remembered between visits.
 *
 * Collapse is stored rather than defaulted by size. A rule like "fold anything over six"
 * would put VisA away for the person who works on VisA and leave it open for the person
 * who does not, and neither could tell why. Everything opens expanded the first time; what
 * you close stays closed.
 *
 * `localStorage` rather than the URL: this is a preference about the reader, not about the
 * page, so it should not travel in a shared link or make two tabs disagree with history.
 */

import { useCallback, useState } from "react";

const KEY = "anomaly-lab:collapsed-collections";

function read(): string[] {
  try {
    const raw = globalThis.localStorage?.getItem(KEY);
    const parsed: unknown = raw ? JSON.parse(raw) : [];
    return Array.isArray(parsed) ? parsed.filter((name) => typeof name === "string") : [];
  } catch {
    // A corrupt or unavailable store is not worth a broken catalogue; everything opens.
    return [];
  }
}

export function useCollapsedGroups() {
  const [collapsed, setCollapsed] = useState<string[]>(read);

  const toggle = useCallback((name: string) => {
    setCollapsed((current) => {
      const next = current.includes(name)
        ? current.filter((item) => item !== name)
        : [...current, name];
      try {
        globalThis.localStorage?.setItem(KEY, JSON.stringify(next));
      } catch {
        // Private mode, or a full quota. The fold still works for this session.
      }
      return next;
    });
  }, []);

  const isCollapsed = useCallback((name: string) => collapsed.includes(name), [collapsed]);

  return { isCollapsed, toggle };
}
