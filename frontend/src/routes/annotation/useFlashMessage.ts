/**
 * The header's transient status line: "Draft saved", "2 regions erased", "MobileSAM is ready".
 *
 * Every seam of the editor reports through it, so it is owned by none of them. A message
 * clears itself after `ms`; a new one restarts the clock.
 */

import { useEffect, useState } from "react";

export type Flash = (message: string | null) => void;

export function useFlashMessage(ms = 2200): [string | null, Flash] {
  const [message, setMessage] = useState<string | null>(null);

  useEffect(() => {
    if (!message) return;
    const timer = globalThis.setTimeout(() => setMessage(null), ms);
    return () => globalThis.clearTimeout(timer);
  }, [message, ms]);

  return [message, setMessage];
}
