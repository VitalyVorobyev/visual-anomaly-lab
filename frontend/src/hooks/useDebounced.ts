/**
 * A value that settles: `value`, once it has stopped changing for `delay` milliseconds.
 *
 * The live Prepare stage re-runs its preview on every control change, and a number typed
 * one digit at a time is several changes. Debouncing the *request* rather than the controls
 * keeps every control responsive while only the settled value reaches the backend.
 */

import { useEffect, useState } from "react";

export function useDebounced<T>(value: T, delay: number): T {
  const [settled, setSettled] = useState(value);
  useEffect(() => {
    const timer = setTimeout(() => setSettled(value), delay);
    return () => clearTimeout(timer);
  }, [value, delay]);
  return settled;
}
