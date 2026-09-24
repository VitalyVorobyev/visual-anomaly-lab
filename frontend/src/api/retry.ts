/**
 * When a failed read is worth asking again.
 *
 * React Query's default retries every failure three times with backoff — about seven seconds
 * of skeleton before any error is shown. The backend here is local and deterministic: a 4xx
 * is the request's fault and says so in its message, and asking again only hides it. A 5xx or
 * a dropped connection may be a sidecar restarting, so it gets one more try.
 */

import { ApiError } from "./client";

export function shouldRetry(failureCount: number, error: unknown): boolean {
  if (error instanceof ApiError && error.status >= 400 && error.status < 500) return false;
  return failureCount < 1;
}
