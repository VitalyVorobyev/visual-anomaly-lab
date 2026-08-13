/**
 * The providers a route needs before it will render at all.
 *
 * Several of these tests exist to assert what a screen looks like *while its data is still
 * in flight* — that the dataset band and its tab strip are already on screen, and do not
 * arrive late and push everything down. That state is only reachable if the query client is
 * real, so it is — and `test-setup.ts` holds every fetch pending, which is exactly the state
 * under test and keeps the suite off the network.
 */

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import type { ReactNode } from "react";

import { TooltipProvider } from "./components/ui";

export function withProviders(
  children: ReactNode,
  /**
   * Query keys to pre-fill, for the tests that are about *settled* data rather than the
   * pending frame. Seeding the cache keeps the screen under test unmodified — no injected
   * props, no mocked hook — while still never touching the network.
   */
  seed: [readonly unknown[], unknown][] = [],
) {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false, gcTime: 0 } },
  });
  for (const [key, value] of seed) client.setQueryData(key, value);

  return (
    <QueryClientProvider client={client}>
      <TooltipProvider>{children}</TooltipProvider>
    </QueryClientProvider>
  );
}
