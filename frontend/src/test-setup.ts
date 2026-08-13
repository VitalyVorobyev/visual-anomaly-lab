/**
 * Unmount rendered components between tests.
 *
 * Testing Library registers this automatically only when vitest's globals are enabled.
 * They are not, deliberately — an explicit `import { describe, it } from "vitest"` says
 * where the test API comes from — so the teardown is wired here instead. Without it,
 * every render accumulates in the same document and queries start finding several
 * matches for what should be one element.
 */

import { cleanup } from "@testing-library/react";
import { afterEach } from "vitest";

afterEach(cleanup);

/**
 * No test talks to a backend, and one that tries should hang rather than reach the network.
 *
 * A never-settling fetch also *is* the state several layout tests are about: a screen whose
 * queries have not answered yet. The band and its tab strip have to be on screen in that
 * state, because arriving late is how they used to push the rest of the page down.
 */
globalThis.fetch = (() => new Promise(() => {})) as typeof fetch;
