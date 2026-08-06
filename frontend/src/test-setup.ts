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
