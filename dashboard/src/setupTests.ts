import "@testing-library/jest-dom/vitest";

import { cleanup } from "@testing-library/react";
import { afterEach, vi } from "vitest";

// @testing-library/react's own auto-cleanup only registers itself when it
// detects test-framework globals (Vitest's `globals: true`) -- this project
// deliberately imports `describe`/`it`/`expect`/etc. explicitly per test
// file instead of turning that on, so cleanup has to be wired by hand here.
//
// vi.resetAllMocks() is here for the same reason: every ../api mock in this
// suite is a module-level `vi.fn()` (via vi.hoisted), so its call history
// AND queued mockResolvedValueOnce/mockImplementation behavior would
// otherwise leak between tests in the same file -- verified live: without
// this, a test asserting `getModel` was called exactly once failed because
// a *previous* test's click had already called it once.
afterEach(() => {
  cleanup();
  vi.resetAllMocks();
});
