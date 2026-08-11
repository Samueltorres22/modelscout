import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { RunPipelineView } from "./RunPipelineView";

const { runPipeline, getPipelineRunStatus } = vi.hoisted(() => ({
  runPipeline: vi.fn(),
  getPipelineRunStatus: vi.fn(),
}));

vi.mock("../api", () => ({
  api: { runPipeline, getPipelineRunStatus },
}));

// Deliberately real timers, not vi.useFakeTimers(): the component polls on
// a real 2s setInterval, and Testing Library's findBy*/waitFor also poll via
// real setTimeout under the hood -- mixing those with a fake clock means the
// two have to be advanced in lockstep by hand, which is fragile. A few real
// seconds of wall-clock time per test is a fine trade for that not being a
// source of flakiness.
const POLL_INTERVAL_MS = 2000;

// Mock reset between tests is handled globally in setupTests.ts.

async function submit(user: ReturnType<typeof userEvent.setup>) {
  await user.click(screen.getByRole("button", { name: /run pipeline/i }));
}

describe("RunPipelineView", () => {
  it("schedules a run (202) and shows the run_id without blocking on completion", async () => {
    const user = userEvent.setup();
    runPipeline.mockResolvedValue({ run_id: "run-123", status: "pending" });
    getPipelineRunStatus.mockReturnValue(new Promise(() => {})); // never resolves in this test

    render(<RunPipelineView />);
    await submit(user);

    expect(await screen.findByText(/run_id: run-123/)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /scheduled/i })).toBeDisabled();
  });

  it(
    "polls status and renders the result once the run completes",
    async () => {
      const user = userEvent.setup();
      runPipeline.mockResolvedValue({ run_id: "run-123", status: "pending" });
      getPipelineRunStatus
        .mockResolvedValueOnce({ run_id: "run-123", status: "running", result: null, error: null })
        .mockResolvedValueOnce({
          run_id: "run-123",
          status: "completed",
          result: {
            profile: "vlm_ocr",
            counts: { ingested: 3, triage_pass: 2, extracted: 2, parse_errors: 0, fact_checked: 2, implausible: 0 },
            digest_markdown_path: "digests/2026-08-01-vlm_ocr.md",
            top_models: [],
          },
          error: null,
        });

      render(<RunPipelineView />);
      await submit(user);

      expect(
        await screen.findByRole("button", { name: /running/i }, { timeout: POLL_INTERVAL_MS + 1000 }),
      ).toBeInTheDocument();

      expect(
        await screen.findByText(/run complete: vlm_ocr/i, {}, { timeout: POLL_INTERVAL_MS + 1000 }),
      ).toBeInTheDocument();
      expect(screen.getByText("digests/2026-08-01-vlm_ocr.md", { exact: false })).toBeInTheDocument();
      expect(screen.getByRole("button", { name: /run pipeline/i })).not.toBeDisabled();
    },
    2 * POLL_INTERVAL_MS + 5000,
  );

  it(
    "stops polling and surfaces the error when a run fails",
    async () => {
      const user = userEvent.setup();
      runPipeline.mockResolvedValue({ run_id: "run-456", status: "pending" });
      getPipelineRunStatus.mockResolvedValue({
        run_id: "run-456",
        status: "failed",
        result: null,
        error: "HF Hub rate limit exceeded",
      });

      render(<RunPipelineView />);
      await submit(user);

      expect(
        await screen.findByText(/hf hub rate limit exceeded/i, {}, { timeout: POLL_INTERVAL_MS + 1000 }),
      ).toBeInTheDocument();
      expect(screen.getByRole("button", { name: /run pipeline/i })).not.toBeDisabled();

      // No further polling once failed -- a second interval tick shouldn't add more calls.
      const callsAtFailure = getPipelineRunStatus.mock.calls.length;
      await new Promise((r) => setTimeout(r, POLL_INTERVAL_MS + 500));
      expect(getPipelineRunStatus.mock.calls.length).toBe(callsAtFailure);
    },
    2 * POLL_INTERVAL_MS + 5000,
  );

  it("shows an error immediately if scheduling the run itself fails (e.g. unknown profile)", async () => {
    const user = userEvent.setup();
    runPipeline.mockRejectedValue(new Error("Interest profile not found: made-up-profile"));

    render(<RunPipelineView />);
    await submit(user);

    expect(await screen.findByText(/interest profile not found/i)).toBeInTheDocument();
    expect(screen.queryByText(/run_id:/)).not.toBeInTheDocument();
  });
});
