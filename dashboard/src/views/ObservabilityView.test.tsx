import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { ObservabilityView } from "./ObservabilityView";

const { observabilitySummary } = vi.hoisted(() => ({ observabilitySummary: vi.fn() }));

vi.mock("../api", () => ({ api: { observabilitySummary } }));

describe("ObservabilityView", () => {
  it("shows a loading state before the summary resolves", () => {
    observabilitySummary.mockReturnValue(new Promise(() => {}));

    render(<ObservabilityView />);

    expect(screen.getByText(/loading/i)).toBeInTheDocument();
  });

  it("shows a hint when no LLM calls have been recorded yet", async () => {
    observabilitySummary.mockResolvedValue({ agents: [] });

    render(<ObservabilityView />);

    expect(await screen.findByText(/no llm calls recorded yet/i)).toBeInTheDocument();
  });

  it("renders per-agent rows and a correctly summed totals row", async () => {
    observabilitySummary.mockResolvedValue({
      agents: [
        {
          agent_name: "extractor",
          n_calls: 2,
          total_input_tokens: 1000,
          total_output_tokens: 500,
          avg_latency_ms: 2000,
          total_cost_usd: 0.01,
        },
        {
          agent_name: "fact_checker",
          n_calls: 3,
          total_input_tokens: 3000,
          total_output_tokens: 1500,
          avg_latency_ms: 1000,
          total_cost_usd: 0.02,
        },
      ],
    });

    render(<ObservabilityView />);

    expect(await screen.findByText("extractor")).toBeInTheDocument();
    expect(screen.getByText("fact_checker")).toBeInTheDocument();

    // Totals row: n_calls 2+3=5, input 1000+3000=4000, output 500+1500=2000, cost 0.01+0.02=0.03.
    // Token counts use toLocaleString() itself rather than a hardcoded
    // separator -- see ModelsView.test.tsx for why.
    expect(screen.getByText("5")).toBeInTheDocument();
    expect(screen.getByText((4000).toLocaleString())).toBeInTheDocument();
    expect(screen.getByText((2000).toLocaleString())).toBeInTheDocument();
    expect(screen.getByText("$0.0300")).toBeInTheDocument();
  });

  it("shows 'n/a' for cost and the pricing hint when no cost data is configured", async () => {
    observabilitySummary.mockResolvedValue({
      agents: [
        {
          agent_name: "extractor",
          n_calls: 1,
          total_input_tokens: 100,
          total_output_tokens: 50,
          avg_latency_ms: 500,
          total_cost_usd: null,
        },
      ],
    });

    render(<ObservabilityView />);

    await screen.findByText("extractor");
    expect(screen.getAllByText("n/a").length).toBeGreaterThan(0);
    expect(screen.getByText(/PRICE_PER_MTOK_INPUT/)).toBeInTheDocument();
  });
});
