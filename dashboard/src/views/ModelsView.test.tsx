import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import type { ModelDetail, ModelSummary } from "../api";
import { ModelsView } from "./ModelsView";

const { listModels, getModel } = vi.hoisted(() => ({
  listModels: vi.fn(),
  getModel: vi.fn(),
}));

vi.mock("../api", () => ({
  api: { listModels, getModel },
}));

function makeModel(overrides: Partial<ModelSummary> = {}): ModelSummary {
  return {
    model_id: "test/model-a",
    pipeline_tag: "text-generation",
    downloads: 1234,
    likes: 5,
    matched_profile: "vlm_ocr",
    ingested_at: "2026-08-01T00:00:00Z",
    is_relevant: true,
    triage_confidence: 0.91,
    params_billion: 7,
    license: "mit",
    architecture_family: "transformer",
    extraction_parse_error: false,
    n_benchmarks: 2,
    fact_check_verdict: "plausible",
    fact_check_confidence: 0.8,
    ...overrides,
  };
}

function makeDetail(overrides: Partial<ModelDetail> = {}): ModelDetail {
  return {
    model_id: "test/model-a",
    pipeline_tag: "text-generation",
    downloads: 1234,
    likes: 5,
    tags: ["text-generation"],
    matched_profile: "vlm_ocr",
    ingested_at: "2026-08-01T00:00:00Z",
    hf_url: "https://huggingface.co/test/model-a",
    is_relevant: true,
    triage_confidence: 0.91,
    params_billion: 7,
    license: "mit",
    architecture_family: "transformer",
    hardware_requirements: "1x A100",
    quantization_available: ["int8"],
    declared_benchmarks: [{ name: "MMLU", metric: "accuracy", score: 0.7 }],
    extraction_parse_error: false,
    fact_check_verdict: "plausible",
    fact_check_confidence: 0.8,
    fact_check_flags: [],
    fact_check_consistency_issues: [],
    fact_check_reasoning: "Numbers check out against reported params.",
    fact_check_parse_error: false,
    ...overrides,
  };
}

describe("ModelsView", () => {
  it("shows a loading state before the models resolve", () => {
    listModels.mockReturnValue(new Promise(() => {}));

    render(<ModelsView />);

    expect(screen.getByText(/loading/i)).toBeInTheDocument();
  });

  it("renders an error box when the API call fails", async () => {
    listModels.mockRejectedValue(new Error("503 Service Unavailable"));

    render(<ModelsView />);

    expect(await screen.findByText(/failed to load models/i)).toBeInTheDocument();
    expect(screen.getByText(/503 Service Unavailable/)).toBeInTheDocument();
  });

  it("shows an empty-state hint when no models are ingested yet", async () => {
    listModels.mockResolvedValue({ models: [] });

    render(<ModelsView />);

    expect(await screen.findByText(/no models ingested yet/i)).toBeInTheDocument();
  });

  it("renders one row per model with its catalog fields", async () => {
    listModels.mockResolvedValue({ models: [makeModel()] });

    render(<ModelsView />);

    expect(await screen.findByText("test/model-a")).toBeInTheDocument();
    // Uses toLocaleString() itself rather than a hardcoded "1,234" -- the
    // thousands separator is locale-dependent (this project's runtime
    // formats it as "1.234" under an es-* default locale, "1,234" under
    // en-US), and the point of this assertion is "downloads got formatted",
    // not "formatted for en-US specifically".
    expect(screen.getByText((1234).toLocaleString())).toBeInTheDocument();
    expect(screen.getByText("mit")).toBeInTheDocument();
    expect(screen.getByText("7B")).toBeInTheDocument();
  });

  it("expands a row on click and loads the model detail panel", async () => {
    const user = userEvent.setup();
    listModels.mockResolvedValue({ models: [makeModel()] });
    getModel.mockResolvedValue(makeDetail());

    render(<ModelsView />);

    const row = await screen.findByText("test/model-a");
    await user.click(row);

    expect(getModel).toHaveBeenCalledWith("test/model-a");
    expect(await screen.findByText(/numbers check out against reported params/i)).toBeInTheDocument();
    expect(screen.getByText("1x A100")).toBeInTheDocument();
  });

  it("collapses the row again on a second click, without a second fetch", async () => {
    const user = userEvent.setup();
    listModels.mockResolvedValue({ models: [makeModel()] });
    getModel.mockResolvedValue(makeDetail());

    render(<ModelsView />);

    const row = await screen.findByText("test/model-a");
    await user.click(row);
    await screen.findByText(/numbers check out against reported params/i);
    await user.click(row);

    await waitFor(() => {
      expect(screen.queryByText(/numbers check out against reported params/i)).not.toBeInTheDocument();
    });
    expect(getModel).toHaveBeenCalledTimes(1);
  });
});
