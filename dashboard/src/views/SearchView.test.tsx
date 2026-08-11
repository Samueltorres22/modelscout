import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { SearchView } from "./SearchView";

const { search } = vi.hoisted(() => ({ search: vi.fn() }));

vi.mock("../api", () => ({ api: { search } }));

describe("SearchView", () => {
  it("does nothing on submit when the query box is empty", async () => {
    const user = userEvent.setup();
    render(<SearchView />);

    await user.click(screen.getByRole("button", { name: /search/i }));

    expect(search).not.toHaveBeenCalled();
  });

  it("calls api.search with the typed query and default k, then renders results", async () => {
    const user = userEvent.setup();
    search.mockResolvedValue({
      query: "vision model for OCR",
      results: [
        {
          model_id: "test/vision-model",
          chunk_text: "A lightweight vision-language model for document OCR.",
          score: 0.87,
          pipeline_tag: "image-text-to-text",
          downloads: 999,
          hf_url: "https://huggingface.co/test/vision-model",
        },
      ],
    });

    render(<SearchView />);
    await user.type(screen.getByPlaceholderText(/lightweight vision model/i), "vision model for OCR");
    await user.click(screen.getByRole("button", { name: /search/i }));

    expect(search).toHaveBeenCalledWith("vision model for OCR", 5);
    expect(await screen.findByText("test/vision-model")).toBeInTheDocument();
    expect(screen.getByText("score 0.870")).toBeInTheDocument();
  });

  it("shows a 'no results' hint when the API returns an empty array", async () => {
    const user = userEvent.setup();
    search.mockResolvedValue({ query: "nonexistent", results: [] });

    render(<SearchView />);
    await user.type(screen.getByPlaceholderText(/lightweight vision model/i), "nonexistent");
    await user.click(screen.getByRole("button", { name: /search/i }));

    expect(await screen.findByText(/no results/i)).toBeInTheDocument();
  });

  it("renders the disabled-in-production error message from the API", async () => {
    const user = userEvent.setup();
    search.mockRejectedValue(
      new Error("Semantic search is disabled on this deployment (ENABLE_ML_FEATURES=false). Run locally instead."),
    );

    render(<SearchView />);
    await user.type(screen.getByPlaceholderText(/lightweight vision model/i), "anything");
    await user.click(screen.getByRole("button", { name: /search/i }));

    expect(await screen.findByText(/disabled on this deployment/i)).toBeInTheDocument();
  });
});
