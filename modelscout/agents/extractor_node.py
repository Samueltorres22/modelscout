"""Extractor Agent: the only function that calls the Anthropic API. Called
by process_model_node (see graph.py) only when triage_node marked a model
relevant -- see that module's docstring for why this is an explicit if/else
rather than a graph-structural guarantee, and pipeline.py for the concrete
`n_extracted == n_triage_pass` check that verifies it held on every run.

Uses forced tool-use (tool_choice pinned to the one tool) rather than asking
the model to "reply in JSON" -- when it works, tool_use.input arrives as an
already-parsed object with nothing left to parse. The 3-level fallback below
mirrors the pattern built for the n8n github-pr-issue-triage project:
tool_use.input -> brace-matched text-JSON extraction -> safe default with
parse_error=True. Never raises out of extract_specs(); never crashes the
pipeline over one bad model card.
"""

from __future__ import annotations

import json
import logging
import threading

import anthropic

from modelscout.agents.schemas import ExtractedModelSpecs
from modelscout.agents.state import ModelState
from modelscout.config import settings

logger = logging.getLogger(__name__)

_client: anthropic.Anthropic | None = None
_client_lock = threading.Lock()


def _get_client() -> anthropic.Anthropic:
    # extractor_node runs concurrently across Send-fanned-out branches (see
    # the identical race documented in agents/triage_node.py) -- double-
    # checked locking avoids constructing redundant client instances.
    global _client
    if _client is None:
        with _client_lock:
            if _client is None:
                _client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    return _client


EXTRACT_TOOL = {
    "name": "extract_model_specs",
    "description": "Extract structured specifications from a Hugging Face model card README.",
    "input_schema": {
        "type": "object",
        "properties": {
            "params_billion": {
                "type": ["number", "null"],
                "description": "Total parameter count in billions, if stated or clearly inferable.",
            },
            "license": {"type": ["string", "null"]},
            "architecture_family": {
                "type": ["string", "null"],
                "description": "e.g. 'LLaVA', 'Qwen2-VL', 'SigLIP+Llama'.",
            },
            "hardware_requirements": {
                "type": ["string", "null"],
                "description": "Free text, e.g. '16GB VRAM (fp16)'.",
            },
            "quantization_available": {
                "type": "array",
                "items": {"type": "string"},
                "description": "e.g. ['4-bit', 'GGUF']. Empty array if none mentioned.",
            },
            "declared_benchmarks": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "metric": {"type": ["string", "null"]},
                        "score": {"type": ["number", "string", "null"]},
                    },
                    "required": ["name"],
                },
                "description": "Benchmarks the card itself claims, with whatever score is reported. Empty array if none.",
            },
        },
        "required": [
            "params_billion",
            "license",
            "architecture_family",
            "hardware_requirements",
            "quantization_available",
            "declared_benchmarks",
        ],
    },
}

_SYSTEM_PROMPT = (
    "You extract structured specs from Hugging Face model card READMEs. "
    "Ground every field in what the README actually says -- if a field isn't stated, "
    "use null (or an empty array for list fields) rather than guessing. "
    "Call the extract_model_specs tool exactly once with your extraction."
)


def _bracket_match_json(text: str) -> dict | None:
    """Find the first balanced {...} in text and parse it. Bracket-counting,
    not regex -- naive regex mishandles nested braces in declared_benchmarks.
    """
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start : i + 1]
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    return None
    return None


def parse_claude_response(response: anthropic.types.Message, model_id: str) -> ExtractedModelSpecs:
    """3-level tolerant parse. Never raises."""
    # Level 1: forced tool_use, input is already a real object.
    for block in response.content:
        if getattr(block, "type", None) == "tool_use" and block.name == "extract_model_specs":
            try:
                return ExtractedModelSpecs.model_validate({**block.input, "model_id": model_id})
            except Exception as exc:  # noqa: BLE001
                logger.warning("tool_use.input failed validation for %s: %s", model_id, exc)
                return ExtractedModelSpecs(
                    model_id=model_id,
                    parse_error=True,
                    parse_error_detail=f"tool_use validation error: {exc}",
                    raw_model_response=block.input,
                )

    # Level 2: no tool_use block -- salvage JSON out of any text reply.
    text_parts = [b.text for b in response.content if getattr(b, "type", None) == "text"]
    raw_text = "\n".join(text_parts)
    cleaned = raw_text.strip()
    for fence in ("```json", "```"):
        if cleaned.startswith(fence):
            cleaned = cleaned[len(fence) :]
    cleaned = cleaned.strip().removesuffix("```").strip()

    parsed = None
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        parsed = _bracket_match_json(cleaned)

    if isinstance(parsed, dict) and "license" in parsed:
        try:
            return ExtractedModelSpecs.model_validate({**parsed, "model_id": model_id})
        except Exception as exc:  # noqa: BLE001
            logger.warning("text-JSON extraction failed validation for %s: %s", model_id, exc)

    # Level 3: nothing usable. Fail open with a safe default.
    return ExtractedModelSpecs(
        model_id=model_id,
        parse_error=True,
        parse_error_detail="no_valid_tool_use_or_json",
        raw_model_response=raw_text[:2000] if raw_text else None,
    )


def extract_specs(model_id: str, readme_text: str) -> ExtractedModelSpecs:
    client = _get_client()
    response = client.messages.create(
        model=settings.anthropic_extractor_model,
        max_tokens=2048,
        system=_SYSTEM_PROMPT,
        tools=[EXTRACT_TOOL],
        tool_choice={"type": "tool", "name": "extract_model_specs"},
        messages=[
            {
                "role": "user",
                "content": f"Extract specs from this model card:\n\n{(readme_text or '')[:12000]}",
            }
        ],
    )
    return parse_claude_response(response, model_id)


def _base_result(state: ModelState) -> dict:
    """Fields common to every outcome, carried from the triage step. This is
    also the actual fan-in point into PipelineState.results (operator.add) --
    everything ModelState accumulated for this one model gets folded into a
    single dict here, since nothing downstream reads per-branch ModelState
    directly (notifier_node only reads PipelineState.results).
    """
    return {
        "model_id": state["model_id"],
        "downloads": state.get("downloads", 0),
        "triage_label": state.get("triage_label"),
        "triage_confidence": state.get("triage_confidence", 0.0),
        "is_relevant": state.get("is_relevant", False),
    }


def extractor_node(state: ModelState) -> dict:
    specs = extract_specs(state["model_id"], state["readme_text"])
    result = {**_base_result(state), "extracted": specs.model_dump(), "parse_error": specs.parse_error}
    return {"results": [result]}


def skip_node(state: ModelState) -> dict:
    """Trivial pass-through for models that didn't pass triage. No LLM/HF
    calls here -- this is the cheap side of the cost-aware routing split.
    """
    result = {**_base_result(state), "extracted": None, "parse_error": False}
    return {"results": [result]}
