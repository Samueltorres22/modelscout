"""Fact-Checker Agent: LLM-as-judge plausibility/consistency read of the
Extractor's declared_benchmarks claims. NOT a re-run of the actual
benchmarks -- infeasible in this project's scope (would mean downloading
and running arbitrary multi-billion-parameter models). This is the same
kind of judgment a human reviewer makes skimming a paper's claims for red
flags without independently reproducing every experiment: is this
plausible for a model this size, is it internally consistent, is it
specific enough to check at all, does it show cherry-picking patterns.

Gated like Extractor is gated by Triage: only runs when there's something
to check (declared_benchmarks non-empty) -- a model with no benchmark
claims has nothing for a fact-checker to fact-check, so skip the API call
rather than spend it on an empty list. See process_model_node in graph.py
for where that gate lives.

Reuses extractor_node's Anthropic client and bracket-matching JSON
extractor rather than duplicating them -- same tolerant-parse philosophy
(tool_use.input -> brace-matched text-JSON -> safe default), so a second
copy would just be two places to keep in sync.
"""

from __future__ import annotations

import json
import logging

from modelscout.agents.extractor_node import _bracket_match_json, _get_client
from modelscout.agents.schemas import ExtractedModelSpecs, FactCheckResult
from modelscout.config import settings

logger = logging.getLogger(__name__)

FACT_CHECK_TOOL = {
    "name": "submit_fact_check",
    "description": "Submit a plausibility/consistency fact-check of this model's declared benchmark claims.",
    "input_schema": {
        "type": "object",
        "properties": {
            "flags": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Work through this FIRST. Specific, concrete red flags (e.g. 'OCRBench 99.8 is above any published score for a 4B model'), one per item. Empty array if none.",
            },
            "consistency_issues": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Specific internal contradictions between the benchmark claims and the stated params/architecture/hardware. Empty array if none.",
            },
            "reasoning": {
                "type": "string",
                "minLength": 1,
                "description": "2-4 sentences synthesizing the flags/consistency_issues above into an overall read, in plain language -- the summary someone would read if they read nothing else. Must not be empty.",
            },
            "confidence": {
                "type": "number",
                "minimum": 0.05,
                "maximum": 1.0,
                "description": "How confident you are in the verdict you're about to give, anchored: 0.8-1.0 = clear-cut (obvious fabrication, or nothing suspicious at all); 0.5-0.8 = a genuine mixed case like some numbers plausible and others not, still a real judgment call; 0.05-0.3 = mostly guessing, too little to go on. Never 0 -- you always have SOME basis for the verdict.",
            },
            "verdict": {
                "type": "string",
                "enum": ["plausible", "questionable", "implausible"],
                "description": "Decide this LAST, after everything above -- it must be consistent with your own flags/reasoning. If your reasoning concludes something looks fabricated, mislabeled, or implausible for the model's size, the verdict must be 'implausible', not 'plausible'.",
            },
        },
        "required": ["flags", "consistency_issues", "reasoning", "confidence", "verdict"],
    },
}

_SYSTEM_PROMPT = """You are a skeptical peer reviewer fact-checking self-reported benchmark claims on a Hugging Face model card. You cannot re-run the benchmarks yourself -- your job is a plausibility and consistency read, the same judgment a reviewer makes skimming a paper's claims for red flags without reproducing every experiment.

Check for:
1. Plausibility given model size: do the claimed scores make sense for a model this size and architecture? A small model claiming to match or beat much larger frontier models on hard reasoning benchmarks is a red flag.
2. Internal consistency: do the benchmark claims contradict the model's own stated params/architecture/hardware requirements (e.g. a claimed hardware footprint implausibly small for the stated parameter count, with no quantization mentioned)?
3. Verifiability: are claims backed by named, recognizable benchmarks with numeric scores, or are they vague ("great results", "outperforms competitors") with nothing concrete to check?
4. Cherry-picking / red-flag patterns: uniformly perfect or near-perfect scores across every benchmark, or a single obscure benchmark with no baseline comparison, are both worth flagging even if not necessarily false.

Ground every flag in something specific you can point to -- no generic "benchmarks should be independently verified" boilerplate. If nothing here is suspicious, say so plainly rather than inventing a concern; mixed/modest results across multiple named benchmarks are typically a sign of an honest card, not a problem.

Work through the fields in the order the tool asks for them: list flags and consistency_issues first, then synthesize them into `reasoning` (never leave it blank), then set `confidence`, and only then commit to `verdict` as the last field. The verdict is a conclusion FROM your reasoning, not a separate judgment -- if your own reasoning says something looks fabricated, mislabeled, or implausible for the model's size, `verdict` must be "implausible", not "plausible". Call the submit_fact_check tool exactly once."""


def _build_user_content(model_id: str, extracted: ExtractedModelSpecs) -> str:
    benchmarks = "\n".join(
        f"- {b.name}: {b.score}" + (f" ({b.metric})" if b.metric else "")
        for b in extracted.declared_benchmarks
    )
    return f"""Model: {model_id}
Parameters: {extracted.params_billion}B
Architecture: {extracted.architecture_family or 'unknown'}
Hardware requirements stated: {extracted.hardware_requirements or 'unknown'}
Quantization available: {', '.join(extracted.quantization_available) or 'none stated'}

Declared benchmarks:
{benchmarks}"""


def _normalize_result(result: FactCheckResult) -> FactCheckResult:
    """Two fields the schema marks required/bounded still occasionally come
    back empty/zero under forced tool-use, verified empirically across
    repeated live calls -- and more often on models with very large
    declared_benchmarks lists (Claude appears to spend its "attention" on
    the itemized flags and shortchange the summary fields as input size
    grows, the same class of behavior that motivated reordering verdict to
    be decided last). required/enum/type ARE reliably enforced; minLength
    and numeric minimum/maximum are NOT. Rather than keep spending live API
    calls chasing 100% prompt-only compliance, patch it here -- same
    tolerant-fallback philosophy as everywhere else in this project: never
    surface an empty/misleading field when there's a sane derivable default.
    """
    if result.confidence == 0.0 and not result.parse_error:
        result.confidence = 0.5

    if not result.reasoning.strip() and not result.parse_error:
        if result.flags:
            result.reasoning = f"{result.verdict.capitalize()} verdict. " + result.flags[0]
        else:
            result.reasoning = f"{result.verdict.capitalize()} verdict (no reasoning text returned)."

    return result


def parse_fact_check_response(response, model_id: str) -> FactCheckResult:
    """3-level tolerant parse, identical philosophy to extractor_node's
    parse_claude_response: tool_use.input -> brace-matched text-JSON ->
    safe default. Never raises.
    """

    def fallback(reason: str, raw_text: str | None) -> FactCheckResult:
        return FactCheckResult(
            model_id=model_id,
            verdict="questionable",
            confidence=0.0,
            flags=[],
            consistency_issues=[],
            reasoning="Fact-check could not be completed automatically; needs manual review.",
            parse_error=True,
            parse_error_detail=reason,
            raw_model_response=(raw_text or "")[:500],
        )

    httpError = getattr(response, "error", None)
    if httpError:
        return fallback(f"claude_api_error: {str(httpError)[:300]}", None)

    content = getattr(response, "content", [])

    for block in content:
        if getattr(block, "type", None) == "tool_use" and block.name == "submit_fact_check":
            try:
                return _normalize_result(
                    FactCheckResult.model_validate({**block.input, "model_id": model_id})
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("tool_use.input failed validation for %s: %s", model_id, exc)
                return fallback(f"tool_use validation error: {exc}", json.dumps(block.input))

    text_parts = [b.text for b in content if getattr(b, "type", None) == "text"]
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

    if isinstance(parsed, dict) and "verdict" in parsed:
        try:
            return _normalize_result(FactCheckResult.model_validate({**parsed, "model_id": model_id}))
        except Exception as exc:  # noqa: BLE001
            logger.warning("text-JSON fact-check extraction failed validation for %s: %s", model_id, exc)

    return fallback("no_valid_tool_use_or_json", raw_text)


def fact_check(model_id: str, extracted: ExtractedModelSpecs) -> FactCheckResult:
    client = _get_client()
    response = client.messages.create(
        model=settings.anthropic_extractor_model,
        max_tokens=1024,
        system=_SYSTEM_PROMPT,
        tools=[FACT_CHECK_TOOL],
        tool_choice={"type": "tool", "name": "submit_fact_check"},
        messages=[{"role": "user", "content": _build_user_content(model_id, extracted)}],
    )
    return parse_fact_check_response(response, model_id)
