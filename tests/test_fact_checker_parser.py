"""Unit tests for the Fact-Checker's tolerant parser. No API calls -- mirrors
tests/test_parser.py's approach for the Extractor.
"""

from types import SimpleNamespace

from modelscout.agents.fact_checker_node import parse_fact_check_response

VALID_INPUT = {
    "verdict": "implausible",
    "confidence": 0.9,
    "flags": ["MMLU 98.7 for a 0.5B model is above any published frontier-model score"],
    "consistency_issues": [],
    "reasoning": "A 0.5B model claiming near-perfect MMLU is not plausible given known scaling behavior.",
}


def _tool_use_response(input_dict):
    block = SimpleNamespace(type="tool_use", name="submit_fact_check", input=input_dict)
    return SimpleNamespace(content=[block])


def _text_response(text):
    block = SimpleNamespace(type="text", text=text)
    return SimpleNamespace(content=[block])


def test_level1_clean_tool_use():
    response = _tool_use_response(VALID_INPUT)
    result = parse_fact_check_response(response, "org/model")
    assert result.parse_error is False
    assert result.verdict == "implausible"
    assert len(result.flags) == 1


def test_level2_markdown_fenced_json_text():
    import json

    text = "Here's my assessment:\n```json\n" + json.dumps(VALID_INPUT) + "\n```"
    response = _text_response(text)
    result = parse_fact_check_response(response, "org/model")
    assert result.parse_error is False
    assert result.confidence == 0.9


def test_level3_garbage_falls_back_safely():
    response = _text_response("I'm not sure how to assess this one, sorry.")
    result = parse_fact_check_response(response, "org/model")
    assert result.parse_error is True
    assert result.model_id == "org/model"
    assert result.flags == []


def test_level1_invalid_tool_input_falls_back_with_error_flagged():
    bad_input = {**VALID_INPUT, "confidence": "not-a-number"}
    response = _tool_use_response(bad_input)
    result = parse_fact_check_response(response, "org/model")
    assert result.parse_error is True
    assert result.parse_error_detail is not None


def test_confidence_zero_is_normalized_to_neutral_midpoint():
    # Verified empirically: Claude's forced tool-use doesn't enforce the
    # schema's minimum=0.05 on `confidence` the way it enforces
    # required/enum/type -- a bare 0.0 sometimes comes back anyway. That's
    # never a real calibrated value under this rubric, so it should be
    # normalized rather than surfaced as-is.
    zero_confidence_input = {**VALID_INPUT, "confidence": 0.0}
    response = _tool_use_response(zero_confidence_input)
    result = parse_fact_check_response(response, "org/model")
    assert result.parse_error is False
    assert result.confidence == 0.5


def test_genuine_nonzero_confidence_is_left_untouched():
    response = _tool_use_response({**VALID_INPUT, "confidence": 0.9})
    result = parse_fact_check_response(response, "org/model")
    assert result.confidence == 0.9


def test_empty_reasoning_is_synthesized_from_first_flag():
    # Verified empirically, especially on models with very long
    # declared_benchmarks lists: reasoning sometimes comes back empty
    # despite minLength=1 in the schema and explicit prompt instructions.
    # Should never surface a blank field when a flag is available to build
    # a fallback synthesis from.
    empty_reasoning_input = {**VALID_INPUT, "reasoning": ""}
    response = _tool_use_response(empty_reasoning_input)
    result = parse_fact_check_response(response, "org/model")
    assert result.parse_error is False
    assert result.reasoning != ""
    assert "Implausible" in result.reasoning


def test_empty_reasoning_with_no_flags_gets_generic_fallback():
    empty_everything = {**VALID_INPUT, "reasoning": "", "flags": []}
    response = _tool_use_response(empty_everything)
    result = parse_fact_check_response(response, "org/model")
    assert result.reasoning != ""


def test_genuine_reasoning_is_left_untouched():
    response = _tool_use_response(VALID_INPUT)
    result = parse_fact_check_response(response, "org/model")
    assert result.reasoning == VALID_INPUT["reasoning"]
