"""Unit tests for the Extractor Agent's 3-level tolerant parser. No API calls
-- constructs synthetic anthropic.types.Message-shaped objects directly.
"""

from types import SimpleNamespace

from modelscout.agents.extractor_node import parse_claude_response

VALID_INPUT = {
    "params_billion": 2.0,
    "license": "apache-2.0",
    "architecture_family": "LLaVA",
    "hardware_requirements": "8GB VRAM",
    "quantization_available": ["4-bit"],
    "declared_benchmarks": [{"name": "DocVQA", "metric": "accuracy", "score": 82.3}],
}


def _tool_use_response(input_dict):
    block = SimpleNamespace(type="tool_use", name="extract_model_specs", input=input_dict)
    return SimpleNamespace(content=[block])


def _text_response(text):
    block = SimpleNamespace(type="text", text=text)
    return SimpleNamespace(content=[block])


def test_level1_clean_tool_use():
    response = _tool_use_response(VALID_INPUT)
    specs = parse_claude_response(response, "org/model")
    assert specs.parse_error is False
    assert specs.license == "apache-2.0"
    assert specs.declared_benchmarks[0].name == "DocVQA"


def test_level2_markdown_fenced_json_text():
    text = "Sure, here's the extraction:\n```json\n" + _to_json(VALID_INPUT) + "\n```"
    response = _text_response(text)
    specs = parse_claude_response(response, "org/model")
    assert specs.parse_error is False
    assert specs.architecture_family == "LLaVA"


def test_level2_brace_matched_json_with_surrounding_prose():
    text = "I looked at the card and here is what I found: " + _to_json(VALID_INPUT) + " Hope that helps!"
    response = _text_response(text)
    specs = parse_claude_response(response, "org/model")
    assert specs.parse_error is False
    assert specs.hardware_requirements == "8GB VRAM"


def test_level3_garbage_falls_back_safely():
    response = _text_response("I couldn't determine the specs from this README, sorry about that.")
    specs = parse_claude_response(response, "org/model")
    assert specs.parse_error is True
    assert specs.model_id == "org/model"
    # Must never raise, and must always be a valid, safe default.
    assert specs.declared_benchmarks == []


def test_level1_invalid_tool_input_falls_back_with_error_flagged():
    # params_billion isn't coercible to float|None -- Pydantic validation
    # should fail cleanly rather than raise out of parse_claude_response.
    bad_input = {**VALID_INPUT, "params_billion": "not-a-number"}
    response = _tool_use_response(bad_input)
    specs = parse_claude_response(response, "org/model")
    assert specs.parse_error is True
    assert specs.parse_error_detail is not None


def _to_json(d: dict) -> str:
    import json

    return json.dumps(d)
