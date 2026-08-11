"""Unit tests for content-addressed prompt versioning (modelscout/agents/
prompts.py) -- pure logic, no DB or network.
"""

from __future__ import annotations

from modelscout.agents.extractor_node import _PROMPT_VERSION as EXTRACTOR_PROMPT_VERSION
from modelscout.agents.extractor_node import _SYSTEM_PROMPT as EXTRACTOR_SYSTEM_PROMPT
from modelscout.agents.fact_checker_node import _PROMPT_VERSION as FACT_CHECKER_PROMPT_VERSION
from modelscout.agents.fact_checker_node import _SYSTEM_PROMPT as FACT_CHECKER_SYSTEM_PROMPT
from modelscout.agents.prompts import prompt_version


def test_prompt_version_is_deterministic():
    assert prompt_version("hello") == prompt_version("hello")


def test_prompt_version_changes_with_the_text():
    assert prompt_version("hello") != prompt_version("hello!")


def test_prompt_version_is_a_short_hex_string():
    version = prompt_version("some system prompt")
    assert len(version) == 12
    assert all(c in "0123456789abcdef" for c in version)


def test_extractor_prompt_version_matches_its_own_prompt_text():
    # Regression-proofs the wiring itself, not just the hash function --
    # catches the module computing _PROMPT_VERSION from stale/copy-pasted
    # text instead of the actual _SYSTEM_PROMPT it sends to Claude.
    assert prompt_version(EXTRACTOR_SYSTEM_PROMPT) == EXTRACTOR_PROMPT_VERSION


def test_fact_checker_prompt_version_matches_its_own_prompt_text():
    assert prompt_version(FACT_CHECKER_SYSTEM_PROMPT) == FACT_CHECKER_PROMPT_VERSION


def test_extractor_and_fact_checker_have_different_prompt_versions():
    assert EXTRACTOR_PROMPT_VERSION != FACT_CHECKER_PROMPT_VERSION
