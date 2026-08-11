"""Content-addressed prompt versioning.

Each agent's system prompt gets a short version string derived from a hash
of its own text, computed once at import time (see extractor_node.py's and
fact_checker_node.py's `_PROMPT_VERSION`). No manual semver bumping to
forget or drift out of sync -- the version changes automatically and only
when the prompt text actually changes, and it's the same string in every
environment running that code, since it's a pure function of the text, not
a timestamp or git SHA that has to be looked up separately.

Recorded on every LLM call (observability.record_llm_call writes it to
llm_calls.prompt_version) and printed by scripts/run_golden_eval.py, so a
pass-rate number -- or a cost/latency figure from scripts/cost_report.py --
can always be traced back to the exact prompt that produced it.
"""

from __future__ import annotations

import hashlib


def prompt_version(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
