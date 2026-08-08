#!/usr/bin/env python
"""Golden regression eval for the Fact-Checker judge.

Deliberately NOT a pytest test -- this makes real Anthropic API calls (one
per golden example) and costs real money, so it should run manually or in
CI when the judge's prompt/model changes, not on every `pytest` invocation.
This is the actual "eval-driven development" loop: change the prompt, run
this, see if the verdict distribution still matches the hand-labeled
expectations before you ship the change.

    python scripts/run_golden_eval.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml

from modelscout.agents.fact_checker_node import fact_check
from modelscout.agents.schemas import ExtractedModelSpecs

GOLDEN_PATH = Path(__file__).resolve().parent.parent / "config" / "golden" / "fact_check_examples.yaml"


def main() -> int:
    with GOLDEN_PATH.open("r", encoding="utf-8") as f:
        examples = yaml.safe_load(f)

    results = []
    for ex in examples:
        specs = ExtractedModelSpecs.model_validate(ex["extracted"])
        judged = fact_check(specs.model_id, specs)
        # Coarse bucket check: does the ACTUAL verdict match the EXPECTED
        # bucket? Confidence/exact wording will vary run to run -- that's
        # expected and fine, the verdict bucket is the regression signal.
        passed = judged.verdict == ex["expected_verdict"]
        results.append((ex["id"], ex["expected_verdict"], judged.verdict, judged.confidence, passed))

    print(f"\n{'ID':<45} {'EXPECTED':<13} {'ACTUAL':<13} {'CONF':<6} RESULT")
    print("-" * 90)
    n_passed = 0
    for id_, expected, actual, confidence, passed in results:
        status = "PASS" if passed else "FAIL"
        n_passed += passed
        print(f"{id_:<45} {expected:<13} {actual:<13} {confidence:<6.2f} {status}")

    print("-" * 90)
    print(f"{n_passed}/{len(results)} golden examples matched their expected verdict bucket.\n")

    if n_passed < len(results):
        print("Some examples missed their expected bucket. This isn't necessarily a bug -- an")
        print("LLM judge's verdict on a borderline case can legitimately shift between runs.")
        print("But if a change you just made caused several to flip, that's the regression")
        print("this script exists to catch: re-check the prompt/model change before shipping.\n")

    return 0 if n_passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
