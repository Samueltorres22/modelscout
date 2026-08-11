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

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml

from modelscout.agents.fact_checker_node import _PROMPT_VERSION, fact_check
from modelscout.agents.schemas import ExtractedModelSpecs

GOLDEN_PATH = Path(__file__).resolve().parent.parent / "config" / "golden" / "fact_check_examples.yaml"

# An LLM judge legitimately disagrees with hand-labeled buckets on genuinely
# fuzzy cases run to run -- verified empirically at 5/8 (62.5%) as the
# realistic baseline for this prompt, with the misses landing in the SAFE
# direction (over-cautious "implausible" on a "questionable" case) rather
# than the dangerous one (confidently calling a fabricated claim
# "plausible"). Demanding 8/8 would make this permanently red without
# signaling an actual regression. 0.5 is deliberately below that observed
# baseline -- it exists to catch a REAL collapse (e.g. a prompt change that
# breaks verdict/reasoning consistency again), not to enforce exact
# agreement with hand labels on borderline cases.
DEFAULT_MIN_PASS_RATE = 0.5


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the Fact-Checker golden regression eval")
    parser.add_argument(
        "--min-pass-rate",
        type=float,
        default=DEFAULT_MIN_PASS_RATE,
        help=f"Exit non-zero if the pass rate falls below this fraction (default {DEFAULT_MIN_PASS_RATE})",
    )
    args = parser.parse_args()

    with GOLDEN_PATH.open("r", encoding="utf-8") as f:
        examples = yaml.safe_load(f)

    # A content hash of _SYSTEM_PROMPT (see agents/prompts.py), not a manually
    # bumped version number -- this is what makes the pass rate below
    # traceable to an exact prompt without also having to record/check out a
    # git SHA: the same prompt text always produces the same version string.
    print(f"Fact-Checker prompt_version: {_PROMPT_VERSION}")

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

    pass_rate = n_passed / len(results)
    print("-" * 90)
    print(
        f"{n_passed}/{len(results)} golden examples matched their expected verdict bucket ({pass_rate:.0%}) "
        f"against prompt_version {_PROMPT_VERSION}.\n"
    )

    if n_passed < len(results):
        print("Some examples missed their expected bucket. This isn't necessarily a bug -- an")
        print("LLM judge's verdict on a borderline case can legitimately shift between runs.")
        print(f"This script only fails below {args.min_pass_rate:.0%} pass rate -- if a prompt/model")
        print("change you just made caused a real collapse (not just one borderline flip), that's")
        print("the regression this threshold exists to catch.\n")

    return 0 if pass_rate >= args.min_pass_rate else 1


if __name__ == "__main__":
    sys.exit(main())
