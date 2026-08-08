#!/usr/bin/env python
"""CLI entrypoint: runs the full pipeline locally without needing the API up.

    python cli.py run --profile vlm_ocr --limit 20
"""

from __future__ import annotations

import argparse
import logging

from modelscout.pipeline import run as run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="ModelScout CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run the ingest -> triage -> extract -> digest pipeline")
    run_parser.add_argument("--profile", required=True, help="Interest profile name (config/interest_profiles/<name>.yaml)")
    run_parser.add_argument("--limit", type=int, default=20, help="Max candidates to pull per pipeline_tag")
    run_parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    # Always show our own progress logs regardless of -v (third-party libs stay quiet by default).
    logging.getLogger("modelscout").setLevel(logging.INFO)

    if args.command == "run":
        summary = run_pipeline(args.profile, limit_per_tag=args.limit)
        counts = summary["counts"]
        print(f"\nProfile: {summary['profile']}")
        print(
            f"Ingested {counts['ingested']} -> "
            f"Triage pass (local, $0): {counts['triage_pass']} -> "
            f"Extracted via Claude: {counts['extracted']} "
            f"({counts['parse_errors']} parse error(s)) -> "
            f"Fact-checked: {counts['fact_checked']} "
            f"({counts['implausible']} flagged implausible)"
        )
        print(f"Digest written to: {summary['digest_markdown_path']}")


if __name__ == "__main__":
    main()
