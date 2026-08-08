#!/usr/bin/env python
"""Summarize LLM call telemetry recorded by modelscout/observability.py.

    python scripts/cost_report.py               # all-time summary
    python scripts/cost_report.py --since 24h    # last 24 hours
"""

from __future__ import annotations

import argparse
import sys
from datetime import timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modelscout.db import get_connection

_UNIT_SECONDS = {"h": 3600, "d": 86400, "m": 60}


def _parse_since(value: str) -> timedelta:
    unit = value[-1]
    if unit not in _UNIT_SECONDS:
        raise ValueError(f"--since must end in h/d/m, e.g. '24h', '7d'. Got: {value}")
    return timedelta(seconds=int(value[:-1]) * _UNIT_SECONDS[unit])


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize LLM call cost/latency telemetry")
    parser.add_argument("--since", help="Only include calls in the last N h/d/m, e.g. '24h', '7d'")
    args = parser.parse_args()

    where_clause = ""
    params: tuple = ()
    if args.since:
        where_clause = "WHERE called_at >= now() - %s"
        params = (_parse_since(args.since),)

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT
                    agent_name,
                    count(*) AS n_calls,
                    sum(input_tokens) AS total_input_tokens,
                    sum(output_tokens) AS total_output_tokens,
                    avg(latency_ms) AS avg_latency_ms,
                    sum(estimated_cost_usd) AS total_cost_usd
                FROM llm_calls
                {where_clause}
                GROUP BY agent_name
                ORDER BY agent_name
                """,
                params,
            )
            rows = cur.fetchall()

    if not rows:
        print("No LLM calls recorded yet (llm_calls table is empty for this window).")
        return 0

    print(f"\n{'AGENT':<15} {'CALLS':<7} {'IN TOK':<10} {'OUT TOK':<10} {'AVG LATENCY':<13} {'EST. COST':<10}")
    print("-" * 70)

    total_calls = total_in = total_out = 0
    total_cost = 0.0
    any_cost = False

    for agent_name, n_calls, in_tok, out_tok, avg_latency, cost_usd in rows:
        cost_str = f"${cost_usd:.4f}" if cost_usd is not None else "n/a"
        if cost_usd is not None:
            total_cost += cost_usd
            any_cost = True
        print(f"{agent_name:<15} {n_calls:<7} {in_tok:<10} {out_tok:<10} {avg_latency:<13.0f} {cost_str:<10}")
        total_calls += n_calls
        total_in += in_tok
        total_out += out_tok

    print("-" * 70)
    cost_total_str = f"${total_cost:.4f}" if any_cost else "n/a (set PRICE_PER_MTOK_INPUT/OUTPUT in .env to estimate)"
    print(f"{'TOTAL':<15} {total_calls:<7} {total_in:<10} {total_out:<10} {'':<13} {cost_total_str}")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
