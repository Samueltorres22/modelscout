"""Lightweight LLM-call observability: tokens, latency, and (optionally)
estimated cost per agent call, persisted to Postgres.

Self-built rather than integrating a hosted product (e.g. Langfuse) --
this project already runs entirely self-hosted/local via Docker Compose,
and the actual skill being demonstrated here (knowing what to measure and
why: the cost-aware-routing story needs real cost/latency numbers to check
against, not just a docstring claim) doesn't require standing up a second
service. Cost estimation is opt-in via ANTHROPIC_PRICE_PER_MTOK_INPUT/OUTPUT
in .env -- left unset by default rather than hardcoding pricing figures
that go stale.

Observability must never break the pipeline it's observing: a failure to
write telemetry is logged and swallowed, never raised.
"""

from __future__ import annotations

import logging

from modelscout.config import settings
from modelscout.db import get_connection

logger = logging.getLogger(__name__)

_INSERT_SQL = """
INSERT INTO llm_calls (agent_name, model_id, model, input_tokens, output_tokens, latency_ms, estimated_cost_usd)
VALUES (%s, %s, %s, %s, %s, %s, %s)
"""


def estimate_cost_usd(input_tokens: int, output_tokens: int) -> float | None:
    if settings.price_per_mtok_input is None or settings.price_per_mtok_output is None:
        return None
    return (
        input_tokens / 1_000_000 * settings.price_per_mtok_input
        + output_tokens / 1_000_000 * settings.price_per_mtok_output
    )


def record_llm_call(agent_name: str, model_id: str | None, model: str, response, latency_ms: int) -> None:
    usage = getattr(response, "usage", None)
    input_tokens = getattr(usage, "input_tokens", 0) if usage else 0
    output_tokens = getattr(usage, "output_tokens", 0) if usage else 0
    cost = estimate_cost_usd(input_tokens, output_tokens)

    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    _INSERT_SQL,
                    (agent_name, model_id, model, input_tokens, output_tokens, latency_ms, cost),
                )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to record LLM call telemetry (pipeline continues): %s", exc)
