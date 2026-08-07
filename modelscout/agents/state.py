"""LangGraph state schemas.

Two levels: ModelState flows through the per-model fan-out branches
(triage -> extract/skip); PipelineState is the graph-level state that
accumulates results from every branch before Notifier runs.
"""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict


class ModelState(TypedDict):
    model_id: str
    readme_text: str
    downloads: int
    tags: list[str]
    interest_profile: dict

    triage_label: str
    triage_confidence: float
    is_relevant: bool
    raw_labels: dict

    extracted: dict | None
    parse_error: bool


class PipelineState(TypedDict):
    interest_profile: dict
    candidates: list[dict]  # [{"model_id", "readme_text", "downloads"}, ...] from ingestion
    results: Annotated[list[dict], operator.add]
    digest_markdown: str | None
