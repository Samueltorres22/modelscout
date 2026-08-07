"""Single orchestrator function shared by cli.py and the FastAPI
POST /pipeline/run endpoint -- deliberately one implementation so both
entrypoints are directly comparable (same counts on the same profile) rather
than two pipelines that can silently drift apart.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

from modelscout.agents.graph import build_graph
from modelscout.config import load_interest_profile
from modelscout.db import get_connection
from modelscout.ingestion.filters import filter_candidates
from modelscout.ingestion.hf_client import search_candidate_models
from modelscout.ingestion.persist import upsert_models
from modelscout.rag.index import index_model_readmes

logger = logging.getLogger(__name__)

_INSERT_TRIAGE_SQL = """
INSERT INTO triage_results (model_id, profile_name, is_relevant, confidence, raw_labels)
VALUES (%s, %s, %s, %s, %s)
"""

_INSERT_EXTRACTED_SQL = """
INSERT INTO extracted_specs (
    model_id, params_billion, license, architecture_family, hardware_requirements,
    quantization_available, declared_benchmarks, parse_error, parse_error_detail,
    raw_model_response
) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
"""

_INSERT_DIGEST_RUN_SQL = """
INSERT INTO digest_runs (
    profile_name, n_ingested, n_triage_pass, n_extracted, n_parse_errors, digest_markdown_path
) VALUES (%s, %s, %s, %s, %s, %s)
"""


def _persist_results(conn, profile_name: str, results: list[dict]) -> None:
    from psycopg.types.json import Jsonb

    with conn.cursor() as cur:
        for r in results:
            cur.execute(
                _INSERT_TRIAGE_SQL,
                (
                    r["model_id"],
                    profile_name,
                    r.get("is_relevant", False),
                    r.get("triage_confidence", 0.0),
                    Jsonb(r.get("raw_labels", {}) or {}),
                ),
            )

            extracted = r.get("extracted")
            if extracted is None:
                continue

            cur.execute(
                _INSERT_EXTRACTED_SQL,
                (
                    r["model_id"],
                    extracted.get("params_billion"),
                    extracted.get("license"),
                    extracted.get("architecture_family"),
                    extracted.get("hardware_requirements"),
                    Jsonb(extracted.get("quantization_available", [])),
                    Jsonb(extracted.get("declared_benchmarks", [])),
                    extracted.get("parse_error", False),
                    extracted.get("parse_error_detail"),
                    Jsonb(extracted.get("raw_model_response"))
                    if extracted.get("raw_model_response") is not None
                    else None,
                ),
            )


def run(profile_name: str, limit_per_tag: int = 20) -> dict:
    profile = load_interest_profile(profile_name)

    logger.info("Ingesting candidates for profile '%s'...", profile.name)
    raw_candidates = search_candidate_models(profile, limit_per_tag=limit_per_tag)
    filtered = filter_candidates(raw_candidates, profile)
    logger.info(
        "Ingested %d raw candidates, %d passed keyword/size filters",
        len(raw_candidates),
        len(filtered),
    )

    with get_connection() as conn:
        upsert_models(conn, filtered, profile.name)
        n_chunks = index_model_readmes(conn, [c.model_id for c in filtered])
        logger.info("Indexed %d RAG chunks", n_chunks)

    graph = build_graph()
    initial_state = {
        "interest_profile": profile.model_dump(),
        "candidates": [
            {
                "model_id": c.model_id,
                "readme_text": c.readme_text,
                "downloads": c.downloads,
                "tags": c.tags,
            }
            for c in filtered
        ],
        "results": [],
        "digest_markdown": None,
    }

    logger.info("Running Triage -> Extractor -> Notifier graph over %d candidates...", len(filtered))
    final_state = graph.invoke(initial_state)
    results = final_state["results"]

    n_triage_pass = sum(1 for r in results if r.get("is_relevant"))
    n_extracted = sum(1 for r in results if r.get("extracted") is not None)
    n_parse_errors = sum(
        1 for r in results if r.get("extracted") and r["extracted"].get("parse_error")
    )
    # This is the core cost-aware-routing claim, checked concretely rather
    # than just asserted in a docstring.
    assert n_extracted == n_triage_pass, (
        f"extractor invocation count ({n_extracted}) must equal triage-pass count "
        f"({n_triage_pass}) -- routing guarantee violated"
    )
    logger.info(
        "%d ingested -> %d passed triage locally ($0) -> %d sent to Claude (%d parse errors)",
        len(filtered),
        n_triage_pass,
        n_extracted,
        n_parse_errors,
    )

    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    digest_path = Path("digests") / f"{date_str}-{profile.name}.md"

    with get_connection() as conn:
        _persist_results(conn, profile.name, results)
        with conn.cursor() as cur:
            cur.execute(
                _INSERT_DIGEST_RUN_SQL,
                (
                    profile.name,
                    len(filtered),
                    n_triage_pass,
                    n_extracted,
                    n_parse_errors,
                    str(digest_path),
                ),
            )

    ranked = sorted(
        results, key=lambda r: (-r.get("triage_confidence", 0.0), -r.get("downloads", 0))
    )
    top_models = [
        {
            "model_id": r["model_id"],
            "is_relevant": r.get("is_relevant", False),
            "triage_confidence": r.get("triage_confidence", 0.0),
            "extracted": r.get("extracted"),
        }
        for r in ranked[: profile.notify.top_n]
    ]

    return {
        "profile": profile.name,
        "counts": {
            "ingested": len(filtered),
            "triage_pass": n_triage_pass,
            "extracted": n_extracted,
            "parse_errors": n_parse_errors,
        },
        "digest_markdown_path": str(digest_path),
        "top_models": top_models,
    }
