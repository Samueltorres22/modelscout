"""Read queries backing the dashboard's new endpoints. Kept separate from
main.py so routing stays thin and these are independently readable/testable.

triage_results/extracted_specs/fact_checks can all have multiple rows per
model_id (re-ingestion is idempotent on `models`, but each pipeline run adds
a fresh row to these three) -- every query here uses a LATERAL join with
ORDER BY ... LIMIT 1 to get the latest row per model, per table. This is the
read-side of the same idempotent-reingestion reality pipeline.py already
handles on the write side.
"""

from __future__ import annotations

_MODELS_CATALOG_SQL = """
SELECT
    m.model_id, m.pipeline_tag, m.downloads, m.likes, m.matched_profile, m.ingested_at,
    t.is_relevant, t.confidence AS triage_confidence,
    e.params_billion, e.license, e.architecture_family, e.parse_error AS extraction_parse_error,
    jsonb_array_length(COALESCE(e.declared_benchmarks, '[]'::jsonb)) AS n_benchmarks,
    f.verdict AS fact_check_verdict, f.confidence AS fact_check_confidence
FROM models m
LEFT JOIN LATERAL (
    SELECT * FROM triage_results tr WHERE tr.model_id = m.model_id ORDER BY tr.created_at DESC LIMIT 1
) t ON true
LEFT JOIN LATERAL (
    SELECT * FROM extracted_specs es WHERE es.model_id = m.model_id ORDER BY es.extracted_at DESC LIMIT 1
) e ON true
LEFT JOIN LATERAL (
    SELECT * FROM fact_checks fc WHERE fc.model_id = m.model_id ORDER BY fc.checked_at DESC LIMIT 1
) f ON true
ORDER BY m.ingested_at DESC
"""

_MODEL_DETAIL_SQL = """
SELECT
    m.model_id, m.pipeline_tag, m.downloads, m.likes, m.tags, m.matched_profile, m.ingested_at,
    t.is_relevant, t.confidence AS triage_confidence,
    e.params_billion, e.license, e.architecture_family, e.hardware_requirements,
    e.quantization_available, e.declared_benchmarks, e.parse_error AS extraction_parse_error,
    f.verdict AS fact_check_verdict, f.confidence AS fact_check_confidence,
    f.flags AS fact_check_flags, f.consistency_issues AS fact_check_consistency_issues,
    f.reasoning AS fact_check_reasoning, f.parse_error AS fact_check_parse_error
FROM models m
LEFT JOIN LATERAL (
    SELECT * FROM triage_results tr WHERE tr.model_id = m.model_id ORDER BY tr.created_at DESC LIMIT 1
) t ON true
LEFT JOIN LATERAL (
    SELECT * FROM extracted_specs es WHERE es.model_id = m.model_id ORDER BY es.extracted_at DESC LIMIT 1
) e ON true
LEFT JOIN LATERAL (
    SELECT * FROM fact_checks fc WHERE fc.model_id = m.model_id ORDER BY fc.checked_at DESC LIMIT 1
) f ON true
WHERE m.model_id = %(model_id)s
"""

_DIGEST_RUNS_SQL = """
SELECT id, profile_name, run_at, n_ingested, n_triage_pass, n_extracted,
       n_parse_errors, n_fact_checked, n_implausible, digest_markdown_path
FROM digest_runs
ORDER BY run_at DESC
LIMIT 50
"""

# Same aggregation as scripts/cost_report.py -- reused, not forked, so the
# dashboard and the CLI report can never silently disagree.
_OBSERVABILITY_SUMMARY_SQL = """
SELECT
    agent_name,
    count(*) AS n_calls,
    sum(input_tokens) AS total_input_tokens,
    sum(output_tokens) AS total_output_tokens,
    avg(latency_ms) AS avg_latency_ms,
    sum(estimated_cost_usd) AS total_cost_usd
FROM llm_calls
GROUP BY agent_name
ORDER BY agent_name
"""


def _rows_to_dicts(cur) -> list[dict]:
    columns = [desc[0] for desc in cur.description]
    return [dict(zip(columns, row, strict=True)) for row in cur.fetchall()]


def get_models_catalog(conn) -> list[dict]:
    with conn.cursor() as cur:
        cur.execute(_MODELS_CATALOG_SQL)
        return _rows_to_dicts(cur)


def get_model_detail(conn, model_id: str) -> dict | None:
    with conn.cursor() as cur:
        cur.execute(_MODEL_DETAIL_SQL, {"model_id": model_id})
        rows = _rows_to_dicts(cur)
    return rows[0] if rows else None


def get_digest_runs(conn) -> list[dict]:
    with conn.cursor() as cur:
        cur.execute(_DIGEST_RUNS_SQL)
        return _rows_to_dicts(cur)


def get_observability_summary(conn) -> list[dict]:
    with conn.cursor() as cur:
        cur.execute(_OBSERVABILITY_SUMMARY_SQL)
        return _rows_to_dicts(cur)
