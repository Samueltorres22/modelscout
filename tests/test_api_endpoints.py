"""Integration tests for the dashboard's read endpoints (modelscout/api/main.py).

Same skip-if-unreachable pattern as test_persist_integration.py: these hit a
real Postgres through modelscout.db.get_connection, so they're skipped rather
than failed when DATABASE_URL isn't reachable, and always run for real in CI
(tests.yml's Postgres service). Seeds its own rows under a "test/" namespaced
model_id (cascades to triage_results/extracted_specs/fact_checks via FK) and
a distinctly-named profile/agent for digest_runs/llm_calls, which have no FK
to models -- cleanup deletes all four explicitly so nothing lingers among the
real demo data.

Uses FastAPI's TestClient (in-process, no server process needed) against the
real `app` object, so these exercise the actual route -> queries.py -> DB
path, not a mocked one.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from modelscout.api.main import app
from modelscout.db import get_connection

TEST_MODEL_ID = "test/api-endpoint-check-do-not-use-as-real-model-id"
TEST_PROFILE = "test_api_endpoint_profile_do_not_use"
TEST_AGENT = "test_api_endpoint_agent_do_not_use"


def _db_available() -> bool:
    try:
        with get_connection() as conn, conn.cursor() as cur:
            cur.execute("SELECT 1")
        return True
    except Exception:  # noqa: BLE001
        return False


pytestmark = pytest.mark.skipif(not _db_available(), reason="Postgres not reachable (DATABASE_URL)")

client = TestClient(app)


@pytest.fixture(autouse=True)
def _cleanup_test_rows():
    yield
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute("DELETE FROM models WHERE model_id = %s", (TEST_MODEL_ID,))
        cur.execute("DELETE FROM digest_runs WHERE profile_name = %s", (TEST_PROFILE,))
        cur.execute("DELETE FROM llm_calls WHERE agent_name = %s", (TEST_AGENT,))


def _seed_model_with_full_pipeline():
    from psycopg.types.json import Jsonb

    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO models (model_id, pipeline_tag, downloads, likes, tags, card_data,
                                 readme_text, matched_profile)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                TEST_MODEL_ID,
                "text-generation",
                42,
                7,
                Jsonb(["test", "text-generation"]),
                Jsonb({"license": "mit"}),
                "test readme",
                TEST_PROFILE,
            ),
        )
        cur.execute(
            """
            INSERT INTO triage_results (model_id, profile_name, is_relevant, confidence, raw_labels)
            VALUES (%s, %s, %s, %s, %s)
            """,
            (TEST_MODEL_ID, TEST_PROFILE, True, 0.91, Jsonb({"relevant": 0.91})),
        )
        cur.execute(
            """
            INSERT INTO extracted_specs (model_id, params_billion, license, architecture_family,
                                          hardware_requirements, quantization_available,
                                          declared_benchmarks, parse_error)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                TEST_MODEL_ID,
                7.0,
                "mit",
                "transformer",
                "1x A100 40GB",
                Jsonb(["int4", "int8"]),
                Jsonb([{"name": "MMLU", "metric": "accuracy", "score": 0.71}]),
                False,
            ),
        )
        cur.execute(
            """
            INSERT INTO fact_checks (model_id, verdict, confidence, flags, consistency_issues,
                                      reasoning, parse_error)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            """,
            (
                TEST_MODEL_ID,
                "plausible",
                0.8,
                Jsonb([]),
                Jsonb([]),
                "benchmark numbers are consistent with reported params",
                False,
            ),
        )


def test_list_models_includes_seeded_row_with_joined_fields():
    _seed_model_with_full_pipeline()

    resp = client.get("/models")
    assert resp.status_code == 200

    rows = {m["model_id"]: m for m in resp.json()["models"]}
    assert TEST_MODEL_ID in rows
    row = rows[TEST_MODEL_ID]
    assert row["is_relevant"] is True
    assert row["fact_check_verdict"] == "plausible"
    assert row["n_benchmarks"] == 1


def test_get_model_detail_includes_full_benchmark_and_reasoning():
    _seed_model_with_full_pipeline()

    resp = client.get(f"/models/{TEST_MODEL_ID}")
    assert resp.status_code == 200

    body = resp.json()
    assert body["model_id"] == TEST_MODEL_ID
    assert body["hf_url"] == f"https://huggingface.co/{TEST_MODEL_ID}"
    assert body["declared_benchmarks"] == [{"name": "MMLU", "metric": "accuracy", "score": 0.71}]
    assert body["fact_check_reasoning"] == "benchmark numbers are consistent with reported params"


def test_get_model_detail_404_for_unknown_model():
    resp = client.get("/models/test/does-not-exist-in-db")
    assert resp.status_code == 404


def test_get_model_detail_handles_model_with_no_downstream_rows():
    # Regression test: a model that never passed triage (or passed triage but
    # had zero declared_benchmarks, so the Fact-Checker never ran -- see
    # process_model_node's gate in graph.py) has no extracted_specs/
    # fact_checks row at all, which means every LEFT JOIN LATERAL column for
    # those tables comes back NULL, not an empty list/array. ModelDetail's
    # list fields (quantization_available, declared_benchmarks,
    # fact_check_flags, fact_check_consistency_issues) don't accept None --
    # found live in the deployed dashboard (a 500 on every model except the
    # one with a real fact-check) before queries.py wrapped those four
    # columns in COALESCE(..., '[]'::jsonb).
    from psycopg.types.json import Jsonb

    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(
            "INSERT INTO models (model_id, pipeline_tag, downloads, likes, tags, card_data, readme_text) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s)",
            (TEST_MODEL_ID, "text-generation", 1, 0, Jsonb([]), Jsonb({}), ""),
        )

    resp = client.get(f"/models/{TEST_MODEL_ID}")
    assert resp.status_code == 200

    body = resp.json()
    assert body["quantization_available"] == []
    assert body["declared_benchmarks"] == []
    assert body["fact_check_flags"] == []
    assert body["fact_check_consistency_issues"] == []


def test_list_runs_includes_seeded_row():
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO digest_runs (profile_name, n_ingested, n_triage_pass, n_extracted,
                                      n_parse_errors, n_fact_checked, n_implausible,
                                      digest_markdown_path)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (TEST_PROFILE, 10, 4, 4, 0, 4, 1, "digests/test-run.md"),
        )

    resp = client.get("/runs")
    assert resp.status_code == 200

    profiles = [r["profile_name"] for r in resp.json()["runs"]]
    assert TEST_PROFILE in profiles


def test_observability_summary_aggregates_seeded_calls():
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO llm_calls (agent_name, model_id, model, input_tokens, output_tokens,
                                    latency_ms, estimated_cost_usd)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            """,
            (TEST_AGENT, TEST_MODEL_ID, "claude-test", 100, 50, 1200, 0.01),
        )
        cur.execute(
            """
            INSERT INTO llm_calls (agent_name, model_id, model, input_tokens, output_tokens,
                                    latency_ms, estimated_cost_usd)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            """,
            (TEST_AGENT, TEST_MODEL_ID, "claude-test", 200, 100, 800, 0.02),
        )

    resp = client.get("/observability/summary")
    assert resp.status_code == 200

    agents = {a["agent_name"]: a for a in resp.json()["agents"]}
    assert TEST_AGENT in agents
    summary = agents[TEST_AGENT]
    assert summary["n_calls"] == 2
    assert summary["total_input_tokens"] == 300
    assert summary["total_output_tokens"] == 150
    assert summary["avg_latency_ms"] == pytest.approx(1000.0)
    assert summary["total_cost_usd"] == pytest.approx(0.03)
